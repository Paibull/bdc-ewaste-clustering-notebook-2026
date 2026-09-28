"""Jalankan eksperimen semantic-discovery G0 yang dipreregistrasikan."""
from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
LENSES = ("function", "material", "condition")
BACKBONES = (
    "dinov3", "radio", "aimv2", "siglip2",
    "dinov2", "siglip", "convnext", "eva02",
)
SEED = 0
N_ROWS = 4_179
N_HOLDOUT = 600
N_DISCOVERY = 3_579
N_CLUSTERS = 12
N_PERMUTATIONS = 200
OUTPUTS = {
    "metrics": SEMI / "semantic_g0_metrics.csv",
    "partitions": SEMI / "semantic_g0_partitions.npz",
    "profiles": SEMI / "semantic_g0_cluster_profiles.csv",
    "pairwise": SEMI / "semantic_g0_pairwise.csv",
}


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def fit_pca_space(discovery, holdout, n_components):
    """Fit L2 -> randomized PCA -> L2 tanpa melihat holdout."""
    discovery = l2(discovery)
    pca = PCA(
        n_components=n_components,
        whiten=False,
        svd_solver="randomized",
        random_state=SEED,
    ).fit(discovery)
    transformed_discovery = l2(pca.transform(discovery))
    transformed_holdout = l2(pca.transform(l2(holdout)))
    return transformed_discovery, transformed_holdout, pca


def evidence_accumulation(partitions, n_clusters=N_CLUSTERS):
    """Average-linkage atas jarak 1 - fraksi co-assignment."""
    partitions = [np.asarray(labels) for labels in partitions]
    if not partitions or any(labels.shape != partitions[0].shape for labels in partitions):
        raise ValueError("partisi evidence accumulation harus non-kosong dan sejajar")
    similarity = np.zeros((len(partitions[0]), len(partitions[0])), dtype=np.float32)
    for labels in partitions:
        similarity += labels[:, None] == labels[None, :]
    similarity /= len(partitions)
    distances = squareform(1.0 - similarity, checks=False)
    return cut_tree(linkage(distances, method="average"), n_clusters=n_clusters).reshape(-1).astype(np.int32)


def semantic_concentration(labels, scores):
    labels = np.asarray(labels)
    keep = labels != -1
    if not keep.any():
        return np.nan
    total = 0.0
    for cluster in sorted(np.unique(labels[keep])):
        members = labels == cluster
        total += int(members.sum()) * float(scores[members].mean(axis=0).max())
    return total / int(keep.sum())


def semantic_profile(labels, scores, concepts, label_ids, lenses):
    labels = np.asarray(labels)
    rows = []
    for cluster in sorted(label for label in np.unique(labels) if label != -1):
        members = labels == cluster
        means = scores[members].mean(axis=0)
        order = np.argsort(-means, kind="stable")
        top, second = int(order[0]), int(order[1])
        rows.append({
            "cluster": int(cluster),
            "size": int(members.sum()),
            "top_concept": str(concepts[top]),
            "top_label_id": str(label_ids[top]),
            "lens": str(lenses[top]),
            "mean_z": float(means[top]),
            "second_mean_z": float(means[second]),
            "margin": float(means[top] - means[second]),
        })
    non_noise = sum(row["size"] for row in rows)
    if not rows:
        return rows, (np.nan, np.nan, np.nan)
    concentration = sum(row["size"] * row["mean_z"] for row in rows) / non_noise
    margin = sum(row["size"] * row["margin"] for row in rows) / non_noise
    coverage = len({row["top_concept"] for row in rows}) / len(rows)
    return rows, (float(concentration), float(margin), float(coverage))


def semantic_permutation_pvalue(labels, scores, observed, n_permutations=N_PERMUTATIONS):
    rng = np.random.default_rng(SEED)
    null = np.asarray([
        semantic_concentration(rng.permutation(labels), scores)
        for _ in range(n_permutations)
    ])
    if not np.isfinite(observed):
        return 1.0, null
    return float((1 + np.count_nonzero(null >= observed)) / (n_permutations + 1)), null


def benjamini_hochberg(pvalues, alpha=0.05):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues, kind="stable")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    qvalues = np.empty_like(ranked)
    qvalues[order] = np.clip(ranked, 0.0, 1.0)
    return qvalues, qvalues <= alpha


def assign_holdout(discovery, holdout, labels):
    """Assign holdout ke centroid non-noise ternormalisasi dengan cosine distance."""
    clusters = sorted(label for label in np.unique(labels) if label != -1)
    n = len(holdout)
    if len(clusters) < 2:
        return (
            np.full(n, -1, dtype=np.int32),
            np.full(n, np.nan, dtype=np.float32),
            np.full(n, np.nan, dtype=np.float32),
            np.full(n, np.nan, dtype=np.float32),
        )
    centroids = l2(np.stack([discovery[labels == cluster].mean(axis=0) for cluster in clusters]))
    distances = 1.0 - l2(holdout) @ centroids.T
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :2]
    rows = np.arange(n)
    d1 = distances[rows, nearest[:, 0]].astype(np.float32)
    d2 = distances[rows, nearest[:, 1]].astype(np.float32)
    assigned = np.asarray(clusters, dtype=np.int32)[nearest[:, 0]]
    return assigned, d1, d2, (d2 - d1).astype(np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions():
    packages = (
        "numpy", "pandas", "scipy", "scikit-learn", "hdbscan",
        "scikit-learn-extra", "igraph", "leidenalg", "Pillow",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def _require_method_dependencies():
    try:
        import hdbscan
    except Exception as exc:
        raise RuntimeError(f"dependency hdbscan gagal diimpor: {exc}") from exc
    try:
        from sklearn_extra.cluster import KMedoids
    except Exception as exc:
        raise RuntimeError(
            f"dependency scikit-learn-extra gagal diimpor: {exc}. "
            "Pasang build yang kompatibel dengan versi NumPy environment ini."
        ) from exc
    try:
        import igraph
        import leidenalg
    except Exception as exc:
        raise RuntimeError(f"dependency igraph/leidenalg gagal diimpor: {exc}") from exc
    try:
        from interpret_lenses import standardized_scores
    except Exception as exc:
        raise RuntimeError(f"dependency semantic profiling gagal diimpor: {exc}") from exc
    return hdbscan, KMedoids, igraph, leidenalg, standardized_scores


def _validate_inputs(embedding, subset, imgstats, axes, bank_path):
    keys = embedding["key"].astype(str)
    if len(keys) != N_ROWS or len(np.unique(keys)) != N_ROWS:
        raise ValueError(f"embedding harus memiliki {N_ROWS} key unik")
    for backbone in BACKBONES:
        if backbone not in embedding.files:
            raise ValueError(f"embedding backbone {backbone} hilang")
    if len(embedding["split"]) != N_ROWS:
        raise ValueError("split embedding tidak sejajar dengan key")

    if len(subset) != N_HOLDOUT or subset["key"].astype(str).nunique() != N_HOLDOUT:
        raise ValueError(f"subset validasi harus memiliki {N_HOLDOUT} key unik")
    holdout_keys = subset["key"].astype(str)
    holdout_mask = np.isin(keys, holdout_keys)
    if int(holdout_mask.sum()) != N_HOLDOUT:
        raise ValueError("key subset validasi tidak seluruhnya ada di embedding")
    discovery_mask = ~holdout_mask
    if int(discovery_mask.sum()) != N_DISCOVERY:
        raise ValueError(f"discovery harus memiliki {N_DISCOVERY} baris")

    split_by_key = pd.Series(embedding["split"].astype(str), index=keys)
    expected_split = split_by_key.loc[holdout_keys].to_numpy()
    if "split" not in subset or not np.array_equal(subset["split"].astype(str), expected_split):
        raise ValueError("split subset validasi tidak sejajar dengan embedding")

    stats_keys = imgstats["key"].astype(str)
    if len(imgstats) != N_ROWS or stats_keys.nunique() != N_ROWS or set(stats_keys) != set(keys):
        raise ValueError("imgstats harus memiliki key unik yang sama dengan embedding")
    stats = imgstats.assign(key=stats_keys).set_index("key").loc[keys]
    corpus = ((stats["W"].to_numpy() == 150) & (stats["H"].to_numpy() == 150)).astype(np.int8)

    stored_hash = str(np.asarray(axes["bank_sha256"]).item())
    actual_hash = _sha256(bank_path)
    if stored_hash != actual_hash:
        raise ValueError(f"SHA-256 concept bank tidak cocok: axes={stored_hash}, yaml={actual_hash}")
    if sum(len(axes[f"concept_{lens}"]) for lens in LENSES) != 31:
        raise ValueError("concept axes harus memuat tepat 31 anchor")
    return keys, discovery_mask, holdout_mask, corpus, actual_hash


def _fit_spaces(embedding, discovery_mask, holdout_mask):
    spaces = {}
    fused_discovery, fused_holdout = [], []
    for backbone in BACKBONES:
        print(f"[transform] {backbone}", flush=True)
        raw = np.asarray(embedding[backbone], dtype=np.float32)
        if raw.shape[0] != N_ROWS:
            raise ValueError(f"embedding backbone {backbone} tidak sejajar")
        discovery, holdout = raw[discovery_mask], raw[holdout_mask]
        del raw
        single_discovery, single_holdout, _ = fit_pca_space(discovery, holdout, 100)
        block_discovery, block_holdout, _ = fit_pca_space(discovery, holdout, 32)
        spaces[backbone] = (single_discovery, single_holdout)
        fused_discovery.append(block_discovery)
        fused_holdout.append(block_holdout)
    fused_discovery = np.concatenate(fused_discovery, axis=1)
    fused_holdout = np.concatenate(fused_holdout, axis=1)
    spaces["fused"] = fit_pca_space(fused_discovery, fused_holdout, 100)[:2]
    return spaces


def _kmeans_candidate(x):
    start = time.perf_counter()
    runs = [
        KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=seed).fit_predict(x).astype(np.int32)
        for seed in range(5)
    ]
    stability = np.mean([
        adjusted_rand_score(runs[left], runs[right])
        for left, right in itertools.combinations(range(5), 2)
    ])
    return runs[0], float(stability), time.perf_counter() - start


def _leiden_graph(x, igraph):
    neighbors = NearestNeighbors(n_neighbors=31, metric="cosine").fit(x)
    distances, indices = neighbors.kneighbors(x)
    edges = {}
    for source, (row_distances, row_indices) in enumerate(zip(distances, indices)):
        used = 0
        for distance, target in zip(row_distances, row_indices):
            target = int(target)
            if target == source:
                continue
            edge = (min(source, target), max(source, target))
            edges[edge] = max(edges.get(edge, 0.0), max(0.0, 1.0 - float(distance)))
            used += 1
            if used == 30:
                break
    ordered = sorted(edges)
    graph = igraph.Graph(n=len(x), edges=ordered, directed=False)
    return graph, [edges[edge] for edge in ordered]


def _cluster_candidates(spaces, dependencies):
    hdbscan, KMedoids, igraph, leidenalg = dependencies[:4]
    candidates = []
    total = 19

    def record(name, method, representation, parameters, fit):
        print(f"[{len(candidates) + 1}/{total}] {name}", flush=True)
        start = time.perf_counter()
        labels = np.asarray(fit(), dtype=np.int32)
        candidates.append({
            "candidate": name,
            "method": method,
            "representation": representation,
            "parameters": json.dumps(parameters, sort_keys=True, separators=(",", ":")),
            "labels": labels,
            "optimizer_stability_ari": np.nan,
            "runtime_seconds": time.perf_counter() - start,
        })
        print(f"    selesai: {len(np.unique(labels[labels != -1]))} cluster", flush=True)

    for representation in (*BACKBONES, "fused"):
        name = f"kmeans_{representation}"
        print(f"[{len(candidates) + 1}/{total}] {name}", flush=True)
        labels, stability, runtime = _kmeans_candidate(spaces[representation][0])
        candidates.append({
            "candidate": name,
            "method": "kmeans",
            "representation": representation,
            "parameters": json.dumps(
                {"k": 12, "n_init": 20, "stored_seed": 0, "stability_seeds": [0, 1, 2, 3, 4]},
                sort_keys=True, separators=(",", ":"),
            ),
            "labels": labels,
            "optimizer_stability_ari": stability,
            "runtime_seconds": runtime,
        })
        print(f"    selesai: 12 cluster, mean seed ARI={stability:.3f}", flush=True)

    fused = spaces["fused"][0]
    record(
        "ward_fused", "ward", "fused", {"k": 12, "linkage": "ward"},
        lambda: AgglomerativeClustering(n_clusters=12, linkage="ward").fit_predict(fused),
    )
    record(
        "gmm_diag_fused", "gmm_diag", "fused",
        {"k": 12, "covariance_type": "diag", "n_init": 5, "seed": 0},
        lambda: GaussianMixture(
            n_components=12, covariance_type="diag", n_init=5, random_state=0,
        ).fit_predict(fused),
    )
    record(
        "kmedoids_pam_fused", "kmedoids_pam", "fused",
        {"k": 12, "metric": "euclidean", "method": "pam", "seed": 0},
        lambda: KMedoids(
            n_clusters=12, metric="euclidean", method="pam", random_state=0,
        ).fit_predict(fused),
    )
    for min_cluster_size in (40, 80, 120):
        record(
            f"hdbscan_fused_mcs_{min_cluster_size}", "hdbscan", "fused",
            {"min_cluster_size": min_cluster_size, "min_samples": 10},
            lambda size=min_cluster_size: hdbscan.HDBSCAN(
                min_cluster_size=size, min_samples=10,
            ).fit_predict(fused),
        )

    graph_start = time.perf_counter()
    graph, weights = _leiden_graph(fused, igraph)
    graph_runtime = time.perf_counter() - graph_start
    for resolution in (0.5, 1.0, 1.5):
        name_resolution = str(resolution).replace(".", "_")
        record(
            f"leiden_fused_res_{name_resolution}", "leiden", "fused",
            {"knn_k": 30, "metric": "cosine", "resolution": resolution, "seed": 0},
            lambda value=resolution: leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=value,
                seed=0,
            ).membership,
        )
        candidates[-1]["runtime_seconds"] += graph_runtime

    backbone_partitions = [candidates[index]["labels"] for index in range(len(BACKBONES))]
    record(
        "evidence_accumulation", "evidence_accumulation", "fused",
        {"sources": list(BACKBONES), "distance": "1-coassignment", "linkage": "average", "k": 12},
        lambda: evidence_accumulation(backbone_partitions),
    )
    return candidates


def _candidate_metrics(candidate, x, corpus, support_partitions, hdbscan):
    labels = candidate["labels"]
    non_noise = labels != -1
    clusters, sizes = np.unique(labels[non_noise], return_counts=True)
    has_noise = not non_noise.all()
    entropy = 0.0
    if len(clusters) > 1:
        proportions = sizes / sizes.sum()
        entropy = float(-(proportions * np.log(proportions)).sum() / np.log(len(clusters)))
    valid_partition = len(clusters) > 1 and len(clusters) < len(labels)
    centroid_metrics_allowed = candidate["method"] != "hdbscan" and not has_noise and valid_partition
    row = {
        key: candidate[key]
        for key in ("candidate", "method", "representation", "parameters", "runtime_seconds")
    }
    row.update({
        "n_clusters_non_noise": int(len(clusters)),
        "min_cluster_size_non_noise": int(sizes.min()) if len(sizes) else 0,
        "normalized_cluster_size_entropy": entropy if len(sizes) else np.nan,
        "noise_fraction": float((labels == -1).mean()),
        "silhouette": float(silhouette_score(x, labels)) if centroid_metrics_allowed else np.nan,
        "calinski_harabasz": float(calinski_harabasz_score(x, labels)) if centroid_metrics_allowed else np.nan,
        "davies_bouldin": float(davies_bouldin_score(x, labels)) if centroid_metrics_allowed else np.nan,
        "dbcv": (
            float(hdbscan.validity.validity_index(np.asarray(x, np.float64), labels))
            if candidate["method"] == "hdbscan" and valid_partition else np.nan
        ),
        "ami_corpus": float(adjusted_mutual_info_score(corpus, labels)),
        "nmi_corpus": float(normalized_mutual_info_score(corpus, labels)),
        "optimizer_stability_ari": candidate["optimizer_stability_ari"],
    })
    support = []
    for backbone, reference in zip(BACKBONES, support_partitions):
        value = float(adjusted_mutual_info_score(reference, labels))
        row[f"ami_support_{backbone}"] = value
        support.append(value)
    row["mean_cross_backbone_support"] = float(np.mean(support))
    return row


def main():
    embedding_path = SEMI / "ewaste_emb_harm_resjpeg.npz"
    subset_path = SEMI / "subset_validasi.csv"
    imgstats_path = SEMI / "imgstats.csv"
    axes_path = SEMI / "concept_axes.npz"
    bank_path = SEMI / "concept_bank.yaml"
    for path in (embedding_path, subset_path, imgstats_path, axes_path, bank_path):
        if not path.exists():
            raise FileNotFoundError(f"input wajib tidak ditemukan: {path}")

    embedding = np.load(embedding_path, allow_pickle=False)
    subset = pd.read_csv(subset_path)
    imgstats = pd.read_csv(imgstats_path)
    axes = np.load(axes_path, allow_pickle=False)
    keys, discovery_mask, holdout_mask, corpus, bank_sha256 = _validate_inputs(
        embedding, subset, imgstats, axes, bank_path,
    )
    dependencies = _require_method_dependencies()
    versions = _package_versions()
    versions_json = json.dumps(versions, sort_keys=True, separators=(",", ":"))
    print(
        f"Input valid: {len(keys)} citra, {discovery_mask.sum()} discovery, "
        f"{holdout_mask.sum()} holdout, concept_bank_sha256={bank_sha256}",
        flush=True,
    )

    spaces = _fit_spaces(embedding, discovery_mask, holdout_mask)
    candidates = _cluster_candidates(spaces, dependencies)
    hdbscan = dependencies[0]
    support_partitions = [candidate["labels"] for candidate in candidates[:len(BACKBONES)]]

    print("[semantic] menghitung standardized scores setelah seluruh clustering", flush=True)
    standardized_scores = dependencies[4]
    image = np.asarray(embedding["siglip2"], dtype=np.float32)[discovery_mask]
    score_sets = {
        lens: standardized_scores(image, np.asarray(axes[f"anchor_{lens}"], dtype=np.float32))
        for lens in LENSES
    }
    scores = np.concatenate([score_sets[lens] for lens in LENSES], axis=1)
    concepts = np.concatenate([axes[f"concept_{lens}"].astype(str) for lens in LENSES])
    label_ids = np.concatenate([axes[f"label_id_{lens}"].astype(str) for lens in LENSES])
    concept_lenses = np.concatenate([
        np.repeat(lens, len(axes[f"concept_{lens}"])) for lens in LENSES
    ])

    metrics, profiles = [], []
    partition_payload = {
        "keys": keys,
        "split": embedding["split"].astype(str),
        "discovery_mask": discovery_mask,
        "holdout_mask": holdout_mask,
        "candidate_names": np.asarray([candidate["candidate"] for candidate in candidates]),
    }
    pvalues = []
    for candidate in candidates:
        name = candidate["candidate"]
        discovery, holdout = spaces[candidate["representation"]]
        row = _candidate_metrics(candidate, discovery, corpus[discovery_mask], support_partitions, hdbscan)
        candidate_profiles, semantic = semantic_profile(
            candidate["labels"], scores, concepts, label_ids, concept_lenses,
        )
        concentration, margin, coverage = semantic
        pvalue, _ = semantic_permutation_pvalue(candidate["labels"], scores, concentration)
        row.update({
            "semantic_concentration": concentration,
            "semantic_mean_margin": margin,
            "semantic_name_coverage": coverage,
            "semantic_permutation_pvalue": pvalue,
            "concept_bank_sha256": bank_sha256,
            "package_versions": versions_json,
        })
        metrics.append(row)
        pvalues.append(pvalue)
        for profile in candidate_profiles:
            profiles.append({"candidate": name, **profile})

        assigned, d1, d2, assignment_margin = assign_holdout(
            discovery, holdout, candidate["labels"],
        )
        full_labels = np.full(len(keys), -1, dtype=np.int32)
        full_labels[discovery_mask] = candidate["labels"]
        full_labels[holdout_mask] = assigned
        full_d1 = np.full(len(keys), np.nan, dtype=np.float32)
        full_d2 = np.full(len(keys), np.nan, dtype=np.float32)
        full_margin = np.full(len(keys), np.nan, dtype=np.float32)
        full_d1[holdout_mask], full_d2[holdout_mask], full_margin[holdout_mask] = d1, d2, assignment_margin
        partition_payload[f"labels__{name}"] = full_labels
        partition_payload[f"holdout_nearest_distance__{name}"] = full_d1
        partition_payload[f"holdout_second_distance__{name}"] = full_d2
        partition_payload[f"holdout_margin__{name}"] = full_margin

    qvalues, rejected = benjamini_hochberg(pvalues)
    for row, qvalue, reject in zip(metrics, qvalues, rejected):
        row["semantic_bh_qvalue"] = float(qvalue)
        row["semantic_bh_reject_0_05"] = bool(reject)

    pairwise = []
    for left, right in itertools.combinations(candidates, 2):
        pairwise.append({
            "candidate_left": left["candidate"],
            "candidate_right": right["candidate"],
            "ami": float(adjusted_mutual_info_score(left["labels"], right["labels"])),
            "ari": float(adjusted_rand_score(left["labels"], right["labels"])),
        })

    pd.DataFrame(metrics).to_csv(OUTPUTS["metrics"], index=False)
    np.savez_compressed(OUTPUTS["partitions"], **partition_payload)
    pd.DataFrame(profiles).to_csv(OUTPUTS["profiles"], index=False)
    pd.DataFrame(pairwise).to_csv(OUTPUTS["pairwise"], index=False)

    print("Output semantic G0:")
    for path in OUTPUTS.values():
        print(f"- {path.relative_to(ROOT)}: sha256={_sha256(path)}")
    axes.close()
    embedding.close()


if __name__ == "__main__":
    main()
