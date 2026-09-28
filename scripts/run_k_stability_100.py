"""Local 100-seed and chooseR stability study for fused 8-view K-Means."""
from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_samples,
)
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
ART = SEMI / "artifacts"
RESULTS = SEMI / "results"
SWEEP = RESULTS / "k_resolution_sweep" / "20260927"
OUT = RESULTS / "k_stability_100"
BACKBONES = ("dinov3", "radio", "aimv2", "siglip2", "dinov2", "siglip", "convnext", "eva02")
KS = (12, 13, 14, 15, 16)
N_SEEDS = 100
N_SUBSAMPLES = 100
SUBSAMPLE_SEED = 20260927
SUBSAMPLE_KMEANS_SEED = 42
N_BOOTSTRAP = 25_000


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def fit_fused(views: dict[str, np.ndarray], rows: np.ndarray | None = None) -> np.ndarray:
    """Same randomized PCA32-per-view then PCA100 fusion as the existing runner."""
    blocks = []
    for name in BACKBONES:
        x = views[name] if rows is None else views[name][rows]
        x = l2(x)
        blocks.append(PCA(n_components=32, svd_solver="randomized", whiten=False,
                          random_state=0).fit_transform(x).astype(np.float32, copy=False))
    joined = l2(np.concatenate(blocks, axis=1))
    return l2(PCA(n_components=100, svd_solver="randomized", whiten=False,
                  random_state=0).fit_transform(joined).astype(np.float32, copy=False))


def load_inputs():
    emb = np.load(ART / "ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    g0 = np.load(ART / "semantic_g0_partitions.npz", allow_pickle=False)
    keys = emb["key"].astype(str)
    if len(keys) != 4179 or len(np.unique(keys)) != 4179:
        raise ValueError("expected 4,179 unique canonical keys")
    if not np.array_equal(g0["keys"].astype(str), keys):
        raise ValueError("embedding and G0 key order differs")
    discovery = g0["discovery_mask"].astype(bool)
    holdout = g0["holdout_mask"].astype(bool)
    if discovery.sum() != 3579 or holdout.sum() != 600 or np.any(discovery & holdout):
        raise ValueError("locked discovery/holdout masks are invalid")
    if not np.all(discovery | holdout):
        raise ValueError("locked masks do not cover all rows")
    views = {}
    for name in BACKBONES:
        x = np.asarray(emb[name], dtype=np.float32)
        if x.shape[0] != len(keys) or not np.isfinite(x).all():
            raise ValueError(f"invalid embedding array {name}: {x.shape}")
        views[name] = x[discovery]
    discovery_keys = keys[discovery]
    del emb

    ref_path = SWEEP / "remote" / "classic" / "kmeans_fused" / "k_14" / "seed_42" / "labels.npz"
    ref = np.load(ref_path, allow_pickle=False)
    reference_labels = np.asarray(ref["labels"], dtype=np.int32)
    if "key" in ref.files and not np.array_equal(ref["key"].astype(str), discovery_keys):
        raise ValueError("stored K=14 assignment key order differs")
    if reference_labels.shape != (3579,) or np.unique(reference_labels).size != 14:
        raise ValueError("stored K=14 seed-42 assignment is malformed")

    stats_path = RESULTS / "imgstats.csv"
    with stats_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {r["key"]: r for r in csv.DictReader(stream)}
    if set(discovery_keys) - rows.keys():
        raise ValueError("imgstats.csv is missing discovery keys")
    source = np.asarray([
        max(float(rows[k]["W"]), float(rows[k]["H"])) > 150 for k in discovery_keys
    ], dtype=np.int8)
    return discovery_keys, views, reference_labels, source


def quantiles(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "q2_5": float(np.quantile(x, .025)),
        "median": float(np.median(x)),
        "mean": float(np.mean(x)),
        "q97_5": float(np.quantile(x, .975)),
        "minimum": float(np.min(x)),
    }


def atomic_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def full_stability(x: np.ndarray, source: np.ndarray, discovery_keys: np.ndarray):
    distance = pairwise_distances(x, metric="euclidean", n_jobs=1).astype(np.float32, copy=False)
    np.fill_diagonal(distance, 0.0)
    labels_by_k, medoids, ari_matrices, ami_matrices = [], [], [], []
    rows, summaries = [], {}
    rng = np.random.default_rng(20260928)
    checkpoint = OUT / "assignments.npz"
    saved_labels = np.full((len(KS), N_SEEDS, len(discovery_keys)), -1, dtype=np.int8)
    done = np.zeros(len(KS), dtype=bool)
    if checkpoint.exists():
        saved = np.load(checkpoint, allow_pickle=False)
        if (not np.array_equal(saved["discovery_keys"].astype(str), discovery_keys)
                or not np.array_equal(saved["k_values"], np.asarray(KS, dtype=np.int8))):
            raise ValueError("existing assignments.npz is not this locked experiment; refusing to overwrite")
        if "optimizer_labels" in saved.files and saved["optimizer_labels"].shape == saved_labels.shape:
            saved_labels[:] = saved["optimizer_labels"]
        if "optimizer_done" in saved.files:
            done[:] = saved["optimizer_done"].astype(bool)
        elif np.all(saved_labels >= 0):
            done[:] = True
    for ki, k in enumerate(KS):
        started = time.perf_counter()
        if done[ki]:
            labels = saved_labels[ki].copy()
            print(f"OPTIMIZER_RESUME k={k} seeds=100/100 from assignments.npz", flush=True)
        else:
            labels = np.empty((N_SEEDS, len(discovery_keys)), dtype=np.int8)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(
                    KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict, x
                ): seed for seed in range(N_SEEDS)}
                completed = 0
                for future in as_completed(futures):
                    seed = futures[future]
                    labels[seed] = future.result().astype(np.int8)
                    completed += 1
                    if completed % 20 == 0 or completed == N_SEEDS:
                        print(f"OPTIMIZER_SEEDS k={k} completed={completed}/100", flush=True)
        if not np.all([np.unique(y).size == k for y in labels]):
            raise AssertionError(f"KMeans collapsed for K={k}")
        ari = np.eye(N_SEEDS, dtype=np.float32)
        ami = np.eye(N_SEEDS, dtype=np.float32)
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                ari[i, j] = ari[j, i] = adjusted_rand_score(labels[i], labels[j])
                ami[i, j] = ami[j, i] = adjusted_mutual_info_score(labels[i], labels[j])
        means = (ari.sum(axis=1) - 1.0) / (N_SEEDS - 1)
        medoid_i = int(np.argmax(means))
        medoid = labels[medoid_i]
        ari_ref = np.asarray([adjusted_rand_score(y, medoid) for y in labels])
        ami_ref = np.asarray([adjusted_mutual_info_score(y, medoid) for y in labels])
        run_metrics = []
        for seed, y in enumerate(labels):
            sil = float(silhouette_samples(distance, y, metric="precomputed").mean())
            sizes = np.bincount(y, minlength=k)
            record = {
                "k": k, "seed": seed, "is_medoid": seed == medoid_i,
                "ari_to_medoid": float(ari_ref[seed]), "ami_to_medoid": float(ami_ref[seed]),
                "silhouette": sil, "davies_bouldin": float(davies_bouldin_score(x, y)),
                "calinski_harabasz": float(calinski_harabasz_score(x, y)),
                "source_nmi": float(normalized_mutual_info_score(source, y)),
                "min_cluster_size": int(sizes.min()), "max_cluster_size": int(sizes.max()),
            }
            run_metrics.append(record)
        observed = float(np.median(ari_ref))
        null = np.empty(100, dtype=np.float64)
        for i in range(len(null)):
            null[i] = adjusted_rand_score(medoid, rng.permutation(medoid))
        p_null = (1 + int(np.sum(null >= observed))) / (len(null) + 1)
        ari_stats, ami_stats = quantiles(ari_ref), quantiles(ami_ref)
        mean = float(np.mean(ari_ref))
        margin = float(student_t.ppf(.975, N_SEEDS - 1) * np.std(ari_ref, ddof=1) / np.sqrt(N_SEEDS))
        summary = {
            "k": k, "medoid_seed": medoid_i, "pairwise_ari_mean": float((ari.sum() - N_SEEDS) / (N_SEEDS*(N_SEEDS-1))),
            "pairwise_ami_mean": float((ami.sum() - N_SEEDS) / (N_SEEDS*(N_SEEDS-1))),
            "ari_to_medoid": ari_stats, "ami_to_medoid": ami_stats,
            "ari_to_medoid_mean_ci95": [mean - margin, mean + margin],
            "ari_ge_0_90_pct": float(np.mean(ari_ref >= .90)),
            "ari_ge_0_95_pct": float(np.mean(ari_ref >= .95)),
            "ari_ge_0_99_pct": float(np.mean(ari_ref >= .99)),
            "null_ari_to_medoid": quantiles(null), "null_empirical_p_vs_observed_median": p_null,
            "silhouette": quantiles(np.asarray([r["silhouette"] for r in run_metrics])),
            "davies_bouldin": quantiles(np.asarray([r["davies_bouldin"] for r in run_metrics])),
            "calinski_harabasz": quantiles(np.asarray([r["calinski_harabasz"] for r in run_metrics])),
            "source_nmi": quantiles(np.asarray([r["source_nmi"] for r in run_metrics])),
            "min_cluster_size": quantiles(np.asarray([r["min_cluster_size"] for r in run_metrics])),
            "max_cluster_size": quantiles(np.asarray([r["max_cluster_size"] for r in run_metrics])),
            "runtime_seconds": time.perf_counter() - started,
        }
        for r in run_metrics:
            r["pairwise_ari_mean"] = summary["pairwise_ari_mean"]
            r["pairwise_ami_mean"] = summary["pairwise_ami_mean"]
            r["null_empirical_p_vs_observed_median"] = p_null
            rows.append(r)
        labels_by_k.append(labels)
        medoids.append(medoid)
        ari_matrices.append(ari)
        ami_matrices.append(ami)
        summaries[k] = summary
        if not done[ki]:
            saved_labels[ki] = labels
            done[ki] = True
            atomic_npz(checkpoint, discovery_keys=discovery_keys, k_values=np.asarray(KS, dtype=np.int8),
                       optimizer_labels=saved_labels, optimizer_done=done, parity_ari=np.asarray(1.0))
        print(f"OPTIMIZER_DONE k={k} seeds=100/100 medoid_seed={medoid_i} mean_ARI={mean:.6f} elapsed={summary['runtime_seconds']:.1f}s", flush=True)
    return (np.stack(labels_by_k), np.stack(medoids), np.stack(ari_matrices),
            np.stack(ami_matrices), rows, summaries)


def fit_subsample(views: dict[str, np.ndarray], ids: np.ndarray, position: int):
    x = fit_fused(views, ids)
    labels = np.stack([
        KMeans(n_clusters=k, n_init=20, random_state=SUBSAMPLE_KMEANS_SEED).fit_predict(x).astype(np.int8)
        for k in KS
    ])
    return position, ids.astype(np.int16), labels


def cluster_medians_and_stability(subsets, sublabels, medoids, semantic_names):
    n = len(medoids[0])
    denominator = np.zeros((n, n), dtype=np.uint8)
    for ids in subsets:
        denominator[np.ix_(ids, ids)] += 1
    if np.any(denominator == 0):
        raise AssertionError("some row pairs were never co-sampled")
    sil_by_k, cluster_rows = [], []
    for ki, k in enumerate(KS):
        numerator = np.zeros((n, n), dtype=np.uint8)
        for ids, run_labels in zip(subsets, sublabels[ki]):
            for cluster in np.unique(run_labels):
                members = ids[run_labels == cluster]
                numerator[np.ix_(members, members)] += 1
        frequency = numerator.astype(np.float32)
        np.divide(frequency, denominator, out=frequency)
        distance = 1.0 - frequency
        np.fill_diagonal(distance, 0.0)
        medoid = medoids[ki].astype(np.int32)
        sil = silhouette_samples(distance, medoid, metric="precomputed")
        sil_by_k.append(sil.astype(np.float32))
        del frequency, distance, numerator

        jaccards = [[] for _ in range(k)]
        split_flags = [[] for _ in range(k)]
        merge_flags = [[] for _ in range(k)]
        for ids, run_labels in zip(subsets, sublabels[ki]):
            ref = medoid[ids]
            overlap = np.bincount(ref * k + run_labels, minlength=k * k).reshape(k, k)
            rr, cc = linear_sum_assignment(-overlap)
            match = dict(zip(rr.tolist(), cc.tolist()))
            for ref_cluster in range(k):
                sub_cluster = match[ref_cluster]
                inter = int(overlap[ref_cluster, sub_cluster])
                ref_count = int(np.sum(ref == ref_cluster))
                sub_count = int(np.sum(run_labels == sub_cluster))
                union = ref_count + sub_count - inter
                jaccards[ref_cluster].append(inter / union if union else 0.0)
                row_total = int(overlap[ref_cluster].sum())
                col_total = int(overlap[:, sub_cluster].sum())
                split_flags[ref_cluster].append(1.0 - inter / row_total if row_total else 0.0)
                merge_flags[ref_cluster].append(1.0 - inter / col_total if col_total else 0.0)
        names = semantic_names.get(k, {})
        for cluster in range(k):
            vals = np.asarray(jaccards[cluster])
            row = {
                "k": k, "cluster": cluster, "semantic_node": names.get(cluster, ""),
                **{f"jaccard_{key}": value for key, value in quantiles(vals).items()},
                "fraction_jaccard_below_0_60": float(np.mean(vals < .60)),
                "fraction_consistently_split": float(np.mean(np.asarray(split_flags[cluster]) > .20)),
                "fraction_consistently_merged": float(np.mean(np.asarray(merge_flags[cluster]) > .20)),
            }
            if row["fraction_jaccard_below_0_60"] > .50:
                row["stability_status"] = "unstable_or_dissolved"
            elif row["jaccard_q2_5"] >= .60:
                row["stability_status"] = "stable"
            else:
                row["stability_status"] = "mixed"
            cluster_rows.append(row)
        print(f"CONSENSUS_DONE k={k} rows={n} co-samples=100 silhouette_median={float(np.median(sil)):.5f}", flush=True)

    # Shared vectorized bootstrap samples give paired CIs across K.
    rng = np.random.default_rng(20260929)
    boot = np.empty((len(KS), N_BOOTSTRAP), dtype=np.float32)
    chunk = 250
    n = len(sil_by_k[0])
    for start in range(0, N_BOOTSTRAP, chunk):
        stop = min(start + chunk, N_BOOTSTRAP)
        draws = rng.integers(0, n, size=(stop - start, n), dtype=np.int32)
        for ki, values in enumerate(sil_by_k):
            boot[ki, start:stop] = np.median(values[draws], axis=1)
        if stop % 5000 == 0:
            print(f"BOOTSTRAP_DONE {stop}/{N_BOOTSTRAP}", flush=True)

    lower = np.quantile(boot, .025, axis=1)
    upper = np.quantile(boot, .975, axis=1)
    threshold = float(np.max(lower))
    chooser = []
    for ki, k in enumerate(KS):
        medoid = medoids[ki]
        cluster_medians = np.asarray([np.median(sil_by_k[ki][medoid == c]) for c in range(k)])
        pass_count = int(np.sum(cluster_medians > threshold))
        chooser.append({
            "k": k, "consensus_silhouette_median": float(np.median(sil_by_k[ki])),
            "bootstrap_ci95_low": float(lower[ki]), "bootstrap_ci95_high": float(upper[ki]),
            "chooseR_threshold": threshold, "clusters_above_threshold": pass_count,
            "cluster_median_silhouettes": json.dumps(cluster_medians.tolist(), separators=(",", ":")),
        })
    max_pass = max(row["clusters_above_threshold"] for row in chooser)
    winners = [row["k"] for row in chooser if row["clusters_above_threshold"] == max_pass]
    selected = min(winners)  # conservative deterministic tie-break
    for row in chooser:
        row["chooseR_winner"] = row["k"] == selected
        row["chooseR_supported_tie"] = row["k"] in winners

    pairs = {}
    for a in range(len(KS)):
        for b in range(a + 1, len(KS)):
            diff = boot[a] - boot[b]
            pairs[f"{KS[a]}_minus_{KS[b]}"] = {
                "median_difference": float(np.median(sil_by_k[a]) - np.median(sil_by_k[b])),
                "paired_bootstrap_ci95": [float(np.quantile(diff, .025)), float(np.quantile(diff, .975))],
                "distinguishable": bool(np.quantile(diff, .025) > 0 or np.quantile(diff, .975) < 0),
            }
    return chooser, cluster_rows, selected, winners, pairs


def semantic_names_by_k(full_labels, seed42_labels):
    path = SWEEP / "semantic_nodes.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for k, medoid, ref in zip(KS, full_labels, seed42_labels):
        candidate = f"classic::kmeans_fused::k{k}"
        table = {int(r["cluster"]): r["top_concept"] for r in rows if r["candidate"] == candidate}
        overlap = np.bincount(medoid.astype(np.int32) * k + ref.astype(np.int32), minlength=k * k).reshape(k, k)
        rr, cc = linear_sum_assignment(-overlap)
        result[k] = {int(a): table.get(int(b), "") for a, b in zip(rr, cc)}
    return result


def main(smoke: bool = False):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("PHASE1 loading locked discovery inputs", flush=True)
    keys, views, reference, source = load_inputs()
    x = fit_fused(views)
    if x.shape != (3579, 100) or not np.isfinite(x).all():
        raise AssertionError(f"bad fused space shape/values: {x.shape}")
    check = KMeans(n_clusters=14, n_init=20, random_state=42).fit_predict(x)
    parity_ari = float(adjusted_rand_score(reference, check))
    print(f"PHASE1_PARITY rows=3579 dims=100 ari={parity_ari:.6f}", flush=True)
    if parity_ari != 1.0:
        raise RuntimeError("required fused K=14 seed-42 reconstruction parity is not ARI=1.000; stopped")
    if smoke:
        print("SMOKE_OK", flush=True)
        return

    with threadpool_limits(limits=1):
        print("PHASE2 optimizer stability: 5 K x 100 seeds", flush=True)
        full_labels, medoids, ari_mats, ami_mats, optimizer_rows, optimizer_summary = full_stability(x, source, keys)
        seed42 = full_labels[:, 42, :]
        node_names = semantic_names_by_k(medoids, seed42)

        subset_size = int(.8 * len(keys))
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        subsets = np.stack([rng.choice(len(keys), subset_size, replace=False) for _ in range(N_SUBSAMPLES)])
        sublabels = np.full((len(KS), N_SUBSAMPLES, subset_size), -1, dtype=np.int8)
        print("PHASE3 chooseR: 100 shared 80% subsets; at most 8 workers", flush=True)
        sub_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(fit_subsample, views, subsets[i], i) for i in range(N_SUBSAMPLES)]
            done = 0
            for future in as_completed(futures):
                position, ids, labels = future.result()
                if not np.array_equal(ids, subsets[position]):
                    raise AssertionError("subsample row order changed in worker")
                sublabels[:, position, :] = labels
                done += 1
                if done % 10 == 0 or done == N_SUBSAMPLES:
                    print(f"SUBSAMPLE_DONE {done}/{N_SUBSAMPLES} elapsed={time.perf_counter()-sub_started:.1f}s", flush=True)

        chooser_rows, cluster_rows, chooseR_k, chooseR_ties, pairwise_ci = cluster_medians_and_stability(
            subsets, sublabels, medoids, node_names
        )

        optimizer_winner = max(KS, key=lambda k: optimizer_summary[k]["ari_to_medoid"]["mean"])
        winner_pair = pairwise_ci.get(f"{chooseR_k}_minus_{14 if chooseR_k != 14 else 15}")
        chooseR_single = len(chooseR_ties) == 1
        neighbor_14_15 = pairwise_ci["14_minus_15"]
        phase4_needed = (not chooseR_single or not neighbor_14_15["distinguishable"]
                         or (chooseR_k != optimizer_winner))
        ps_rows = []
        phase4_reason = "not needed: unique chooseR winner; K14/K15 separated; optimizer winner agrees"
        if phase4_needed:
            phase4_reason = "triggered by a non-unique chooseR result, indistinguishable K14/K15, or optimizer disagreement"
            print("PHASE4 triggered: chooseR/14-vs-15/optimizer requires resolution check; 50 splits", flush=True)
            ps_rows = prediction_strength(views, keys)
        else:
            print("PHASE4 skipped: chooseR is unique, K14/K15 distinguished, no optimizer conflict", flush=True)

    # Preserve only requested compact result artifacts.
    import pandas as pd
    pd.DataFrame(optimizer_rows).to_csv(OUT / "optimizer_stability.csv", index=False)
    pd.DataFrame(chooser_rows).to_csv(OUT / "chooser_summary.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "cluster_stability.csv", index=False)
    if ps_rows:
        pd.DataFrame(ps_rows).to_csv(OUT / "prediction_strength.csv", index=False)
    atomic_npz(
        OUT / "assignments.npz", discovery_keys=keys, k_values=np.asarray(KS, dtype=np.int8),
        optimizer_labels=full_labels, medoid_labels=medoids,
        pairwise_ari=ari_mats, pairwise_ami=ami_mats,
        subsample_indices=subsets.astype(np.int16), subsample_labels=sublabels,
        parity_ari=np.asarray(parity_ari),
    )
    all_pairs_indistinguishable = all(not pair["distinguishable"] for pair in pairwise_ci.values())
    if all_pairs_indistinguishable or (len(chooseR_ties) > 1 and 14 in chooseR_ties):
        decision = "B_MULTI_RESOLUTION_PLATEAU_12_16"
    elif chooseR_single and chooseR_k == 14:
        decision = "A_K14_STABLE_BEST"
    else:
        decision = f"C_REPLACE_K14_WITH_K{chooseR_k}"
    summary = {
        "decision": decision, "decision_basis": "chooseR and stability only; semantic metrics were not used to choose K",
        "phase4_reason": phase4_reason, "parity_ari_k14_seed42": parity_ari,
        "n_discovery": len(keys), "k_values": list(KS), "optimizer_seed_count": N_SEEDS,
        "chooseR_subsample_count": N_SUBSAMPLES, "subsample_fraction": .8,
        "subsample_seed": SUBSAMPLE_SEED, "chooseR_kmeans_seed": SUBSAMPLE_KMEANS_SEED,
        "chooseR_selected_k": chooseR_k, "chooseR_supported_ties": chooseR_ties,
        "optimizer_stability_winner_by_mean_ari_to_medoid": optimizer_winner,
        "paired_consensus_silhouette_cis": pairwise_ci,
        "prediction_strength_ran": bool(ps_rows), "prediction_strength": ps_rows,
        "optimizer_summary": optimizer_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"COMPLETE elapsed={summary['runtime_seconds']:.1f}s output={OUT}", flush=True)


def prediction_strength(views: dict[str, np.ndarray], keys: np.ndarray):
    """50 split prediction-strength diagnostic in the train-fitted fused space."""
    from sklearn.metrics import pairwise_distances_argmin

    rng = np.random.default_rng(20260930)
    n = len(keys)
    out = {k: np.empty(50, dtype=np.float32) for k in KS}
    started = time.perf_counter()
    for rep in range(50):
        perm = rng.permutation(n)
        train_ids, test_ids = perm[:n // 2], perm[n // 2:]
        blocks_train, blocks_test = [], []
        for name in BACKBONES:
            train = l2(views[name][train_ids])
            test = l2(views[name][test_ids])
            pca = PCA(n_components=32, svd_solver="randomized", whiten=False, random_state=0).fit(train)
            blocks_train.append(pca.transform(train).astype(np.float32, copy=False))
            blocks_test.append(pca.transform(test).astype(np.float32, copy=False))
        fused_train = l2(np.concatenate(blocks_train, axis=1))
        fused_test = l2(np.concatenate(blocks_test, axis=1))
        pca_fused = PCA(n_components=100, svd_solver="randomized", whiten=False, random_state=0).fit(fused_train)
        fused_train = l2(pca_fused.transform(fused_train).astype(np.float32, copy=False))
        fused_test = l2(pca_fused.transform(fused_test).astype(np.float32, copy=False))
        for k in KS:
            train_model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(fused_train)
            test_labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(fused_test)
            predicted = pairwise_distances_argmin(fused_test, train_model.cluster_centers_, metric="euclidean")
            per_cluster = []
            for c in range(k):
                idx = np.flatnonzero(test_labels == c)
                if len(idx) < 2:
                    continue
                counts = np.bincount(predicted[idx], minlength=k)
                pairs_same = int(np.sum(counts * (counts - 1)) // 2)
                total = len(idx) * (len(idx) - 1) // 2
                per_cluster.append(pairs_same / total)
            out[k][rep] = min(per_cluster) if per_cluster else 0.0
        if (rep + 1) % 5 == 0:
            print(f"PREDICTION_STRENGTH_DONE {rep+1}/50 elapsed={time.perf_counter()-started:.1f}s", flush=True)
    rows = []
    for k in KS:
        vals = out[k]
        rows.append({
            "k": k, "n_splits": 50, "prediction_strength_median": float(np.median(vals)),
            "q2_5": float(np.quantile(vals, .025)), "q97_5": float(np.quantile(vals, .975)),
            "mean": float(np.mean(vals)), "passes_0_80": bool(np.median(vals) >= .80),
            "runtime_seconds": time.perf_counter() - started,
        })
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(smoke=parser.parse_args().smoke)
