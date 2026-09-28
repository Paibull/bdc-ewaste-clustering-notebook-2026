"""Compare AMF and existing clustering frameworks with one fixed evaluator."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from interpret_lenses import standardized_scores
from run_semantic_g0 import (
    benjamini_hochberg,
    semantic_permutation_pvalue,
    semantic_profile,
)


ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
ARTIFACTS = SEMI / "artifacts"
RESULTS = SEMI / "results"
KEYS = ARTIFACTS / "ewaste_keys.npz"
AMF_DEFAULT = RESULTS / "amf_dinov2" / "full-1790306969" / "full-1790306969"
BACKBONES = (
    "dinov3", "radio", "aimv2", "siglip2",
    "dinov2", "siglip", "convnext", "eva02",
)
LENSES = ("function", "material", "condition", "data_driven")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_labels(values: object) -> np.ndarray:
    """Encode arbitrary cluster IDs while preserving noise=-1."""
    series = pd.Series(values, dtype="string")
    codes, _ = pd.factorize(series, sort=True)
    codes = codes.astype(np.int32)
    return codes


def aligned_labels(values: object, keys: np.ndarray, candidate_keys: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values)
    if candidate_keys is None:
        if len(values) != len(keys):
            raise ValueError("label length tidak sama dengan canonical manifest")
        return values.astype(np.int32)
    frame = pd.DataFrame({"key": candidate_keys.astype(str), "label": values})
    frame = frame.drop_duplicates("key").set_index("key")
    if len(frame) != len(candidate_keys) or not set(keys).issubset(frame.index):
        raise ValueError("candidate keys tidak sejajar dengan canonical manifest")
    return frame.loc[keys, "label"].to_numpy(np.int32)


def add_candidate(candidates: dict[str, dict], name: str, labels: np.ndarray, family: str, method: str, representation: str, **meta: object) -> None:
    if name in candidates:
        raise ValueError(f"candidate duplicate: {name}")
    candidates[name] = {
        "labels": np.asarray(labels, dtype=np.int32),
        "family": family,
        "method": method,
        "representation": representation,
        **meta,
    }


def load_candidates(amf_dir: Path, keys: np.ndarray) -> tuple[dict[str, dict], dict[str, float]]:
    candidates: dict[str, dict] = {}
    stability: dict[str, float] = {}

    g0_path = ARTIFACTS / "semantic_g0_partitions.npz"
    g0_metrics = pd.read_csv(RESULTS / "semantic_g0_metrics.csv").set_index("candidate")
    with np.load(g0_path, allow_pickle=False) as g0:
        for name in g0["candidate_names"].astype(str):
            add_candidate(
                candidates,
                name,
                aligned_labels(g0[f"labels__{name}"], keys, g0["keys"]),
                "existing_g0",
                str(g0_metrics.loc[name, "method"]),
                str(g0_metrics.loc[name, "representation"]),
            )
            stability[name] = float(g0_metrics.loc[name, "optimizer_stability_ari"])

    turtle_rows = pd.read_csv(RESULTS / "turtle_evaluation.csv").set_index("seed")
    turtle_labels = {}
    for seed in (42, 43, 44):
        name = f"turtle_seed{seed}"
        with np.load(ARTIFACTS / f"turtle_ewaste_seed{seed}.npz", allow_pickle=False) as artifact:
            turtle_labels[name] = artifact["labels"].astype(np.int32)
        add_candidate(candidates, name, turtle_labels[name], "turtle", "turtle", "multi-space")
    turtle_stability = float(np.mean([
        adjusted_rand_score(turtle_labels[left], turtle_labels[right])
        for left, right in itertools.combinations(turtle_labels, 2)
    ]))
    stability.update({name: turtle_stability for name in turtle_labels})

    final = pd.read_csv(RESULTS / "clusters_lenses_final.csv")
    final_keys = final["key"].astype(str).to_numpy()
    final = final.set_index("key").loc[keys]
    for lens in LENSES:
        name = f"canonical_{lens}"
        add_candidate(candidates, name, final[lens].to_numpy(np.int32), "canonical", "kmeans", lens)

    amf_metrics = pd.read_csv(amf_dir / "metrics.csv").set_index("arm")
    for arm in amf_metrics.index.astype(str):
        with np.load(amf_dir / f"result_{arm}.npz", allow_pickle=False) as artifact:
            name = f"amf_{arm}"
            add_candidate(candidates, name, artifact["labels"], "amf", "amf", "dinov2_multistage", beta=float(amf_metrics.loc[arm, "beta"]))
        stability[name] = float(amf_metrics.loc[arm, "stability_ari_mean"])

    divisive = pd.read_csv(amf_dir / "divisive_assignments.csv")
    divisive_keys = divisive["key"].astype(str).to_numpy()
    add_candidate(
        candidates,
        "amf_divisive_literal_adaptive",
        aligned_labels(divisive["leaf"].astype(str), keys, divisive_keys),
        "amf",
        "amf_divisive",
        "dinov2_multistage",
    )
    return candidates, stability


def dbcv_score(space: np.ndarray, labels: np.ndarray) -> float:
    try:
        from hdbscan.validity import validity_index
    except ImportError:
        return float("nan")
    keep = labels != -1
    if keep.sum() == 0 or len(np.unique(labels[keep])) < 2:
        return float("nan")
    return float(validity_index(np.asarray(space[keep], dtype=np.float64), labels[keep]))


def summarize_candidate(name: str, item: dict, space: np.ndarray, source: np.ndarray, final: pd.DataFrame, scores: np.ndarray, concepts: np.ndarray, label_ids: np.ndarray, concept_lenses: np.ndarray, stability: dict[str, float]) -> tuple[dict, list[dict]]:
    labels = item["labels"]
    keep = labels != -1
    cluster_ids, sizes = np.unique(labels[keep], return_counts=True)
    valid = keep.all() and len(cluster_ids) > 1
    sample = np.flatnonzero(keep)
    if len(sample) > 3000:
        sample = np.random.default_rng(42).choice(sample, 3000, replace=False)
    row = {
        "candidate": name,
        "family": item["family"],
        "method": item["method"],
        "representation": item["representation"],
        "n_clusters": int(len(cluster_ids)),
        "min_cluster_size": int(sizes.min()) if len(sizes) else 0,
        "max_cluster_size": int(sizes.max()) if len(sizes) else 0,
        "noise_fraction": float((labels == -1).mean()),
        "source_nmi": float(normalized_mutual_info_score(source, labels)),
        "silhouette_data_driven": float(silhouette_score(space[sample], labels[sample])) if valid else np.nan,
        "davies_bouldin_data_driven": float(davies_bouldin_score(space, labels)) if valid else np.nan,
        "dbcv_data_driven": dbcv_score(space, labels) if item["method"] == "hdbscan" else np.nan,
        "stability_ari_mean": float(stability.get(name, np.nan)),
    }
    for lens in LENSES:
        row[f"ami_{lens}"] = float(adjusted_mutual_info_score(final[lens].to_numpy(), labels))
    profile, semantic = semantic_profile(labels, scores, concepts, label_ids, concept_lenses)
    concentration, margin, coverage = semantic
    row.update({
        "semantic_concentration": concentration,
        "semantic_mean_margin": margin,
        "semantic_name_coverage": coverage,
        "semantic_permutation_pvalue": semantic_permutation_pvalue(labels, scores, concentration, 200)[0],
    })
    profile_rows = [{"candidate": name, **entry} for entry in profile]
    return row, profile_rows


def main(amf_dir: Path) -> None:
    embedding = np.load(ARTIFACTS / "ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    spaces = np.load(ARTIFACTS / "lens_spaces.npz", allow_pickle=False)
    axes = np.load(ARTIFACTS / "concept_axes.npz", allow_pickle=False)
    with np.load(ARTIFACTS / "semantic_g0_partitions.npz", allow_pickle=False) as g0:
        evaluation_mask = g0["discovery_mask"].astype(bool)
    keys = embedding["key"].astype(str)
    if not np.array_equal(keys, spaces["key"].astype(str)):
        raise ValueError("embedding dan lens_spaces tidak sejajar")
    if not np.array_equal(keys, np.load(KEYS, allow_pickle=False)["key"].astype(str)):
        raise ValueError("canonical manifest tidak sejajar")

    imgstats = pd.read_csv(RESULTS / "imgstats.csv").set_index("key").loc[keys]
    source = (np.maximum(imgstats["W"].to_numpy(), imgstats["H"].to_numpy()) > 150).astype(np.int8)
    final = pd.read_csv(RESULTS / "clusters_lenses_final.csv").set_index("key").loc[keys]
    source = source[evaluation_mask]
    final = final.iloc[np.flatnonzero(evaluation_mask)]
    image = np.asarray(embedding["siglip2"], dtype=np.float32)[evaluation_mask]
    space = np.asarray(spaces["data_driven"], dtype=np.float32)[evaluation_mask]
    lens_scores = [standardized_scores(image, np.asarray(axes[f"anchor_{lens}"], dtype=np.float32)) for lens in ("function", "material", "condition")]
    scores = np.concatenate(lens_scores, axis=1)
    concepts = np.concatenate([axes[f"concept_{lens}"].astype(str) for lens in ("function", "material", "condition")])
    label_ids = np.concatenate([axes[f"label_id_{lens}"].astype(str) for lens in ("function", "material", "condition")])
    concept_lenses = np.concatenate([np.repeat(lens, len(axes[f"concept_{lens}"])) for lens in ("function", "material", "condition")])

    candidates, stability = load_candidates(amf_dir, keys)
    rows, profiles = [], []
    for name, item in candidates.items():
        if len(item["labels"]) != len(keys):
            raise ValueError(f"candidate {name} length mismatch")
        evaluated = {**item, "labels": item["labels"][evaluation_mask]}
        row, profile = summarize_candidate(name, evaluated, space, source, final, scores, concepts, label_ids, concept_lenses, stability)
        rows.append(row)
        profiles.extend(profile)

    metrics = pd.DataFrame(rows)
    qvalues, rejects = benjamini_hochberg(metrics["semantic_permutation_pvalue"].to_numpy())
    metrics["semantic_bh_qvalue"] = qvalues
    metrics["semantic_bh_reject_0_05"] = rejects
    metrics = metrics.sort_values(["family", "candidate"]).reset_index(drop=True)
    pairwise = pd.DataFrame([
        {
            "candidate_left": left,
            "candidate_right": right,
            "ami": adjusted_mutual_info_score(candidates[left]["labels"][evaluation_mask], candidates[right]["labels"][evaluation_mask]),
            "ari": adjusted_rand_score(candidates[left]["labels"][evaluation_mask], candidates[right]["labels"][evaluation_mask]),
        }
        for left, right in itertools.combinations(candidates, 2)
    ])

    metrics_path = RESULTS / "framework_comparison.csv"
    profiles_path = RESULTS / "framework_semantic_profiles.csv"
    pairwise_path = RESULTS / "framework_pairwise.csv"
    metadata_path = RESULTS / "framework_comparison_metadata.json"
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame(profiles).to_csv(profiles_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    metadata_path.write_text(json.dumps({
        "amf_dir": str(amf_dir.relative_to(ROOT)),
        "amf_metrics_sha256": sha256(amf_dir / "metrics.csv"),
        "reference": "artifacts/lens_spaces.npz:data_driven (100D)",
        "evaluation_rows": int(evaluation_mask.sum()),
        "evaluation": "semantic_g0 discovery mask; 600 holdout excluded",
        "semantic_scores": "SigLIP2 concept_axes; proxy, not human ground truth",
        "n_candidates": len(candidates),
        "candidate_names": list(candidates),
        "final_partition_modified": False,
    }, indent=2), encoding="utf-8")
    print(metrics[["candidate", "method", "n_clusters", "noise_fraction", "silhouette_data_driven", "dbcv_data_driven", "semantic_concentration", "semantic_mean_margin", "semantic_name_coverage"]].to_string(index=False))
    for path in (metrics_path, profiles_path, pairwise_path, metadata_path):
        print(f"{path.relative_to(ROOT)} sha256={sha256(path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amf-dir", type=Path, default=AMF_DEFAULT)
    args = parser.parse_args()
    main(args.amf_dir)
