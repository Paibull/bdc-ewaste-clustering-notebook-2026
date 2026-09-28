from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon


ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
MATERIALS = ("plastic", "metal", "glass", "rubber", "paper_cardboard", "other_unknown")
SEED = 20260927
N_PERMUTATIONS = 999
OUT = SEMI / "results" / "fused_cluster_material"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate() -> dict[str, object]:
    fraction_rows = read_csv(SEMI / "results" / "dms46_full" / "fraction_table.csv")
    semantic_rows = read_csv(SEMI / "results" / "semantic_g0_cluster_profiles.csv")
    with np.load(SEMI / "artifacts" / "semantic_g0_partitions.npz", allow_pickle=False) as parts:
        keys = parts["keys"].astype(str)
        labels = parts["labels__kmeans_fused"].astype(np.int32)
        discovery = parts["discovery_mask"].astype(bool)
    with np.load(SEMI / "results" / "dms46_full" / "descriptors.npz", allow_pickle=False) as desc:
        desc_keys = desc["key"].astype(str)
        sources = desc["source"].astype(str)

    fraction_keys = np.asarray([row["key"] for row in fraction_rows], dtype=str)
    if any(len(set(values)) != 4179 for values in (keys, desc_keys, fraction_keys)):
        raise ValueError("partition, descriptor, and fraction inputs must each have 4,179 unique keys")
    if set(keys) != set(desc_keys) or set(keys) != set(fraction_keys):
        raise ValueError("material join does not contain the same 4,179 keys")

    part_index = {key: i for i, key in enumerate(keys)}
    desc_index = {key: i for i, key in enumerate(desc_keys)}
    labels_joined = np.asarray([labels[part_index[key]] for key in fraction_keys], dtype=np.int32)
    source_joined = np.asarray([sources[desc_index[key]] for key in fraction_keys], dtype=str)
    x = np.asarray([[float(row[name]) for name in MATERIALS] for row in fraction_rows], dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("material fractions must be finite and nonnegative")
    if np.max(np.abs(x.sum(axis=1) - 1.0)) > 1e-5:
        raise ValueError("six material fractions must sum to one")
    clusters = np.sort(np.unique(labels_joined))
    if len(clusters) != 12:
        raise ValueError(f"expected 12 fused clusters, got {len(clusters)}")

    profile_lookup = {
        int(row["cluster"]): row["top_concept"]
        for row in semantic_rows if row["candidate"] == "kmeans_fused"
    }
    global_mean = x.mean(axis=0)
    profile_rows = []
    means = []
    for cluster in clusters:
        members = labels_joined == cluster
        comp = x[members]
        mean = comp.mean(axis=0)
        means.append(mean)
        entropy = -np.sum(np.where(comp > 0, comp * np.log(np.clip(comp, 1e-15, 1)), 0), axis=1)
        row: dict[str, object] = {
            "cluster": int(cluster), "N": int(members.sum()),
            "semantic_name": profile_lookup.get(int(cluster), ""),
            "largest_mean_fraction_material": MATERIALS[int(np.argmax(mean))],
            "mean_shannon_entropy_nats": float(entropy.mean()),
        }
        for i, material in enumerate(MATERIALS):
            row[f"mean_{material}"] = float(mean[i])
            row[f"median_{material}"] = float(np.median(comp[:, i]))
            row[f"enrichment_{material}_vs_global_mean"] = float(mean[i] / global_mean[i]) if global_mean[i] else None
        profile_rows.append(row)

    means_array = np.asarray(means, dtype=np.float64)
    js_rows = []
    for i, cluster in enumerate(clusters):
        row = {"cluster": int(cluster)}
        row.update({str(int(other)): float(jensenshannon(means_array[i], means_array[j], base=2))
                    for j, other in enumerate(clusters)})
        js_rows.append(row)

    x_discovery = x[discovery]
    labels_discovery = labels_joined[discovery]
    source_discovery = source_joined[discovery]
    transformed = np.sqrt(x_discovery)
    transformed /= np.clip(np.linalg.norm(transformed, axis=1, keepdims=True), 1e-12, None)
    grand = transformed.mean(axis=0)
    total_ss = float(np.square(transformed - grand).sum())

    def between_r2(cluster_labels: np.ndarray) -> float:
        between = 0.0
        for cluster in np.unique(cluster_labels):
            members = cluster_labels == cluster
            delta = transformed[members].mean(axis=0) - grand
            between += int(members.sum()) * float(np.dot(delta, delta))
        return between / total_ss

    observed = between_r2(labels_discovery)
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERMUTATIONS, dtype=np.float64)
    strata = [np.flatnonzero(source_discovery == source) for source in np.unique(source_discovery)]
    for i in range(N_PERMUTATIONS):
        shuffled = labels_discovery.copy()
        for idx in strata:
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[i] = between_r2(shuffled)
    p_value = float((1 + np.count_nonzero(null >= observed)) / (N_PERMUTATIONS + 1))

    OUT.mkdir(parents=True, exist_ok=True)
    profile_fields = ["cluster", "N", "semantic_name", "largest_mean_fraction_material"]
    for material in MATERIALS:
        profile_fields.extend((f"mean_{material}", f"median_{material}", f"enrichment_{material}_vs_global_mean"))
    profile_fields.append("mean_shannon_entropy_nats")
    write_csv(OUT / "cluster_profiles.csv", profile_rows, profile_fields)
    write_csv(OUT / "js_distance_matrix.csv", js_rows, ["cluster", *[str(int(c)) for c in clusters]])
    metrics = {
        "partition": "labels__kmeans_fused",
        "n_joined": int(len(keys)),
        "n_clusters": int(len(clusters)),
        "materials": list(MATERIALS),
        "global_mean_fraction": {material: float(global_mean[i]) for i, material in enumerate(MATERIALS)},
        "six_fraction_max_abs_sum_error": float(np.max(np.abs(x.sum(axis=1) - 1.0))),
        "discovery_material_r2": {
            "n": int(discovery.sum()), "transform": "sqrt(material fractions), row-wise L2 normalized",
            "between_cluster_r2": observed,
            "permutation": {
                "draws": N_PERMUTATIONS, "seed": SEED,
                "stratification": "permutation of fused cluster labels within source (web/field)",
                "p_value": p_value, "null_mean": float(null.mean()),
                "null_95_quantile": float(np.quantile(null, 0.95)),
            },
        },
        "js_distance": "SciPy Jensen-Shannon distance (base 2) on each cluster's mean six-material composition",
    }
    with (OUT / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return metrics


if __name__ == "__main__":
    print(json.dumps(calculate(), ensure_ascii=False))
