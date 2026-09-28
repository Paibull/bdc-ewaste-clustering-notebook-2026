from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SEMI = ROOT / "semifinal"
BACKBONES = ("dinov3", "radio", "aimv2", "siglip2", "dinov2", "siglip", "convnext", "eva02")
FAMILIES = ("battery", "mobile", "pcb", "input_peripheral")
CLASS_TO_FAMILY = {
    "Battery Waste": "battery",
    "Mobile": "mobile",
    "PCB": "pcb",
    "Keyboard": "input_peripheral",
    "Mouse": "input_peripheral",
}
K14_MAP = {0: "battery", 5: "mobile", 12: "pcb", 6: "input_peripheral", 9: "input_peripheral"}
RESULTS = SEMI / "results/bangladesh_frozen_transfer_k14_sensitivity"


def l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fused_spaces(discovery: np.ndarray, external: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    source = np.load(SEMI / "artifacts/ewaste_emb_harm_resjpeg.npz", allow_pickle=False)
    internal_blocks, external_blocks = [], []
    for name in BACKBONES:
        internal = l2(np.asarray(source[name], dtype=np.float32)[discovery])
        outside = l2(external[name])
        pca = PCA(n_components=32, svd_solver="randomized", whiten=False, random_state=0)
        internal_blocks.append(pca.fit_transform(internal))
        external_blocks.append(pca.transform(outside))
    internal_joined = l2(np.concatenate(internal_blocks, axis=1))
    external_joined = l2(np.concatenate(external_blocks, axis=1))
    pca = PCA(n_components=100, svd_solver="randomized", whiten=False, random_state=0)
    return l2(pca.fit_transform(internal_joined)), l2(pca.transform(external_joined))


def metric_values(y: np.ndarray, distances: np.ndarray) -> dict[str, object]:
    order = np.argsort(distances, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1, kind="stable")
    true_rank = ranks[np.arange(len(y)), y] + 1
    pred = order[:, 0]
    recall = {family: float(np.mean(pred[y == i] == i)) for i, family in enumerate(FAMILIES)}
    return {
        "micro_top_1": float(np.mean(pred == y)),
        "macro_top_1": float(np.mean(list(recall.values()))),
        "top_2": float(np.mean(true_rank <= 2)),
        "mrr": float(np.mean(1.0 / true_rank)),
        "mean_rank": float(np.mean(true_rank)),
        "recall_per_family": recall,
    }


def bootstrap(y: np.ndarray, candidates: dict[str, np.ndarray], draws: int = 2000) -> dict[str, object]:
    rng = np.random.default_rng(20260927)
    groups = [np.flatnonzero(y == i) for i in range(len(FAMILIES))]
    metrics = ("micro_top_1", "macro_top_1", "top_2", "mrr")
    values = {name: {metric: [] for metric in metrics} for name in candidates}
    deltas = {f"k14_minus_{name}": {metric: [] for metric in metrics} for name in candidates if name != "fused_k14"}
    for _ in range(draws):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        points = {name: metric_values(y[sample], distance[sample]) for name, distance in candidates.items()}
        for name in candidates:
            for metric in metrics:
                values[name][metric].append(points[name][metric])
        for name in candidates:
            if name == "fused_k14":
                continue
            for metric in metrics:
                deltas[f"k14_minus_{name}"][metric].append(points["fused_k14"][metric] - points[name][metric])

    def interval(items: list[float]) -> list[float]:
        return [float(x) for x in np.quantile(items, [0.025, 0.975])]

    return {
        "draws": draws,
        "seed": 20260927,
        "ci": {name: {metric: interval(items) for metric, items in rows.items()} for name, rows in values.items()},
        "delta_ci": {name: {metric: interval(items) for metric, items in rows.items()} for name, rows in deltas.items()},
    }


def main() -> None:
    partitions = np.load(SEMI / "artifacts/semantic_g0_partitions.npz", allow_pickle=False)
    discovery = partitions["discovery_mask"].astype(bool)
    external_npz = np.load(SEMI / "results/bangladesh_frozen_transfer/external_embeddings.npz", allow_pickle=False)
    external = {name: np.asarray(external_npz[name], dtype=np.float32) for name in BACKBONES}
    internal_space, external_space = fused_spaces(discovery, external)

    model = KMeans(n_clusters=14, n_init=20, random_state=42).fit(internal_space)
    saved = np.load(
        SEMI / "results/k_resolution_sweep/20260927/remote/classic/kmeans_fused/k_14/seed_42/labels.npz",
        allow_pickle=False,
    )
    discovery_keys = partitions["keys"].astype(str)[discovery]
    assert np.array_equal(saved["key"].astype(str), discovery_keys)
    frozen_labels = saved["labels"].astype(np.int32)
    reconstruction_ari = float(adjusted_rand_score(frozen_labels, model.labels_))
    overlap = np.zeros((14, 14), dtype=np.int64)
    np.add.at(overlap, (frozen_labels, model.labels_), 1)
    frozen_ids, fit_ids = linear_sum_assignment(-overlap)
    fit_to_frozen = {int(fit): int(frozen) for frozen, fit in zip(frozen_ids, fit_ids)}
    centers = np.empty_like(model.cluster_centers_)
    for fit_id, frozen_id in fit_to_frozen.items():
        centers[frozen_id] = model.cluster_centers_[fit_id]
    assert reconstruction_ari == 1.0

    cluster_distances = 1.0 - external_space @ centers.T
    k14_distances = np.column_stack(
        [np.min(cluster_distances[:, [cluster for cluster, family in K14_MAP.items() if family == name]], axis=1)
         for name in FAMILIES]
    )

    keys = external_npz["key"].astype(str)
    source_classes = external_npz["source_class"].astype(str)
    known = np.asarray([name in CLASS_TO_FAMILY for name in source_classes])
    y = np.asarray([FAMILIES.index(CLASS_TO_FAMILY[name]) for name in source_classes[known]], dtype=np.int32)

    with (SEMI / "results/bangladesh_frozen_transfer/predictions.csv").open(encoding="utf-8", newline="") as handle:
        previous = {row["key"]: row for row in csv.DictReader(handle)}
    assert set(previous) == set(keys)
    k12_distances = np.asarray(
        [[float(previous[key][f"fused_distance_{family}"]) for family in FAMILIES] for key in keys], dtype=np.float32
    )
    radio_distances = np.asarray(
        [[float(previous[key][f"radio_distance_{family}"]) for family in FAMILIES] for key in keys], dtype=np.float32
    )
    candidates = {
        "fused_k14": k14_distances[known],
        "fused_k12_frozen": k12_distances[known],
        "radio_k12_selected": radio_distances[known],
    }
    point = {name: metric_values(y, values) for name, values in candidates.items()}
    bootstrap_result = bootstrap(y, candidates)

    unknown = ~known
    novelty = {}
    for name, values in (("fused_k14", k14_distances), ("fused_k12_frozen", k12_distances),
                         ("radio_k12_selected", radio_distances)):
        ordered = np.sort(values, axis=1)
        scores = {
            "nearest_known_family_distance": ordered[:, 0],
            "negative_known_family_margin": -(ordered[:, 1] - ordered[:, 0]),
        }
        novelty[name] = {
            signal: {
                "auroc": float(roc_auc_score(unknown, score)),
                "average_precision": float(average_precision_score(unknown, score)),
            }
            for signal, score in scores.items()
        }

    payload = {
        "status": "SENSITIVITY_ONLY",
        "protocol": "K=14 and mapping selected from BDC only; cached Bangladesh embeddings; no refit on Bangladesh",
        "n_known": int(known.sum()),
        "n_unknown": int(unknown.sum()),
        "reconstruction_ari": reconstruction_ari,
        "fused_k14_mapping": {str(key): value for key, value in K14_MAP.items()},
        "mapping_source": "semantic_nodes.csv candidate classic::kmeans_fused::k14",
        "point_metrics": point,
        "bootstrap": bootstrap_result,
        "novelty_unknown_positive": novelty,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
