from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np

try:
    import modal
except ImportError:
    modal = None


SCRIPT = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT.parents if (parent / "semifinal").is_dir()), Path("/root"))
SEMI = ROOT / "semifinal"
BDC_EMBEDDINGS = SEMI / "artifacts" / "ewaste_emb_harm_resjpeg.npz"
PARTITIONS = SEMI / "artifacts" / "semantic_g0_partitions.npz"
SUBSET = SEMI / "results" / "subset_validasi.csv"
RESULTS = SEMI / "results" / "bangladesh_frozen_transfer"
DATASET_ROOT = Path("/data/externals/bangladesh_ewaste_v1/extracted")
REMOTE_ROOT = Path("/data/ckpt/bangladesh-frozen-transfer-v1")
HARMONIZE_PATH = ROOT / "penyisihan" / "modal-train" / "harmonize.py"
APP_NAME = "bdc-bangladesh-frozen-transfer-20260927"
VOLUME_NAME = "bdc-data"
BACKBONES = (
    ("dinov3", "vit_large_patch16_dinov3.lvd1689m", 224),
    ("radio", "HF:nvidia/C-RADIOv4-SO400M", 224),
    ("aimv2", "aimv2_large_patch14_224.apple_pt", 224),
    ("siglip2", "OC:timm/ViT-SO400M-16-SigLIP2-384", 384),
    ("dinov2", "vit_large_patch14_reg4_dinov2.lvd142m", 224),
    ("siglip", "vit_so400m_patch14_siglip_384", 378),
    ("convnext", "convnextv2_large.fcmae_ft_in22k_in1k", 224),
    ("eva02", "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 448),
)
FAMILIES = ("battery", "mobile", "pcb", "input_peripheral")
KNOWN_CLASS_TO_FAMILY = {
    "Battery Waste": "battery", "Mobile": "mobile", "PCB": "pcb",
    "Keyboard": "input_peripheral", "Mouse": "input_peripheral",
}
SOURCE_CLASSES = (
    "Battery Waste", "Glass Waste", "Keyboard", "Light Bulb", "Medical Waste",
    "Metal Waste", "Mobile", "Mouse", "Organic Waste", "Paper Waste", "PCB", "Plastic Waste",
)
FROZEN_MAPS = {
    "fused": {1: "battery", 5: "mobile", 8: "pcb", 0: "input_peripheral", 7: "input_peripheral"},
    "radio": {4: "battery", 3: "mobile", 10: "pcb", 2: "input_peripheral", 11: "input_peripheral"},
}
N_BOOTSTRAPS = 2000
SEED = 20260927


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_g0_helpers():
    sys.path.insert(0, str(SEMI / "scripts"))
    from run_semantic_g0 import BACKBONES as G0_BACKBONES, fit_pca_space, l2
    if tuple(G0_BACKBONES) != tuple(name for name, _, _ in BACKBONES):
        raise ValueError("backbone order differs from frozen G0 implementation")
    return fit_pca_space, l2


def align_reconstructed_centroids(frozen: np.ndarray, reconstructed: np.ndarray,
                                  centers: np.ndarray) -> tuple[float, dict[int, int], np.ndarray]:
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import adjusted_rand_score

    overlap = np.zeros((12, 12), dtype=np.int64)
    np.add.at(overlap, (frozen.astype(int), reconstructed.astype(int)), 1)
    ref_ids, fit_ids = linear_sum_assignment(-overlap)
    fit_to_ref = {int(fit): int(ref) for ref, fit in zip(ref_ids, fit_ids)}
    aligned = np.empty_like(centers)
    for fit_id, ref_id in fit_to_ref.items():
        aligned[ref_id] = centers[fit_id]
    return float(adjusted_rand_score(frozen, reconstructed)), fit_to_ref, aligned


def reconstruct_frozen(external: dict[str, np.ndarray] | None = None) -> dict[str, object]:
    from sklearn.cluster import KMeans

    fit_pca_space, l2 = local_g0_helpers()
    with np.load(BDC_EMBEDDINGS, allow_pickle=False) as src:
        bdc_keys = src["key"].astype(str)
        bdc_by_key = {key: i for i, key in enumerate(bdc_keys)}
        if len(bdc_keys) != 4179 or len(bdc_by_key) != 4179:
            raise ValueError("BDC embedding must contain 4,179 unique keys")
        with np.load(PARTITIONS, allow_pickle=False) as part:
            keys = part["keys"].astype(str)
            discovery = part["discovery_mask"].astype(bool)
            frozen_fused = part["labels__kmeans_fused"].astype(np.int32)
            frozen_radio = part["labels__kmeans_radio"].astype(np.int32)
        if len(keys) != 4179 or len(set(keys)) != 4179 or set(keys) != set(bdc_keys):
            raise ValueError("BDC embedding keys and frozen partition keys differ")
        if int(discovery.sum()) != 3579 or int((~discovery).sum()) != 600:
            raise ValueError("frozen discovery/holdout split must be 3,579/600")
        bdc_order = np.asarray([bdc_by_key[key] for key in keys], dtype=np.int64)
        holdout_rows = read_csv(SUBSET)
        holdout_keys = {row["key"] for row in holdout_rows}
        if len(holdout_keys) != 600 or holdout_keys != set(keys[~discovery]):
            raise ValueError("subset_validasi keys do not match the frozen 600-row holdout")

        fused_discovery_blocks, fused_holdout_blocks, fused_external_blocks = [], [], []
        radio_raw = None
        for name, _, _ in BACKBONES:
            raw = np.asarray(src[name][bdc_order], dtype=np.float32)
            disc_space, hold_space, pca = fit_pca_space(raw[discovery], raw[~discovery], 32)
            fused_discovery_blocks.append(disc_space)
            fused_holdout_blocks.append(hold_space)
            if external is not None:
                ext = np.asarray(external[name], dtype=np.float32)
                fused_external_blocks.append(l2(pca.transform(l2(ext))))
            if name == "radio":
                radio_raw = raw

    fused_disc_concat = np.concatenate(fused_discovery_blocks, axis=1)
    fused_hold_concat = np.concatenate(fused_holdout_blocks, axis=1)
    fused_disc, fused_hold, fused_pca = fit_pca_space(fused_disc_concat, fused_hold_concat, 100)
    fused_ext = (l2(fused_pca.transform(l2(np.concatenate(fused_external_blocks, axis=1))))
                 if external is not None else None)
    radio_disc, radio_hold, radio_pca = fit_pca_space(radio_raw[discovery], radio_raw[~discovery], 100)
    radio_ext = None
    if external is not None:
        radio_ext = l2(radio_pca.transform(l2(np.asarray(external["radio"], dtype=np.float32))))

    result = {}
    for name, train_space, hold_space, ext_space, frozen in (
        ("fused", fused_disc, fused_hold, fused_ext, frozen_fused),
        ("radio", radio_disc, radio_hold, radio_ext, frozen_radio),
    ):
        model = KMeans(n_clusters=12, n_init=20, random_state=0).fit(train_space)
        train_labels = model.labels_.astype(np.int32)
        ari, fit_to_ref, centers = align_reconstructed_centroids(frozen[discovery], train_labels,
                                                                  model.cluster_centers_)
        record: dict[str, object] = {
            "ari": ari,
            "n_discovery": int(discovery.sum()),
            "cluster_id_alignment_reconstructed_to_frozen": {str(k): v for k, v in fit_to_ref.items()},
            "frozen_mapping": {str(k): v for k, v in FROZEN_MAPS[name].items()},
            "holdout_transformed_rows": int(len(hold_space)),
            "reconstruction_pass": bool(ari >= 0.9999),
        }
        if ari < 0.9999:
            raise ReconstructionMismatch(name, record)
        if ext_space is not None:
            centers = centers / np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
            distances = 1.0 - np.asarray(ext_space, dtype=np.float32) @ centers.T
            if not np.isfinite(distances).all():
                raise ValueError(f"non-finite cosine distance for {name}")
            record["family_distances"] = family_distances(distances, FROZEN_MAPS[name])
        result[name] = record
    return result


class ReconstructionMismatch(RuntimeError):
    def __init__(self, estimator: str, record: dict[str, object]):
        super().__init__(f"{estimator} reconstruction ARI={record['ari']:.8f} < 0.9999")
        self.estimator = estimator
        self.record = record


def family_distances(cluster_distances: np.ndarray, mapping: dict[int, str]) -> np.ndarray:
    return np.column_stack([
        np.min(cluster_distances[:, [cluster for cluster, family in mapping.items() if family == name]], axis=1)
        for name in FAMILIES
    ])


def metric_values(y: np.ndarray, distances: np.ndarray) -> dict[str, object]:
    ranks = np.argsort(np.argsort(distances, axis=1, kind="stable"), axis=1, kind="stable")
    true_rank = ranks[np.arange(len(y)), y] + 1
    pred = distances.argmin(axis=1)
    recall = {family: float(np.mean(pred[y == i] == i)) for i, family in enumerate(FAMILIES)}
    return {
        "micro_top_1": float(np.mean(pred == y)),
        "macro_top_1": float(np.mean(list(recall.values()))),
        "top_2": float(np.mean(true_rank <= 2)),
        "mrr": float(np.mean(1.0 / true_rank)),
        "mean_rank": float(np.mean(true_rank)),
        "recall_per_family": recall,
    }


def bootstrap_metrics(y: np.ndarray, fused: np.ndarray, radio: np.ndarray) -> tuple[dict, dict]:
    rng = np.random.default_rng(SEED)
    draws = {name: {metric: np.empty(N_BOOTSTRAPS, dtype=np.float64)
                    for metric in ("micro_top_1", "macro_top_1", "top_2", "mrr", "mean_rank")}
             for name in ("fused", "radio")}
    recall_draws = {name: {family: np.empty(N_BOOTSTRAPS, dtype=np.float64) for family in FAMILIES}
                    for name in ("fused", "radio")}
    deltas = {metric: np.empty(N_BOOTSTRAPS, dtype=np.float64)
              for metric in ("micro_top_1", "macro_top_1", "mrr")}
    by_family = [np.flatnonzero(y == i) for i in range(len(FAMILIES))]
    for b in range(N_BOOTSTRAPS):
        ix = np.concatenate([rng.choice(items, size=len(items), replace=True) for items in by_family])
        a, r = metric_values(y[ix], fused[ix]), metric_values(y[ix], radio[ix])
        for key in draws["fused"]:
            draws["fused"][key][b] = a[key]
            draws["radio"][key][b] = r[key]
        for family in FAMILIES:
            recall_draws["fused"][family][b] = a["recall_per_family"][family]
            recall_draws["radio"][family][b] = r["recall_per_family"][family]
        for key in deltas:
            deltas[key][b] = a[key] - r[key]

    def ci(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    metric_ci = {model: {key: ci(values) for key, values in metrics.items()} for model, metrics in draws.items()}
    for model in ("fused", "radio"):
        metric_ci[model]["recall_per_family"] = {
            family: ci(values) for family, values in recall_draws[model].items()
        }
    delta_ci = {key: ci(values) for key, values in deltas.items()}
    return metric_ci, delta_ci


def score_outputs() -> dict[str, object]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    RESULTS.mkdir(parents=True, exist_ok=True)
    embedding_path = RESULTS / "external_embeddings.npz"
    if not embedding_path.is_file():
        raise FileNotFoundError(f"external embeddings not found: {embedding_path}")
    with np.load(embedding_path, allow_pickle=False) as source:
        keys = source["key"].astype(str)
        classes = source["source_class"].astype(str)
        splits = source["split"].astype(str)
        external = {name: np.asarray(source[name], dtype=np.float32) for name, _, _ in BACKBONES}
        metadata = json.loads(str(source["metadata_json"].item()))
    if len(keys) != 2153 or len(set(keys)) != 2153:
        raise ValueError("external embeddings must contain 2,153 unique keys")
    if not np.isfinite(np.concatenate([value.ravel()[:10] for value in external.values()])).all():
        raise ValueError("external embeddings contain non-finite values")
    if set(classes) - set(SOURCE_CLASSES):
        raise ValueError("external embeddings contain unknown source classes")
    expected_splits = np.asarray(["known" if name in KNOWN_CLASS_TO_FAMILY else "unknown" for name in classes])
    if not np.array_equal(splits, expected_splits):
        raise ValueError("stored known/unknown split does not match the frozen class taxonomy")
    known = splits == "known"
    unknown = ~known
    if int(known.sum()) != 710 or int(unknown.sum()) != 1443:
        raise ValueError(f"expected 710 known and 1,443 out-of-taxonomy rows, got {known.sum()}/{unknown.sum()}")
    if any(not np.isfinite(values).all() for values in external.values()):
        raise ValueError("external embeddings contain NaN or infinity")

    try:
        reconstruction = reconstruct_frozen(external)
    except ReconstructionMismatch as exc:
        payload = {"state": "RECONSTRUCTION_MISMATCH", "failed_estimator": exc.estimator,
                   "failed_reconstruction": exc.record, "mapping_changed": False}
        (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False))
        return payload
    y = np.asarray([FAMILIES.index(KNOWN_CLASS_TO_FAMILY[name]) for name in classes[known]], dtype=np.int32)
    fused_dist = np.asarray(reconstruction["fused"]["family_distances"])[known]
    radio_dist = np.asarray(reconstruction["radio"]["family_distances"])[known]
    point = {"fused": metric_values(y, fused_dist), "radio": metric_values(y, radio_dist)}
    metric_ci, delta_ci = bootstrap_metrics(y, fused_dist, radio_dist)
    novelty = {}
    unknown_truth = unknown.astype(np.int8)
    for name, all_distances in (("fused", reconstruction["fused"]["family_distances"]),
                                ("radio", reconstruction["radio"]["family_distances"])):
        ordered = np.sort(all_distances, axis=1)
        signals = {
            "nearest_known_family_distance": ordered[:, 0],
            "negative_known_family_margin": -(ordered[:, 1] - ordered[:, 0]),
        }
        novelty[name] = {
            metric: {"auroc_unknown_positive": float(roc_auc_score(unknown_truth, values)),
                     "average_precision_unknown_positive": float(average_precision_score(unknown_truth, values)),
                     "unknown_prevalence_ap_chance": float(unknown.mean()),
                     "score_direction": "larger values indicate out-of-taxonomy"}
            for metric, values in signals.items()
        }

    metric_order = {
        "micro_top_1": "micro Top-1", "macro_top_1": "macro Top-1", "top_2": "Top-2",
        "mrr": "MRR", "mean_rank": "mean rank",
    }
    primary_results = {}
    for name in ("fused", "radio"):
        primary_results[name] = {**point[name], "bootstrap_95_ci": metric_ci[name]}
    delta_points = {key: point["fused"][key] - point["radio"][key] for key in delta_ci}
    fused_macro_ci = metric_ci["fused"]["macro_top_1"]
    if delta_ci["macro_top_1"][0] > 0:
        status = "FUSION_IMPROVES_TRANSFER"
    elif fused_macro_ci[0] > 0.25:
        status = "TRANSFER_SUPPORTED_NO_FUSION_GAIN"
    else:
        status = "NOT_SUPPORTED"

    prediction_rows = []
    confusion_rows = []
    family_rows = []
    estimator_distances = {"fused": reconstruction["fused"]["family_distances"],
                           "radio": reconstruction["radio"]["family_distances"]}
    for i, family in enumerate(FAMILIES):
        for name in ("fused", "radio"):
            family_rows.append({"estimator": name, "family": family,
                                "support": int(np.sum(y == i)),
                                "recall": point[name]["recall_per_family"][family],
                                "recall_bootstrap_95_ci": json.dumps(metric_ci[name]["recall_per_family"][family])})
    for name, distances in estimator_distances.items():
        pred = distances.argmin(axis=1)
        known_indices = np.flatnonzero(known)
        for known_i, row_i in enumerate(known_indices):
            order = np.argsort(distances[row_i], kind="stable")
            true_family = FAMILIES[y[known_i]]
            out = {"key": keys[row_i], "source_class": classes[row_i], "split": splits[row_i],
                   "true_family": true_family, f"{name}_predicted_family": FAMILIES[int(pred[row_i])],
                   f"{name}_rank": int(np.flatnonzero(order == FAMILIES.index(true_family))[0]) + 1}
            for j, family in enumerate(FAMILIES):
                out[f"{name}_distance_{family}"] = float(distances[row_i, j])
            prediction_rows.append(out)
        # Export out-of-taxonomy predictions too for novelty review.
        for row_i in np.flatnonzero(unknown):
            order = np.argsort(distances[row_i], kind="stable")
            out = {"key": keys[row_i], "source_class": classes[row_i], "split": splits[row_i],
                   "true_family": "", f"{name}_predicted_family": FAMILIES[int(pred[row_i])],
                   f"{name}_rank": ""}
            for j, family in enumerate(FAMILIES):
                out[f"{name}_distance_{family}"] = float(distances[row_i, j])
            prediction_rows.append(out)

    # Coalesce per-estimator rows produced above into one row per image.
    pred_by_key: dict[str, dict[str, object]] = {}
    for row in prediction_rows:
        if row["key"] in pred_by_key:
            pred_by_key[row["key"]].update(row)
        else:
            pred_by_key[row["key"]] = row
    prediction_rows = [pred_by_key[key] for key in keys]
    for name, distances in estimator_distances.items():
        pred = distances[:]
        # Distances include all rows; confusion counts use only the known subset.
        pred_known = pred[known].argmin(axis=1)
        for true_i, true_family in enumerate(FAMILIES):
            for pred_i, predicted_family in enumerate(FAMILIES):
                confusion_rows.append({"estimator": name, "true_family": true_family,
                                       "predicted_family": predicted_family,
                                       "count": int(np.sum((y == true_i) & (pred_known == pred_i)))})

    for row_i, row in enumerate(prediction_rows):
        for name, distances in estimator_distances.items():
            ordered = np.sort(distances[row_i])
            row[f"{name}_nearest_known_family_distance"] = float(ordered[0])
            row[f"{name}_known_family_margin"] = float(ordered[1] - ordered[0])

    reconstruction_payload = {
        name: {key: value for key, value in record.items() if key != "family_distances"}
        for name, record in reconstruction.items()
    }
    payload = {
        "state": "complete", "verdict": status,
        "reconstruction": reconstruction_payload,
        "external_counts": {"total_annotated_unique": len(keys), "known": int(known.sum()),
                            "out_of_taxonomy": int(unknown.sum()), "excluded_without_annotation": 4},
        "primary_known_set": {
            "n": int(known.sum()), "families": list(FAMILIES), "chance_macro_top_1": 0.25,
            "top_2_definition": "Top-2 among the four frozen families: battery, mobile, pcb, input_peripheral",
            "bootstrap": {"draws": N_BOOTSTRAPS, "seed": SEED,
                          "method": "paired, stratified by true family; same resampled keys for fused and RADIO"},
            "estimators": primary_results,
            "paired_deltas_fused_minus_radio": {
                key: {"point": float(delta_points[key]), "bootstrap_95_ci": delta_ci[key]}
                for key in delta_ci
            },
            "verdict_rule": "FUSION_IMPROVES_TRANSFER if paired macro Top-1 delta lower CI > 0; else TRANSFER_SUPPORTED_NO_FUSION_GAIN if fused macro Top-1 lower CI > 0.25; else NOT_SUPPORTED",
        },
        "out_of_taxonomy_novelty": {
            "positive_class": "unknown (out-of-taxonomy)",
            "signals": novelty,
            "threshold_tuning": False,
        },
        "mapping_frozen": FROZEN_MAPS,
        "mapping_changed_after_scoring": False,
        "embedding_metadata": metadata,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(RESULTS / "predictions.csv", prediction_rows)
    write_csv(RESULTS / "per_family_metrics.csv", family_rows)
    write_csv(RESULTS / "confusion_matrix.csv", confusion_rows)
    print(json.dumps({"verdict": status, "reconstruction": {k: v["ari"] for k, v in reconstruction.items()},
                      "primary": {k: {m: v[m] for m in metric_order} for k, v in point.items()},
                      "paired_deltas": {k: {"point": delta_points[k], "ci": delta_ci[k]} for k in delta_ci},
                      "novelty": novelty}, ensure_ascii=False))
    return payload


def run_reconstruction_check() -> dict[str, object]:
    result = reconstruct_frozen()
    summary = {name: {k: v for k, v in row.items() if k != "family_distances"} for name, row in result.items()}
    print(json.dumps({"state": "RECONSTRUCTION_PASS" if all(row["reconstruction_pass"] for row in result.values())
                      else "RECONSTRUCTION_MISMATCH", "estimators": summary}, ensure_ascii=False))
    return result


if modal is not None:
    VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    IMAGE = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("torch==2.5.0", "torchvision==0.20.0", "timm>=1.0.20",
                          "transformers>=4.45", "open-clip-torch", "numpy", "pillow", "pyyaml", "einops")
             .env({"HF_HOME": "/data/hf-cache", "HF_HUB_CACHE": "/data/hf-cache/hub",
                   "HUGGINGFACE_HUB_CACHE": "/data/hf-cache/hub", "TORCH_HOME": "/data/hf-cache/torch"})
             .add_local_file(str(HARMONIZE_PATH), "/root/harmonize.py"))
    APP = modal.App(APP_NAME)

    @APP.function(image=IMAGE, gpu="A10G", cpu=8, memory=24576, timeout=4 * 60 * 60,
                  retries=0, volumes={"/data": VOLUME})
    def embed_external() -> None:
        import importlib.metadata
        import re
        from collections import Counter

        import torch
        import torchvision.transforms as T
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        import timm
        from timm.data import create_transform, resolve_model_data_config
        import sys as remote_sys

        remote_sys.path.insert(0, "/root")
        from harmonize import harmonize

        started = time.time()
        REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
        status_path = REMOTE_ROOT / "status.json"

        def status(stage: str, **values: object) -> None:
            status_path.write_text(json.dumps({"state": stage, "updated_unix": time.time(), **values}, indent=2),
                                   encoding="utf-8")
            VOLUME.commit()

        try:
            if not DATASET_ROOT.is_dir():
                raise FileNotFoundError(f"Bangladesh dataset missing at {DATASET_ROOT}")
            yaml_path = next(DATASET_ROOT.rglob("data.yaml"), None)
            if yaml_path is None:
                raise FileNotFoundError("Bangladesh YOLO data.yaml not found")
            names_text = next((line.partition(":")[2].strip() for line in yaml_path.read_text(encoding="utf-8").splitlines()
                               if line.strip().startswith("names:")), "")
            class_names = [str(name).replace("_", " ") for name in ast.literal_eval(names_text)]
            if len(class_names) != 12 or set(class_names) != set(SOURCE_CLASSES):
                raise ValueError(f"unexpected YOLO class list: {class_names}")
            image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
            media = sorted(path for path in DATASET_ROOT.rglob("*")
                           if path.is_file() and path.suffix.lower() in image_suffixes)
            rows, missing = [], []
            for path in media:
                relative = path.relative_to(DATASET_ROOT).as_posix()
                parts = PurePosixPath(relative).parts
                if len(parts) < 3 or parts[-2] != "images":
                    continue
                annotation = DATASET_ROOT.joinpath(*parts[:-2], "labels", f"{path.stem}.txt")
                if not annotation.is_file() or not annotation.read_text(encoding="utf-8").strip():
                    missing.append(relative)
                    continue
                ids = {int(line.split()[0]) for line in annotation.read_text(encoding="utf-8").splitlines() if line.strip()}
                if len(ids) != 1:
                    raise ValueError(f"image must have one unique YOLO class ID: {relative} -> {sorted(ids)}")
                class_id = next(iter(ids))
                if not 0 <= class_id < len(class_names):
                    raise ValueError(f"YOLO class ID out of range in {relative}: {class_id}")
                source_class = class_names[class_id]
                rows.append({"key": relative, "path": str(path), "source_class": source_class,
                             "split": "known" if source_class in KNOWN_CLASS_TO_FAMILY else "unknown"})
            if len(media) != 2157 or len(rows) != 2153 or len(missing) != 4:
                raise ValueError(f"expected 2,157 images, 2,153 annotated, 4 without annotations; got {len(media)}, {len(rows)}, {len(missing)}")
            counts = Counter(row["source_class"] for row in rows)
            known_n = sum(row["split"] == "known" for row in rows)
            unknown_n = len(rows) - known_n
            if known_n != 710 or unknown_n != 1443 or set(counts) != set(SOURCE_CLASSES):
                raise ValueError(f"external taxonomy counts differ from frozen expectation: known={known_n}, unknown={unknown_n}, class_counts={dict(counts)}")
            if len({row["key"] for row in rows}) != len(rows):
                raise ValueError("external annotated image keys are not unique")
            status("INVENTORY_COMPLETE", total_images=len(media), annotated_images=len(rows),
                   known=known_n, out_of_taxonomy=unknown_n, without_annotation=len(missing),
                   class_counts=dict(counts), missing_annotation_keys=missing)

            dev = "cuda"
            completed = 0
            last_update = time.time()
            out: dict[str, np.ndarray] = {
                "key": np.asarray([row["key"] for row in rows]),
                "source_class": np.asarray([row["source_class"] for row in rows]),
                "split": np.asarray([row["split"] for row in rows]),
            }

            class ExternalDataset(Dataset):
                def __init__(self, transform):
                    self.transform = transform

                def __len__(self):
                    return len(rows)

                def __getitem__(self, index):
                    with Image.open(rows[index]["path"]) as opened:
                        image = harmonize(opened.convert("RGB"), jpeg=True)
                    return self.transform(image), index

            def encode(model, transform, feature_fn, backbone: str):
                nonlocal completed, last_update
                loader = DataLoader(ExternalDataset(transform), batch_size=16, num_workers=8,
                                    pin_memory=True, shuffle=False)
                vectors = None
                model_started = time.time()
                with torch.inference_mode():
                    for batch_index, (images, indexes) in enumerate(loader, start=1):
                        images = images.to(dev, non_blocking=True)
                        with torch.autocast("cuda"):
                            features = feature_fn(model, images).float()
                        features = torch.nn.functional.normalize(features, dim=1).cpu().numpy()
                        if not np.isfinite(features).all():
                            raise ValueError(f"non-finite embeddings from {backbone}")
                        if vectors is None:
                            vectors = np.zeros((len(rows), features.shape[1]), dtype=np.float32)
                        vectors[indexes.numpy()] = features
                        completed += len(features)
                        now = time.time()
                        if now - last_update >= 50 or batch_index == len(loader):
                            status("EMBEDDING", backbone=backbone, completed_image_backbone_units=completed,
                                   total_image_backbone_units=len(rows) * len(BACKBONES),
                                   current_backbone_images=int(indexes[-1]) + 1, total_images=len(rows),
                                   runtime_seconds=round(now - started, 2),
                                   current_backbone_runtime_seconds=round(now - model_started, 2))
                            print(json.dumps({"stage": "embedding", "backbone": backbone,
                                              "completed": completed, "total": len(rows) * len(BACKBONES),
                                              "runtime_seconds": round(now - started, 1)}), flush=True)
                            last_update = now
                if vectors is None or vectors.shape[0] != len(rows):
                    raise RuntimeError(f"empty or incomplete embedding matrix for {backbone}")
                return vectors.astype(np.float16)

            def radio_loader(repo: str, image_size: int):
                from transformers import AutoModel
                model = AutoModel.from_pretrained(repo, trust_remote_code=True).to(dev).eval()
                transform = T.Compose([T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
                                        T.ToTensor()])
                return model, transform, lambda m, batch: m(batch)[0]

            def openclip_loader(hf_id: str):
                import open_clip
                model, transform = open_clip.create_model_from_pretrained(f"hf-hub:{hf_id}")
                return model.to(dev).eval(), transform, lambda m, batch: m.encode_image(batch)

            model_metadata = {}
            for backbone, model_id, image_size in BACKBONES:
                status("MODEL_LOADING", backbone=backbone, completed_image_backbone_units=completed,
                       total_image_backbone_units=len(rows) * len(BACKBONES), runtime_seconds=round(time.time() - started, 2))
                if model_id.startswith("HF:"):
                    model, transform, feature_fn = radio_loader(model_id[3:], image_size)
                elif model_id.startswith("OC:"):
                    model, transform, feature_fn = openclip_loader(model_id[3:])
                else:
                    extra = {"dynamic_img_size": True} if model_id.startswith("vit") else {}
                    model = timm.create_model(model_id, pretrained=True, num_classes=0, **extra).to(dev).eval()
                    data_config = resolve_model_data_config(model)
                    data_config["input_size"] = (3, image_size, image_size)
                    data_config["crop_pct"] = 1.0
                    transform = create_transform(**data_config, is_training=False)
                    feature_fn = lambda m, batch: m(batch)
                out[backbone] = encode(model, transform, feature_fn, backbone)
                model_metadata[backbone] = {"model": model_id, "input_size": image_size,
                                            "embedding_dim": int(out[backbone].shape[1])}
                del model
                torch.cuda.empty_cache()
                print(json.dumps({"backbone_complete": backbone, "shape": list(out[backbone].shape),
                                  "runtime_seconds": round(time.time() - started, 1)}), flush=True)

            metadata = {
                "state": "complete", "app": APP_NAME, "volume": VOLUME_NAME,
                "dataset_path": str(DATASET_ROOT), "annotated_images": len(rows),
                "known": known_n, "out_of_taxonomy": unknown_n, "without_annotation": len(missing),
                "harmonization": "reuse penyisihan/modal-train/harmonize.py: center crop square, resize 150x150 LANCZOS, JPEG q75 4:2:0, RGB",
                "embedding": "frozen model features, per-row L2 normalized, stored float16",
                "backbones": model_metadata, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
                "timm": importlib.metadata.version("timm"), "runtime_seconds": round(time.time() - started, 2),
            }
            out["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
            output_path = REMOTE_ROOT / "external_embeddings.npz"
            with output_path.open("wb") as stream:
                np.savez_compressed(stream, **out)
            VOLUME.commit()
            status("COMPLETE", output_path=str(output_path), completed_image_backbone_units=completed,
                   total_image_backbone_units=len(rows) * len(BACKBONES), runtime_seconds=metadata["runtime_seconds"],
                   gpu=metadata["gpu"])
            print(json.dumps({"state": "COMPLETE", "output_path": str(output_path),
                              "runtime_seconds": metadata["runtime_seconds"], "gpu": metadata["gpu"]}), flush=True)
        except Exception as exc:
            status("FAILED", error_type=type(exc).__name__, error=str(exc), runtime_seconds=round(time.time() - started, 2))
            raise

    @APP.local_entrypoint()
    def launch() -> None:
        call = embed_external.spawn()
        print(json.dumps({"app": APP_NAME, "function_call_id": call.object_id,
                          "dataset_path": str(DATASET_ROOT), "output_path": str(REMOTE_ROOT / "external_embeddings.npz")}),
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.reconstruction_only:
        run_reconstruction_check()
    elif args.score_only:
        score_outputs()
    else:
        raise SystemExit("Use `modal run --detach semifinal/scripts/run_bangladesh_frozen_transfer.py` for external embeddings.")
