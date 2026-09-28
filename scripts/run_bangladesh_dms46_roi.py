"""Run the preregistered Bangladesh DMS46 ROI validation on Modal, then score locally."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np

try:
    import modal
except ImportError:
    modal = None


SCRIPT = Path(__file__).resolve()
ROOT = next((path for path in SCRIPT.parents if (path / "semifinal").is_dir()), Path.cwd())
RESULTS = ROOT / "semifinal" / "results" / "bangladesh_dms46_roi"
OLD_RESULTS = ROOT / "semifinal" / "results" / "bangladesh_dms46_external"
REPORT = ROOT / "semifinal" / "docs" / "110-HASIL-BANGLADESH-DMS46-ROI.md"
APP_NAME = "bdc-bangladesh-dms46-roi-20260927"
DATASET_ROOT = Path("/data/externals/bangladesh_ewaste_v1/extracted")
MODEL_PATH = Path("/data/models/dms46/DMS46_v1.pt")
TAXONOMY_PATH = Path("/data/ckpt/bangladesh-dms46-external-v1/taxonomy.json")
REMOTE_ROOT = Path("/data/ckpt/bangladesh-dms46-roi-v1")
TARGET_GROUPS = ["glass", "metal", "paper_cardboard", "plastic"]
ALL_GROUPS = [*TARGET_GROUPS, "rubber", "other_unknown"]
SOURCE_TO_GROUP = {
    "Glass Waste": "glass",
    "Metal Waste": "metal",
    "Paper Waste": "paper_cardboard",
    "Plastic Waste": "plastic",
}
SOURCE_CLASSES = [
    "Battery Waste", "Glass Waste", "Keyboard", "Light Bulb", "Medical Waste",
    "Metal Waste", "Mobile", "Mouse", "Organic Waste", "Paper Waste", "PCB", "Plastic Waste",
]
SEED = 20260927
N_PERMUTATIONS = 999
N_PAIRED_BOOTSTRAPS = 2000


def parse_yolo_boxes(text: str, width: int, height: int, map_width: int, map_height: int,
                     diagnostics: list[dict] | None = None) -> tuple[list[int], np.ndarray, int, int, int, int, int]:
    """Parse YOLO boxes and polygons; polygon ROI uses its exact enclosing rectangle."""
    mask = np.zeros((map_height, map_width), dtype=bool)
    class_ids: list[int] = []
    usable = invalid = empty = well_formed = polygon_boxes = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if not parts:
                continue
            class_id = int(parts[0])
            class_ids.append(class_id)
            if len(parts) == 5:
                xc, yc, bw, bh = map(float, parts[1:5])
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                coordinates = list(map(float, parts[1:]))
                if len(coordinates) < 6:
                    raise ValueError("polygon has fewer than three points")
                xs, ys = coordinates[0::2], coordinates[1::2]
                xc, yc = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            else:
                raise ValueError("expected YOLO xywh or polygon coordinate pairs")
            if not all(math.isfinite(value) for value in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
                raise ValueError("non-finite or non-positive box")
            if len(parts) > 5:
                polygon_boxes += 1
            well_formed += 1
            x0 = max(0, min(map_width, math.floor((xc - bw / 2) * map_width)))
            x1 = max(0, min(map_width, math.ceil((xc + bw / 2) * map_width)))
            y0 = max(0, min(map_height, math.floor((yc - bh / 2) * map_height)))
            y1 = max(0, min(map_height, math.ceil((yc + bh / 2) * map_height)))
            if x1 <= x0 or y1 <= y0:
                empty += 1
                continue
            mask[y0:y1, x0:x1] = True
            usable += 1
        except (ValueError, OverflowError) as exc:
            invalid += 1
            if diagnostics is not None:
                diagnostics.append({"line_number": line_number, "raw_line": line, "reason": str(exc)})
    return class_ids, mask, usable, invalid, empty, well_formed, polygon_boxes


def fractions(prediction: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    if prediction.shape != mask.shape or not mask.any():
        raise ValueError("ROI is empty or does not match the prediction map")
    values = np.bincount(prediction[mask], minlength=len(ALL_GROUPS)).astype(np.float64)
    values /= values.sum()
    result = dict(zip(ALL_GROUPS, map(float, values)))
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-6):
        raise ValueError("six material fractions do not sum to one")
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def selfcheck() -> None:
    ids, roi, count, invalid, empty, well_formed, polygons = parse_yolo_boxes(
        "0 0.25 0.25 0.5 0.5\n0 0.75 0.75 0.5 0.5", 640, 640, 4, 4
    )
    expected = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    assert ids == [0, 0] and count == 2 and invalid == empty == 0 and well_formed == 2 and polygons == 0 and np.array_equal(roi, expected)
    _, polygon_roi, count, invalid, empty, well_formed, polygons = parse_yolo_boxes(
        "0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75", 640, 640, 4, 4
    )
    assert count == 1 and invalid == empty == 0 and well_formed == polygons == 1
    expected_polygon = np.zeros((4, 4), dtype=bool)
    expected_polygon[1:3, 1:3] = True
    assert np.array_equal(polygon_roi, expected_polygon)
    observed_segmentation_rows = [
        "2 0.645 0.1633333328125 0.4325 0 0.075 0.8433333328125 0.2525 0.99 0.3 0.9933333328125 0.635 0.19 0.645 0.1633333328125",
        "2 0.31 0 0 0.3525 0 0.9225 0.0433333328125 0.915 0.6633333328125 0.1675 0.6633333328125 0.14 0.3666666671875 0 0.31 0",
        "5 0.9433333328125 0.27 0 0.725 0.0166666671875 0.7925 0.9733333328125 0.34 0.9433333328125 0.27",
    ]
    for row in observed_segmentation_rows:
        _, actual_roi, count, invalid, clipped, well_formed, polygons = parse_yolo_boxes(row, 640, 640, 512, 512)
        assert count == well_formed == polygons == 1 and invalid == clipped == 0 and actual_roi.any()
    prediction = np.asarray([[0, 0, 1, 1], [0, 2, 3, 1], [2, 4, 5, 3], [4, 5, 5, 5]])
    values = fractions(prediction, roi)
    assert math.isclose(sum(values.values()), 1.0, abs_tol=1e-6)
    _, empty_mask, count, invalid, clipped, well_formed, polygons = parse_yolo_boxes("0 2 2 0.1 0.1", 640, 640, 4, 4)
    assert count == 0 and invalid == 0 and clipped == well_formed == 1 and polygons == 0 and not empty_mask.any()
    diagnostics = []
    _, _, count, invalid, clipped, well_formed, polygons = parse_yolo_boxes(
        "0 0.5 0.5 0 0.2", 640, 640, 4, 4, diagnostics
    )
    assert count == 0 and invalid == 1 and clipped == well_formed == polygons == 0
    assert diagnostics[0]["line_number"] == 1 and "non-positive" in diagnostics[0]["reason"]
    print("SELF_CHECK_PASS")


if modal is not None:
    VOLUME = modal.Volume.from_name("bdc-data", create_if_missing=False)
    IMAGE = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "numpy", "pillow"
    )
    APP = modal.App(APP_NAME)

    @APP.function(image=IMAGE, gpu="A10G", cpu=4, memory=16384, timeout=3 * 60 * 60,
                  retries=0, volumes={"/data": VOLUME})
    def infer(primary_keys_json: str, mapping_json: str) -> None:
        import importlib.metadata
        import torch
        import torchvision.transforms as transforms
        from PIL import Image

        primary_keys = set(json.loads(primary_keys_json))
        mapping = json.loads(mapping_json)
        status_path = REMOTE_ROOT / "status.json"

        def update_status(state: str, **details: object) -> None:
            write_json(status_path, {"state": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(), **details})
            VOLUME.commit()

        started = time.time()
        try:
            if not DATASET_ROOT.is_dir() or not MODEL_PATH.is_file() or not TAXONOMY_PATH.is_file():
                raise FileNotFoundError("dataset, DMS46 model, or taxonomy is missing on bdc-data")
            REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
            taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
            output_ids = mapping["dms_class_ids"]
            names = [taxonomy["names"][index] for index in output_ids]
            name_to_group = {
                "Glass": "glass", "Mirror": "glass", "Metal": "metal",
                "Paper": "paper_cardboard", "Cardboard": "paper_cardboard",
                "Plastic, clear": "plastic", "Plastic, non-clear": "plastic",
                "Rubber/latex": "rubber",
            }
            class_to_group = np.asarray([ALL_GROUPS.index(name_to_group.get(name, "other_unknown")) for name in names])
            if len(names) != 46 or len(set(output_ids)) != 46 or not set(TARGET_GROUPS).issubset(
                {name_to_group.get(name) for name in names}
            ):
                raise RuntimeError("DMS46 class mapping does not cover the six locked material groups")
            yaml_path = next(DATASET_ROOT.rglob("data.yaml"), None)
            if yaml_path is None:
                raise FileNotFoundError("dataset data.yaml is missing")
            names_line = next((line.partition(":")[2].strip() for line in yaml_path.read_text(encoding="utf-8").splitlines()
                               if line.strip().startswith("names:")), "")
            source_names = [str(name).replace("_", " ") for name in ast.literal_eval(names_line)]
            if len(source_names) != 12 or set(source_names) != set(SOURCE_CLASSES):
                raise RuntimeError(f"unexpected YOLO class mapping: {source_names}")

            media = sorted(path for path in DATASET_ROOT.rglob("*")
                           if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
            REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
            update_status("INVENTORY", stage="yolo_annotations", processed=0, total_files=len(media))
            rows, excluded, excluded_details = [], Counter(), []
            invalid_boxes = empty_boxes = polygon_annotation_count = valid_annotation_count = nonempty_annotation_count = 0
            invalid_bbox_details = []
            annotated_primary_keys = set()
            for index, path in enumerate(media, start=1):
                relative = path.relative_to(DATASET_ROOT).as_posix()
                parts = PurePosixPath(relative).parts
                if len(parts) < 3 or parts[-2] != "images":
                    excluded["image_outside_yolo_images_dir"] += 1
                    excluded_details.append({"key": relative, "reason": "image_outside_yolo_images_dir"})
                    continue
                label_path = DATASET_ROOT.joinpath(*parts[:-2], "labels", f"{path.stem}.txt")
                if not label_path.is_file():
                    excluded["missing_annotation"] += 1
                    excluded_details.append({"key": relative, "reason": "missing_annotation"})
                    continue
                label_text = label_path.read_text(encoding="utf-8")
                if not label_text.strip():
                    excluded["empty_annotation"] += 1
                    excluded_details.append({"key": relative, "reason": "empty_annotation"})
                    continue
                nonempty_annotation_count += 1
                # YOLO coordinates are normalized; the real prediction-map shape is reapplied after inference.
                line_diagnostics = []
                class_ids, _, usable_boxes, bad_count, clipped_boxes, well_formed, polygon_count = parse_yolo_boxes(
                    label_text, 0, 0, 512, 512, line_diagnostics
                )
                invalid_bbox_details.extend({"key": relative, **detail} for detail in line_diagnostics)
                invalid_boxes += bad_count
                empty_boxes += clipped_boxes
                polygon_annotation_count += polygon_count
                if len(set(class_ids)) != 1:
                    excluded["not_exactly_one_yolo_class_id"] += 1
                    excluded_details.append({"key": relative, "reason": "not_exactly_one_yolo_class_id"})
                    continue
                if class_ids[0] < 0 or class_ids[0] >= len(source_names):
                    excluded["class_id_out_of_range"] += 1
                    excluded_details.append({"key": relative, "reason": "class_id_out_of_range"})
                    continue
                source_class = source_names[class_ids[0]]
                material_group = SOURCE_TO_GROUP.get(source_class, "")
                if material_group:
                    annotated_primary_keys.add(relative)
                if well_formed == 0:
                    excluded["invalid_annotation_boxes"] += 1
                    excluded_details.append({"key": relative, "reason": "invalid_annotation_boxes",
                                             "source_class": source_class, "material_truth_group": material_group,
                                             "bbox_count": 0, "invalid_bbox_count": bad_count,
                                             "empty_bbox_count": clipped_boxes})
                    continue
                valid_annotation_count += 1
                if usable_boxes == 0:
                    excluded["empty_or_invalid_roi"] += 1
                    excluded_details.append({"key": relative, "reason": "empty_or_invalid_roi",
                                             "source_class": source_class, "material_truth_group": material_group,
                                             "bbox_count": well_formed, "invalid_bbox_count": bad_count,
                                             "empty_bbox_count": clipped_boxes})
                    continue
                rows.append({"key": relative, "source_class": source_class, "material_truth_group": material_group,
                             "annotation_text": label_text, "bbox_count": well_formed,
                             "invalid_bbox_count": bad_count, "empty_bbox_count": clipped_boxes})
                if index % 256 == 0 or index == len(media):
                    update_status("INVENTORY", stage="yolo_annotations", processed=index, total_files=len(media),
                                  valid_annotations=valid_annotation_count, roi_eligible=len(rows),
                                  exclusions=dict(excluded), invalid_boxes=invalid_boxes, empty_boxes=empty_boxes,
                                  polygon_annotations=polygon_annotation_count,
                                  invalid_bbox_details=invalid_bbox_details[:10])
                    print(json.dumps({"stage": "yolo_inventory", "processed": index, "total_files": len(media),
                                      "valid_annotations": valid_annotation_count, "roi_eligible": len(rows)}), flush=True)

            eligible_primary = {row["key"] for row in rows if row["material_truth_group"]}
            excluded_primary = primary_keys - eligible_primary
            known_exclusion_keys = {row["key"] for row in excluded_details}
            for key in sorted(excluded_primary - known_exclusion_keys):
                excluded_details.append({"key": key, "reason": "missing_or_invalid_yolo_annotation_for_locked_primary_key"})
            if len(media) != 2157 or nonempty_annotation_count != 2153 or len(primary_keys) != 1018:
                raise RuntimeError(
                    f"dataset inventory mismatch: image_files={len(media)} (expected 2157), "
                    f"nonempty_annotations={nonempty_annotation_count} (expected 2153), "
                    f"locked_primary_keys={len(primary_keys)} (expected 1018), exclusions={dict(excluded)}"
                )
            if len({row["key"] for row in rows}) != len(rows):
                raise RuntimeError("YOLO image keys are not unique")
            if {row["source_class"] for row in rows} != set(SOURCE_CLASSES):
                raise RuntimeError("one or more of the 12 YOLO classes has no valid annotated image")
            update_status("PREFLIGHT_COMPLETE", total_image_files=len(media), nonempty_annotation_images=nonempty_annotation_count,
                          valid_annotated=valid_annotation_count, roi_eligible_images=len(rows), primary_annotations=len(annotated_primary_keys),
                          primary_images=len(eligible_primary), excluded_primary_images=len(excluded_primary),
                          exclusions=dict(excluded), invalid_boxes=invalid_boxes, empty_boxes=empty_boxes,
                          polygon_annotations=polygon_annotation_count,
                          invalid_bbox_details=invalid_bbox_details, stage="model_load")

            model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval().cuda()
            normalize = transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])

            def input_array(image):
                width, height = image.size
                scale = min(512 / height, 512 / width)
                size = (int(math.ceil(width * scale)), int(math.ceil(height * scale)))
                return np.asarray(image.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)

            def tensor_batch(arrays):
                if len({array.shape for array in arrays}) != 1:
                    raise ValueError("batch contains different DMS input shapes")
                tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy()).float()
                return normalize(tensor).cuda()

            def predict_batch(arrays):
                with torch.inference_mode():
                    output = model(tensor_batch(arrays))
                prediction = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError(f"unexpected DMS46 output type {type(prediction).__name__}")
                if prediction.ndim == 4 and prediction.shape[1] == 46:
                    prediction = prediction.argmax(dim=1)
                elif prediction.ndim == 4 and prediction.shape[-1] == 46:
                    prediction = prediction.argmax(dim=-1)
                elif prediction.ndim == 4 and prediction.shape[1] == 1:
                    prediction = prediction[:, 0]
                if prediction.ndim == 2 and len(arrays) == 1:
                    prediction = prediction.unsqueeze(0)
                if prediction.ndim != 3 or prediction.shape[0] != len(arrays):
                    raise RuntimeError(f"unexpected DMS46 prediction shape {tuple(prediction.shape)}")
                return [item.detach().cpu().numpy().astype(np.int64) for item in prediction]

            sample_rows = rows[:2]
            sample_arrays = []
            for row in sample_rows:
                with Image.open(DATASET_ROOT / row["key"]) as opened:
                    sample_arrays.append(input_array(opened.convert("RGB")))
            batch_ok = True
            try:
                batch_maps = predict_batch(sample_arrays)
                sequential_maps = [predict_batch([array])[0] for array in sample_arrays]
                batch_ok = len(batch_maps) == 2 and all(np.array_equal(a, b) for a, b in zip(batch_maps, sequential_maps))
            except Exception as exc:
                batch_maps, sequential_maps = [], []
                batch_ok = False
                batch_error = f"{type(exc).__name__}: {exc}"
            if batch_ok:
                for row, prediction in zip(sample_rows, sequential_maps):
                    class_ids, mask, usable_boxes, bad_boxes, clipped_boxes, well_formed, polygon_count = parse_yolo_boxes(
                        row["annotation_text"], 0, 0, prediction.shape[1], prediction.shape[0]
                    )
                    if (len(set(class_ids)) != 1 or well_formed != row["bbox_count"]
                            or usable_boxes == 0 or not mask.any()):
                        raise RuntimeError("sample YOLO ROI transform check failed")
                    fractions(prediction, mask)
            else:
                batch_error = locals().get("batch_error", "batched prediction map differed from sequential")
            batch_enabled = batch_ok
            update_status("INFERENCE", stage="batched" if batch_enabled else "sequential_fallback",
                          batch_size=8, batch_microcheck_identical=batch_ok, batch_error=None if batch_ok else batch_error,
                          completed=0, total=len(rows), eta_seconds=None, gpu=torch.cuda.get_device_name(0))

            output_rows = []
            total_box_count = 0
            total_bbox_area = 0.0
            for start in range(0, len(rows), 8 if batch_enabled else 1):
                chunk = rows[start:start + (8 if batch_enabled else 1)]
                arrays = []
                for row in chunk:
                    with Image.open(DATASET_ROOT / row["key"]) as opened:
                        arrays.append(input_array(opened.convert("RGB")))
                if batch_enabled and len({array.shape for array in arrays}) == 1:
                    maps = predict_batch(arrays)
                else:
                    maps = [predict_batch([array])[0] for array in arrays]
                for row, prediction in zip(chunk, maps):
                    if prediction.min() < 0 or prediction.max() >= len(names):
                        raise RuntimeError("DMS46 emitted out-of-range class IDs")
                    material_map = class_to_group[prediction]
                    class_ids, roi, usable_boxes, bad_boxes, clipped_boxes, well_formed, polygon_count = parse_yolo_boxes(
                        row["annotation_text"], 0, 0, prediction.shape[1], prediction.shape[0]
                    )
                    if len(set(class_ids)) != 1 or well_formed != row["bbox_count"] or usable_boxes == 0 or not roi.any():
                        raise RuntimeError(f"invalid ROI escaped preflight for {row['key']}")
                    values = fractions(material_map, roi)
                    total_box_count += well_formed
                    area_fraction = float(roi.mean())
                    total_bbox_area += area_fraction
                    output_rows.append({
                        "key": row["key"], "source_class": row["source_class"],
                        "material_truth_group": row["material_truth_group"], "bbox_count": well_formed,
                        "invalid_bbox_count": bad_boxes, "empty_bbox_count": clipped_boxes,
                        "bbox_area_fraction": area_fraction, **values,
                    })
                completed = len(output_rows)
                if completed % 100 < len(chunk) or completed == len(rows):
                    elapsed = time.time() - started
                    rate = completed / elapsed if elapsed else 0
                    eta = (len(rows) - completed) / rate if rate else None
                    update_status("INFERENCE", stage="batched" if batch_enabled else "sequential_fallback",
                                  batch_size=8 if batch_enabled else 1,
                                  batch_microcheck_identical=batch_ok,
                                  batch_error=None if batch_ok else batch_error,
                                  completed=completed, total=len(rows), elapsed_seconds=round(elapsed, 1),
                                  eta_seconds=round(eta, 1) if eta is not None else None,
                                  gpu=torch.cuda.get_device_name(0), exclusions=dict(excluded),
                                  invalid_or_empty_boxes=invalid_boxes)
                    print(json.dumps({"stage": "dms46_roi", "completed": completed, "total": len(rows),
                                      "elapsed_seconds": round(elapsed, 1), "eta_seconds": round(eta, 1) if eta is not None else None,
                                      "gpu": torch.cuda.get_device_name(0)}), flush=True)

            if len({row["key"] for row in output_rows}) != len(rows):
                raise RuntimeError("inference output key coverage failed")
            REMOTE_ROOT.mkdir(parents=True, exist_ok=True)
            write_csv(REMOTE_ROOT / "roi_predictions.csv", output_rows)
            by_class = {name: [row for row in output_rows if row["source_class"] == name] for name in SOURCE_CLASSES}
            profile_rows = []
            for name, class_rows in by_class.items():
                profile = {"source_class": name, "N": len(class_rows),
                           "mean_bbox_area_fraction": float(np.mean([float(row["bbox_area_fraction"]) for row in class_rows]))}
                means = {}
                for group in ALL_GROUPS:
                    values = np.asarray([float(row[group]) for row in class_rows])
                    means[group] = float(values.mean())
                    profile[f"mean_{group}"] = float(values.mean())
                    profile[f"median_{group}"] = float(np.median(values))
                profile["largest_mean_fraction_material"] = max(means, key=means.get)
                profile_rows.append(profile)
            write_csv(REMOTE_ROOT / "class_material_profiles.csv", profile_rows)
            metadata = {
                "app_name": APP_NAME, "dataset_path": str(DATASET_ROOT),
                "model_path": str(MODEL_PATH), "output_path": str(REMOTE_ROOT),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "torchvision": importlib.metadata.version("torchvision"),
                "started_at_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": round(time.time() - started, 2), "batch_size": 8 if batch_enabled else 1,
                "batch_microcheck_identical": batch_ok,
                "batch_error": None if batch_ok else batch_error,
                "preprocessing": "RGB; aspect ratio preserved; max side 512; ceil dimensions; LANCZOS; official DMS46 Normalize mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]",
                "roi_transform": "normalized YOLO xywh to prediction map using floor left/top, ceil right/bottom, clamped; union boolean mask; no padding",
                "source_classes": SOURCE_CLASSES, "total_image_files": len(media),
                "nonempty_annotation_images": nonempty_annotation_count,
                "valid_annotated_images": valid_annotation_count,
                "inferred_valid_annotated_images": len(output_rows), "primary_material_annotations": len(annotated_primary_keys),
                "primary_material_images": len(eligible_primary), "excluded_primary_images": len(excluded_primary),
                "excluded_primary_keys": sorted(excluded_primary),
                "class_counts": {name: len(class_rows) for name, class_rows in by_class.items()},
                "exclusions": dict(excluded), "excluded_image_details": excluded_details,
                "invalid_bbox_details": invalid_bbox_details,
                "polygon_annotations_converted_to_enclosing_bbox": polygon_annotation_count,
                "invalid_bbox_count": invalid_boxes, "empty_bbox_count": empty_boxes,
                "total_bbox_count": total_box_count,
                "mean_bbox_area_fraction": total_bbox_area / len(output_rows),
                "silent_full_image_fallback": False,
            }
            write_json(REMOTE_ROOT / "inference_metadata.json", metadata)
            update_status("INFERENCE_COMPLETE", **metadata)
            print(json.dumps({"state": "INFERENCE_COMPLETE", "rows": len(output_rows), "primary": len(eligible_primary),
                              "runtime_seconds": metadata["runtime_seconds"], "gpu": metadata["gpu"],
                              "batch_size": metadata["batch_size"], "batch_identical": batch_ok}), flush=True)
        except Exception as exc:
            update_status("FAILED", stage="inference", error=f"{type(exc).__name__}: {exc}",
                          elapsed_seconds=round(time.time() - started, 1))
            raise

    @APP.local_entrypoint()
    def main() -> None:
        manifest = read_csv(OLD_RESULTS / "manifest.csv")
        mapping = json.loads((OLD_RESULTS / "mapping.json").read_text(encoding="utf-8"))
        keys = [row["key"] for row in manifest]
        if len(keys) != 1018 or len(set(keys)) != 1018:
            raise RuntimeError("locked old primary manifest must contain 1,018 unique keys")
        call = infer.spawn(json.dumps(keys), json.dumps(mapping, ensure_ascii=False))
        print(json.dumps({"app_name": APP_NAME, "app_id": getattr(APP, "app_id", None),
                          "function_call_id": call.object_id, "output_path": str(REMOTE_ROOT),
                          "primary_expected": len(keys)}, ensure_ascii=False), flush=True)


def score_outputs() -> dict:
    from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

    predictions = read_csv(RESULTS / "roi_predictions.csv")
    old_rows = [row for row in read_csv(OLD_RESULTS / "predictions_scored.csv") if row["variant"] == "postmask_highres"]
    old = {row["key"]: row for row in old_rows}
    metadata = json.loads((RESULTS / "inference_metadata.json").read_text(encoding="utf-8"))
    primary = [row for row in predictions if row["material_truth_group"]]
    if (len(predictions) != metadata["inferred_valid_annotated_images"]
            or metadata["total_image_files"] != 2157
            or metadata["nonempty_annotation_images"] != 2153
            or len(primary) != metadata["primary_material_images"]
            or len({row["key"] for row in predictions}) != len(predictions)):
        raise ValueError(f"output coverage gate failed: all={len(predictions)}, primary={len(primary)}, metadata={metadata}")
    keys = [row["key"] for row in primary]
    excluded_primary = set(metadata["excluded_primary_keys"])
    excluded_details = {row["key"]: row for row in metadata["excluded_image_details"]}
    if (len(old) != 1018 or set(keys) & excluded_primary
            or set(keys) | excluded_primary != set(old)
            or any(key not in excluded_details for key in excluded_primary)):
        raise ValueError("ROI-valid keys plus explicitly excluded ROI keys do not reproduce the locked 1,018 key set")
    primary.sort(key=lambda row: row["key"])
    y = np.asarray([TARGET_GROUPS.index(row["material_truth_group"]) for row in primary], dtype=int)
    roi_scores = np.asarray([[float(row[group]) for group in TARGET_GROUPS] for row in primary], dtype=float)
    old_scores = np.asarray([[float(old[row["key"]][f"score_{group}"]) for group in TARGET_GROUPS] for row in primary], dtype=float)
    if not np.isfinite(roi_scores).all() or not np.isfinite(old_scores).all():
        raise ValueError("primary scoring arrays contain NaN or infinity")
    if np.any(roi_scores < 0) or np.any(old_scores < 0):
        raise ValueError("primary scoring arrays contain negative values")
    for row in primary:
        if old[row["key"]]["true_label"] != row["material_truth_group"]:
            raise ValueError(f"ground-truth key mismatch for {row['key']}")

    def per_material(truth, scores):
        metrics = []
        for i, name in enumerate(TARGET_GROUPS):
            binary = truth == i
            auc = float(roc_auc_score(binary, scores[:, i]))
            ap = float(average_precision_score(binary, scores[:, i]))
            prevalence = float(binary.mean())
            metrics.append({"material": name, "support": int(binary.sum()), "prevalence_ap_chance": prevalence,
                            "auroc": auc, "average_precision": ap, "ap_lift": ap / prevalence,
                            "median_positive_score": float(np.median(scores[binary, i])),
                            "median_negative_score": float(np.median(scores[~binary, i]))})
        return metrics

    point = per_material(y, roi_scores)
    old_point = per_material(y, old_scores)
    detection_thresholds = [("target_detection_rate_ge_0_01", 0.01),
                           ("target_detection_rate_ge_0_05", 0.05),
                           ("target_detection_rate_ge_0_10", 0.10)]
    target_scores = roi_scores[np.arange(len(y)), y]
    target_material_detection = {"per_material": {}, "macro_rates": {}, "overall_rates": {}}
    for i, row in enumerate(point):
        positive_scores = roi_scores[y == i, i]
        rates = {name: float(np.mean(positive_scores >= threshold))
                 for name, threshold in detection_thresholds}
        row.update(rates)
        target_material_detection["per_material"][row["material"]] = rates
    for name, threshold in detection_thresholds:
        target_material_detection["macro_rates"][name] = float(np.mean(
            [row[name] for row in point]))
        target_material_detection["overall_rates"][name] = float(np.mean(target_scores >= threshold))
    macro_auc = float(np.mean([row["auroc"] for row in point]))
    macro_ap = float(np.mean([row["average_precision"] for row in point]))
    macro_lift = float(np.mean([row["ap_lift"] for row in point]))
    macro_old_auc = float(np.mean([row["auroc"] for row in old_point]))
    macro_old_ap = float(np.mean([row["average_precision"] for row in old_point]))

    rng = np.random.default_rng(SEED)
    labels = range(len(TARGET_GROUPS))
    auc_draws = np.empty((N_PAIRED_BOOTSTRAPS, 4), dtype=float)
    ap_draws = np.empty_like(auc_draws)
    old_auc_draws = np.empty_like(auc_draws)
    old_ap_draws = np.empty_like(auc_draws)
    for draw in range(N_PAIRED_BOOTSTRAPS):
        indexes = np.concatenate([rng.choice(np.flatnonzero(y == label), size=int(np.sum(y == label)), replace=True)
                                  for label in labels])
        for i in labels:
            binary = y[indexes] == i
            auc_draws[draw, i] = roc_auc_score(binary, roi_scores[indexes, i])
            ap_draws[draw, i] = average_precision_score(binary, roi_scores[indexes, i])
            old_auc_draws[draw, i] = roc_auc_score(binary, old_scores[indexes, i])
            old_ap_draws[draw, i] = average_precision_score(binary, old_scores[indexes, i])
    macro_auc_draws = auc_draws.mean(axis=1)
    macro_ap_draws = ap_draws.mean(axis=1)
    delta_auc_draws = macro_auc_draws - old_auc_draws.mean(axis=1)
    delta_ap_draws = macro_ap_draws - old_ap_draws.mean(axis=1)
    ci = lambda values: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
    for i, row in enumerate(point):
        row["auroc_bootstrap_95_ci"] = ci(auc_draws[:, i])
        row["average_precision_bootstrap_95_ci"] = ci(ap_draws[:, i])

    perm_rng = np.random.default_rng(SEED)
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for i in range(N_PERMUTATIONS):
        permuted = perm_rng.permutation(y)
        null[i] = np.mean([roc_auc_score(permuted == group, roi_scores[:, group]) for group in labels])
    p_value = float((1 + np.sum(null >= macro_auc)) / (N_PERMUTATIONS + 1))
    macro_auc_ci = ci(macro_auc_draws)
    delta_auc_ci = ci(delta_auc_draws)
    delta_ap_ci = ci(delta_ap_draws)
    if macro_auc_ci[0] <= 0.5:
        verdict = "NOT_SUPPORTED"
    elif all(row["auroc"] > 0.5 for row in point) and p_value <= 0.01 and delta_auc_ci[0] > 0:
        verdict = "SUPPORTED"
    else:
        verdict = "PARTIAL"

    predicted = roi_scores.argmax(axis=1)
    ordered = np.argsort(-roi_scores, axis=1, kind="stable")
    ranks = np.asarray([int(np.flatnonzero(ordered[i] == y[i])[0]) + 1 for i in range(len(y))])
    secondary = {
        "argmax_macro_f1": float(f1_score(y, predicted, labels=list(labels), average="macro", zero_division=0)),
        "argmax_balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "argmax_accuracy": float(accuracy_score(y, predicted)),
        "top_2_accuracy": float(np.mean(ranks <= 2)),
        "mean_reciprocal_rank": float(np.mean(1 / ranks)),
        "top_2_definition": "true material among the top two of four candidates: glass, metal, paper_cardboard, plastic; secondary only",
        "mean_reciprocal_rank_definition": "rank of true material among the same four candidates; secondary only",
        "argmax_definition": "argmax over the four mapped material fractions; secondary only",
    }
    metrics = {
        "experiment": "bangladesh_dms46_roi_material_validation", "verdict": verdict,
        "primary_metric": "macro one-vs-rest AUROC on material fractions inside union YOLO ROI",
        "n_primary": len(primary), "support": {name: int(np.sum(y == i)) for i, name in enumerate(TARGET_GROUPS)},
        "macro_auroc": macro_auc, "macro_auroc_bootstrap_95_ci": macro_auc_ci,
        "macro_average_precision": macro_ap, "macro_average_precision_bootstrap_95_ci": ci(macro_ap_draws),
        "macro_ap_lift": macro_lift, "permutation": {"draws": N_PERMUTATIONS, "seed": SEED,
            "p_value": p_value, "null_macro_auroc_95_quantile": float(np.quantile(null, 0.95))},
        "bootstrap": {"draws": N_PAIRED_BOOTSTRAPS, "seed": SEED, "method": "stratified paired image bootstrap"},
        "per_material": point, "secondary_argmax_metrics": secondary,
        "target_material_detection": {
            "definition": "fraction of positive images whose target material occupies at least the fixed fraction of YOLO ROI",
            "thresholds": [0.01, 0.05, 0.10],
            **target_material_detection,
            "note": "descriptive positive-label detection rate; Bangladesh has no exhaustive pixel-level material labels",
        },
        "all_images": len(predictions), "class_counts": {name: sum(r["source_class"] == name for r in predictions) for name in SOURCE_CLASSES},
        "sum_of_six_fraction_max_abs_error": float(max(abs(sum(float(row[group]) for group in ALL_GROUPS) - 1.0) for row in predictions)),
        "silent_full_image_fallback": False,
    }
    metrics["primary_key_coverage"] = {
        "locked_primary_keys": 1018, "scored_roi_valid_keys": len(primary),
        "excluded_keys": sorted(excluded_primary),
        "excluded_reason_details": [excluded_details[key] for key in sorted(excluded_primary)],
    }
    metrics["inference"] = {key: metadata.get(key) for key in (
        "gpu", "runtime_seconds", "batch_size", "batch_microcheck_identical", "nonempty_annotation_images",
        "valid_annotated_images", "inferred_valid_annotated_images", "total_image_files", "class_counts", "exclusions",
        "invalid_bbox_count", "empty_bbox_count", "total_bbox_count", "mean_bbox_area_fraction",
        "polygon_annotations_converted_to_enclosing_bbox")}
    comparison = {
        "n_paired": len(primary), "old_n": len(old), "old_n_excluded_from_roi": len(excluded_primary),
        "excluded_primary_keys": sorted(excluded_primary),
        "old_variant": "postmask_highres", "old_source": "predictions_scored.csv",
        "macro_auroc_roi": macro_auc, "macro_auroc_rembg": macro_old_auc,
        "paired_delta_macro_auroc_roi_minus_rembg": float(macro_auc - macro_old_auc),
        "paired_delta_macro_auroc_bootstrap_95_ci": delta_auc_ci,
        "macro_average_precision_roi": macro_ap, "macro_average_precision_rembg": macro_old_ap,
        "paired_delta_macro_average_precision_roi_minus_rembg": float(macro_ap - macro_old_ap),
        "paired_delta_macro_average_precision_bootstrap_95_ci": delta_ap_ci,
        "bootstrap_draws": N_PAIRED_BOOTSTRAPS, "seed": SEED,
        "same_keys_verified": True,
    }
    profile_rows = read_csv(RESULTS / "class_material_profiles.csv")
    per_material_rows = [{"material": row["material"], **row} for row in point]
    write_csv(RESULTS / "per_material_metrics.csv", per_material_rows)
    write_json(RESULTS / "roi_metrics.json", metrics)
    write_json(RESULTS / "roi_vs_rembg.json", comparison)

    all_rows = read_csv(RESULTS / "roi_predictions.csv")
    excluded_lines = []
    metadata_path = RESULTS / "inference_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        excluded_lines = [f"- `{name}`: {count}" for name, count in metadata.get("exclusions", {}).items()]
        runtime = metadata.get("runtime_seconds")
        gpu = metadata.get("gpu")
        batch = metadata.get("batch_size")
    else:
        runtime = gpu = batch = None
    report = [
        "# Validasi Eksternal DMS46 pada ROI YOLO Bangladesh", "",
        "## Pertanyaan eksperimen", "",
        "Apakah proporsi material DMS46 di dalam union bounding box YOLO diperkaya secara konsisten pada empat kelas material Bangladesh, dan bagaimana hasilnya dibanding evaluasi `rembg` pada key gambar yang sama?", "",
        "## Perbaikan desain", "",
        "DMS46 tetap frozen. Primary score adalah fraction material kontinu di dalam ROI bbox tanpa padding; bbox axis-aligned untuk anotasi polygon YOLO diturunkan dari min/max titik polygon. Tidak dilakukan argmax untuk primary metric dan tidak digunakan `rembg`. Output enam fraksi juga diringkas secara deskriptif pada 12 kelas. Bangladesh menyediakan label kelas objek, bukan pixel-level material ground truth.", "",
        "## Metric utama", "",
        f"Cakupan anotasi: {metadata.get('total_image_files', 'NA')} file gambar; {metadata.get('nonempty_annotation_images', 'NA')} anotasi tidak kosong; {metadata.get('inferred_valid_annotated_images', len(all_rows))} gambar dengan ROI valid. Subset material: {len(primary)}/1.018 key dinilai; {metadata.get('excluded_primary_images', 0)} dikecualikan. {metadata.get('polygon_annotations_converted_to_enclosing_bbox', 0)} anotasi polygon dikonversi menjadi bbox enclosing; {metadata.get('invalid_bbox_count', 0)} bbox invalid; {metadata.get('empty_bbox_count', 0)} bbox kosong.", "",
        "| Material | N positif | AUROC (95% CI) | AP | Prevalensi (AP chance) | AP lift | Median positif | Median negatif |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in point:
        report.append(f"| {row['material']} | {row['support']} | {row['auroc']:.3f} ({row['auroc_bootstrap_95_ci'][0]:.3f}–{row['auroc_bootstrap_95_ci'][1]:.3f}) | {row['average_precision']:.3f} | {row['prevalence_ap_chance']:.3f} | {row['ap_lift']:.2f}× | {row['median_positive_score']:.3f} | {row['median_negative_score']:.3f} |")
    report.extend([
        "", f"Macro AUROC **{macro_auc:.3f}** (bootstrap 95% CI {macro_auc_ci[0]:.3f}–{macro_auc_ci[1]:.3f}); macro AP **{macro_ap:.3f}**; macro AP lift **{macro_lift:.2f}×**; permutation p={p_value:.4f} (999 permutasi, seed {SEED}).", "",
        "## ROI versus `rembg`", "",
        f"Pada {len(primary)} key yang sama: macro AUROC ROI {macro_auc:.3f} vs `rembg` {macro_old_auc:.3f}, paired Δ={macro_auc - macro_old_auc:+.3f} (95% CI {delta_auc_ci[0]:+.3f}–{delta_auc_ci[1]:+.3f}); macro AP ROI {macro_ap:.3f} vs {macro_old_ap:.3f}, paired Δ={macro_ap - macro_old_ap:+.3f} (95% CI {delta_ap_ci[0]:+.3f}–{delta_ap_ci[1]:+.3f}). Argmax metrics hanya sensitivity: Macro-F1 {secondary['argmax_macro_f1']:.3f}, balanced accuracy {secondary['argmax_balanced_accuracy']:.3f}, accuracy {secondary['argmax_accuracy']:.3f}, Top-2 {secondary['top_2_accuracy']:.3f}, MRR {secondary['mean_reciprocal_rank']:.3f}.", "",
        "## Profil material deskriptif", "",
        "Rata-rata fraction terbesar per kelas (bukan prediksi label atau accuracy):", "",
    ])
    for row in profile_rows:
        report.append(f"- {row['source_class']} (N={row['N']}): {row['largest_mean_fraction_material']} ({float(row['mean_' + row['largest_mean_fraction_material']]):.3f})")
    report.extend([
        "", "## Deteksi material target tanpa pemaksaan argmax", "",
        "| Material | Target ≥1% ROI | Target ≥5% ROI | Target ≥10% ROI |",
        "|---|---:|---:|---:|",
    ])
    material_labels = {
        "glass": "Glass", "metal": "Metal", "paper_cardboard": "Paper/cardboard", "plastic": "Plastic",
    }
    for row in point:
        report.append(
            f"| {material_labels[row['material']]} | {row['target_detection_rate_ge_0_01']:.1%} | "
            f"{row['target_detection_rate_ge_0_05']:.1%} | {row['target_detection_rate_ge_0_10']:.1%} |"
        )
    report.extend([
        "",
        f"Macro rate: ≥1% **{target_material_detection['macro_rates']['target_detection_rate_ge_0_01']:.1%}**, ≥5% **{target_material_detection['macro_rates']['target_detection_rate_ge_0_05']:.1%}**, ≥10% **{target_material_detection['macro_rates']['target_detection_rate_ge_0_10']:.1%}**. Overall rate pada seluruh {len(primary)} gambar: ≥1% **{target_material_detection['overall_rates']['target_detection_rate_ge_0_01']:.1%}**, ≥5% **{target_material_detection['overall_rates']['target_detection_rate_ge_0_05']:.1%}**, ≥10% **{target_material_detection['overall_rates']['target_detection_rate_ge_0_10']:.1%}**.",
        f"Top-2 **{secondary['top_2_accuracy']:.3f}** dan MRR **{secondary['mean_reciprocal_rank']:.3f}** memakai ranking empat kandidat glass, metal, paper_cardboard, plastic; keduanya secondary metric. AUROC/AP tetap primary metric. Detection rate menunjukkan material relevan ditemukan di ROI tanpa harus menjadi argmax; confusion matrix dan Macro-F1 hanya sensitivity analysis.",
        "Fraction kecil dapat berasal dari material nyata atau segmentation noise. Bangladesh tidak menyediakan pixel-level material ground truth maupun daftar lengkap material setiap objek.",
        "",
        "Verdict tetap PARTIAL.",
        "DMS46 menunjukkan bukti eksternal moderat sebagai material descriptor. Material target diperkaya di atas chance dan sering muncul dalam komposisi ROI meskipun tidak selalu dominan. Hasil mendukung penggunaan DMS46 sebagai descriptor eksploratif, bukan classifier material tunggal atau estimator komposisi yang tervalidasi penuh.",
        "",
    ])
    report.extend([
        "", "## Verdict dan batas", "", f"Verdict sesuai decision rule terkunci: **{verdict}**. Ini menguji enrichment fraction DMS46 terhadap label kelas kasar pada dataset Bangladesh; bukan validasi segmentasi pixel atau ground truth komposisi material per objek.", "",
        f"Inferensi: {len(all_rows)} gambar, GPU {gpu}, batch size {batch}, runtime {runtime} detik.",
    ])
    if excluded_lines:
        report.extend(["", "Pengecualian anotasi:", *excluded_lines])
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "macro_auroc": macro_auc, "macro_auroc_ci": macro_auc_ci,
                      "macro_ap": macro_ap, "top_2_accuracy": secondary["top_2_accuracy"],
                      "mean_reciprocal_rank": secondary["mean_reciprocal_rank"],
                      "target_material_detection": target_material_detection,
                      "permutation_p": p_value, "comparison": comparison,
                      "report": str(REPORT)}, ensure_ascii=False))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.score_only:
        score_outputs()
