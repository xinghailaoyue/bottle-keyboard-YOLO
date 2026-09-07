#!/usr/bin/env python3
"""Bottle/keyboard YOLO project: all project entry functions in one file.

This file consolidates the project's own orchestration code. Ultralytics,
OpenCV, PyTorch and Pillow remain external libraries and are not copied here.

Subcommands:
    prepare        Build an Open Images dataset with optional hard negatives.
    train          Train the baseline or conservatively fine-tune the model.
    validate       Evaluate one model on a YOLO validation split.
    smoke          Run a deterministic per-class sample test.
    compare        Compare multiple models on the same validation split.
    hard-negative Compare false detections on person-only images.
    export         Export a model to ONNX or TensorRT.
    camera         Probe cameras or run live detection, snapshots and recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

import cv2  # noqa: E402  (environment is configured before importing Ultralytics)
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402


# -----------------------------------------------------------------------------
# Project defaults
# -----------------------------------------------------------------------------

BASELINE_MODEL = ROOT / "exports" / "bottle_keyboard_yolo11n_best.pt"
EXPANDED_MODEL = ROOT / "exports" / "bottle_keyboard_yolo11n_expanded_best.pt"
DATA_V1 = ROOT / "data.yaml"
DATA_V2 = ROOT / "data_v2.yaml"
CAMERA_OUTPUT_DIR = ROOT / "runs" / "camera_test"

TARGETS = {
    "/m/04dr76w": (0, "bottle"),
    "/m/01m2v": (1, "keyboard"),
}
PERSON_MID = "/m/01g317"

CAMERA_BACKENDS = {
    "msmf": (cv2.CAP_MSMF, "Media Foundation"),
    "dshow": (cv2.CAP_DSHOW, "DirectShow"),
    "auto": (cv2.CAP_ANY, "OpenCV Auto"),
}


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a user path relative to the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def model_argument(value: str | Path) -> str:
    """Resolve local model files while still allowing names such as yolo11n.pt."""
    path = Path(value)
    candidate = path if path.is_absolute() else ROOT / path
    return str(candidate.resolve()) if candidate.exists() else str(value)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_model_specs(specs: list[str] | None, defaults: dict[str, Path]) -> dict[str, Path]:
    """Parse repeated NAME=PATH model arguments."""
    if not specs:
        return defaults
    parsed = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model specification must be NAME=PATH: {spec}")
        name, value = spec.split("=", 1)
        if not name.strip():
            raise ValueError(f"Model name is empty: {spec}")
        parsed[name.strip()] = resolve_project_path(value.strip())
    return parsed


# -----------------------------------------------------------------------------
# 1. Dataset preparation
# -----------------------------------------------------------------------------

def load_openimages_annotations(metadata_dir: Path):
    """Read Bottle, Computer keyboard and Person boxes from Open Images CSVs."""
    target_records = defaultdict(list)
    class_images = defaultdict(set)
    person_images = set()

    for source_split in ("validation", "test"):
        annotation_path = metadata_dir / f"{source_split}-annotations-bbox.csv"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                mid = row["LabelName"]
                key = (source_split, row["ImageID"])
                if mid == PERSON_MID and row.get("IsDepiction") != "1":
                    person_images.add(key)
                if mid not in TARGETS:
                    continue
                if row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1":
                    continue
                class_id, class_name = TARGETS[mid]
                target_records[key].append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "xmin": float(row["XMin"]),
                        "xmax": float(row["XMax"]),
                        "ymin": float(row["YMin"]),
                        "ymax": float(row["YMax"]),
                    }
                )
                class_images[class_id].add(key)

    # These images contain an annotated person but no target-class box.
    negative_candidates = person_images - set(target_records)
    return target_records, class_images, negative_candidates


def sample_items(items, count: int, seed: int):
    values = sorted(items)
    random.Random(seed).shuffle(values)
    return values[:count]


def validation_items(items, fraction: float, seed: int):
    values = list(items)
    random.Random(seed).shuffle(values)
    if not values:
        return set()
    count = max(1, round(len(values) * fraction))
    return set(values[:count])


def load_openimages_metadata(metadata_dir: Path, selected):
    wanted = defaultdict(set)
    for source_split, image_id in selected:
        wanted[source_split].add(image_id)
    metadata = {}
    for source_split in ("validation", "test"):
        path = metadata_dir / f"{source_split}-images.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["ImageID"] in wanted[source_split]:
                    metadata[(source_split, row["ImageID"])] = row
    return metadata


def openimages_url(source_split: str, image_id: str) -> str:
    return f"https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg"


def valid_image(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def fetch_openimages_image(key, destination: Path, reuse_dir: Path):
    """Reuse an existing image when possible; otherwise download with retries."""
    source_split, image_id = key
    stem = f"{source_split}_{image_id}"
    url = openimages_url(source_split, image_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_image(destination):
        return url, "existing", None

    for subset in ("train", "val"):
        reusable = reuse_dir / "images" / subset / f"{stem}.jpg"
        if valid_image(reusable):
            shutil.copy2(reusable, destination)
            return url, "reuse-copy", None

    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                destination.write_bytes(response.read())
            if not valid_image(destination):
                raise ValueError("downloaded file is not a valid image")
            return url, "download", None
        except Exception as exc:
            last_error = str(exc)
            destination.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    return url, "failed", last_error


def boxes_to_yolo_lines(boxes) -> str:
    """Convert normalized xmin/xmax/ymin/ymax boxes to YOLO xywh lines."""
    lines = []
    for box in boxes:
        width = box["xmax"] - box["xmin"]
        height = box["ymax"] - box["ymin"]
        x_center = box["xmin"] + width / 2
        y_center = box["ymin"] + height / 2
        lines.append(
            f'{box["class_id"]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        )
    return "\n".join(lines) + ("\n" if lines else "")


def command_prepare(args: argparse.Namespace) -> None:
    metadata_dir = resolve_project_path(args.metadata_dir)
    output_dir = resolve_project_path(args.output_dir)
    reuse_dir = resolve_project_path(args.reuse_dir)
    external_dir = output_dir / "external" / "openimages"
    external_dir.mkdir(parents=True, exist_ok=True)

    records, class_images, negative_candidates = load_openimages_annotations(metadata_dir)
    groups = {
        "bottle": sample_items(class_images[0], args.bottle_images, args.seed + 1),
        "keyboard": sample_items(class_images[1], args.keyboard_images, args.seed + 2),
        "person_negative": sample_items(negative_candidates, args.negative_images, args.seed + 3),
    }
    requested = {
        "bottle": args.bottle_images,
        "keyboard": args.keyboard_images,
        "person_negative": args.negative_images,
    }
    for role, values in groups.items():
        if len(values) < requested[role]:
            raise RuntimeError(f"Only {len(values)} samples are available for {role}")

    selected = set().union(*(set(values) for values in groups.values()))
    roles = defaultdict(set)
    for role, values in groups.items():
        for key in values:
            roles[key].add(role)

    val_keys = set()
    for offset, values in enumerate(groups.values()):
        val_keys.update(validation_items(values, args.val_fraction, args.seed + 100 + offset))
    metadata = load_openimages_metadata(metadata_dir, selected)

    jobs = []
    for key in sorted(selected):
        subset = "val" if key in val_keys else "train"
        source_split, image_id = key
        stem = f"{source_split}_{image_id}"
        destination = output_dir / "images" / subset / f"{stem}.jpg"
        jobs.append((key, subset, stem, destination))

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(fetch_openimages_image, key, destination, reuse_dir):
                (key, subset, stem, destination)
            for key, subset, stem, destination in jobs
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            key, subset, stem, destination = future_map[future]
            url, method, error = future.result()
            if error:
                failures.append({"source_split": key[0], "image_id": key[1], "error": error})
            else:
                label_path = output_dir / "labels" / subset / f"{stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(boxes_to_yolo_lines(records.get(key, [])), encoding="utf-8")
                successes.append((key, subset, stem, destination, url, method))
            if index % 50 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)} images")

    manifest_fields = [
        "local_file", "dataset_split", "selection_role", "openimages_split", "image_id",
        "acquisition", "download_url", "original_url", "original_landing_url", "license",
        "author", "title",
    ]
    manifest_path = external_dir / "source_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for key, subset, stem, destination, url, method in sorted(successes):
            source = metadata.get(key, {})
            writer.writerow(
                {
                    "local_file": destination.relative_to(ROOT).as_posix(),
                    "dataset_split": subset,
                    "selection_role": "+".join(sorted(roles[key])),
                    "openimages_split": key[0],
                    "image_id": key[1],
                    "acquisition": method,
                    "download_url": url,
                    "original_url": source.get("OriginalURL", ""),
                    "original_landing_url": source.get("OriginalLandingURL", ""),
                    "license": source.get("License", ""),
                    "author": source.get("Author", ""),
                    "title": source.get("Title", ""),
                }
            )

    image_counts = Counter(item[1] for item in successes)
    method_counts = Counter(item[5] for item in successes)
    object_counts = Counter()
    negative_count = 0
    for key, *_ in successes:
        boxes = records.get(key, [])
        if not boxes:
            negative_count += 1
        object_counts.update(box["class_name"] for box in boxes)

    stats = {
        "images": dict(sorted(image_counts.items())),
        "total_images": len(successes),
        "objects": dict(sorted(object_counts.items())),
        "hard_negative_images": negative_count,
        "requested_roles": requested,
        "unique_selected_images": len(selected),
        "acquisition": dict(sorted(method_counts.items())),
        "failed_downloads": len(failures),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    write_json(output_dir / "stats.json", stats)
    if failures:
        write_json(external_dir / "download_failures.json", {"failures": failures})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 2. Model training
# -----------------------------------------------------------------------------

TRAINING_PROFILES = {
    "baseline": {
        "model": "yolo11n.pt",
        "data": DATA_V1,
        "epochs": 60,
        "name": "baseline_yolo11n_all_in_one",
        "optimizer": "auto",
        "patience": 15,
        "seed": 42,
        "extra": {"pretrained": True, "close_mosaic": 10},
    },
    "expanded": {
        "model": BASELINE_MODEL,
        "data": DATA_V2,
        "epochs": 60,
        "name": "expanded_low_lr_yolo11n_all_in_one",
        "optimizer": "AdamW",
        "patience": 20,
        "seed": 84,
        "extra": {
            "lr0": 0.0002,
            "lrf": 0.1,
            "cos_lr": True,
            "weight_decay": 0.0005,
            "mosaic": 0.2,
            "scale": 0.25,
            "translate": 0.05,
            "close_mosaic": 8,
            "cls": 0.7,
        },
    },
}


def command_train(args: argparse.Namespace) -> None:
    profile = TRAINING_PROFILES[args.profile]
    source_model = args.model or profile["model"]
    data = resolve_project_path(args.data or profile["data"])
    epochs = args.epochs or profile["epochs"]
    name = args.name or profile["name"]

    model = YOLO(model_argument(source_model))
    settings = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(ROOT / "runs" / "detect"),
        "name": name,
        "exist_ok": True,
        "optimizer": profile["optimizer"],
        "patience": profile["patience"],
        "seed": profile["seed"],
        "deterministic": True,
        "plots": True,
        "verbose": True,
    }
    settings.update(profile["extra"])
    model.train(**settings)


# -----------------------------------------------------------------------------
# 3. Standard validation and model comparison
# -----------------------------------------------------------------------------

def metrics_from_validation(result, weights: Path | str) -> dict:
    per_class = {}
    for index, class_name in result.names.items():
        per_class[class_name] = {
            "precision": round(float(result.box.p[index]), 6),
            "recall": round(float(result.box.r[index]), 6),
            "map50": round(float(result.box.ap50[index]), 6),
            "map50_95": round(float(result.box.ap[index]), 6),
        }
    return {
        "weights": str(weights),
        "all": {
            "precision": round(float(result.box.mp), 6),
            "recall": round(float(result.box.mr), 6),
            "map50": round(float(result.box.map50), 6),
            "map50_95": round(float(result.box.map), 6),
        },
        "classes": per_class,
    }


def evaluate_model(weights: Path, data: Path, run_name: str, args: argparse.Namespace) -> dict:
    result = YOLO(str(weights)).val(
        data=str(data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        project=str(ROOT / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    return metrics_from_validation(result, weights)


def command_validate(args: argparse.Namespace) -> None:
    weights = resolve_project_path(args.model)
    data = resolve_project_path(args.data)
    report = evaluate_model(weights, data, args.name, args)
    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_compare(args: argparse.Namespace) -> None:
    defaults = {"baseline": BASELINE_MODEL, "expanded": EXPANDED_MODEL}
    models = parse_model_specs(args.models, defaults)
    data = resolve_project_path(args.data)
    report = {"evaluation_dataset": str(data), "models": {}, "deltas_vs_first": {}}
    for name, weights in models.items():
        report["models"][name] = evaluate_model(weights, data, f"compare_{name}_all_in_one", args)

    first_name = next(iter(models))
    baseline_metrics = report["models"][first_name]
    for name in list(models)[1:]:
        candidate = report["models"][name]
        report["deltas_vs_first"][name] = {
            "all": {
                metric: round(candidate["all"][metric] - baseline_metrics["all"][metric], 6)
                for metric in baseline_metrics["all"]
            },
            "classes": {
                class_name: {
                    metric: round(
                        candidate["classes"][class_name][metric]
                        - baseline_metrics["classes"][class_name][metric],
                        6,
                    )
                    for metric in baseline_metrics["classes"][class_name]
                }
                for class_name in baseline_metrics["classes"]
            },
        }

    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 4. Deterministic per-class smoke test
# -----------------------------------------------------------------------------

def read_yolo_labels(path: Path):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = map(float, line.split())
        labels.append((int(class_id), x, y, width, height))
    return labels


def normalized_to_xyxy(label, image_width: int, image_height: int):
    _, x, y, width, height = label
    return (
        (x - width / 2) * image_width,
        (y - height / 2) * image_height,
        (x + width / 2) * image_width,
        (y + height / 2) * image_height,
    )


def box_iou(box_a, box_b) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def image_for_label(images_dir: Path, stem: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for label stem: {stem}")


def command_smoke(args: argparse.Namespace) -> None:
    dataset_dir = resolve_project_path(args.dataset_dir)
    labels_dir = dataset_dir / "labels" / "val"
    images_dir = dataset_dir / "images" / "val"
    rng = random.Random(args.seed)
    class_names = {0: "bottle", 1: "keyboard"}

    selected = {}
    for class_id in class_names:
        candidates = [
            path for path in sorted(labels_dir.glob("*.txt"))
            if any(label[0] == class_id for label in read_yolo_labels(path))
        ]
        rng.shuffle(candidates)
        selected[class_id] = candidates[:args.per_class]

    model = YOLO(str(resolve_project_path(args.model)))
    report = {
        "criteria": f"correct class and IoU >= {args.match_iou:.2f} at conf >= {args.conf:.2f}",
        "classes": {},
    }
    for class_id, label_paths in selected.items():
        image_paths = [image_for_label(images_dir, path.stem) for path in label_paths]
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            save=args.save_images,
            project=str(ROOT / "runs" / "detect"),
            name=f"smoke_{class_names[class_id]}_all_in_one",
            exist_ok=True,
            verbose=False,
        )
        rows = []
        for label_path, image_path, result in zip(label_paths, image_paths, results):
            image_height, image_width = result.orig_shape
            truths = [
                normalized_to_xyxy(label, image_width, image_height)
                for label in read_yolo_labels(label_path)
                if label[0] == class_id
            ]
            predictions = []
            if result.boxes is not None:
                for xyxy, predicted_class, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    if int(predicted_class) == class_id:
                        predictions.append((xyxy, confidence))
            best_iou = max(
                (box_iou(prediction[0], truth) for prediction in predictions for truth in truths),
                default=0.0,
            )
            rows.append(
                {
                    "image": image_path.name,
                    "success": best_iou >= args.match_iou,
                    "best_iou": round(best_iou, 4),
                    "best_confidence": round(
                        max((prediction[1] for prediction in predictions), default=0.0), 4
                    ),
                }
            )
        successes = sum(row["success"] for row in rows)
        report["classes"][class_names[class_id]] = {
            "successes": successes,
            "total": len(rows),
            "rate": round(successes / len(rows), 6) if rows else 0.0,
            "images": rows,
        }

    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 5. Hard-negative evaluation
# -----------------------------------------------------------------------------

def evaluate_false_detections(weights: Path, images: list[Path], args: argparse.Namespace) -> dict:
    model = YOLO(str(weights))
    false_boxes = Counter()
    images_with_false_detection = Counter()
    results = model.predict(
        source=[str(path) for path in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        verbose=False,
    )
    any_false_images = 0
    for result in results:
        seen = set()
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                class_name = result.names[class_id]
                false_boxes[class_name] += 1
                seen.add(class_name)
        if seen:
            any_false_images += 1
        for class_name in seen:
            images_with_false_detection[class_name] += 1
    return {
        "weights": str(weights),
        "negative_images": len(images),
        "confidence": args.conf,
        "images_with_any_false_detection": any_false_images,
        "images_with_false_detection_by_class": dict(images_with_false_detection),
        "false_detection_boxes_by_class": dict(false_boxes),
    }


def command_hard_negative(args: argparse.Namespace) -> None:
    manifest = resolve_project_path(args.manifest)
    images = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if "person_negative" in row["selection_role"].split("+"):
                images.append(resolve_project_path(row["local_file"]))
    images.sort()
    if not images:
        raise RuntimeError("No person_negative images were found in the manifest")

    defaults = {"baseline": BASELINE_MODEL, "expanded": EXPANDED_MODEL}
    models = parse_model_specs(args.models, defaults)
    report = {
        name: evaluate_false_detections(weights, images, args)
        for name, weights in models.items()
    }
    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 6. Model export
# -----------------------------------------------------------------------------

def command_export(args: argparse.Namespace) -> None:
    weights = resolve_project_path(args.model)
    model = YOLO(str(weights))
    export_device = "0" if args.format == "engine" and args.device == "cpu" else args.device
    settings = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "device": export_device,
    }
    if args.format == "onnx":
        settings["opset"] = args.opset
    elif args.precision == "fp16":
        settings["quantize"] = 16
    elif args.precision == "int8":
        settings["quantize"] = 8
        settings["data"] = str(resolve_project_path(args.data))
    output = model.export(**settings)
    print(f"Exported model: {output}")


# -----------------------------------------------------------------------------
# 7. Camera probe, live inference, snapshots and annotated recording
# -----------------------------------------------------------------------------

def frame_stats(frame):
    return round(float(frame.mean()), 2), round(float(frame.std()), 2)


def open_camera_backend(index: int, backend_key: str):
    backend_id, backend_name = CAMERA_BACKENDS[backend_key]
    camera = cv2.VideoCapture(index, backend_id)
    if not camera.isOpened():
        camera.release()
        return None, backend_name, None
    frame = None
    for _ in range(30):
        ok, candidate = camera.read()
        if ok and candidate is not None and candidate.size:
            frame = candidate
        time.sleep(0.01)
    if frame is None:
        camera.release()
        return None, backend_name, None
    return camera, backend_name, frame


def open_camera(index: int, preference: str):
    keys = [preference] if preference != "auto" else ["msmf", "dshow", "auto"]
    dark_fallback = None
    for key in keys:
        camera, name, frame = open_camera_backend(index, key)
        if camera is None:
            continue
        mean, std = frame_stats(frame)
        is_black = mean < 3.0 and std < 3.0
        print(
            f"Camera candidate {index}/{name}: {frame.shape[1]}x{frame.shape[0]}, "
            f"brightness={mean}, contrast={std}, black={is_black}"
        )
        if not is_black:
            if dark_fallback is not None:
                dark_fallback[0].release()
            return camera, name, frame
        if dark_fallback is None:
            dark_fallback = (camera, name, frame)
        else:
            camera.release()
    return dark_fallback if dark_fallback is not None else (None, None, None)


def probe_cameras(max_index: int, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for index in range(max_index + 1):
        for backend_key in ("msmf", "dshow", "auto"):
            camera, backend, frame = open_camera_backend(index, backend_key)
            if camera is None:
                continue
            height, width = frame.shape[:2]
            mean, std = frame_stats(frame)
            preview = output_dir / f"probe_camera{index}_{backend_key}.jpg"
            cv2.imwrite(str(preview), frame)
            found.append(
                {
                    "index": index,
                    "backend": backend,
                    "width": width,
                    "height": height,
                    "brightness": mean,
                    "contrast": std,
                    "appears_black": mean < 3.0 and std < 3.0,
                    "preview": str(preview),
                }
            )
            camera.release()
    print(json.dumps({"cameras": found}, ensure_ascii=False, indent=2))
    return 0 if found else 1


def create_video_writer(frame, fps: float, output_dir: Path):
    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for codec, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        video_path = output_dir / f"recording_{timestamp}{suffix}"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            return writer, {
                "file": str(video_path.relative_to(ROOT)),
                "codec": codec,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "frames": 0,
            }
        writer.release()
    return None, None


def command_camera(args: argparse.Namespace) -> int:
    output_dir = resolve_project_path(args.output_dir)
    if args.probe:
        return probe_cameras(args.max_camera_index, output_dir)

    model_path = resolve_project_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    camera, backend, frame = open_camera(args.camera, args.backend)
    if camera is None:
        raise RuntimeError(
            f"Cannot open camera {args.camera}. Close other camera apps or use --probe."
        )
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"
    model = YOLO(str(model_path))

    window_name = "Bottle + Keyboard | R record | S snapshot | Q/Esc quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    start_time = time.perf_counter()
    previous_time = start_time
    smoothed_fps = 0.0
    frame_count = 0
    frames_with_class = Counter()
    total_detections = Counter()
    snapshots = []
    recordings = []
    video_writer = None
    current_recording = None
    auto_record_pending = args.record
    recording_fps = float(camera.get(cv2.CAP_PROP_FPS))
    if not 1.0 <= recording_fps <= 120.0:
        recording_fps = 30.0

    print(f"Camera {args.camera} opened with {backend}")
    print(f"Model: {model_path}")
    print(f"Device: {device}; confidence threshold: {args.conf}")

    try:
        while True:
            if frame_count > 0:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Camera frame read failed")
                    break

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.7,
                device=device,
                verbose=False,
            )[0]
            annotated = result.plot(line_width=2, labels=True, conf=True)

            detected_this_frame = set()
            if result.boxes is not None:
                for class_id in result.boxes.cls.int().cpu().tolist():
                    class_name = result.names[class_id]
                    detected_this_frame.add(class_name)
                    total_detections[class_name] += 1
            for class_name in detected_this_frame:
                frames_with_class[class_name] += 1

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-9)
            smoothed_fps = (
                instantaneous_fps
                if frame_count == 0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )
            previous_time = now
            frame_count += 1

            cv2.putText(
                annotated,
                f"FPS {smoothed_fps:.1f} | conf {args.conf:.2f} | device {device}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "R record/stop   S snapshot   Q/Esc quit",
                (15, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if auto_record_pending:
                video_writer, current_recording = create_video_writer(
                    annotated, recording_fps, output_dir
                )
                auto_record_pending = False
                if video_writer is None:
                    print("Cannot create video file. Recording was not started.")
                else:
                    print(f"Started recording: {ROOT / current_recording['file']}")

            if video_writer is not None:
                cv2.putText(
                    annotated, "REC", (15, 98), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA,
                )
                video_writer.write(annotated)
                current_recording["frames"] += 1

            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot = output_dir / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                snapshots.append(str(snapshot.relative_to(ROOT)))
                print(f"Saved snapshot: {snapshot}")
            if key == ord("r"):
                if video_writer is None:
                    video_writer, current_recording = create_video_writer(
                        annotated, recording_fps, output_dir
                    )
                    if video_writer is None:
                        print("Cannot create video file. Recording was not started.")
                    else:
                        print(f"Started recording: {ROOT / current_recording['file']}")
                else:
                    video_writer.release()
                    current_recording["duration_seconds"] = round(
                        current_recording["frames"] / current_recording["fps"], 3
                    )
                    recordings.append(current_recording)
                    print(f"Stopped recording: {ROOT / current_recording['file']}")
                    video_writer = None
                    current_recording = None
    finally:
        if video_writer is not None:
            video_writer.release()
            current_recording["duration_seconds"] = round(
                current_recording["frames"] / current_recording["fps"], 3
            )
            recordings.append(current_recording)
        camera.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    report = {
        "model": str(model_path),
        "camera_index": args.camera,
        "camera_backend": backend,
        "device": str(device),
        "confidence_threshold": args.conf,
        "frames": frame_count,
        "elapsed_seconds": round(elapsed, 3),
        "average_fps": round(frame_count / elapsed, 2) if elapsed else 0.0,
        "frames_with_class": dict(frames_with_class),
        "total_detections": dict(total_detections),
        "snapshots": snapshots,
        "recordings": recordings,
    }
    report_path = output_dir / "camera_test_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")
    return 0


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=str(DATA_V2), help="YOLO data YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="All-in-one bottle/keyboard YOLO project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build the Open Images dataset")
    prepare.add_argument(
        "--metadata-dir", default="dataset/external/openimages/metadata",
        help="directory containing Open Images metadata CSV files",
    )
    prepare.add_argument("--output-dir", default="dataset_v2")
    prepare.add_argument("--reuse-dir", default="dataset")
    prepare.add_argument("--bottle-images", type=int, default=300)
    prepare.add_argument("--keyboard-images", type=int, default=240)
    prepare.add_argument("--negative-images", type=int, default=60)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=84)
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="train or fine-tune a YOLO model")
    train.add_argument("--profile", choices=sorted(TRAINING_PROFILES), default="expanded")
    train.add_argument("--model", help="override starting model")
    train.add_argument("--data", help="override YOLO data YAML")
    train.add_argument("--epochs", type=int, help="override epoch count")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--name", help="override run name")
    train.set_defaults(func=command_train)

    validate = subparsers.add_parser("validate", help="validate one model")
    validate.add_argument("--model", default=str(EXPANDED_MODEL))
    validate.add_argument("--name", default="validation_all_in_one")
    validate.add_argument("--output", default="runs/detect/validation_all_in_one.json")
    add_validation_arguments(validate)
    validate.set_defaults(func=command_validate)

    smoke = subparsers.add_parser("smoke", help="run a per-class sample test")
    smoke.add_argument("--model", default=str(EXPANDED_MODEL))
    smoke.add_argument("--dataset-dir", default="dataset_v2")
    smoke.add_argument("--per-class", type=int, default=10)
    smoke.add_argument("--conf", type=float, default=0.25)
    smoke.add_argument("--match-iou", type=float, default=0.5)
    smoke.add_argument("--imgsz", type=int, default=640)
    smoke.add_argument("--device", default="0")
    smoke.add_argument("--seed", type=int, default=20260824)
    smoke.add_argument(
        "--save-images", action=argparse.BooleanOptionalAction, default=True
    )
    smoke.add_argument("--output", default="runs/detect/smoke_test_all_in_one.json")
    smoke.set_defaults(func=command_smoke)

    compare = subparsers.add_parser("compare", help="compare models on one validation set")
    compare.add_argument(
        "--models", action="append", metavar="NAME=PATH",
        help="repeat for each model; defaults to baseline and expanded",
    )
    compare.add_argument("--output", default="runs/detect/model_comparison_all_in_one.json")
    add_validation_arguments(compare)
    compare.set_defaults(func=command_compare)

    hard = subparsers.add_parser(
        "hard-negative", help="compare false detections on person-only images"
    )
    hard.add_argument(
        "--models", action="append", metavar="NAME=PATH",
        help="repeat for each model; defaults to baseline and expanded",
    )
    hard.add_argument(
        "--manifest", default="dataset_v2/external/openimages/source_manifest.csv"
    )
    hard.add_argument("--conf", type=float, default=0.25)
    hard.add_argument("--imgsz", type=int, default=640)
    hard.add_argument("--batch", type=int, default=16)
    hard.add_argument("--device", default="0")
    hard.add_argument("--output", default="runs/detect/hard_negative_all_in_one.json")
    hard.set_defaults(func=command_hard_negative)

    export = subparsers.add_parser("export", help="export to ONNX or TensorRT")
    export.add_argument("--model", default=str(EXPANDED_MODEL))
    export.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--batch", type=int, default=1)
    export.add_argument("--device", default="cpu")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    export.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    export.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    export.add_argument("--data", default=str(DATA_V2), help="INT8 calibration data")
    export.set_defaults(func=command_export)

    camera = subparsers.add_parser(
        "camera", help="probe or run live camera detection, snapshots and recording"
    )
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--backend", choices=tuple(CAMERA_BACKENDS), default="auto")
    camera.add_argument("--conf", type=float, default=0.25)
    camera.add_argument("--imgsz", type=int, default=640)
    camera.add_argument("--model", default=str(EXPANDED_MODEL))
    camera.add_argument("--device", default="auto")
    camera.add_argument("--record", action="store_true", help="record immediately")
    camera.add_argument("--probe", action="store_true", help="probe cameras and exit")
    camera.add_argument("--max-camera-index", type=int, default=4)
    camera.add_argument("--output-dir", default="runs/camera_test")
    camera.set_defaults(func=command_camera)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    result = args.func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
