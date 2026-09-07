"""Compare model versions on held-out horizontal-bottle COCO validation images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "v1_baseline": ROOT / "exports" / "bottle_keyboard_yolo11n_best.pt",
    "v2_expanded": ROOT / "exports" / "bottle_keyboard_yolo11n_expanded_best.pt",
    "v3_horizontal": ROOT / "exports" / "bottle_keyboard_yolo11n_horizontal_v3_best.pt",
}


def read_bottle_truths(label_path: Path, image_width: int, image_height: int):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, x, y, width, height = map(float, line.split())
        if int(class_id) != 0:
            continue
        boxes.append(
            (
                (x - width / 2) * image_width,
                (y - height / 2) * image_height,
                (x + width / 2) * image_width,
                (y + height / 2) * image_height,
            )
        )
    return boxes


def iou(box_a, box_b) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def parse_models(values: list[str] | None):
    if not values:
        return DEFAULT_MODELS
    models = {}
    for value in values:
        name, path = value.split("=", 1)
        models[name] = Path(path).resolve()
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", action="append", metavar="NAME=PATH")
    parser.add_argument(
        "--manifest", default=str(ROOT / "dataset_v3" / "external" / "coco" / "source_manifest.csv")
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--output", default=str(ROOT / "runs" / "detect" / "horizontal_model_comparison.json")
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    image_paths = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["selection_role"] == "horizontal_bottle" and row["dataset_split"] == "val":
                image_paths.append(ROOT / row["local_file"])
    image_paths.sort()
    if not image_paths:
        raise RuntimeError("No held-out horizontal_bottle validation images found")

    report = {
        "dataset": "COCO 2017 train, held-out V3 horizontal_bottle validation subset",
        "images": len(image_paths),
        "confidence": args.conf,
        "match_iou": args.match_iou,
        "models": {},
    }
    for name, weights in parse_models(args.models).items():
        model = YOLO(str(weights))
        results = model.predict(
            source=[str(path) for path in image_paths],
            conf=args.conf,
            iou=0.7,
            imgsz=args.imgsz,
            device=args.device,
            batch=16,
            verbose=False,
        )
        rows = []
        for image_path, result in zip(image_paths, results):
            image_height, image_width = result.orig_shape
            label_path = Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
            truths = read_bottle_truths(label_path, image_width, image_height)
            bottle_predictions = []
            keyboard_boxes = 0
            if result.boxes is not None:
                for box, class_id, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.int().cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    if class_id == 0:
                        bottle_predictions.append((box, confidence))
                    elif class_id == 1:
                        keyboard_boxes += 1
            best_iou = max(
                (iou(prediction[0], truth) for prediction in bottle_predictions for truth in truths),
                default=0.0,
            )
            rows.append(
                {
                    "image": image_path.name,
                    "has_bottle_detection": bool(bottle_predictions),
                    "matched": best_iou >= args.match_iou,
                    "best_iou": round(best_iou, 6),
                    "best_confidence": round(
                        max((prediction[1] for prediction in bottle_predictions), default=0.0), 6
                    ),
                    "keyboard_false_boxes": keyboard_boxes,
                }
            )
        detected = sum(row["has_bottle_detection"] for row in rows)
        matched = sum(row["matched"] for row in rows)
        report["models"][name] = {
            "weights": str(weights),
            "images_with_bottle_detection": detected,
            "detection_rate": round(detected / len(rows), 6),
            "images_matched_at_iou": matched,
            "matched_rate": round(matched / len(rows), 6),
            "mean_best_iou": round(sum(row["best_iou"] for row in rows) / len(rows), 6),
            "keyboard_false_boxes": sum(row["keyboard_false_boxes"] for row in rows),
            "per_image": rows,
        }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
