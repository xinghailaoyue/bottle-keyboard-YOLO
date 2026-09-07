"""Compare false detections on person-only hard-negative images."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights: Path, images: list[Path]) -> dict:
    model = YOLO(str(weights))
    false_boxes = Counter()
    images_with_false_detection = Counter()
    results = model.predict(
        source=[str(path) for path in images],
        conf=0.25,
        imgsz=640,
        device=0,
        batch=16,
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
        "confidence": 0.25,
        "images_with_any_false_detection": any_false_images,
        "images_with_false_detection_by_class": dict(images_with_false_detection),
        "false_detection_boxes_by_class": dict(false_boxes),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    manifest = root / "dataset_v2" / "external" / "openimages" / "source_manifest.csv"
    images = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if "person_negative" in row["selection_role"].split("+"):
                images.append(root / row["local_file"])
    images.sort()
    report = {
        "baseline": evaluate(
            root / "exports" / "bottle_keyboard_yolo11n_best.pt", images
        ),
        "expanded_low_lr": evaluate(
            root / "runs" / "detect" / "expanded_low_lr_yolo11n" / "weights" / "best.pt",
            images,
        ),
    }
    destination = root / "runs" / "detect" / "hard_negative_comparison.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
