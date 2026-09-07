"""Run a deterministic 10-image-per-class validation smoke test."""

from __future__ import annotations

import json
import random
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
CLASS_NAMES = {0: "bottle", 1: "keyboard"}


def read_labels(path: Path):
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
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


def box_iou(a, b) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def main() -> None:
    labels_dir = ROOT / "dataset" / "labels" / "val"
    images_dir = ROOT / "dataset" / "images" / "val"
    rng = random.Random(20260824)

    selected = {}
    for class_id in CLASS_NAMES:
        candidates = [
            path for path in sorted(labels_dir.glob("*.txt"))
            if any(label[0] == class_id for label in read_labels(path))
        ]
        rng.shuffle(candidates)
        selected[class_id] = candidates[:10]

    model = YOLO(ROOT / "runs" / "detect" / "baseline_yolo11n" / "weights" / "best.pt")
    report = {"criteria": "correct class and IoU >= 0.50 at confidence >= 0.25", "classes": {}}

    for class_id, label_paths in selected.items():
        image_paths = [images_dir / f"{path.stem}.jpg" for path in label_paths]
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=640,
            conf=0.25,
            iou=0.7,
            device=0,
            save=True,
            project=str(ROOT / "runs" / "detect"),
            name=f"smoke_test_{CLASS_NAMES[class_id]}",
            exist_ok=True,
            verbose=False,
        )

        rows = []
        for label_path, image_path, result in zip(label_paths, image_paths, results):
            image_height, image_width = result.orig_shape
            ground_truth = [
                normalized_to_xyxy(label, image_width, image_height)
                for label in read_labels(label_path)
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
                (box_iou(predicted[0], truth) for predicted in predictions for truth in ground_truth),
                default=0.0,
            )
            best_confidence = max((prediction[1] for prediction in predictions), default=0.0)
            rows.append({
                "image": image_path.name,
                "success": best_iou >= 0.5,
                "best_iou": round(best_iou, 4),
                "best_confidence": round(best_confidence, 4),
            })

        successes = sum(row["success"] for row in rows)
        report["classes"][CLASS_NAMES[class_id]] = {
            "successes": successes,
            "total": len(rows),
            "rate": successes / len(rows) if rows else 0.0,
            "images": rows,
        }

    output = ROOT / "runs" / "detect" / "baseline_yolo11n" / "smoke_test_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
