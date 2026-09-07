"""Evaluate baseline and expanded weights on the same expanded validation set."""

from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights: Path, data: Path, run_name: str) -> dict:
    result = YOLO(str(weights)).val(
        data=str(data),
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        plots=False,
        project=str(data.parent / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
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


def main() -> None:
    root = Path(__file__).resolve().parent
    data = root / "data_v2.yaml"
    report = {
        "evaluation_dataset": str(data),
        "baseline": evaluate(
            root / "exports" / "bottle_keyboard_yolo11n_best.pt",
            data,
            "comparison_baseline_on_v2",
        ),
        "expanded": evaluate(
            root / "runs" / "detect" / "expanded_yolo11n" / "weights" / "best.pt",
            data,
            "comparison_expanded_on_v2",
        ),
        "expanded_low_lr": evaluate(
            root / "runs" / "detect" / "expanded_low_lr_yolo11n" / "weights" / "best.pt",
            data,
            "comparison_expanded_low_lr_on_v2",
        ),
    }
    for scope in ("all",):
        report["delta_expanded_minus_baseline"] = {
            key: round(report["expanded"][scope][key] - report["baseline"][scope][key], 6)
            for key in report["baseline"][scope]
        }
    report["class_deltas"] = {}
    for class_name in report["baseline"]["classes"]:
        report["class_deltas"][class_name] = {
            key: round(
                report["expanded"]["classes"][class_name][key]
                - report["baseline"]["classes"][class_name][key],
                6,
            )
            for key in report["baseline"]["classes"][class_name]
        }
    report["low_lr_delta_expanded_minus_baseline"] = {
        key: round(report["expanded_low_lr"]["all"][key] - report["baseline"]["all"][key], 6)
        for key in report["baseline"]["all"]
    }
    report["low_lr_class_deltas"] = {}
    for class_name in report["baseline"]["classes"]:
        report["low_lr_class_deltas"][class_name] = {
            key: round(
                report["expanded_low_lr"]["classes"][class_name][key]
                - report["baseline"]["classes"][class_name][key],
                6,
            )
            for key in report["baseline"]["classes"][class_name]
        }
    destination = root / "runs" / "detect" / "model_comparison_v2.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
