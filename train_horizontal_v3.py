"""Fine-tune the V2 model on the 1,000-image horizontal-bottle V3 dataset."""

from pathlib import Path
import shutil

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    source_model = root / "exports" / "bottle_keyboard_yolo11n_expanded_best.pt"
    model = YOLO(str(source_model))
    model.train(
        data=str(root / "data_v3.yaml"),
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(root / "runs" / "detect"),
        name="horizontal_v3_yolo11n",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        cos_lr=True,
        weight_decay=0.0005,
        patience=20,
        seed=20260901,
        deterministic=True,
        mosaic=0.25,
        mixup=0.05,
        degrees=12.0,
        scale=0.30,
        translate=0.08,
        fliplr=0.5,
        flipud=0.10,
        close_mosaic=8,
        cls=0.7,
        plots=True,
        verbose=True,
    )
    best = Path(model.trainer.best)
    export_path = root / "exports" / "bottle_keyboard_yolo11n_horizontal_v3_best.pt"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, export_path)
    print(f"V3 best model copied to: {export_path}")


if __name__ == "__main__":
    main()
