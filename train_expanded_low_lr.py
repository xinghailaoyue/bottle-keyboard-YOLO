"""Conservative fine-tuning on expanded data to preserve baseline accuracy."""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    model = YOLO(str(root / "exports" / "bottle_keyboard_yolo11n_best.pt"))
    model.train(
        data=str(root / "data_v2.yaml"),
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(root / "runs" / "detect"),
        name="expanded_low_lr_yolo11n",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        cos_lr=True,
        weight_decay=0.0005,
        patience=20,
        seed=84,
        deterministic=True,
        mosaic=0.2,
        scale=0.25,
        translate=0.05,
        close_mosaic=8,
        cls=0.7,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
