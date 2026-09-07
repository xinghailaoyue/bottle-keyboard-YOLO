"""Train the bottle/keyboard baseline detector on Windows."""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(root / "data.yaml"),
        epochs=60,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project=str(root / "runs" / "detect"),
        name="baseline_yolo11n",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        patience=15,
        seed=42,
        deterministic=True,
        close_mosaic=10,
        plots=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
