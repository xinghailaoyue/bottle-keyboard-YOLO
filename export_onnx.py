"""Export the trained detector to a Jetson-transferable ONNX model."""

from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent
    weights = root / "runs" / "detect" / "baseline_yolo11n" / "weights" / "best.pt"
    model = YOLO(weights)
    output = model.export(
        format="onnx",
        imgsz=640,
        batch=1,
        dynamic=False,
        simplify=True,
        opset=17,
        device="cpu",
    )
    print(output)


if __name__ == "__main__":
    main()
