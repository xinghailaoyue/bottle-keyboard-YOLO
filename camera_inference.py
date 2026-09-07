"""Run real-time bottle/keyboard detection with a local Windows camera."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "exports" / "bottle_keyboard_yolo11n_expanded_best.pt"
OUTPUT_DIR = ROOT / "runs" / "camera_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Windows camera index")
    parser.add_argument(
        "--backend",
        choices=("auto", "dshow", "msmf"),
        default="auto",
        help="camera backend; auto tries Media Foundation then DirectShow",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--record",
        action="store_true",
        help="start recording annotated video immediately",
    )
    parser.add_argument("--probe", action="store_true", help="list usable camera indexes and exit")
    return parser.parse_args()


BACKENDS = {
    "msmf": (cv2.CAP_MSMF, "Media Foundation"),
    "dshow": (cv2.CAP_DSHOW, "DirectShow"),
    "auto": (cv2.CAP_ANY, "OpenCV Auto"),
}


def frame_stats(frame):
    return round(float(frame.mean()), 2), round(float(frame.std()), 2)


def open_backend(index: int, backend_key: str):
    backend_id, backend_name = BACKENDS[backend_key]
    camera = cv2.VideoCapture(index, backend_id)
    if not camera.isOpened():
        camera.release()
        return None, backend_name, None

    # Laptop cameras often return several empty/dark frames while exposure settles.
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


def try_open(index: int, backend_preference: str = "auto"):
    # Media Foundation usually handles integrated Windows cameras best. DirectShow
    # remains the fallback for older drivers. Do not force 1280x720: some laptop
    # drivers accept that setting but subsequently output black frames.
    backend_keys = (
        [backend_preference]
        if backend_preference != "auto"
        else ["msmf", "dshow", "auto"]
    )
    dark_fallback = None
    for backend_key in backend_keys:
        camera, name, frame = open_backend(index, backend_key)
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
    if dark_fallback is not None:
        return dark_fallback
    return None, None, None


def probe_cameras() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    found = []
    for index in range(5):
        for backend_key in ("msmf", "dshow", "auto"):
            camera, backend, frame = open_backend(index, backend_key)
            if camera is None:
                continue
            height, width = frame.shape[:2]
            mean, std = frame_stats(frame)
            preview = OUTPUT_DIR / f"probe_camera{index}_{backend_key}.jpg"
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


def create_video_writer(frame, fps: float):
    """Create an annotated-video writer, preferring MP4 and falling back to AVI."""
    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    formats = (
        ("mp4v", ".mp4"),
        ("MJPG", ".avi"),
    )
    for codec, suffix in formats:
        video_path = OUTPUT_DIR / f"recording_{timestamp}{suffix}"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            recording = {
                "file": str(video_path.relative_to(ROOT)),
                "codec": codec,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "frames": 0,
            }
            return writer, recording
        writer.release()
    return None, None


def main() -> int:
    args = parse_args()
    if args.probe:
        return probe_cameras()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    camera, backend, frame = try_open(args.camera, args.backend)
    if camera is None:
        raise RuntimeError(
            f"Cannot open camera {args.camera}. Close other camera apps or run with --probe."
        )

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    window_name = "Bottle + Keyboard Detection | R: record | S: snapshot | Q/Esc: quit"
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
                instantaneous_fps if frame_count == 0 else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )
            previous_time = now
            frame_count += 1

            status = f"FPS {smoothed_fps:.1f} | conf {args.conf:.2f} | device {device}"
            cv2.putText(
                annotated, status, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated, "R record/stop   S snapshot   Q/Esc quit", (15, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA,
            )

            if auto_record_pending:
                video_writer, current_recording = create_video_writer(annotated, recording_fps)
                auto_record_pending = False
                if video_writer is None:
                    print("Cannot create video file. Recording was not started.")
                else:
                    print(f"Started recording: {ROOT / current_recording['file']}")

            if video_writer is not None:
                cv2.putText(
                    annotated, "REC", (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA,
                )
                video_writer.write(annotated)
                current_recording["frames"] += 1

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot = OUTPUT_DIR / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                snapshots.append(str(snapshot.relative_to(ROOT)))
                print(f"Saved snapshot: {snapshot}")
            if key == ord("r"):
                if video_writer is None:
                    video_writer, current_recording = create_video_writer(annotated, recording_fps)
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
            print(f"Stopped recording: {ROOT / current_recording['file']}")
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
    report_path = OUTPUT_DIR / "camera_test_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
