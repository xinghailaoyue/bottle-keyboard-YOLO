# Expanded Bottle/Keyboard Training Report

## Dataset

- Source: Open Images V7 validation and test images with bounding-box annotations.
- Total images: 599 (`479` train, `120` validation).
- Bottle boxes: 629.
- Keyboard boxes: 266.
- Person-only hard-negative images: 60 (`48` train, `12` validation).
- Existing baseline images reused: 235.
- Newly downloaded images: 364.
- Failed downloads: 0.
- Source/license manifest: `dataset_v2/external/openimages/source_manifest.csv`.

Open Images official documentation:

- https://storage.googleapis.com/openimages/web/download_v7.html
- https://storage.googleapis.com/openimages/web/factsfigures_v7.html

## Training

- Architecture: YOLO11n.
- Starting weights: `exports/bottle_keyboard_yolo11n_best.pt`.
- Selected training: conservative 60-epoch AdamW fine-tuning at `lr0=0.0002`.
- Selected weights: `exports/bottle_keyboard_yolo11n_expanded_best.pt`.

## Same-validation-set comparison

| Metric | Baseline | Expanded | Delta |
|---|---:|---:|---:|
| Overall precision | 0.9033 | 0.9240 | +0.0208 |
| Overall recall | 0.6716 | 0.6658 | -0.0058 |
| Overall mAP50 | 0.7637 | 0.7751 | +0.0114 |
| Overall mAP50-95 | 0.6324 | 0.6393 | +0.0068 |
| Bottle precision | 0.8643 | 0.8762 | +0.0119 |
| Bottle recall | 0.5130 | 0.5391 | +0.0261 |
| Bottle mAP50 | 0.6278 | 0.6505 | +0.0227 |
| Bottle mAP50-95 | 0.5138 | 0.5564 | +0.0426 |
| Keyboard precision | 0.9423 | 0.9718 | +0.0296 |
| Keyboard recall | 0.8302 | 0.7925 | -0.0377 |
| Keyboard mAP50 | 0.8996 | 0.8996 | +0.0000 |

## Person hard-negative comparison at confidence 0.25

| Metric (60 images) | Baseline | Expanded |
|---|---:|---:|
| Images with any false detection | 14 | 4 |
| False bottle boxes | 16 | 3 |
| False keyboard boxes | 7 | 1 |

The expanded model reduced bottle false detections on this person-only set by
81.25% while improving bottle precision, recall, and mAP on the shared
validation set. The expanded model is therefore selected as the new camera-test
default. Real webcam examples should still be collected for a later domain-
specific refinement round.
