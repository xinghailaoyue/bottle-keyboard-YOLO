#!/usr/bin/env python3
"""Build dataset_v3 by adding COCO horizontal bottles to dataset_v2.

The resulting YOLO dataset contains exactly 1,000 images by default:
  - 599 existing Open Images samples from dataset_v2
  - 280 COCO 2017 train images with a clearly horizontal bottle
  - 80 COCO 2017 train images containing a keyboard
  - 41 COCO 2017 train person-only hard-negative images

COCO images are selected deterministically and downloaded individually from the
official images.cocodataset.org host. A source manifest preserves the original
COCO image ID, URL and per-image license metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COCO_TO_YOLO = {44: (0, "bottle"), 76: (1, "keyboard")}
PERSON_CATEGORY_ID = 1


def valid_image(path: Path) -> bool:
    """Return True only for a non-empty image that Pillow can verify."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def copy_v2_dataset(source: Path, destination: Path) -> dict[str, int]:
    """Copy the 599 V2 images and labels without changing their split."""
    counts = Counter()
    for subset in ("train", "val"):
        source_images = source / "images" / subset
        source_labels = source / "labels" / subset
        target_images = destination / "images" / subset
        target_labels = destination / "labels" / subset
        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_images.iterdir()):
            if not image_path.is_file():
                continue
            label_path = source_labels / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing V2 label: {label_path}")
            shutil.copy2(image_path, target_images / image_path.name)
            shutil.copy2(label_path, target_labels / label_path.name)
            counts[subset] += 1
    return dict(counts)


def clear_generated_coco_samples(destination: Path) -> None:
    """Remove only files created by this builder, leaving V2 and metadata untouched."""
    for kind in ("images", "labels"):
        for subset in ("train", "val"):
            directory = destination / kind / subset
            if not directory.exists():
                continue
            for path in directory.glob("coco_train2017_*"):
                if path.is_file():
                    path.unlink()


def sample_exact(items, count: int, seed: int):
    """Deterministically sample exactly count items from an unordered set."""
    values = sorted(items)
    random.Random(seed).shuffle(values)
    if len(values) < count:
        raise RuntimeError(f"Requested {count} items, but only {len(values)} are available")
    return values[:count]


def stratified_subsets(groups: dict[str, list[int]], val_fraction: float, seed: int):
    """Assign each selection role independently to train/val."""
    subsets = {}
    for offset, (role, image_ids) in enumerate(groups.items()):
        shuffled = list(image_ids)
        random.Random(seed + offset).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_ids = set(shuffled[:val_count])
        for image_id in image_ids:
            subsets[image_id] = "val" if image_id in val_ids else "train"
    return subsets


def load_and_select(
    annotation_path: Path,
    horizontal_count: int,
    keyboard_count: int,
    negative_count: int,
    ratio: float,
    min_width: float,
    min_height: float,
    min_area_fraction: float,
    seed: int,
):
    """Load COCO JSON and select three non-overlapping image groups."""
    print(f"Loading COCO annotations: {annotation_path}")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in payload["images"]}
    licenses = {item["id"]: item for item in payload.get("licenses", [])}
    targets_by_image = defaultdict(list)
    person_images = set()
    horizontal_candidates = set()
    keyboard_candidates = set()

    for annotation in payload["annotations"]:
        if annotation.get("iscrowd"):
            continue
        category_id = annotation["category_id"]
        image_id = annotation["image_id"]
        if category_id == PERSON_CATEGORY_ID:
            person_images.add(image_id)
        if category_id not in COCO_TO_YOLO:
            continue
        targets_by_image[image_id].append(annotation)
        if category_id == 76:
            keyboard_candidates.add(image_id)
        if category_id == 44:
            x, y, width, height = annotation["bbox"]
            image = images[image_id]
            area_fraction = width * height / (image["width"] * image["height"])
            if (
                width / max(height, 1e-9) >= ratio
                and width >= min_width
                and height >= min_height
                and area_fraction >= min_area_fraction
            ):
                horizontal_candidates.add(image_id)

    horizontal = sample_exact(horizontal_candidates, horizontal_count, seed + 1)
    used = set(horizontal)
    keyboard = sample_exact(keyboard_candidates - used, keyboard_count, seed + 2)
    used.update(keyboard)
    person_negative_candidates = person_images - set(targets_by_image) - used
    person_negative = sample_exact(person_negative_candidates, negative_count, seed + 3)

    groups = {
        "horizontal_bottle": horizontal,
        "keyboard": keyboard,
        "person_negative": person_negative,
    }
    return payload, images, licenses, targets_by_image, groups


def official_image_url(image: dict) -> str:
    """Use the stable official COCO object URL instead of the original Flickr URL."""
    return f"http://images.cocodataset.org/train2017/{image['file_name']}"


def fetch_image(image: dict, destination: Path):
    """Download one selected image with retries and resumable existing-file reuse."""
    url = official_image_url(image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if valid_image(destination):
        return "existing", url, None
    last_error = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
            if not valid_image(destination):
                raise ValueError("downloaded file is not a valid image")
            return "download", url, None
        except Exception as exc:
            last_error = str(exc)
            destination.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    return "failed", url, last_error


def yolo_label_text(annotations: list[dict], image: dict) -> str:
    """Convert COCO absolute [x, y, width, height] boxes into YOLO normalized labels."""
    lines = []
    image_width = image["width"]
    image_height = image["height"]
    for annotation in annotations:
        category_id = annotation["category_id"]
        if category_id not in COCO_TO_YOLO:
            continue
        class_id, _ = COCO_TO_YOLO[category_id]
        x, y, width, height = annotation["bbox"]
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        normalized_width = width / image_width
        normalized_height = height / image_height
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def make_contact_sheet(
    rows: list[dict],
    images: dict[int, dict],
    targets_by_image: dict[int, list[dict]],
    output_path: Path,
) -> None:
    """Create an annotated preview so horizontal selections can be visually audited."""
    chosen = [row for row in rows if row["selection_role"] == "horizontal_bottle"][:20]
    thumb_width, thumb_height = 320, 240
    sheet = Image.new("RGB", (thumb_width * 5, thumb_height * 4), "white")
    for index, row in enumerate(chosen):
        image_id = int(row["coco_image_id"])
        source_path = ROOT / row["local_file"]
        with Image.open(source_path).convert("RGB") as source:
            scale = min(thumb_width / source.width, (thumb_height - 24) / source.height)
            resized = source.resize((round(source.width * scale), round(source.height * scale)))
            canvas = Image.new("RGB", (thumb_width, thumb_height), "#202020")
            left = (thumb_width - resized.width) // 2
            top = (thumb_height - 24 - resized.height) // 2
            canvas.paste(resized, (left, top))
            draw = ImageDraw.Draw(canvas)
            for annotation in targets_by_image[image_id]:
                if annotation["category_id"] != 44:
                    continue
                x, y, width, height = annotation["bbox"]
                draw.rectangle(
                    (
                        left + x * scale,
                        top + y * scale,
                        left + (x + width) * scale,
                        top + (y + height) * scale,
                    ),
                    outline="#00ff4c",
                    width=3,
                )
            draw.text((6, thumb_height - 20), f"COCO {image_id}", fill="white")
            sheet.paste(canvas, ((index % 5) * thumb_width, (index // 5) * thumb_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v2", default=str(ROOT / "dataset_v2"))
    parser.add_argument("--output", default=str(ROOT / "dataset_v3"))
    parser.add_argument(
        "--annotations",
        default=str(
            ROOT / "dataset_v3" / "external" / "coco" / "annotations" / "instances_train2017.json"
        ),
    )
    parser.add_argument("--horizontal-images", type=int, default=280)
    parser.add_argument("--keyboard-images", type=int, default=80)
    parser.add_argument("--negative-images", type=int, default=41)
    parser.add_argument("--horizontal-ratio", type=float, default=1.5)
    parser.add_argument("--min-width", type=float, default=40.0)
    parser.add_argument("--min-height", type=float, default=15.0)
    parser.add_argument("--min-area-fraction", type=float, default=0.005)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    source_v2 = Path(args.source_v2).resolve()
    output = Path(args.output).resolve()
    annotation_path = Path(args.annotations).resolve()
    if not annotation_path.exists():
        raise FileNotFoundError(f"COCO annotation JSON not found: {annotation_path}")

    clear_generated_coco_samples(output)
    v2_counts = copy_v2_dataset(source_v2, output)
    payload, images, licenses, targets_by_image, groups = load_and_select(
        annotation_path,
        args.horizontal_images,
        args.keyboard_images,
        args.negative_images,
        args.horizontal_ratio,
        args.min_width,
        args.min_height,
        args.min_area_fraction,
        args.seed,
    )
    subsets = stratified_subsets(groups, args.val_fraction, args.seed + 100)
    roles = {image_id: role for role, image_ids in groups.items() for image_id in image_ids}

    jobs = []
    for image_id in sorted(roles):
        subset = subsets[image_id]
        stem = f"coco_train2017_{image_id:012d}"
        destination = output / "images" / subset / f"{stem}.jpg"
        jobs.append((image_id, subset, stem, destination))

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(fetch_image, images[image_id], destination):
                (image_id, subset, stem, destination)
            for image_id, subset, stem, destination in jobs
        }
        for completed, future in enumerate(as_completed(future_map), start=1):
            image_id, subset, stem, destination = future_map[future]
            method, url, error = future.result()
            if error:
                failures.append({"image_id": image_id, "url": url, "error": error})
            else:
                label_path = output / "labels" / subset / f"{stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    yolo_label_text(targets_by_image.get(image_id, []), images[image_id]),
                    encoding="utf-8",
                )
                successes.append((image_id, subset, stem, destination, method, url))
            if completed % 40 == 0 or completed == len(jobs):
                print(f"COCO images processed: {completed}/{len(jobs)}")

    if failures:
        failure_path = output / "external" / "coco" / "download_failures.json"
        failure_path.write_text(
            json.dumps({"failures": failures}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failures)} COCO downloads failed; rerun to retry")

    manifest_rows = []
    for image_id, subset, stem, destination, method, url in sorted(successes):
        image = images[image_id]
        license_info = licenses.get(image.get("license"), {})
        horizontal_boxes = 0
        for annotation in targets_by_image.get(image_id, []):
            if annotation["category_id"] == 44:
                width, height = annotation["bbox"][2:4]
                if width / max(height, 1e-9) >= args.horizontal_ratio:
                    horizontal_boxes += 1
        manifest_rows.append(
            {
                "local_file": destination.relative_to(ROOT).as_posix(),
                "dataset_split": subset,
                "selection_role": roles[image_id],
                "source_dataset": "COCO 2017 train",
                "coco_image_id": image_id,
                "coco_file_name": image["file_name"],
                "acquisition": method,
                "download_url": url,
                "flickr_url": image.get("flickr_url", ""),
                "license_id": image.get("license", ""),
                "license_name": license_info.get("name", ""),
                "license_url": license_info.get("url", ""),
                "horizontal_bottle_boxes": horizontal_boxes,
            }
        )

    manifest_path = output / "external" / "coco" / "source_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    old_manifest = source_v2 / "external" / "openimages" / "source_manifest.csv"
    copied_old_manifest = output / "external" / "openimages" / "source_manifest_v2.csv"
    copied_old_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old_manifest, copied_old_manifest)

    new_split_counts = Counter(row["dataset_split"] for row in manifest_rows)
    object_counts = Counter()
    for image_id in roles:
        for annotation in targets_by_image.get(image_id, []):
            if annotation["category_id"] in COCO_TO_YOLO:
                object_counts[COCO_TO_YOLO[annotation["category_id"]][1]] += 1
    stats = {
        "total_images": sum(v2_counts.values()) + len(successes),
        "images": {
            "train": v2_counts.get("train", 0) + new_split_counts.get("train", 0),
            "val": v2_counts.get("val", 0) + new_split_counts.get("val", 0),
        },
        "source_images": {
            "Open Images V2 retained": sum(v2_counts.values()),
            "COCO 2017 train added": len(successes),
        },
        "coco_selection_roles": {role: len(values) for role, values in groups.items()},
        "coco_added_objects": dict(object_counts),
        "horizontal_selection": {
            "minimum_width_height_ratio": args.horizontal_ratio,
            "minimum_box_width_pixels": args.min_width,
            "minimum_box_height_pixels": args.min_height,
            "minimum_box_area_fraction": args.min_area_fraction,
        },
        "val_fraction_for_new_coco_images": args.val_fraction,
        "seed": args.seed,
        "annotation_source": "COCO 2017 instances_train2017.json",
    }
    (output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    make_contact_sheet(
        manifest_rows,
        images,
        targets_by_image,
        output / "external" / "coco" / "horizontal_samples_contact_sheet.jpg",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
