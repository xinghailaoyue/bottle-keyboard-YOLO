#!/usr/bin/env python3
"""Bottle/keyboard YOLO project: all project entry functions in one file.

This file consolidates the project's own orchestration code. Ultralytics,
OpenCV, PyTorch and Pillow remain external libraries and are not copied here.

Subcommands:
    prepare        Build an Open Images dataset with optional hard negatives.
    train          Train the baseline or conservatively fine-tune the model.
    validate       Evaluate one model on a YOLO validation split.
    smoke          Run a deterministic per-class sample test.
    compare        Compare multiple models on the same validation split.
    hard-negative Compare false detections on person-only images.
    export         Export a model to ONNX or TensorRT.
    camera         Probe cameras or run live detection, snapshots and recording.
"""

# =============================================================================
# 文件阅读指南
# =============================================================================
# 这是原版 yolo_project_all_in_one.py 的“详细中文注释版”。它把本项目自己的
# 任务编排代码集中在一个入口文件中，但 YOLO 网络结构、损失函数和底层训练循环
# 仍由 ultralytics 包提供；摄像头与视频编解码由 OpenCV 提供；CUDA 判断由
# PyTorch 提供；下载图片的完整性检查由 Pillow 提供。
#
# 程序的共同执行路径是：
#   1. main() 解析命令行；
#   2. build_parser() 根据子命令生成 args；
#   3. args.func(args) 分派到 command_prepare / command_train / ...；
#   4. 对应命令完成任务，并把模型、图片、视频或 JSON 报告写入项目目录。
#
# 建议讲解顺序：先看最下方的 build_parser() 与 main()，理解“如何选择功能”；
# 再按编号阅读数据准备、训练、验证、冒烟测试、负样本评估、导出和摄像头模块。

from __future__ import annotations

# argparse 负责把 PowerShell 中的参数转换为 Python 对象 args。
import argparse
# csv/json 用于读取 Open Images 元数据及生成实验报告。
import csv
import json
# os 用于在导入 Ultralytics 前设置其配置目录。
import os
# random 仅创建带固定种子的局部随机数生成器，保证抽样可以复现。
import random
# shutil 用于复用旧数据集中的图片，避免重复下载。
import shutil
import time
# urllib.request 使用 Open Images 公共 URL 下载图片。
import urllib.request
# Counter 统计类别/检测框数量；defaultdict 简化“一张图对应多个框”的收集。
from collections import Counter, defaultdict
# 数据集图片可以并行下载，as_completed 用于按完成顺序处理任务。
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


# ROOT 永远指向“当前源码文件所在目录”。这样无论从哪个工作目录启动脚本，
# 默认的数据、模型和输出路径都仍然相对于 E:\YOLO 项目定位。
ROOT = Path(__file__).resolve().parent
# 必须在导入 ultralytics 之前设置。它把 Ultralytics 的 settings.json 等配置
# 限制在项目内的 .ultralytics 目录，避免依赖用户主目录或产生权限问题。
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

# OpenCV：摄像头、画面显示、文字叠加、截图和录像。
import cv2  # noqa: E402  (environment is configured before importing Ultralytics)
# PyTorch：这里只用来判断 CUDA 是否可用，真正的推理也由 Ultralytics 调用它。
import torch  # noqa: E402
# Pillow：下载后验证文件确实是可解码图片，而不只是非空文件。
from PIL import Image  # noqa: E402
# YOLO 是本文件调用训练、验证、推理和导出的统一高层接口。
from ultralytics import YOLO  # noqa: E402


# -----------------------------------------------------------------------------
# Project defaults
# -----------------------------------------------------------------------------

# 两个权重分别代表扩充前基线模型和扩充数据后再训练的模型。
BASELINE_MODEL = ROOT / "exports" / "bottle_keyboard_yolo11n_best.pt"
EXPANDED_MODEL = ROOT / "exports" / "bottle_keyboard_yolo11n_expanded_best.pt"
# data.yaml 描述训练/验证图片目录和类别名称；V2 对应扩充后的数据集。
DATA_V1 = ROOT / "data.yaml"
DATA_V2 = ROOT / "data_v2.yaml"
# 摄像头截图、录像和运行报告的默认保存位置。
CAMERA_OUTPUT_DIR = ROOT / "runs" / "camera_test"

# Open Images 使用机器 ID（MID）表示类别。本项目把 MID 映射为 YOLO 类别编号：
# 0=bottle，1=keyboard。编号必须和 data.yaml 中 names 的顺序一致。
TARGETS = {
    "/m/04dr76w": (0, "bottle"),
    "/m/01m2v": (1, "keyboard"),
}
# Person 不作为要识别的类别；它用于挑选“有人但没有瓶子/键盘”的困难负样本，
# 以测量模型是否会把人或人体局部误报成 bottle。
PERSON_MID = "/m/01g317"

# Windows 摄像头可能通过不同后端工作。auto 模式会按 MSMF、DShow、CAP_ANY
# 依次尝试，以绕开某个后端打开成功但画面全黑的问题。
CAMERA_BACKENDS = {
    "msmf": (cv2.CAP_MSMF, "Media Foundation"),
    "dshow": (cv2.CAP_DSHOW, "DirectShow"),
    "auto": (cv2.CAP_ANY, "OpenCV Auto"),
}


# ---- 通用路径与报告辅助函数 -------------------------------------------------
def resolve_project_path(value: str | Path) -> Path:
    """Resolve a user path relative to the project root."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def model_argument(value: str | Path) -> str:
    """Resolve local model files while still allowing names such as yolo11n.pt."""
    path = Path(value)
    # 本地文件存在时传绝对路径；否则保留 yolo11n.pt 之类的官方模型名称，
    # 让 Ultralytics 按自身规则查找缓存或下载预训练权重。
    candidate = path if path.is_absolute() else ROOT / path
    return str(candidate.resolve()) if candidate.exists() else str(value)


def write_json(path: Path, payload: dict) -> None:
    # parents=True 会连同上级目录一起创建；ensure_ascii=False 保留中文可读性。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_model_specs(specs: list[str] | None, defaults: dict[str, Path]) -> dict[str, Path]:
    """Parse repeated NAME=PATH model arguments."""
    if not specs:
        return defaults
    parsed = {}
    # --models 可以重复出现，例如：
    # --models baseline=exports/a.pt --models expanded=exports/b.pt
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Model specification must be NAME=PATH: {spec}")
        name, value = spec.split("=", 1)
        if not name.strip():
            raise ValueError(f"Model name is empty: {spec}")
        parsed[name.strip()] = resolve_project_path(value.strip())
    return parsed


# -----------------------------------------------------------------------------
# 1. Dataset preparation
# -----------------------------------------------------------------------------
# 本模块把 Open Images 的官方 CSV 标注转换成 Ultralytics 可直接训练的目录：
#   dataset_v2/
#     images/train、images/val     原始图片
#     labels/train、labels/val     与图片同名的 YOLO 标签文本
#     external/openimages          来源清单与下载失败记录
#     stats.json                   数据量统计
#
# 正样本来自 Bottle 和 Computer keyboard；困难负样本来自 Person 标注图片，
# 且这些图片不能同时具有目标类别框。负样本仍会创建同名的空标签文件。

def load_openimages_annotations(metadata_dir: Path):
    """Read Bottle, Computer keyboard and Person boxes from Open Images CSVs."""
    # target_records: (官方数据分区, ImageID) -> 该图片的全部目标框。
    target_records = defaultdict(list)
    # class_images: YOLO 类别编号 -> 包含该类别的图片键集合，用于分层抽样。
    class_images = defaultdict(set)
    # person_images 只记录“确实标注了 Person”的图，之后用于构造困难负样本。
    person_images = set()

    # Open Images 的 validation 与 test 图片都有稳定的公开下载 URL。
    # 这里将两个官方分区作为候选池；稍后会重新划分本项目的 train/val。
    for source_split in ("validation", "test"):
        annotation_path = metadata_dir / f"{source_split}-annotations-bbox.csv"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                mid = row["LabelName"]
                # 加上 source_split 可避免两个官方分区中相同 ImageID 发生冲突。
                key = (source_split, row["ImageID"])
                # IsDepiction=1 表示绘画/图示而非真实照片，不纳入真人负样本。
                if mid == PERSON_MID and row.get("IsDepiction") != "1":
                    person_images.add(key)
                if mid not in TARGETS:
                    continue
                # 目标类同样排除 depiction；GroupOf 是一框多个密集物体，
                # 对本项目的单物体定位训练不够明确，因此也过滤掉。
                if row.get("IsDepiction") == "1" or row.get("IsGroupOf") == "1":
                    continue
                class_id, class_name = TARGETS[mid]
                # Open Images 的 X/Y 坐标本身已经归一化到 0~1，先原样保存。
                target_records[key].append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "xmin": float(row["XMin"]),
                        "xmax": float(row["XMax"]),
                        "ymin": float(row["YMin"]),
                        "ymax": float(row["YMax"]),
                    }
                )
                class_images[class_id].add(key)

    # 集合差保证负样本“有人、但没有任何 bottle/keyboard 目标框”。
    negative_candidates = person_images - set(target_records)
    return target_records, class_images, negative_candidates


def sample_items(items, count: int, seed: int):
    # 先排序再打乱：set 的遍历顺序不稳定，排序可使相同 seed 每次得到相同样本。
    values = sorted(items)
    random.Random(seed).shuffle(values)
    return values[:count]


def validation_items(items, fraction: float, seed: int):
    # 每个角色单独抽取验证集，防止某个小类别全部落入训练集。
    values = list(items)
    random.Random(seed).shuffle(values)
    if not values:
        return set()
    # 非空组至少保留 1 张验证图。
    count = max(1, round(len(values) * fraction))
    return set(values[:count])


def load_openimages_metadata(metadata_dir: Path, selected):
    # 元数据 CSV 很大，因此先按官方分区构建 wanted 集合，只保存已选图片的行。
    wanted = defaultdict(set)
    for source_split, image_id in selected:
        wanted[source_split].add(image_id)
    metadata = {}
    for source_split in ("validation", "test"):
        path = metadata_dir / f"{source_split}-images.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["ImageID"] in wanted[source_split]:
                    metadata[(source_split, row["ImageID"])] = row
    return metadata


def openimages_url(source_split: str, image_id: str) -> str:
    # Open Images 官方 S3 对象路径可以由“分区 + ImageID”直接拼出。
    return f"https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg"


def valid_image(path: Path) -> bool:
    # 第一道检查排除不存在或 0 字节文件。
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        # verify() 检查文件结构，不需要把整张图解码到内存。
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def fetch_openimages_image(key, destination: Path, reuse_dir: Path):
    """Reuse an existing image when possible; otherwise download with retries."""
    source_split, image_id = key
    stem = f"{source_split}_{image_id}"
    url = openimages_url(source_split, image_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # 优先级 1：目标目录已经存在有效图片，断点续传时直接跳过。
    if valid_image(destination):
        return url, "existing", None

    # 优先级 2：旧版数据集已有同一张图时本地复制，节省网络流量。
    for subset in ("train", "val"):
        reusable = reuse_dir / "images" / subset / f"{stem}.jpg"
        if valid_image(reusable):
            shutil.copy2(reusable, destination)
            return url, "reuse-copy", None

    last_error = None
    # 优先级 3：最多下载三次。每次失败后删除残缺文件并逐渐延长等待。
    for attempt in range(3):
        try:
            # 常见浏览器 User-Agent 可避免部分公共存储服务拒绝默认 Python 请求头。
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                destination.write_bytes(response.read())
            if not valid_image(destination):
                raise ValueError("downloaded file is not a valid image")
            return url, "download", None
        except Exception as exc:
            last_error = str(exc)
            destination.unlink(missing_ok=True)
            time.sleep(1 + attempt)
    return url, "failed", last_error


def boxes_to_yolo_lines(boxes) -> str:
    """Convert normalized xmin/xmax/ymin/ymax boxes to YOLO xywh lines."""
    lines = []
    # YOLO 每行格式：class_id x_center y_center width height，全部为 0~1 坐标。
    for box in boxes:
        width = box["xmax"] - box["xmin"]
        height = box["ymax"] - box["ymin"]
        x_center = box["xmin"] + width / 2
        y_center = box["ymin"] + height / 2
        lines.append(
            f'{box["class_id"]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}'
        )
    # 无目标框时返回空字符串，从而生成合法的负样本空标签文件。
    return "\n".join(lines) + ("\n" if lines else "")


def command_prepare(args: argparse.Namespace) -> None:
    # 第一阶段：解析目录并创建来源记录目录。
    metadata_dir = resolve_project_path(args.metadata_dir)
    output_dir = resolve_project_path(args.output_dir)
    reuse_dir = resolve_project_path(args.reuse_dir)
    external_dir = output_dir / "external" / "openimages"
    external_dir.mkdir(parents=True, exist_ok=True)

    # 第二阶段：加载候选框，然后按角色和独立随机种子抽样。
    records, class_images, negative_candidates = load_openimages_annotations(metadata_dir)
    groups = {
        "bottle": sample_items(class_images[0], args.bottle_images, args.seed + 1),
        "keyboard": sample_items(class_images[1], args.keyboard_images, args.seed + 2),
        "person_negative": sample_items(negative_candidates, args.negative_images, args.seed + 3),
    }
    requested = {
        "bottle": args.bottle_images,
        "keyboard": args.keyboard_images,
        "person_negative": args.negative_images,
    }
    # 需求数量不足时立即停止，避免用户以为已得到“约 600 张”的完整数据集。
    for role, values in groups.items():
        if len(values) < requested[role]:
            raise RuntimeError(f"Only {len(values)} samples are available for {role}")

    # 同一张图片可能既含瓶子又含键盘；用集合去重，但 roles 会保留它的全部角色。
    selected = set().union(*(set(values) for values in groups.values()))
    roles = defaultdict(set)
    for role, values in groups.items():
        for key in values:
            roles[key].add(role)

    # 第三阶段：对每个角色做分层验证集抽样，再合并成最终 val_keys。
    val_keys = set()
    for offset, values in enumerate(groups.values()):
        val_keys.update(validation_items(values, args.val_fraction, args.seed + 100 + offset))
    metadata = load_openimages_metadata(metadata_dir, selected)

    # 第四阶段：把每张图片预先映射为 train/val 目标文件，形成下载任务表。
    jobs = []
    for key in sorted(selected):
        subset = "val" if key in val_keys else "train"
        source_split, image_id = key
        stem = f"{source_split}_{image_id}"
        destination = output_dir / "images" / subset / f"{stem}.jpg"
        jobs.append((key, subset, stem, destination))

    successes = []
    failures = []
    # 第五阶段：线程池并发执行 I/O。future_map 保存 Future 与原任务的对应关系。
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(fetch_openimages_image, key, destination, reuse_dir):
                (key, subset, stem, destination)
            for key, subset, stem, destination in jobs
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            key, subset, stem, destination = future_map[future]
            url, method, error = future.result()
            if error:
                failures.append({"source_split": key[0], "image_id": key[1], "error": error})
            else:
                # 图片成功后才写标签，防止留下“有标签但图片损坏”的孤立样本。
                label_path = output_dir / "labels" / subset / f"{stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(boxes_to_yolo_lines(records.get(key, [])), encoding="utf-8")
                successes.append((key, subset, stem, destination, url, method))
            if index % 50 == 0 or index == len(jobs):
                print(f"Processed {index}/{len(jobs)} images")

    # 第六阶段：生成可追溯清单。除了本地路径，还保留授权、作者和原始页面。
    manifest_fields = [
        "local_file", "dataset_split", "selection_role", "openimages_split", "image_id",
        "acquisition", "download_url", "original_url", "original_landing_url", "license",
        "author", "title",
    ]
    manifest_path = external_dir / "source_manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for key, subset, stem, destination, url, method in sorted(successes):
            source = metadata.get(key, {})
            writer.writerow(
                {
                    "local_file": destination.relative_to(ROOT).as_posix(),
                    "dataset_split": subset,
                    "selection_role": "+".join(sorted(roles[key])),
                    "openimages_split": key[0],
                    "image_id": key[1],
                    "acquisition": method,
                    "download_url": url,
                    "original_url": source.get("OriginalURL", ""),
                    "original_landing_url": source.get("OriginalLandingURL", ""),
                    "license": source.get("License", ""),
                    "author": source.get("Author", ""),
                    "title": source.get("Title", ""),
                }
            )

    # 第七阶段：统计实际成功图片、各类别目标框、负样本和获取方式。
    image_counts = Counter(item[1] for item in successes)
    method_counts = Counter(item[5] for item in successes)
    object_counts = Counter()
    negative_count = 0
    for key, *_ in successes:
        boxes = records.get(key, [])
        if not boxes:
            negative_count += 1
        object_counts.update(box["class_name"] for box in boxes)

    stats = {
        "images": dict(sorted(image_counts.items())),
        "total_images": len(successes),
        "objects": dict(sorted(object_counts.items())),
        "hard_negative_images": negative_count,
        "requested_roles": requested,
        "unique_selected_images": len(selected),
        "acquisition": dict(sorted(method_counts.items())),
        "failed_downloads": len(failures),
        "val_fraction": args.val_fraction,
        "seed": args.seed,
    }
    # stats.json 是后续核对“是否约 600 张、训练/验证比例是否合理”的摘要。
    write_json(output_dir / "stats.json", stats)
    if failures:
        write_json(external_dir / "download_failures.json", {"failures": failures})
    print(json.dumps(stats, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 2. Model training
# -----------------------------------------------------------------------------
# 两套 profile 共用同一个训练函数：
# - baseline：从 COCO 预训练 yolo11n.pt 开始，使用原始数据集训练；
# - expanded：从本项目 baseline 权重继续微调，使用扩充数据集和较低学习率。
# 这样既减少重复代码，又能通过 --profile 明确复现实验配置。

TRAINING_PROFILES = {
    "baseline": {
        # yolo11n.pt 是通用预训练起点；Ultralytics 会解析该模型名。
        "model": "yolo11n.pt",
        "data": DATA_V1,
        "epochs": 60,
        "name": "baseline_yolo11n_all_in_one",
        "optimizer": "auto",
        "patience": 15,
        "seed": 42,
        # 最后 10 个 epoch 关闭 mosaic，让模型在自然图像分布上收敛。
        "extra": {"pretrained": True, "close_mosaic": 10},
    },
    "expanded": {
        # 扩充训练不是从零开始，而是承接已经会识别两类物体的 baseline。
        "model": BASELINE_MODEL,
        "data": DATA_V2,
        "epochs": 60,
        "name": "expanded_low_lr_yolo11n_all_in_one",
        "optimizer": "AdamW",
        "patience": 20,
        "seed": 84,
        "extra": {
            # 低初始学习率 + 余弦下降用于保留原有能力并做温和微调。
            "lr0": 0.0002,
            "lrf": 0.1,
            "cos_lr": True,
            "weight_decay": 0.0005,
            # 降低强增强幅度，避免瓶子形态被过度扭曲而影响小数据集微调。
            "mosaic": 0.2,
            "scale": 0.25,
            "translate": 0.05,
            "close_mosaic": 8,
            # 略提高分类损失权重，强调 bottle 与 keyboard 的类别判别。
            "cls": 0.7,
        },
    },
}


def command_train(args: argparse.Namespace) -> None:
    # profile 提供经过设计的默认组合；用户显式传入的值具有更高优先级。
    profile = TRAINING_PROFILES[args.profile]
    source_model = args.model or profile["model"]
    data = resolve_project_path(args.data or profile["data"])
    epochs = args.epochs or profile["epochs"]
    name = args.name or profile["name"]

    # YOLO(...) 加载网络结构和初始权重，但此时还没有开始训练。
    model = YOLO(model_argument(source_model))
    # settings 中是两套 profile 共用的训练参数。输出统一进入 runs/detect。
    settings = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(ROOT / "runs" / "detect"),
        "name": name,
        "exist_ok": True,
        "optimizer": profile["optimizer"],
        "patience": profile["patience"],
        "seed": profile["seed"],
        # 固定 profile 的 seed 并启用 deterministic，尽量让同配置结果可复现。
        "deterministic": True,
        "plots": True,
        "verbose": True,
    }
    # profile 专属参数最后合并，例如 expanded 的低学习率和弱数据增强。
    settings.update(profile["extra"])
    # Ultralytics 内部完成 DataLoader、前向传播、损失计算、反向传播、
    # 验证、早停和 best.pt/last.pt 保存；本项目代码负责传入统一配置。
    model.train(**settings)


# -----------------------------------------------------------------------------
# 3. Standard validation and model comparison
# -----------------------------------------------------------------------------
# validate 评估一个模型；compare 用完全相同的验证集和参数连续评估多个模型，
# 再计算“候选模型 - 第一个模型”的指标差值，避免跨数据集比较造成误导。

def metrics_from_validation(result, weights: Path | str) -> dict:
    # result.box 是 Ultralytics 验证器计算的检测指标对象。
    # p/r/ap50/ap 分别是逐类别 Precision、Recall、mAP@0.5、mAP@0.5:0.95。
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
        # mp/mr/map50/map 是所有类别汇总后的指标。
        "all": {
            "precision": round(float(result.box.mp), 6),
            "recall": round(float(result.box.mr), 6),
            "map50": round(float(result.box.map50), 6),
            "map50_95": round(float(result.box.map), 6),
        },
        "classes": per_class,
    }


def evaluate_model(weights: Path, data: Path, run_name: str, args: argparse.Namespace) -> dict:
    # 单独封装共同验证流程，使 validate 和 compare 不会使用两套不同参数。
    result = YOLO(str(weights)).val(
        data=str(data),
        split="val",  # 明确使用 data.yaml 指向的验证集，不参与训练。
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=args.plots,
        project=str(ROOT / "runs" / "detect"),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )
    return metrics_from_validation(result, weights)


def command_validate(args: argparse.Namespace) -> None:
    # 解析路径 -> 执行评估 -> 保存结构化 JSON -> 同时打印到终端。
    weights = resolve_project_path(args.model)
    data = resolve_project_path(args.data)
    report = evaluate_model(weights, data, args.name, args)
    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def command_compare(args: argparse.Namespace) -> None:
    # 未传 --models 时自动比较基线与扩充模型；传入后可比较任意 NAME=PATH。
    defaults = {"baseline": BASELINE_MODEL, "expanded": EXPANDED_MODEL}
    models = parse_model_specs(args.models, defaults)
    data = resolve_project_path(args.data)
    report = {"evaluation_dataset": str(data), "models": {}, "deltas_vs_first": {}}
    # 所有模型使用同一 data、imgsz、batch 和 device，保证横向比较公平。
    for name, weights in models.items():
        report["models"][name] = evaluate_model(weights, data, f"compare_{name}_all_in_one", args)

    # Python 字典保留插入顺序，因此第一个模型就是差值计算的参照模型。
    first_name = next(iter(models))
    baseline_metrics = report["models"][first_name]
    for name in list(models)[1:]:
        candidate = report["models"][name]
        # 正数表示候选模型在该指标上高于参照，负数表示退步。
        report["deltas_vs_first"][name] = {
            "all": {
                metric: round(candidate["all"][metric] - baseline_metrics["all"][metric], 6)
                for metric in baseline_metrics["all"]
            },
            "classes": {
                class_name: {
                    metric: round(
                        candidate["classes"][class_name][metric]
                        - baseline_metrics["classes"][class_name][metric],
                        6,
                    )
                    for metric in baseline_metrics["classes"][class_name]
                }
                for class_name in baseline_metrics["classes"]
            },
        }

    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 4. Deterministic per-class smoke test
# -----------------------------------------------------------------------------
# “冒烟测试”不是完整验证，而是从每个类别固定抽取少量图片，快速确认：
# 模型能否输出正确类别，以及预测框与真实框的 IoU 是否达到阈值。
# 固定 seed 和排序步骤使每次检查相同图片，适合训练后快速回归测试。

def read_yolo_labels(path: Path):
    # 把 YOLO 标签文本每行解析为 (class_id, x_center, y_center, w, h)。
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, width, height = map(float, line.split())
        labels.append((int(class_id), x, y, width, height))
    return labels


def normalized_to_xyxy(label, image_width: int, image_height: int):
    # 将 0~1 的中心点宽高坐标还原为像素级左上角/右下角坐标，供 IoU 计算。
    _, x, y, width, height = label
    return (
        (x - width / 2) * image_width,
        (y - height / 2) * image_height,
        (x + width / 2) * image_width,
        (y + height / 2) * image_height,
    )


def box_iou(box_a, box_b) -> float:
    # IoU = 两框交集面积 / 两框并集面积，是目标检测中衡量定位重合度的指标。
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    # 极端情况下两个框都没有面积，返回 0 可避免除零异常。
    return intersection / union if union else 0.0


def image_for_label(images_dir: Path, stem: str) -> Path:
    # 标签只保存文件 stem，因此依次尝试常见图片后缀找到配对图片。
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for label stem: {stem}")


def command_smoke(args: argparse.Namespace) -> None:
    # 第一步：定位扩充数据集的验证标签与图片目录，并固定抽样随机源。
    dataset_dir = resolve_project_path(args.dataset_dir)
    labels_dir = dataset_dir / "labels" / "val"
    images_dir = dataset_dir / "images" / "val"
    rng = random.Random(args.seed)
    class_names = {0: "bottle", 1: "keyboard"}

    selected = {}
    # 每个类别独立筛选“确实含该类真实框”的标签，再随机取 per_class 个。
    for class_id in class_names:
        candidates = [
            path for path in sorted(labels_dir.glob("*.txt"))
            if any(label[0] == class_id for label in read_yolo_labels(path))
        ]
        rng.shuffle(candidates)
        selected[class_id] = candidates[:args.per_class]

    # 第二步：模型只加载一次，随后按类别批量推理，减少重复初始化开销。
    model = YOLO(str(resolve_project_path(args.model)))
    report = {
        "criteria": f"correct class and IoU >= {args.match_iou:.2f} at conf >= {args.conf:.2f}",
        "classes": {},
    }
    for class_id, label_paths in selected.items():
        image_paths = [image_for_label(images_dir, path.stem) for path in label_paths]
        # conf 先过滤低置信度预测；save_images 可输出画好框的抽样结果。
        results = model.predict(
            source=[str(path) for path in image_paths],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            device=args.device,
            save=args.save_images,
            project=str(ROOT / "runs" / "detect"),
            name=f"smoke_{class_names[class_id]}_all_in_one",
            exist_ok=True,
            verbose=False,
        )
        rows = []
        for label_path, image_path, result in zip(label_paths, image_paths, results):
            image_height, image_width = result.orig_shape
            # 只拿当前被测试类别的真实框，避免另一类别干扰成功条件。
            truths = [
                normalized_to_xyxy(label, image_width, image_height)
                for label in read_yolo_labels(label_path)
                if label[0] == class_id
            ]
            predictions = []
            if result.boxes is not None:
                # 推理张量通常位于 GPU；先 .cpu().tolist() 才便于 Python 遍历。
                for xyxy, predicted_class, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.cls.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                ):
                    if int(predicted_class) == class_id:
                        predictions.append((xyxy, confidence))
            # 一张图可能有多个真实框/预测框；取所有同类配对中的最高 IoU。
            best_iou = max(
                (box_iou(prediction[0], truth) for prediction in predictions for truth in truths),
                default=0.0,
            )
            rows.append(
                {
                    "image": image_path.name,
                    # 同类别预测已经由 predictions 筛选，因此达到 IoU 即成功。
                    "success": best_iou >= args.match_iou,
                    "best_iou": round(best_iou, 4),
                    "best_confidence": round(
                        max((prediction[1] for prediction in predictions), default=0.0), 4
                    ),
                }
            )
        # bool 在求和时按 1/0 处理，可直接得到成功图片数和成功率。
        successes = sum(row["success"] for row in rows)
        report["classes"][class_names[class_id]] = {
            "successes": successes,
            "total": len(rows),
            "rate": round(successes / len(rows), 6) if rows else 0.0,
            "images": rows,
        }

    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 5. Hard-negative evaluation
# -----------------------------------------------------------------------------
# 困难负样本中的正确答案是“没有 bottle，也没有 keyboard”。因此模型输出的任意
# 检测框都算误报。本模块同时统计误报框总数和出现误报的图片数：前者反映误报
# 密度，后者反映实际使用中有多少画面会受到影响。

def evaluate_false_detections(weights: Path, images: list[Path], args: argparse.Namespace) -> dict:
    # 每个模型独立加载并对完全相同的负样本列表批量推理。
    model = YOLO(str(weights))
    false_boxes = Counter()
    images_with_false_detection = Counter()
    results = model.predict(
        source=[str(path) for path in images],
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        verbose=False,
    )
    any_false_images = 0
    for result in results:
        # seen 是“当前图片出现过误报的类别集合”；同类多个框只计一次图片事件。
        seen = set()
        if result.boxes is not None:
            for class_id in result.boxes.cls.int().cpu().tolist():
                class_name = result.names[class_id]
                # 框计数保留所有误报，可识别“一张图重复报很多瓶子”的问题。
                false_boxes[class_name] += 1
                seen.add(class_name)
        if seen:
            any_false_images += 1
        for class_name in seen:
            images_with_false_detection[class_name] += 1
    return {
        "weights": str(weights),
        "negative_images": len(images),
        "confidence": args.conf,
        "images_with_any_false_detection": any_false_images,
        "images_with_false_detection_by_class": dict(images_with_false_detection),
        "false_detection_boxes_by_class": dict(false_boxes),
    }


def command_hard_negative(args: argparse.Namespace) -> None:
    # source_manifest.csv 的 selection_role 记录每张图为何被选入数据集。
    manifest = resolve_project_path(args.manifest)
    images = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            # 一张图可能拥有多个角色，以 + 分隔，所以先 split 再精确匹配。
            if "person_negative" in row["selection_role"].split("+"):
                images.append(resolve_project_path(row["local_file"]))
    images.sort()
    if not images:
        raise RuntimeError("No person_negative images were found in the manifest")

    defaults = {"baseline": BASELINE_MODEL, "expanded": EXPANDED_MODEL}
    models = parse_model_specs(args.models, defaults)
    # 对每个模型复用同一 images 列表，报告可直接比较误报改善情况。
    report = {
        name: evaluate_false_detections(weights, images, args)
        for name, weights in models.items()
    }
    destination = resolve_project_path(args.output)
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


# -----------------------------------------------------------------------------
# 6. Model export
# -----------------------------------------------------------------------------
# .pt 适合当前 Python/Ultralytics 环境；ONNX 更通用；TensorRT engine 在 Jetson
# 上推理通常更快，但必须与目标 Jetson 的 TensorRT/CUDA/架构兼容。通常先复制
# .pt 或 ONNX 到板端，再在板端生成 engine，而不要直接搬运电脑生成的 engine。

def command_export(args: argparse.Namespace) -> None:
    # 加载训练得到的 PyTorch 权重，并把命令行选项整理为 export 参数字典。
    weights = resolve_project_path(args.model)
    model = YOLO(str(weights))
    # TensorRT 构建需要 CUDA；若用户仍保留 engine 的默认 cpu，则自动改用 GPU 0。
    export_device = "0" if args.format == "engine" and args.device == "cpu" else args.device
    settings = {
        "format": args.format,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "device": export_device,
    }
    if args.format == "onnx":
        # opset 决定 ONNX 运算符版本，17 对现代运行时具有较好兼容性。
        settings["opset"] = args.opset
    elif args.precision == "fp16":
        # 新版 Ultralytics 用 quantize 位数表达 TensorRT 精度。
        settings["quantize"] = 16
    elif args.precision == "int8":
        settings["quantize"] = 8
        # INT8 必须用代表性图片校准量化范围，所以额外提供 data.yaml。
        settings["data"] = str(resolve_project_path(args.data))
    # export() 完成计算图转换并返回实际生成文件路径。
    output = model.export(**settings)
    print(f"Exported model: {output}")


# -----------------------------------------------------------------------------
# 7. Camera probe, live inference, snapshots and annotated recording
# -----------------------------------------------------------------------------
# 摄像头模块包含四个功能：
# 1. --probe 枚举摄像头索引和 Windows 后端，并保存预览图；
# 2. 实时读取本机摄像头，逐帧运行 YOLO 并显示检测框；
# 3. S 保存当前“已画框”截图；R 开始/停止录制已画框视频；
# 4. 退出后生成 FPS、类别出现帧数、总框数、截图与录像清单报告。

def frame_stats(frame):
    # 全画面均值近似亮度，标准差近似对比度；两者都接近 0 时基本是全黑帧。
    return round(float(frame.mean()), 2), round(float(frame.std()), 2)


def open_camera_backend(index: int, backend_key: str):
    # 明确指定摄像头索引和 Windows 捕获后端，便于对黑屏问题逐个排查。
    backend_id, backend_name = CAMERA_BACKENDS[backend_key]
    camera = cv2.VideoCapture(index, backend_id)
    if not camera.isOpened():
        camera.release()
        return None, backend_name, None
    frame = None
    # 某些摄像头刚打开的前几帧为空或尚未完成自动曝光，所以预热读取 30 次。
    for _ in range(30):
        ok, candidate = camera.read()
        if ok and candidate is not None and candidate.size:
            frame = candidate
        time.sleep(0.01)
    if frame is None:
        camera.release()
        return None, backend_name, None
    return camera, backend_name, frame


def open_camera(index: int, preference: str):
    # auto 会尝试三个后端；显式 --backend 则只尝试用户指定的一个。
    keys = [preference] if preference != "auto" else ["msmf", "dshow", "auto"]
    # 若所有可用后端都是黑屏，仍保留第一个黑屏设备作为回退并打印诊断数据。
    dark_fallback = None
    for key in keys:
        camera, name, frame = open_camera_backend(index, key)
        if camera is None:
            continue
        mean, std = frame_stats(frame)
        # 阈值同时要求低亮度与低对比度，普通暗环境不会轻易被判为纯黑帧。
        is_black = mean < 3.0 and std < 3.0
        print(
            f"Camera candidate {index}/{name}: {frame.shape[1]}x{frame.shape[0]}, "
            f"brightness={mean}, contrast={std}, black={is_black}"
        )
        # 优先返回第一个具有有效画面的后端，并释放之前暂存的黑屏设备句柄。
        if not is_black:
            if dark_fallback is not None:
                dark_fallback[0].release()
            return camera, name, frame
        if dark_fallback is None:
            dark_fallback = (camera, name, frame)
        else:
            camera.release()
    return dark_fallback if dark_fallback is not None else (None, None, None)


def probe_cameras(max_index: int, output_dir: Path) -> int:
    # 对 0..max_index 的每个索引尝试全部后端，而不是找到一个就停止。
    output_dir.mkdir(parents=True, exist_ok=True)
    found = []
    for index in range(max_index + 1):
        for backend_key in ("msmf", "dshow", "auto"):
            camera, backend, frame = open_camera_backend(index, backend_key)
            if camera is None:
                continue
            height, width = frame.shape[:2]
            mean, std = frame_stats(frame)
            # 保存原始预览图，用户即使关闭终端也能核对各组合是否有画面。
            preview = output_dir / f"probe_camera{index}_{backend_key}.jpg"
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
    # 返回码 0 表示至少发现一个可读组合；1 便于脚本/批处理识别失败。
    return 0 if found else 1


def create_video_writer(frame, fps: float, output_dir: Path):
    # VideoWriter 创建时必须固定分辨率和 FPS，后续写入帧需保持相同尺寸。
    height, width = frame.shape[:2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # 优先使用体积较小且易播放的 MP4；系统不支持 mp4v 时回退到 MJPG/AVI。
    for codec, suffix in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        video_path = output_dir / f"recording_{timestamp}{suffix}"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
        )
        if writer.isOpened():
            # 与 writer 一起返回录像元数据，停止时再补充帧数和时长。
            return writer, {
                "file": str(video_path.relative_to(ROOT)),
                "codec": codec,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "frames": 0,
            }
        writer.release()
    return None, None


def command_camera(args: argparse.Namespace) -> int:
    # --probe 是独立诊断路径：不加载模型，完成枚举后立即返回。
    output_dir = resolve_project_path(args.output_dir)
    if args.probe:
        return probe_cameras(args.max_camera_index, output_dir)

    # 实时路径先检查模型，再打开摄像头，失败时给出明确错误。
    model_path = resolve_project_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    camera, backend, frame = open_camera(args.camera, args.backend)
    if camera is None:
        raise RuntimeError(
            f"Cannot open camera {args.camera}. Close other camera apps or use --probe."
        )
    # 请求尽量短的采集缓存，减少“处理的是旧画面”导致的实时显示延迟。
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    # auto 优先使用 CUDA:0；电脑没有可用 CUDA 时自动退回 CPU。
    device = 0 if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and not torch.cuda.is_available():
        device = "cpu"
    # 模型在进入循环前只加载一次，循环内只做逐帧 predict。
    model = YOLO(str(model_path))

    window_name = "Bottle + Keyboard | R record | S snapshot | Q/Esc quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # perf_counter 适合测量耗时；Counter 分别累计“出现该类的帧数”和“框总数”。
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
    # --record 不能在得到首张 annotated 之前创建 Writer，因此用 pending 延迟启动。
    auto_record_pending = args.record
    # 摄像头有时报告 0、负数或异常高 FPS；不可信时使用常见的 30 FPS。
    recording_fps = float(camera.get(cv2.CAP_PROP_FPS))
    if not 1.0 <= recording_fps <= 120.0:
        recording_fps = 30.0

    print(f"Camera {args.camera} opened with {backend}")
    print(f"Model: {model_path}")
    print(f"Device: {device}; confidence threshold: {args.conf}")

    try:
        while True:
            # 第一次循环直接使用 open_camera 已预热得到的 frame；之后再读取新帧。
            if frame_count > 0:
                ok, frame = camera.read()
                if not ok or frame is None:
                    print("Camera frame read failed")
                    break

            # Ultralytics 接受 OpenCV 的 BGR ndarray，并返回一个 Results 对象。
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=0.7,
                device=device,
                verbose=False,
            )[0]
            # plot() 复制画面并绘制边框、类别名与置信度，原始 frame 不被录像使用。
            annotated = result.plot(line_width=2, labels=True, conf=True)

            # set 保证一帧即使有三个 bottle，“含 bottle 的帧数”仍只增加一次。
            detected_this_frame = set()
            if result.boxes is not None:
                for class_id in result.boxes.cls.int().cpu().tolist():
                    class_name = result.names[class_id]
                    detected_this_frame.add(class_name)
                    # 框总数则对每个检测框都累加，用于观察重复框/多目标情况。
                    total_detections[class_name] += 1
            for class_name in detected_this_frame:
                frames_with_class[class_name] += 1

            now = time.perf_counter()
            instantaneous_fps = 1.0 / max(now - previous_time, 1e-9)
            # 指数移动平均抑制单帧抖动：90% 历史值 + 10% 最新瞬时 FPS。
            smoothed_fps = (
                instantaneous_fps
                if frame_count == 0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )
            previous_time = now
            frame_count += 1

            # 在检测结果上叠加运行状态和按键提示；这些文字也会录进视频。
            cv2.putText(
                annotated,
                f"FPS {smoothed_fps:.1f} | conf {args.conf:.2f} | device {device}",
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                "R record/stop   S snapshot   Q/Esc quit",
                (15, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --record 指定时，在第一张完成标注的画面尺寸已知后自动创建录像文件。
            if auto_record_pending:
                video_writer, current_recording = create_video_writer(
                    annotated, recording_fps, output_dir
                )
                auto_record_pending = False
                if video_writer is None:
                    print("Cannot create video file. Recording was not started.")
                else:
                    print(f"Started recording: {ROOT / current_recording['file']}")

            if video_writer is not None:
                # 录制状态下先画红色 REC，再把整张 annotated 写入视频。
                cv2.putText(
                    annotated, "REC", (15, 98), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2, cv2.LINE_AA,
                )
                video_writer.write(annotated)
                current_recording["frames"] += 1

            # imshow 只负责显示；waitKey 同时让窗口处理系统消息并读取键盘。
            cv2.imshow(window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            # Q 或 Esc：退出；S：截图；R：在“开始录制/停止录制”之间切换。
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                # 微秒级时间戳防止快速连续截图发生重名覆盖。
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot = output_dir / f"snapshot_{timestamp}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                snapshots.append(str(snapshot.relative_to(ROOT)))
                print(f"Saved snapshot: {snapshot}")
            if key == ord("r"):
                if video_writer is None:
                    # 当前未录制：创建 Writer 并建立 current_recording 元数据。
                    video_writer, current_recording = create_video_writer(
                        annotated, recording_fps, output_dir
                    )
                    if video_writer is None:
                        print("Cannot create video file. Recording was not started.")
                    else:
                        print(f"Started recording: {ROOT / current_recording['file']}")
                else:
                    # 当前正在录制：释放 Writer，按“帧数/FPS”计算视频时长。
                    video_writer.release()
                    current_recording["duration_seconds"] = round(
                        current_recording["frames"] / current_recording["fps"], 3
                    )
                    recordings.append(current_recording)
                    print(f"Stopped recording: {ROOT / current_recording['file']}")
                    video_writer = None
                    current_recording = None
    # finally 无论正常按键退出还是运行异常都会执行，确保摄像头/文件句柄被释放。
    finally:
        if video_writer is not None:
            video_writer.release()
            current_recording["duration_seconds"] = round(
                current_recording["frames"] / current_recording["fps"], 3
            )
            recordings.append(current_recording)
        camera.release()
        cv2.destroyAllWindows()

    # 资源释放后汇总本次会话，写入固定名称的 JSON（下次运行会更新该报告）。
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
    report_path = output_dir / "camera_test_report.json"
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report saved to: {report_path}")
    return 0


# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------
# argparse 将所有功能统一为：python 本文件.py <子命令> [选项]。
# 每个子解析器最后通过 set_defaults(func=...) 绑定处理函数；因此新增功能通常只需
# 编写 command_xxx(args)，再在 build_parser() 注册一个子命令，无需修改 main()。

def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    # validate 与 compare 的公共参数只定义一次，保证二者默认评估条件一致。
    parser.add_argument("--data", default=str(DATA_V2), help="YOLO data YAML")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)


def build_parser() -> argparse.ArgumentParser:
    # ArgumentDefaultsHelpFormatter 会在 --help 中自动显示每项默认值。
    parser = argparse.ArgumentParser(
        description="All-in-one bottle/keyboard YOLO project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=True 表示用户必须选择一个子命令，不能空运行整个程序。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare：约 600 张数据的抽样、复用/下载、标签转换和来源统计。
    prepare = subparsers.add_parser("prepare", help="build the Open Images dataset")
    prepare.add_argument(
        "--metadata-dir", default="dataset/external/openimages/metadata",
        help="directory containing Open Images metadata CSV files",
    )
    prepare.add_argument("--output-dir", default="dataset_v2")
    prepare.add_argument("--reuse-dir", default="dataset")
    prepare.add_argument("--bottle-images", type=int, default=300)
    prepare.add_argument("--keyboard-images", type=int, default=240)
    prepare.add_argument("--negative-images", type=int, default=60)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--workers", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=84)
    # args.func 是函数对象而不是字符串，main() 可以直接调用。
    prepare.set_defaults(func=command_prepare)

    # train：--profile baseline/expanded 选择两套训练策略，其余参数可覆盖默认值。
    train = subparsers.add_parser("train", help="train or fine-tune a YOLO model")
    train.add_argument("--profile", choices=sorted(TRAINING_PROFILES), default="expanded")
    train.add_argument("--model", help="override starting model")
    train.add_argument("--data", help="override YOLO data YAML")
    train.add_argument("--epochs", type=int, help="override epoch count")
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=16)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--name", help="override run name")
    train.set_defaults(func=command_train)

    # validate：对单个模型运行完整验证集指标，并保存 JSON。
    validate = subparsers.add_parser("validate", help="validate one model")
    validate.add_argument("--model", default=str(EXPANDED_MODEL))
    validate.add_argument("--name", default="validation_all_in_one")
    validate.add_argument("--output", default="runs/detect/validation_all_in_one.json")
    add_validation_arguments(validate)
    validate.set_defaults(func=command_validate)

    # smoke：每类抽少量固定样本，快速检查类别正确性和框的 IoU。
    smoke = subparsers.add_parser("smoke", help="run a per-class sample test")
    smoke.add_argument("--model", default=str(EXPANDED_MODEL))
    smoke.add_argument("--dataset-dir", default="dataset_v2")
    smoke.add_argument("--per-class", type=int, default=10)
    smoke.add_argument("--conf", type=float, default=0.25)
    smoke.add_argument("--match-iou", type=float, default=0.5)
    smoke.add_argument("--imgsz", type=int, default=640)
    smoke.add_argument("--device", default="0")
    smoke.add_argument("--seed", type=int, default=20260824)
    smoke.add_argument(
        "--save-images", action=argparse.BooleanOptionalAction, default=True
    )
    smoke.add_argument("--output", default="runs/detect/smoke_test_all_in_one.json")
    smoke.set_defaults(func=command_smoke)

    # compare：在同一个验证集上比较两个或更多 NAME=PATH 模型。
    compare = subparsers.add_parser("compare", help="compare models on one validation set")
    compare.add_argument(
        "--models", action="append", metavar="NAME=PATH",
        help="repeat for each model; defaults to baseline and expanded",
    )
    compare.add_argument("--output", default="runs/detect/model_comparison_all_in_one.json")
    add_validation_arguments(compare)
    compare.set_defaults(func=command_compare)

    # hard-negative：只看 person_negative 图片上的 bottle/keyboard 误报。
    hard = subparsers.add_parser(
        "hard-negative", help="compare false detections on person-only images"
    )
    hard.add_argument(
        "--models", action="append", metavar="NAME=PATH",
        help="repeat for each model; defaults to baseline and expanded",
    )
    hard.add_argument(
        "--manifest", default="dataset_v2/external/openimages/source_manifest.csv"
    )
    hard.add_argument("--conf", type=float, default=0.25)
    hard.add_argument("--imgsz", type=int, default=640)
    hard.add_argument("--batch", type=int, default=16)
    hard.add_argument("--device", default="0")
    hard.add_argument("--output", default="runs/detect/hard_negative_all_in_one.json")
    hard.set_defaults(func=command_hard_negative)

    # export：将 .pt 转为 ONNX，或在有 CUDA/TensorRT 的设备上生成 engine。
    export = subparsers.add_parser("export", help="export to ONNX or TensorRT")
    export.add_argument("--model", default=str(EXPANDED_MODEL))
    export.add_argument("--format", choices=("onnx", "engine"), default="onnx")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--batch", type=int, default=1)
    export.add_argument("--device", default="cpu")
    export.add_argument("--opset", type=int, default=17)
    export.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=False)
    export.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    export.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    export.add_argument("--data", default=str(DATA_V2), help="INT8 calibration data")
    export.set_defaults(func=command_export)

    # camera：摄像头诊断、实时检测、截图、录像和会话报告。
    camera = subparsers.add_parser(
        "camera", help="probe or run live camera detection, snapshots and recording"
    )
    camera.add_argument("--camera", type=int, default=0)
    camera.add_argument("--backend", choices=tuple(CAMERA_BACKENDS), default="auto")
    camera.add_argument("--conf", type=float, default=0.25)
    camera.add_argument("--imgsz", type=int, default=640)
    camera.add_argument("--model", default=str(EXPANDED_MODEL))
    camera.add_argument("--device", default="auto")
    camera.add_argument("--record", action="store_true", help="record immediately")
    camera.add_argument("--probe", action="store_true", help="probe cameras and exit")
    camera.add_argument("--max-camera-index", type=int, default=4)
    camera.add_argument("--output-dir", default="runs/camera_test")
    camera.set_defaults(func=command_camera)

    return parser


def main() -> int:
    # parse_args() 会处理 --help、类型转换、choices 校验以及缺少参数的错误提示。
    parser = build_parser()
    args = parser.parse_args()
    # func 由所选子命令的 set_defaults 注入，实现统一分派。
    result = args.func(args)
    # 大多数命令无显式返回值(None)，视为成功 0；probe 可返回 0/1 给 PowerShell。
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    # SystemExit 把 main() 的整数返回码传给操作系统；作为模块 import 时不会执行。
    raise SystemExit(main())
