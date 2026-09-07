# Bottle + Keyboard YOLO11n

这是一个识别 `bottle`（瓶子）和 `keyboard`（键盘）的两类别 YOLO11n 项目，包含数据准备、
三版模型训练、验证、模型对比、摄像头实时识别、截图、录像、ONNX/TensorRT 导出等功能。

第三版模型重点解决“竖放瓶子可以识别、横放瓶子容易漏检”的问题。它在第二版模型基础上，
增加了经过宽高比和目标尺寸筛选的 COCO 2017 横放瓶子图片。

## 目录结构

```text
.
├─ yolo_project_all_in_one.py              # 所有功能的统一命令行入口
├─ yolo_project_all_in_one_commented.py    # 功能相同的详细中文注释版
├─ camera_inference.py                     # 独立摄像头推理程序
├─ train_baseline.py                       # V1 基线训练
├─ train_expanded_low_lr.py                # V2 扩充数据低学习率微调
├─ train_horizontal_v3.py                  # V3 横放瓶子强化微调
├─ evaluate_horizontal_models.py           # 三版本横放瓶子公平对比
├─ data.yaml / data_v2.yaml / data_v3.yaml
├─ tools/build_v3_coco_horizontal.py       # 构建 1000 张 V3 数据集
├─ exports/                                # 三版最佳 .pt 权重
├─ manifests/                              # 数据量统计和图片来源/许可清单
└─ docs/                                   # 模型、数据源和 GitHub 上传说明
```

训练图片本身没有提交到 GitHub：它们体积较大，而且需要保留各图片原始许可信息。仓库提供了
来源清单、统计和重建脚本。

## 安装

建议使用 Python 3.12。先按照 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
安装与电脑 CUDA/CPU 相适配的 PyTorch，然后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

本次训练环境为 Python 3.12.13、PyTorch 2.11.0+cu128、Ultralytics 8.4.127、
NVIDIA RTX 5080 Laptop GPU。

## 直接测试模型

查看全部功能：

```powershell
python .\yolo_project_all_in_one.py --help
```

用 V3 模型打开本机摄像头：

```powershell
python .\yolo_project_all_in_one.py camera `
  --model .\exports\bottle_keyboard_yolo11n_horizontal_v3_best.pt `
  --camera 0 --backend auto --conf 0.25
```

摄像头窗口中：`S` 截图、`R` 开始/停止录像、`Q` 或 `Esc` 退出。

切换模型时只需替换 `--model`：

```powershell
# V1
--model .\exports\bottle_keyboard_yolo11n_best.pt

# V2
--model .\exports\bottle_keyboard_yolo11n_expanded_best.pt

# V3
--model .\exports\bottle_keyboard_yolo11n_horizontal_v3_best.pt
```

## 训练与比较

```powershell
# 训练 V1
python .\train_baseline.py

# 从 V1 微调 V2
python .\train_expanded_low_lr.py

# 从 V2 微调 V3
python .\train_horizontal_v3.py

# 在留出的横放瓶子图片上比较三个模型
python .\evaluate_horizontal_models.py
```

更详细的模型差异见 [docs/MODEL_VERSIONS.md](docs/MODEL_VERSIONS.md)，数据来源与许可说明见
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)。

## 许可提示

本项目依赖 Ultralytics YOLO，并包含基于 Ultralytics YOLO11n 训练的模型。Ultralytics 当前将
其开源软件和训练模型默认置于 AGPL-3.0；闭源、商业或嵌入式产品用途可能需要 Enterprise
License。发布和部署前请阅读 [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。
