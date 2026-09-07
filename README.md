# Bottle + Keyboard YOLO — V1 Baseline

这是仓库的第一次发布：使用YOLO11n识别 `bottle` 和 `keyboard`。

## 本次提交内容

- Open Images V7 数据集：360张（288 train、72 val）。
- V1最佳权重：`exports/bottle_keyboard_yolo11n_best.pt`。
- 整体 mAP50：0.775；mAP50-95：0.596。
- 每类10张冒烟测试：bottle 8/10、keyboard 10/10。
- 数据准备、训练、验证、摄像头识别、截图、录像和导出代码。

详细指标见 [`docs/BASELINE_REPORT.md`](docs/BASELINE_REPORT.md)，训练曲线见
[`docs/images/v1_training_results.png`](docs/images/v1_training_results.png)。

## 安装

先根据电脑环境安装PyTorch，再安装项目依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 训练

训练图片不提交到GitHub。准备好 `dataset/` 后运行：

```powershell
python .\train_baseline.py
```

## 摄像头测试

```powershell
python .\camera_inference.py --camera 0 --backend auto --conf 0.25
```

按 `S` 截图、`R` 开始或停止录像、`Q` 或 `Esc` 退出。V1快照中的默认模型就是基线权重。

## 数据与许可

训练图片来自Open Images V7；逐图来源与许可链接见
`manifests/openimages_v1_source_manifest.csv`。项目依赖Ultralytics YOLO，许可提示见
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md)。
