# Bottle + Keyboard YOLO — V2 Expanded

这是仓库的第二次发布。在V1基线模型上扩充数据并进行低学习率微调，重点减少把人物等物体
误报成瓶子的问题。

## 版本历史

| 版本 | 图片数量 | 主要变化 | 权重 |
|---|---:|---|---|
| V1 | 360 | 建立瓶子/键盘基线 | `bottle_keyboard_yolo11n_best.pt` |
| V2 | 599 | 扩充样本并加入60张人物困难负样本 | `bottle_keyboard_yolo11n_expanded_best.pt` |

V2在同一验证集上达到整体 mAP50 0.7751、mAP50-95 0.6393；人物困难负样本上的bottle
误报框从V1的16个下降到3个。完整结果见
[`docs/EXPANDED_TRAINING_REPORT.md`](docs/EXPANDED_TRAINING_REPORT.md)。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 训练

```powershell
# 首次训练V1
python .\train_baseline.py

# 从V1最佳权重微调V2
python .\train_expanded_low_lr.py
```

## 摄像头测试

V2快照默认加载扩充模型：

```powershell
python .\camera_inference.py --camera 0 --backend auto --conf 0.25
```

也可以显式切换模型：

```powershell
python .\camera_inference.py `
  --model .\exports\bottle_keyboard_yolo11n_best.pt
```

## 数据与许可

V1/V2训练图片均来自Open Images V7。图片不提交到GitHub，逐图来源和许可链接保存在
`manifests/`。第三方许可提示见 [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md)。
