# 瓶子与键盘 YOLO11n 基线报告

## 环境

- GPU：NVIDIA GeForce RTX 5080 Laptop GPU（16 GB）
- Python：3.12.13
- PyTorch：2.11.0 + CUDA 12.8
- Ultralytics：8.4.127
- 模型：YOLO11n
- 输入尺寸：640 × 640
- 训练轮数：60

## 数据集

- 训练图片：288
- 验证图片：72
- 训练/验证总图片：360
- 类别：`bottle`、`keyboard`

## 最佳 PyTorch 权重验证指标

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| 全部 | 0.900 | 0.738 | 0.775 | 0.596 |
| bottle | 0.861 | 0.531 | 0.581 | 0.495 |
| keyboard | 0.938 | 0.946 | 0.969 | 0.696 |

键盘检测已经较稳定；瓶子召回率偏低，主要风险是漏检。现有 Open Images `Bottle` 类别范围很宽，
包含酒瓶、饮料瓶、罐状容器等，和桌面矿泉水瓶场景并不完全一致。

## 20 张抽样冒烟测试

测试从现有验证集中固定随机抽取每类 10 张。成功标准：类别正确、置信度不低于 0.25，且预测框
与标注框 IoU 不低于 0.50。

- bottle：8/10
- keyboard：10/10
- 合计：18/20（90%）

详细逐图结果位于 `runs/detect/baseline_yolo11n/smoke_test_report.json`。该测试可以验证训练与
推理流程，但验证集已在训练过程中用于模型选择，因此不能代替真实桌面环境下的独立 20 次验收。

## ONNX 导出验证

- ONNX checker：通过
- Opset：17
- 输入：`1 × 3 × 640 × 640`
- 输出：`1 × 6 × 8400`
- ONNX Runtime CPU mAP50：0.750
- ONNX Runtime CPU mAP50-95：0.577
- ONNX Runtime CPU 推理时间：约 26.3 ms/图（不含预处理和后处理）

## 交付模型

- `exports/bottle_keyboard_yolo11n_best.pt`
- `exports/bottle_keyboard_yolo11n_opset17.onnx`

SHA-256：

```text
PT   2181DB69CFBAFACA734EB3BE976FB6E5F04560D8264B0283D60DACB6A0C3F0BD
ONNX A2A4C26C1E598C8F78099FF5945FA46F7401C57558C136BFBC09DCAD9AE30EB0
```

## Jetson 注意事项

没有在 Windows RTX 电脑上生成 TensorRT engine。普通 TensorRT engine 与构建平台、TensorRT
版本和 GPU 架构相关，应把 `.pt` 或 `.onnx` 复制到目标 Jetson，并在该 Jetson 上根据实际
JetPack/TensorRT 版本构建 engine。
