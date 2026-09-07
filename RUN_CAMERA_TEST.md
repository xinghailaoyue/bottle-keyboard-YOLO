# 本机摄像头实时识别

在 PowerShell 中进入克隆后的仓库目录，运行：

```powershell
$env:YOLO_CONFIG_DIR=(Join-Path (Get-Location) '.ultralytics')
.\.venv\Scripts\python.exe .\camera_inference.py --camera 0 --conf 0.25
```

操作：

- 将矿泉水瓶或键盘放到摄像头前；
- `R`：开始或停止录制带检测框的视频；
- `S`：保存带检测框的截图；
- `Q` 或 `Esc`：退出；
- 录像与截图保存在 `runs/camera_test/`；
- 退出后摘要保存在 `runs/camera_test/camera_test_report.json`。

启动后自动录制：

```powershell
.\.venv\Scripts\python.exe .\camera_inference.py --camera 0 --backend msmf --conf 0.25 --record
```

指定其他模型并自动录制：

```powershell
.\.venv\Scripts\python.exe .\camera_inference.py --model .\exports\bottle_keyboard_yolo11n_best.pt --camera 0 --backend msmf --conf 0.25 --record
```

如果摄像头编号不是 0，可先运行：

```powershell
.\.venv\Scripts\python.exe .\camera_inference.py --probe
```
