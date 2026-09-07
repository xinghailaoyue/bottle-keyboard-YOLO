$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot '.ultralytics'
Set-Location $ProjectRoot

Write-Host '正在启动矿泉水瓶 + 键盘实时识别...' -ForegroundColor Cyan
Write-Host '操作：R 开始/停止录像；S 保存带标注截图；Q 或 Esc 退出。' -ForegroundColor Yellow
Write-Host '如果相机被占用，请先关闭“相机”、微信、会议软件等程序。' -ForegroundColor DarkYellow
Write-Host ''

& '.\.venv\Scripts\python.exe' -u '.\camera_inference.py' --camera 0 --backend msmf --conf 0.25

if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host "识别程序退出，错误代码：$LASTEXITCODE" -ForegroundColor Red
    Read-Host '按 Enter 关闭窗口'
}
