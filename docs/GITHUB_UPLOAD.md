# 从 PowerShell 上传到 GitHub

本目录已经按可发布仓库整理完成。不要上传项目原目录中的 `.venv`、完整数据集、训练缓存或
COCO 241MB 标注压缩包。

## 方法一：Git 命令行

1. 登录 GitHub，点击右上角 `+` → `New repository`。
2. 输入仓库名，例如 `bottle-keyboard-yolo`。
3. 选择 Public 或 Private。
4. 不要勾选自动创建 README、`.gitignore` 或 License，因为本地已经准备好这些文件。
5. 创建仓库后复制其 HTTPS 地址，例如：
   `https://github.com/YOUR_NAME/bottle-keyboard-yolo.git`。
6. 在 PowerShell 执行：

```powershell
cd E:\YOLO\github_release
git init -b main
git add .
git status
git commit -m "Initial release: bottle and keyboard YOLO models"
git remote add origin https://github.com/YOUR_NAME/bottle-keyboard-yolo.git
git remote -v
git push -u origin main
```

把 `YOUR_NAME` 换成您的 GitHub 用户名。如果 GitHub 要求登录，按浏览器提示授权，或使用
Personal Access Token；不要把密码、Token 或密钥写进源码和提交记录。

## 方法二：GitHub CLI

已安装并登录 `gh` 时，可直接执行：

```powershell
cd E:\YOLO\github_release
git init -b main
git add .
git commit -m "Initial release: bottle and keyboard YOLO models"
gh repo create bottle-keyboard-yolo --public --source=. --remote=origin --push
```

如果不希望公开，改用 `--private`。

## 模型文件说明

三份 `.pt` 权重单个约 5.5MB，低于 GitHub 普通 Git 的100MiB单文件上限，可直接提交。
未来若加入超过100MiB的模型，应改用 Git LFS 或 GitHub Releases。

## 上传前检查

```powershell
git status
Get-ChildItem -Recurse -File | Sort-Object Length -Descending |
  Select-Object -First 20 Length, FullName
```

确认没有以下内容：

- `.venv/`
- `dataset/`、`dataset_v2/`、`dataset_v3/` 中的训练图片
- `runs/` 训练中间文件
- API Token、账号密码、私钥
- COCO 完整标注压缩包或大体积原始图片
