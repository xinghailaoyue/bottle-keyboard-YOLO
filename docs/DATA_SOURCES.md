# 数据来源与可追溯性

最终 V3 模型使用了两个公开数据集的图片。

## 1. Open Images V7

- 用途：V1、V2 的 bottle/keyboard 正样本，以及 V2 的 person-only 困难负样本。
- V3 中保留图片：599 张。
- 类别：Bottle、Computer keyboard、Person。
- 来源清单：`manifests/openimages_v2_source_manifest.csv`。
- 每行保留 Open Images ImageID、下载地址、原始页面、作者和图片许可 URL。
- 官方页面：https://storage.googleapis.com/openimages/web/index.html

## 2. COCO 2017 Train

- 用途：V3 横放瓶子强化、键盘补充和人物困难负样本。
- 新增图片：401 张。
- 选择组成：280 横放瓶子 + 80 键盘 + 41 人物负样本。
- COCO 类别 ID：bottle=44、keyboard=76、person=1。
- 横放瓶子不是 COCO 的独立类别，而是根据其 bottle 边界框宽高比自动筛选。
- 来源清单：`manifests/coco_v3_source_manifest.csv`。
- 每行保留 COCO ImageID、官方图片 URL、原 Flickr URL、license ID/名称/URL。
- 官方页面：https://cocodataset.org/
- 官方 API：https://github.com/cocodataset/cocoapi

部分横放瓶子筛选结果（绿色框为COCO bottle标注）可查看
[`docs/images/horizontal_samples_contact_sheet.jpg`](images/horizontal_samples_contact_sheet.jpg)。

## 数据集组成

```text
V3 总图片：1000
├─ Open Images V7（从 V2 保留）：599
└─ COCO 2017 train（新增）：401
   ├─ 横放 bottle：280
   ├─ keyboard：80
   └─ person-only 困难负样本：41

训练：800
验证：200
```

完整统计见 `manifests/dataset_v3_stats.json`。

## 重建 V3 数据集

首先需要已经存在的 `dataset_v2/`。然后下载 COCO 2017 官方实例标注，只解压训练/验证实例
JSON（不需要下载18GB完整COCO训练集；构建脚本只会下载选中的401张图片）：

```powershell
New-Item -ItemType Directory -Force `
  .\dataset_v3\external\coco\annotations | Out-Null

curl.exe -L -o .\dataset_v3\external\coco\annotations_trainval2017.zip `
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

tar -xf .\dataset_v3\external\coco\annotations_trainval2017.zip `
  -C .\dataset_v3\external\coco `
  annotations/instances_train2017.json annotations/instances_val2017.json

python .\tools\build_v3_coco_horizontal.py
```

脚本使用固定随机种子 `20260901`，根据COCO标注筛选并分层划分数据；重复运行可复用已经下载
的有效图片。最终应得到 `dataset_v3/stats.json` 中记录的1000张图片。

## 为什么仓库不直接上传训练图片

训练图片数量和体积较大，而且 Open Images/COCO 图片的许可可能逐图不同。仓库只提交模型、
代码、统计和逐图来源清单；需要重新分发图片时，应逐项核对清单中的许可 URL 和署名要求。
COCO 的标注与图片使用条件也应以其官方 Terms of Use 和每张图片记录的许可为准。
