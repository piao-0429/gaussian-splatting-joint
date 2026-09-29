# 数据集制作指南

本文说明如何从场景照片制作 `gaussian-splatting-joint` 所需的数据集，包括相机重建、背景与物体视角划分、SAM2 掩码、深度图、点云检查，以及训练入口。Franka Coke、Real2Sim 和 franka_object 均可使用这套流程。

依据：原始实验记录 `Full Reproduce.docx`，并按当前 `cm_triple_dev` 工作区程序更新（2026-09-29）。外部脚本按本机 VGGT-X、SAM2、Depth-Anything-V2 的实际代码核对。本文中的命令已更新，旧记录里的参数不能直接混用。

## 1. 准备程序和路径

| 程序 | 本文使用的入口 | 用途 |
| --- | --- | --- |
| VGGT-X | `demo_colmap_new.py` | 估计相机与点云，导出 COLMAP 格式 |
| COLMAP，可选 | 本仓库的 `convert.py` | 使用传统 SfM 重建并去畸变 |
| SAM2 | `notebooks/video_predictor_example.ipynb` | 交互标注、传播和导出物体掩码 |
| 本仓库 | `scripts/sam2_frame_mapping.py` | 建立数字帧、保存映射、恢复掩码原图名 |
| Depth-Anything-V2 | `run.py` | 预测相对深度图 |
| 本仓库 | `utils/make_depth_scale.py` | 将深度预测校准到重建的逆深度尺度 |
| 本仓库 | `miscell_steps/mask_prune_gaussians.py` | 按物体掩码提取点云 |
| 本仓库 | `train.py` | 背景与物体联合训练 |

外部程序需要单独准备，不包含在本仓库里。分别按对应仓库的说明安装依赖和模型权重。本仓库环境安装见 [README](../README.md#setup)。下面用 `dexmirror` 表示当前 README 配置的训练环境，用 `depth-anything` 表示深度环境；可选 VGGT-X 路线使用 `vggt-long`。读者应替换为自己的环境名，SAM2 Notebook 需要使用安装了 SAM2 的内核。

先在终端设置路径，后续代码块在同一终端执行：

```bash
export PROJECTS=/path/to/projects
export REPO="$PROJECTS/gaussian-splatting-joint"
export VGGT_ROOT="$PROJECTS/VGGT-X"
export SAM2_ROOT="$PROJECTS/sam2"
export DEPTH_ROOT="$PROJECTS/Depth-Anything-V2"

export RAW_SCENE=/path/to/data/raw/scene
export DATASET=/path/to/data/raw_vggt/scene
export MODEL_DIR="$REPO/output/scene_joint"
```

`RAW_SCENE` 保存原始输入，`DATASET` 保存制作结果，`MODEL_DIR` 保存训练结果。每个新场景使用单独目录。VGGT-X 的输出位置由 `--post_fix` 决定，见下一节。

## 2. 整理照片并重建相机

### 2.1 输入照片

先将需要共同重建的背景视角和物体视角放到一个目录：

```text
raw/scene/
└── images/
    ├── 00000.jpg
    ├── 00001.jpg
    └── ...
```

照片文件名必须唯一，去掉扩展名后也不能重名，例如不能同时使用 `00001.jpg` 和 `00001.png`。后续深度和掩码都按这个名字匹配。重建后不要单独重命名照片，否则相机记录与图像会失配。

背景视角用于训练背景模型，物体视角用于训练独立物体和组合场景。拍摄时应准备能表达背景的视角，以及物体在目标位置上的多视角照片。用于同一个静态物体模型的照片应保持物体位置一致。用于 SAM2 传播的照片尽量按拍摄顺序排列，视角跳变处后续需要额外标注。

### 2.2 路线 A：VGGT-X

在 VGGT-X 环境执行：

```bash
conda activate vggt-long
cd "$VGGT_ROOT"
python demo_colmap_new.py \
  --scene_dir "$RAW_SCENE" \
  --post_fix _vggt \
  --use_ba
```

当前脚本将后缀加在**输入场景的父目录**上：

```text
输入：/path/to/data/raw/scene
输出：/path/to/data/raw_vggt/scene
```

因此，前面的 `DATASET` 必须与实际输出一致。主要结果是 `sparse/0/cameras.bin`、`images.bin`、`points3D.bin`，以及用于预览的 `sparse/points.ply`。以日志中的 `Saving reconstruction to ...` 为准。

如需指定查询帧，在运行前创建 `prefer_9.txt`，每行写一个完整图片文件名，例如 `00000.jpg`。文件名必须在输入中存在。选择 9 张不同视角的照片后，给上面的命令追加：

```text
--query_frame_num 9 --query_frame_list /absolute/path/to/prefer_9.txt
```

固定相机内参的数据可追加 `--shared_camera`。这些选项应在同一次重建中设置。当前本地 `demo_colmap_new.py` 使用 `--use_ba`，没有旧记录中的 `--use_ga`、`--save_depth` 接口；训练所需深度按第 5 节单独生成。

检查输出中的注册图像及 `vggt_results.txt`：重建可能剔除部分帧，因此不能只按原始照片数量判断数据是否完整。输出图像可能是指向原图的符号链接，移动或打包数据集时应一并保留链接目标，或复制为实际文件。

### 2.3 路线 B：COLMAP

已有质量合适的 COLMAP 重建可以跳过重新估计。若从照片开始使用本仓库的转换脚本，另选一个新的 `DATASET` 目录，并将照片复制到它的 `input/`：

```bash
export DATASET=/path/to/data/colmap/scene
mkdir -p "$DATASET/input"
cp -a "$RAW_SCENE/images/." "$DATASET/input/"

conda activate dexmirror
cd "$REPO"
python convert.py -s "$DATASET"
```

该脚本调用系统中的 `colmap`，输出去畸变后的 `images/` 和 `sparse/0/`。后续掩码、深度都应基于这套输出图像制作。当前训练只接受 `PINHOLE` 或 `SIMPLE_PINHOLE` 相机，不能把有畸变的原图与去畸变后的相机参数混用。

### 2.4 统一训练使用的重建目录

两条路线最终都整理成：

```text
scene/
├── images/
├── sparse/0/                 # 保留重建原始输出
└── aligned_sparse/0/         # 当前训练读取的二进制重建
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

如果还没有做额外坐标变换，可把同一次重建的三个文件复制到新建目录：

```bash
mkdir -p "$DATASET/aligned_sparse/0"
cp -n "$DATASET/sparse/0/cameras.bin" "$DATASET/aligned_sparse/0/"
cp -n "$DATASET/sparse/0/images.bin" "$DATASET/aligned_sparse/0/"
cp -n "$DATASET/sparse/0/points3D.bin" "$DATASET/aligned_sparse/0/"
```

这里的复制只是适配目录，并不执行几何对齐。如果已经有对齐结果，应直接使用那一整套相机和点云，不能混入另一套重建文件。`cp -n` 会保留已有文件；已有目录必须自行确认来源一致。

如果输入只有文本模型，先在独立目录通过 `colmap model_converter --input_path ... --output_path ... --output_type BIN` 转成二进制，再整理到上面的目录。虽然读取器有部分文本回退逻辑，本文统一使用二进制格式，避免相机和点云从不同位置读入。

训练和点云提取脚本会在缺少 `aligned_sparse/0/points3D.ply` 时，从 `points3D.bin` 生成该文件。VGGT-X 的 `sparse/points.ply` 是另一份预览输出，不应直接改名覆盖它。已有 `points3D.ply` 也必须与当前相机、点云来自同一次重建。

## 3. 划分背景和物体视角

当前程序的划分由**图像位置**决定：先寻找 `images/<原文件名>`，找不到才寻找 `images_ft/<原文件名>`；只给 `images_ft` 中的视角加载物体掩码。

```text
scene/
├── images/                   # 背景训练照片
│   ├── 00000.jpg
│   └── ...
└── images_ft/                # 物体及组合场景训练照片
    ├── 00577.jpg
    └── ...
```

同一张照片不要同时存在于两个目录，否则会优先作为背景视角。也不要把全部照片移到 `images_ft`：当前训练必须有背景相机。两组照片应共用前面同一套重建坐标系。

在 `$DATASET/object_frames.txt` 中写好要划入物体组的**已注册照片**，每行一个完整文件名。以下代码先检查整个列表，再移动照片；它不修改相机记录：

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path

root = Path(os.environ["DATASET"])
names = [s.strip() for s in (root / "object_frames.txt").read_text().splitlines() if s.strip()]
assert names and len(names) == len(set(names)), "列表为空或有重复文件名"
for name in names:
    assert Path(name).name == name, "本文示例使用平铺的图片目录"
    assert (root / "images" / name).is_file(), f"找不到原图：{name}"
    assert not (root / "images_ft" / name).exists(), f"目标已存在：{name}"
(root / "images_ft").mkdir(exist_ok=True)
for name in names:
    shutil.move(str(root / "images" / name), str(root / "images_ft" / name))
print(f"已划入物体组：{len(names)} 张")
PY
```

## 4. 使用 SAM2 制作物体掩码

### 4.1 标注与传播

打开 SAM2 仓库的 `notebooks/video_predictor_example.ipynb`。本机另有 `video_predictor_example_r2s.ipynb`，但其中路径和标注点同样需要按新场景修改。Notebook 使用的模型配置为 `configs/sam2.1/sam2.1_hiera_l.yaml`，权重为 `checkpoints/sam2.1_hiera_large.pt`。

操作顺序：

1. 设置 `video_dir`，初始化 `predictor` 和 `inference_state`。
2. 给每个物体分配固定的 `obj_id`，在清楚的视角上添加正点、负点或框。
3. 运行 `propagate_in_video`，获得 `video_segments`。
4. 检查传播结果；在漏标、边界错误和视角跳变处补标，再重新传播。
5. 导出每个物体的二值掩码，并恢复原图文件名。

当前 SAM2 的图片序列加载器要求 JPEG 文件名主体可转成整数，例如 `00000.jpg`。如果照片是 `IMG_0001.JPG`，使用本仓库的 [帧名映射脚本](../scripts/sam2_frame_mapping.py) 建立独立工作目录：

```bash
conda activate dexmirror
cd "$REPO"
python scripts/sam2_frame_mapping.py prepare \
  --images-dir "$DATASET/images_ft" \
  --frames-dir "$DATASET/sam2_work/frames" \
  --manifest "$DATASET/sam2_work/frame_map.json"
```

脚本按文件名自然排序，例如 `IMG_2.JPG` 在 `IMG_10.JPG` 前面。如果文件名顺序与拍摄顺序不同，追加 `--frame-list /absolute/path/to/ordered_frames.txt`；列表每行一个原图完整文件名，脚本严格按列表选帧和排序。随后把 Notebook 中的 `video_dir` 指向 `$DATASET/sam2_work/frames` 的实际绝对路径。

原始 JPEG 按字节复制，PNG 等其他格式会在工作目录转成 JPEG，尺寸不变；脚本不旋转或裁剪图像。**原图不会重命名或覆盖。** 数字帧目录和映射文件必须使用新路径，映射文件保留在帧目录之外。

脚本自动生成的 `frame_map.json` 如下。原图已经是数字文件名时，也可以使用相同流程，保留一份明确的映射记录：

```json
{
  "00000.jpg": "IMG_0444.JPG",
  "00001.jpg": "IMG_0445.JPG"
}
```

也可以使用仓库内的 [generate_sam2_masks.py](../scripts/generate_sam2_masks.py)，通过 JSON 提供各关键帧的物体编号、点或框提示，完成双向传播和多物体 PNG 导出。命令和提示文件示例见 [英文数据集指南的 SAM2 使用步骤](dataset_preparation.md#4-segment-each-object-with-sam2)。该脚本在独立的 `sam2` 环境运行；已有 Notebook 标注可以继续使用下节的导出流程。

### 4.2 导出训练需要的格式

每个物体一个子目录，掩码使用原图主体名、单通道 PNG、背景 `0`、物体 `255`：

```text
masks_ft_obj/
├── obj01/
│   ├── 00577.png
│   └── 00578.png
└── obj02/
    ├── 00577.png
    └── 00578.png
```

不要将着色预览图或所有物体合并后的掩码用于多物体训练。程序按子目录名字排序确定物体顺序；`obj01` 对应第 0 个物体，`obj02` 对应第 1 个物体。目录名使用相同位数，避免 `obj10` 排在 `obj2` 前面。掩码根目录如果没有子目录，会被当作一个物体。

完成传播并得到 `video_segments`、`frame_names` 后，运行 Notebook 的二值掩码导出单元格。如果使用的 Notebook 没有这个单元格，可以追加：

```python
from pathlib import Path
import numpy as np
from PIL import Image

mask_root = Path(video_dir) / "masks_bw" / "separate"
for frame_idx, objects in video_segments.items():
    frame_name = frame_names[frame_idx]
    with Image.open(Path(video_dir) / frame_name) as source:
        width, height = source.size
    for obj_id, mask in objects.items():
        binary = (np.asarray(mask).squeeze() > 0).astype(np.uint8) * 255
        assert binary.shape == (height, width), f"掩码尺寸不匹配：{frame_name}"
        folder = mask_root / f"obj{int(obj_id):02d}"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary).save(folder / f"{Path(frame_name).stem}.png")
```

结果位于 `sam2_work/frames/masks_bw/separate/objXX/`。然后恢复文件名：

```bash
cd "$REPO"
python scripts/sam2_frame_mapping.py restore \
  --masks-dir "$DATASET/sam2_work/frames/masks_bw/separate" \
  --manifest "$DATASET/sam2_work/frame_map.json" \
  --output-dir "$DATASET/masks_ft_obj"
```

例如，`obj01/00000.png` 会复制为 `obj01/IMG_0444.png`。脚本保留物体子目录和掩码文件内容，不生成缺失帧的掩码。如果目标目录已存在，请指定一个新的输出目录，检查后再决定训练使用哪一版。未知帧名、重复映射或目标重名会直接报错，避免静默覆盖。

一个物体可以只在部分视角有掩码，但每个要训练的物体都应有可靠的标注。未标注、物体不可见和标注失败需要人工区分：缺少掩码文件时，该视角不参与该物体的掩码采样；全黑文件仍被视为一张可用掩码，并进入裁剪比例的分母。不要用全黑文件自动填补缺失标注。

### 4.3 检查掩码

逐帧或按连续片段查看原图上的叠加结果，重点检查：

- 物体是否在某一帧突然消失，尤其是视角变化较大的位置。
- 底座、杯口、细杆等结构是否连续保留。
- 物体编号是否在传播中交换，是否误选桌面或机械臂。
- 遮挡前后是否需要补标，掩码是否与原图尺寸、方向和文件名一致。

原实验曾出现底座缺失、扫码器整件漏标，以及相邻帧视角剧变后跟踪丢失。遇到这些问题应先修正标注，再提取点云或训练。

当前物体 RGB 损失会将预测和目标图像同时乘以掩码，因此遮挡主要减少有效监督；点云裁剪则只统计投影落入掩码的次数，没有显式判断机械臂等遮挡物。被遮挡的真实点可能因票数不足被删除。掩码膨胀只能容忍边界误差，不能恢复被遮挡的几何，也不应把机械臂区域直接当成物体标注。

## 5. 生成深度图并校准尺度

深度监督是可选项。需要使用时，背景和物体视角都要有深度图，并配套同一套相机的 `depth_params.json`。

确认权重 `Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth` 已准备好，在该仓库内运行两次，分别处理两组图像：

```bash
conda activate depth-anything
cd "$DEPTH_ROOT"
python run.py \
  --encoder vitl --pred-only --grayscale \
  --img-path "$DATASET/images" \
  --outdir "$DATASET/depth"

python run.py \
  --encoder vitl --pred-only --grayscale \
  --img-path "$DATASET/images_ft" \
  --outdir "$DATASET/depth"
```

输入目录内只保留图片，不放掩码子目录或说明文件；当前 `run.py` 会递归枚举路径。`--pred-only` 避免导出原图与深度拼接图，`--grayscale` 避免导出伪彩色图。

当前本地 `run.py` 保存的是逐图归一化的 8 位灰度预测，文件有三个相同通道。它表示相对深度预测，不能直接当作以米为单位的深度。本仓库按 `PNG 值 / 65536` 读取，再使用 `scale` 和 `offset` 校准；因此需要对**实际用于训练的这批 PNG** 执行：

```bash
conda activate dexmirror
cd "$REPO"
python utils/make_depth_scale.py \
  --base_dir "$DATASET" \
  --depths_dir "$DATASET/depth" \
  --model_type bin
```

结果写入 `aligned_sparse/0/depth_params.json`，键为照片主体名，每项包含 `scale` 和 `offset`。检查参数是否有限、是否覆盖所有注册照片，以及是否有大量零或异常尺度。稀疏点轨迹不足的视角可能无法得到可靠校准，生成 JSON 本身不代表每张深度都有效。

相机、点云、深度编码或图像集合变更后，应重新校准；旧 JSON 会被脚本按键合并，制作新版本时应使用干净的结果目录，避免混入旧条目。VGGT 导出的中间深度数组不能只改扩展名就用在这里。

如果不使用深度，跳过本节，训练时不传 `-d`。只要传入 `-d`，当前程序就会要求存在 `aligned_sparse/0/depth_params.json`，并读取每个相机对应的深度 PNG。

## 6. 检查并提取物体点云

完整场景点云可以直接用于初始化，掩码提取是可选的检查步骤。当前使用 [mask_prune_gaussians.py](../miscell_steps/mask_prune_gaussians.py)：

```bash
conda activate dexmirror
cd "$REPO"
python miscell_steps/mask_prune_gaussians.py \
  -s "$DATASET" \
  -m "$DATASET" \
  --ft_masks "$DATASET/masks_ft_obj" \
  --mask_prune_min_prop 0.6 \
  --mask_prune_expand 2 \
  --output_path "$DATASET/mask_pruned"
```

这里的 `0.6` 和 `2` 是可调整的示例值，并非所有场景的最优值。输出按掩码目录顺序编号：

```text
mask_pruned/
├── point3D_objects.ply       # 所有已提取物体点的并集，用于多物体初始化
├── point3D_objects_gaussian.ply
├── obj_0/object/
│   ├── point3D_object_obj0.ply
│   └── point3D_object_obj0_gaussian.ply
├── obj_1/object/...
├── debug_mask/               # 抽样导出的掩码检查图
└── final/
    └── point3D_remaining.ply # 提取所有物体后剩下的点
```

某个物体未保留任何点时不会生成对应 PLY。先查看日志中的匹配掩码数、阈值和保留点数，再查看点云是否保留完整结构。`final` 下保存的是余下的场景点，不能作为物体点云使用。

当前保留条件为：

```text
阈值 = max(1, ceil(mask_prune_min_prop × 该物体可用掩码视角数))
点的投影落入该物体掩码的次数 ≥ 阈值
```

分母是该物体的全部可用掩码视角数，不是每个点的实际可见视角数。`--mask_prune_expand` 是相机加载后分辨率上的掩码膨胀半径，单位为像素；默认加载会将宽度超过 1600 的图片缩小。因此，它与原始图片上的固定像素误差不一定相同。

当前 `--obj_ply_path` 只接受一个文件，多个物体会从同一个文件分别初始化。单物体场景可以将上面提取的 PLY 传给它；多物体场景可以使用完整场景点云，或提取脚本导出的 `mask_pruned/point3D_objects.ply`。这个并集按原始点索引保存，重叠掩码不会造成重复点；所有物体都为空时不会生成并集文件。不能把仅含物体 0 的 PLY 直接当作所有物体的初始化。

## 7. 训练前检查

最终目录应至少包含：

```text
scene/
├── images/                         # 背景视角，不能为空
├── images_ft/                      # 物体与组合场景视角
├── masks_ft_obj/
│   ├── obj01/*.png
│   └── obj02/*.png
├── aligned_sparse/0/
│   ├── cameras.bin
│   ├── images.bin
│   ├── points3D.bin
│   ├── points3D.ply                # 可由程序首次加载时生成
│   └── depth_params.json          # 开启深度监督时需要
└── depth/*.png                     # 开启深度监督时需要
```

逐项确认：

- 注册照片在 `images/` 或 `images_ft/` 中能找到，两个目录没有同名重复图片。
- 图像、相机、点云来自同一套重建；图像尺寸与相机记录一致。
- 每个物体目录有与 `images_ft` 照片同名的可靠掩码，且前景为白色。
- 根目录中没有被误当成物体的 `debug`、`combine` 等掩码子目录。
- 掩码没有尺寸错误、漏掉整件物体、编号交换或误选遮挡物。
- 使用深度时，所有注册照片都有深度图及校准参数。
- 物体点云和背景点云位于同一坐标系，裁剪后没有整件物体或关键结构丢失。

启动时，读取器会打印 `Finetune mask summary`，训练会打印各物体的 `available_masks` 和 `prune_threshold`。这些统计必须与预期一致。不要在掩码数为零、物体数量错误或背景相机缺失时继续长时间训练。

## 8. 接入当前训练程序

下面给出 30000 步训练示例，先进行物体预训练，再进入联合训练，保存时执行掩码裁剪：

```bash
conda activate dexmirror
cd "$REPO"
python train.py \
  -s "$DATASET" \
  -m "$MODEL_DIR" \
  -d "$DATASET/depth" \
  --ft_masks "$DATASET/masks_ft_obj" \
  --obj_ply_path "$DATASET/aligned_sparse/0/points3D.ply" \
  --object_only_until_iter 8000 \
  --iterations 30000 \
  --mask_prune_on_save \
  --mask_prune_min_prop 0.6 \
  --mask_prune_expand 2 \
  --save_iterations 7000 10000 30000 \
  --checkpoint_iterations 7000 10000 30000 \
  --disable_viewer
```

`--object_only_until_iter 8000` 表示迭代号小于 8000 时只训练物体，从第 8000 步进入联合阶段。初始化 PLY 如果尚未生成，场景加载会先从二进制点云生成它。

如需 100000 步，把 `--iterations` 改为 `100000`，并将 `100000` 加入保存和 checkpoint 列表。本文示例采用全图训练；需要独立测试视角时追加 `--eval`，并重新确认划分后的背景和掩码视角数。未使用测试划分的重建指标不代表未见视角的泛化效果。

如需控制裁剪时机，可以去掉 `--mask_prune_on_save`，改用 `--prune_iterations 30000` 等明确的迭代列表。两者都不设置时，默认不执行基于掩码的物体裁剪；训练自身的增密和清理仍会运行。

训练结果中，背景保存为 `point_cloud/iteration_30000/point_cloud.ply`，第 0 个物体为 `obj.ply`，后续物体为 `obj_1.ply`、`obj_2.ply`。一起保留配置、相机、曝光和模型元数据，方便后续渲染与复现。

## 9. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 找不到相机或点云 | `aligned_sparse/0` 是否包含同一次重建的完整二进制文件 |
| 已经做了掩码，训练却显示 0 张 | 原图是否仍在 `images`，是否应移入 `images_ft`；文件名主体是否匹配 |
| 物体数量不对 | `masks_ft_obj` 的直接子目录是否只包含各物体 |
| SAM2 不能加载 `IMG_*.JPG` | 使用数字帧工作目录，导出时映射回原文件名 |
| 深度文件存在但无法启动 | 是否生成 `depth_params.json`，是否遗漏 `images_ft` 的深度 |
| 裁剪后物体缺底座或整件消失 | 掩码是否漏标、全黑；相机投影是否对齐；比例阈值是否过严 |
| 某些视角出现毛刺或训练后期变模糊 | 对比对应原图、掩码、初始化点云和中间模型，分别排查标注、坐标对齐与训练过程；不能只凭外观确定原因 |

制作新数据集时，建议保留原图文件名、背景/物体划分列表、SAM2 帧名映射、物体编号含义、重建命令、程序版本和最终掩码。这些信息与相机、深度和点云一起构成可复现的数据集。
