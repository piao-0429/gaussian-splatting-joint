# Prepare a DexMirror dataset

[← README](../README.md#build-a-dataset) · [中文指南](dataset_preparation_zh.md)

This guide follows the COLMAP workflow from captured images to training inputs. Complete the [software setup](../README.md#setup) first; the commands switch between the `dexmirror`, `depth-anything`, and `sam2` Conda environments.

Use a scene containing the objects you want to interact with. Capture overlapping views around the scene and each object, including low and high viewpoints. Keep the object poses fixed within a reconstruction. Capture background views with the target objects removed, as well as object-containing views, while keeping the background geometry consistent.

The workflow is: **capture → COLMAP → organize views → depth → object masks → object point clouds**. The [Chinese dataset guide](dataset_preparation_zh.md) includes additional mask checks and an alternative VGGT-X reconstruction route.

Set absolute paths in your working terminal:

```bash
export REPO=/absolute/path/to/gaussian-splatting-joint
export DEPTH_ROOT=/absolute/path/to/Depth-Anything-V2
export SAM2_ROOT=/absolute/path/to/sam2
export DATASET=/absolute/path/to/datasets/my_scene
export MODEL_DIR="$REPO/output/my_scene"
mkdir -p "$DATASET/input"
```

## 1. Capture images and reconstruct cameras

Put all background and object-containing photographs in `input/`. Use unique filenames and unique filename stems throughout the dataset. For a video, one possible starting point is:

```bash
ffmpeg -i /absolute/path/to/capture.mp4 -vf "fps=2" \
  -q:v 2 "$DATASET/input/%06d.jpg"
```

Adjust the sampling rate to preserve overlap. For multiple videos, allocate distinct filename ranges before reconstruction and keep a record of capture order.

Run SfM and undistortion:

```bash
conda activate dexmirror
cd "$REPO"
QT_QPA_PLATFORM=offscreen python convert.py -s "$DATASET" --no_gpu
```

Omit `--no_gpu` with a CUDA-enabled COLMAP build. The script outputs `images/` and `sparse/0/`. Check which images were successfully registered. Use the **undistorted images** for all subsequent masks and depth predictions; the trainer supports `PINHOLE` and `SIMPLE_PINHOLE` cameras.

## 2. Organize the reconstruction and view groups

The trainer reads a binary reconstruction from `aligned_sparse/0/`. If no additional geometric alignment is needed, copy the three files from the same reconstruction into a new directory:

```bash
mkdir -p "$DATASET/aligned_sparse/0"
cp -n "$DATASET/sparse/0/cameras.bin" "$DATASET/aligned_sparse/0/"
cp -n "$DATASET/sparse/0/images.bin" "$DATASET/aligned_sparse/0/"
cp -n "$DATASET/sparse/0/points3D.bin" "$DATASET/aligned_sparse/0/"
```

Copying files only adapts the directory layout. If you have aligned the reconstruction to another coordinate system, use the consistently transformed cameras and points instead. Keep any existing `points3D.ply` consistent with those files.

Keep background views in `images/` and move object-containing views to `images_ft/`. Create `object_frames.txt` in the dataset root, with one **registered, original filename** per line, then run:

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
from utils.read_write_model import read_images_binary

root = Path(os.environ["DATASET"])
names = [line.strip() for line in (root / "object_frames.txt").read_text().splitlines() if line.strip()]
registered = {im.name for im in read_images_binary(root / "aligned_sparse/0/images.bin").values()}
assert names and len(names) == len(set(names)), "Empty or duplicate frame list"
assert set(names) < registered, "Use registered frames and retain background views"
for name in names:
    assert Path(name).name == name, "Use a flat image directory"
    assert (root / "images" / name).is_file(), name
    assert not (root / "images_ft" / name).exists(), name
(root / "images_ft").mkdir(exist_ok=True)
for name in names:
    shutil.move(str(root / "images" / name), str(root / "images_ft" / name))
PY
```

Do not duplicate a view across the two directories. The current loader checks `images/` first and only loads object masks for views resolved through `images_ft/`. Both groups must share the same reconstruction coordinates.

## 3. Predict and calibrate depth

Generate depth for both image groups:

```bash
conda activate depth-anything
cd "$DEPTH_ROOT"
python run.py --encoder vitl --pred-only --grayscale \
  --img-path "$DATASET/images" --outdir "$DATASET/depth"
python run.py --encoder vitl --pred-only --grayscale \
  --img-path "$DATASET/images_ft" --outdir "$DATASET/depth"
```

Keep these input directories free of auxiliary images and subdirectories. The grayscale PNGs contain relative predictions. Fit their scale and offset to the COLMAP inverse depths using the actual PNGs that will be used for training:

```bash
conda activate dexmirror
cd "$REPO"
python utils/make_depth_scale.py \
  --base_dir "$DATASET" --depths_dir "$DATASET/depth" --model_type bin
```

This writes `aligned_sparse/0/depth_params.json`. Check image coverage and invalid or zero scales. Recalibrate after changing the reconstruction or depth encoding. To train without depth supervision, skip this step and omit `-d` from the training command.

## 4. Segment each object with SAM2

Prepare a separate sequence of numeric JPEG filenames, preserving the original dataset:

```bash
conda activate dexmirror
cd "$REPO"
python scripts/sam2_frame_mapping.py prepare \
  --images-dir "$DATASET/images_ft" \
  --frames-dir "$DATASET/sam2_work/frames" \
  --manifest "$DATASET/sam2_work/frame_map.json"
```

The helper uses natural filename order. Add `--frame-list /absolute/path/to/ordered_frames.txt` to specify capture order explicitly. JPEGs are copied without re-encoding; other supported formats are converted to JPEG at the same dimensions.

Copy [sam2_prompts.example.json](sam2_prompts.example.json) to `$DATASET/prompts.json` and **edit it for your images**. Each entry names a numeric frame and an object ID, with either point coordinates plus labels or a bounding box. Coordinates are pixels in the full-size frame; point labels are `1` for foreground and `0` for background. Keep object IDs consistent across keyframes. For example:

```json
[
  {"frame": "00000.jpg", "object_id": 1, "points": [[200, 150], [50, 50]], "labels": [1, 0]},
  {"frame": "00000.jpg", "object_id": 2, "box": [300, 100, 500, 400]}
]
```

Propagate the prompts and export one mask per object and frame:

```bash
conda activate sam2
cd "$SAM2_ROOT"
python "$REPO/scripts/generate_sam2_masks.py" \
  --frames-dir "$DATASET/sam2_work/frames" \
  --prompts "$DATASET/prompts.json" \
  --checkpoint "$SAM2_ROOT/checkpoints/sam2.1_hiera_large.pt" \
  --config configs/sam2.1/sam2.1_hiera_l.yaml \
  --output-dir "$DATASET/sam2_work/masks" \
  --offload-video-to-cpu
```

The exporter propagates forward and, when needed, backward from the first annotated frame. Review the masks, especially around viewpoint jumps, thin structures, and occlusions. Add corrective keyframe prompts and rerun into a new output directory when necessary. All-black predictions need inspection; do not treat failed tracking as a valid absence label.

Restore the original image stems while retaining the object subdirectories:

```bash
conda activate dexmirror
cd "$REPO"
python scripts/sam2_frame_mapping.py restore \
  --masks-dir "$DATASET/sam2_work/masks" \
  --manifest "$DATASET/sam2_work/frame_map.json" \
  --output-dir "$DATASET/masks_ft_obj"
```

For example, `obj01/00000.png` becomes `obj01/IMG_0444.png`. The helper requires a new output directory and rejects ambiguous or missing mappings. A mask must be a binary PNG with foreground `255`, background `0`, and the same geometry as its original image. The trainer assigns object indices by sorted mask-directory name.

## 5. Extract object initialization points

The final dataset layout is:

```text
my_scene/
├── images/                       # Background views
├── images_ft/                    # Object-containing views
├── masks_ft_obj/
│   ├── obj01/*.png
│   └── obj02/*.png
├── depth/*.png                   # Optional depth supervision
└── aligned_sparse/0/
    ├── cameras.bin
    ├── images.bin
    ├── points3D.bin
    ├── points3D.ply              # Generated on first load if missing
    └── depth_params.json         # Required when using -d
```

Use the current mask-based extraction entry point:

```bash
conda activate dexmirror
cd "$REPO"
python miscell_steps/mask_prune_gaussians.py \
  -s "$DATASET" -m "$DATASET" \
  --ft_masks "$DATASET/masks_ft_obj" \
  --mask_prune_min_prop 0.6 --mask_prune_expand 2 \
  --output_path "$DATASET/mask_pruned"
```

This writes per-object point clouds under `mask_pruned/obj_*/object/` and their union to **`mask_pruned/point3D_objects.ply`**. The union can initialize a multi-object run through the single `--obj_ply_path` argument. Overlapping masks do not duplicate points in this union. The `final/` directory contains the remaining scene points.

Inspect each object's point cloud before training. An object with no retained points produces no object PLY. The ratio threshold is computed against **all available mask views for that object**, rather than each point's unoccluded views; occlusion and missing annotations can make a high threshold too strict. `--mask_prune_expand` dilates mask boundaries in pixels at the loaded camera resolution. The values above are starting settings, not universal thresholds.
