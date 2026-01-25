#!/usr/bin/env python3
"""
Generate per-object filtered point clouds from a COLMAP scene using per-object masks.

- Automatically detects object IDs from subdirectories under --masks_dir (e.g., 02/, 03/).
- For each object ID, builds masks for every camera by mapping image index to mask filename.
- Saves both COLMAP-compatible PLY and Gaussian PLY for each object.
- Can load an existing point cloud PLY (e.g., obj.ply) from --obj_ply instead of the COLMAP sparse cloud.

Example:
    python miscell_steps/cut_points_from_colmap_by_obj.py \
        --model_path /path/to/model_dir \
        --masks_dir /cephfs/hp/cm_projects/full_produce_vggt_x/robot_dataset/masks_ft_obj_separate \
        --obj_ply /path/to/model_dir/obj.ply
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

# Ensure project root on PYTHONPATH
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from arguments import ModelParams, OptimizationParams  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly, CameraInfo  # noqa: E402
from utils.camera_utils import cameraList_from_camInfos  # noqa: E402
from train import prune_gaussians_with_object_masks  # noqa: E402
from utils.sh_utils import SH2RGB  # noqa: E402
from utils.graphics_utils import BasicPointCloud  # noqa: E402
from plyfile import PlyData  # noqa: E402

MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _detect_object_entries(masks_dir: str) -> List[Tuple[str, str]]:
    """Return (numeric_id, folder_name) for valid object mask folders.

    Accepts folders named like "02" or "obj02" to match the updated layout
    under masks_bw/separate/objXX.
    """

    entries: List[Tuple[str, str]] = []
    for name in sorted(os.listdir(masks_dir)):
        full = os.path.join(masks_dir, name)
        if not os.path.isdir(full):
            continue

        numeric = None
        if name.isdigit():
            numeric = name
        elif name.lower().startswith("obj") and name[3:].isdigit():
            numeric = name[3:]

        if numeric is not None:
            entries.append((numeric, name))

    return entries


def _detect_pad_and_files(obj_dir: str) -> Tuple[int, List[str]]:
    files = [f for f in sorted(os.listdir(obj_dir)) if os.path.splitext(f)[1].lower() in MASK_EXTS]
    if not files:
        raise RuntimeError(f"No mask files found in {obj_dir}")
    pad = len(os.path.splitext(files[0])[0])
    return pad, files


def _mask_path_for_image(obj_dir: str, image_name: str, pad: int) -> str:
    stem = os.path.splitext(os.path.basename(image_name))[0]
    # First try direct matches like the original image name or stem with common extensions
    candidates = [
        os.path.join(obj_dir, image_name),
        os.path.join(obj_dir, os.path.basename(image_name)),
    ]
    for ext in MASK_EXTS:
        candidates.append(os.path.join(obj_dir, f"{stem}{ext}"))
        candidates.append(os.path.join(obj_dir, f"{os.path.basename(stem)}{ext}"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Fallback: if stem is numeric, try zero-padded numeric filenames
    try:
        idx = int(stem)
    except ValueError:
        return ""
    if pad <= 0:
        mask_name = f"{idx}.png"
    else:
        mask_name = f"{idx:0{pad}d}.png"
    candidate = os.path.join(obj_dir, mask_name)
    return candidate if os.path.exists(candidate) else ""


def parse_args():
    parser = ArgumentParser(description="Cut per-object point clouds using separated masks")
    lp = ModelParams(parser)
    opt_group = OptimizationParams(parser)

    parser.add_argument("--masks_dir", required=True, help="Directory containing per-object mask subfolders (e.g., 02/, 03/)")
    parser.add_argument("--obj_ply", type=str, default=None, help="Use this point cloud (PLY) instead of COLMAP sparse cloud; default: <model_path>/obj.ply if exists")
    parser.add_argument("--output_root", type=str, default=None, help="Root directory to save outputs; default: <model_path>/cut_output_objects")
    parser.add_argument("--load_iteration", type=int, default=None, help="Load specified iteration of model (unused for pure COLMAP init)")

    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = opt_group.extract(args)

    if not getattr(dataset, "model_path", None):
        parser.error("--model_path is required (used for defaults and outputs)")

    if not getattr(dataset, "source_path", None):
        dataset.source_path = dataset.model_path

    if not os.path.isdir(dataset.source_path):
        parser.error(f"source_path missing or not a directory: {dataset.source_path}")

    dataset.model_path = os.path.abspath(dataset.model_path)
    os.makedirs(dataset.model_path, exist_ok=True)
    args.model_path = dataset.model_path

    args.masks_dir = os.path.abspath(args.masks_dir)
    if not os.path.isdir(args.masks_dir):
        parser.error(f"masks_dir missing or not a directory: {args.masks_dir}")

    if args.obj_ply is None:
        candidate = os.path.join(dataset.model_path, "obj.ply")
        args.obj_ply = candidate if os.path.isfile(candidate) else None
    else:
        args.obj_ply = os.path.abspath(args.obj_ply)
        if not os.path.isfile(args.obj_ply):
            parser.error(f"obj_ply not found: {args.obj_ply}")

    args.output_root = args.output_root or os.path.join(dataset.model_path, "cut_output_objects")
    args.output_root = os.path.abspath(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)

    # keep optimizer type consistent
    opt.optimizer_type = getattr(opt, "optimizer_type", "default")

    return args, dataset, opt


def build_cameras_for_object(obj_dir_name: str, pad: int, masks_dir: str, cam_infos: List[CameraInfo], dataset, scene_info):
    obj_dir = os.path.join(masks_dir, obj_dir_name)
    patched_cam_infos = []
    missing = 0
    for cam in cam_infos:
        mask_path = _mask_path_for_image(obj_dir, cam.image_name, pad)
        if not mask_path:
            missing += 1
        mask_paths = getattr(cam, "mask_paths", [])
        mask_paths = [mask_path] if not mask_paths else [mask_path]  # overwrite with per-object mask
        cam_patched = cam._replace(mask_paths=mask_paths)
        patched_cam_infos.append(cam_patched)

    cameras = []
    tlen = len(scene_info.train_cameras) if scene_info.train_cameras else 0
    vlen = len(scene_info.test_cameras) if scene_info.test_cameras else 0

    if tlen:
        cameras += cameraList_from_camInfos(patched_cam_infos[:tlen], 1.0, dataset, scene_info.is_nerf_synthetic, False)
    if vlen:
        start = tlen
        cameras += cameraList_from_camInfos(patched_cam_infos[start:start+vlen], 1.0, dataset, scene_info.is_nerf_synthetic, True)

    remaining = patched_cam_infos[tlen+vlen:]
    if remaining:
        cameras += cameraList_from_camInfos(remaining, 1.0, dataset, scene_info.is_nerf_synthetic, False)
    return cameras, missing


def _load_basic_pcd_from_ply(ply_path: str):
    plydata = PlyData.read(ply_path)
    prop_names = [p.name for p in plydata.elements[0].properties]
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    if any(name.startswith("f_dc_") for name in prop_names):
        # Gaussian-style ply: derive RGB from DC SH coefficients
        fdc0 = np.asarray(plydata.elements[0]["f_dc_0"])
        fdc1 = np.asarray(plydata.elements[0]["f_dc_1"])
        fdc2 = np.asarray(plydata.elements[0]["f_dc_2"])
        fdc = np.stack([fdc0, fdc1, fdc2], axis=1)  # (P,3)
        fdc_t = torch.from_numpy(fdc).float().unsqueeze(1)  # (P,1,3)
        rgb = SH2RGB(fdc_t).clamp(0.0, 1.0).squeeze(1).cpu().numpy()
    elif all(c in prop_names for c in ("red", "green", "blue")):
        rgb = (
            np.vstack(
                [
                    np.asarray(plydata.elements[0]["red"]),
                    np.asarray(plydata.elements[0]["green"]),
                    np.asarray(plydata.elements[0]["blue"]),
                ]
            ).T
            / 255.0
        )
    else:
        rgb = np.zeros_like(xyz)

    normals = (
        np.vstack(
            [
                np.asarray(plydata.elements[0]["nx"]),
                np.asarray(plydata.elements[0]["ny"]),
                np.asarray(plydata.elements[0]["nz"]),
            ]
        ).T
        if all(n in prop_names for n in ("nx", "ny", "nz"))
        else np.zeros_like(xyz)
    )

    return BasicPointCloud(points=xyz, colors=rgb, normals=normals)


def load_point_cloud(args, scene_info):
    # Prefer user-specified obj_ply (or model_path/obj.ply if present); otherwise fall back to COLMAP sparse cloud
    if args.obj_ply and os.path.isfile(args.obj_ply):
        try:
            pcd = _load_basic_pcd_from_ply(args.obj_ply)
            print(f"Loaded point cloud from obj_ply: {args.obj_ply}")
            return pcd
        except Exception as e:
            print(f"[WARN] Failed to load obj_ply ({args.obj_ply}): {e}; falling back to COLMAP sparse cloud")

    if scene_info.point_cloud is None:
        raise RuntimeError("No point cloud available (both obj_ply missing and COLMAP sparse cloud unavailable)")
    print("Using COLMAP sparse point cloud")
    return scene_info.point_cloud


def main():
    args, dataset, opt = parse_args()
    torch.set_grad_enabled(False)

    scene_info = sceneLoadTypeCallbacks["Colmap"](
        dataset.source_path,
        dataset.images,
        dataset.depths,
        dataset.ft_masks,
        dataset.eval,
        dataset.train_test_exp,
    )

    cam_infos = []
    if scene_info.train_cameras:
        cam_infos.extend(scene_info.train_cameras)
    if scene_info.test_cameras:
        cam_infos.extend(scene_info.test_cameras)
    if scene_info.finetune_cameras:
        cam_infos.extend(scene_info.finetune_cameras)

    if not cam_infos:
        print("No cameras found; check source_path and COLMAP outputs.")
        return

    obj_entries = _detect_object_entries(args.masks_dir)
    if not obj_entries:
        print(f"No object IDs found under {args.masks_dir}")
        return

    detected = [f"{obj_id} (dir {folder})" for obj_id, folder in obj_entries]
    print(f"Detected object IDs: {', '.join(detected)}")

    # use first object dir to detect padding
    pad, _ = _detect_pad_and_files(os.path.join(args.masks_dir, obj_entries[0][1]))
    print(f"Using pad width {pad}")

    base_pcd = load_point_cloud(args, scene_info)

    for obj_id, obj_dir_name in obj_entries:
        obj_dir = os.path.join(args.masks_dir, obj_dir_name)
        _, files = _detect_pad_and_files(obj_dir)
        print(f"Obj {obj_id}: {len(files)} masks available")

        cameras, missing = build_cameras_for_object(obj_dir_name, pad, args.masks_dir, cam_infos, dataset, scene_info)
        if missing:
            print(f"[WARN] Obj {obj_id}: {missing} cameras missing masks; those views will not prune points")

        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        gaussians.create_from_pcd(base_pcd, cam_infos, scene_info.nerf_normalization["radius"])
        gaussians.training_setup(opt)

        total_points = gaussians.get_xyz.shape[0]
        pruned, used_th = prune_gaussians_with_object_masks(
            gaussians,
            cameras,
            mask_prune_min_prop=opt.mask_prune_min_prop,
            mask_threshold=opt.mask_prune_threshold,
            mask_blur_sigma=opt.mask_prune_blur_sigma,
        )
        kept = gaussians.get_xyz.shape[0]

        out_dir = os.path.join(args.output_root, f"obj_{obj_id}")
        os.makedirs(out_dir, exist_ok=True)

        xyz = gaussians.get_xyz.detach().cpu().numpy()
        dc = gaussians.get_features_dc.detach().squeeze(1)
        rgb = SH2RGB(dc).clamp(0.0, 1.0)
        rgb = (rgb * 255.0).round().to(torch.uint8).cpu().numpy().astype(np.uint8)

        gaussian_ply_path = os.path.join(out_dir, "point3D_masked_gaussian.ply")
        gaussians.save_ply(gaussian_ply_path)

        print(f"Obj {obj_id}: total {total_points}, pruned {pruned}, kept {kept}")
        print(f"  Gaussian PLY: {gaussian_ply_path}")

    print("Done.")


if __name__ == "__main__":
    main()
