#!/usr/bin/env python3
"""
A minimal utility to prune a Gaussian point cloud with per-view masks using
`prune_gaussians_with_object_masks`.

- Loads cameras (and their masks) via the same pipeline as training to ensure
  mask alignment with images.
- Optionally loads an existing Gaussian PLY (trained model) to prune; otherwise
  initializes from the COLMAP sparse points.
- Saves both a COLMAP-compatible PLY and the Gaussian-format PLY after pruning.
"""

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import math
import numpy as np
import torch
from plyfile import PlyData, PlyElement

# Ensure project root is on PYTHONPATH
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from arguments import ModelParams, OptimizationParams  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly  # noqa: E402
from utils.camera_utils import cameraList_from_camInfos  # noqa: E402
from utils.sh_utils import SH2RGB  # noqa: E402
from train import prune_gaussians_with_object_masks, _get_gaussian_kernel  # noqa: E402

# Local copy to compute prune mask without mutating gaussians (for split saves)
def compute_prune_mask(gaussians, cameras, mask_prune_min_prop=0.5, mask_threshold=0.5, mask_blur_sigma=0.0, mask_index=0):
    if gaussians.get_xyz.numel() == 0:
        return torch.zeros_like(gaussians.get_xyz[:, 0], dtype=torch.bool), 0

    mask_cameras = []
    for cam in cameras:
        mask = getattr(cam, "object_masks", None)
        if isinstance(mask, list):
            if mask_index < len(mask) and mask[mask_index] is not None:
                mask_cameras.append(cam)
        elif getattr(cam, "object_mask", None) is not None:
            mask_cameras.append(cam)
    if len(mask_cameras) == 0:
        return torch.zeros_like(gaussians.get_xyz[:, 0], dtype=torch.bool), 0

    device = gaussians.get_xyz.device
    positions = gaussians.get_xyz.detach()
    homo_positions = torch.cat(
        [positions, torch.ones((positions.shape[0], 1), device=device, dtype=positions.dtype)],
        dim=1,
    )

    seen_counts = torch.zeros(positions.shape[0], dtype=torch.int32, device=device)
    inside_counts = torch.zeros_like(seen_counts)

    for camera in mask_cameras:
        mask_tensor = None
        masks = getattr(camera, "object_masks", None)
        if isinstance(masks, list) and mask_index < len(masks):
            mask_tensor = masks[mask_index]
        if mask_tensor is None:
            mask_tensor = getattr(camera, "object_mask", None)
        if mask_tensor is None:
            continue

        mask_tensor = mask_tensor.squeeze(0)
        if mask_tensor.device != device:
            mask_tensor = mask_tensor.to(device=device)
        mask_tensor = mask_tensor.to(dtype=torch.float32)

        if mask_blur_sigma > 0:
            kernel = _get_gaussian_kernel(mask_blur_sigma, mask_tensor.device)
            pad = kernel.shape[-1] // 2
            mask_tensor = torch.nn.functional.conv2d(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                kernel,
                padding=pad,
            ).squeeze(0).squeeze(0)
            mask_tensor = mask_tensor.clamp(0.0, 1.0)

        full_proj = camera.full_proj_transform
        if full_proj.device != device:
            full_proj = full_proj.to(device)

        clip_coords = torch.matmul(homo_positions, full_proj)
        clip_w = clip_coords[:, 3]
        positive_w = clip_w > 0
        if positive_w.sum() == 0:
            continue

        clip_coords = clip_coords[positive_w]
        indices = torch.nonzero(positive_w, as_tuple=False).squeeze(1)

        ndc = clip_coords[:, :3] / clip_coords[:, 3:4]
        inside_frustum = (
            (ndc[:, 0] >= -1.0)
            & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0)
            & (ndc[:, 1] <= 1.0)
        )

        if inside_frustum.sum() == 0:
            continue

        ndc = ndc[inside_frustum]
        indices = indices[inside_frustum]

        width = int(camera.image_width)
        height = int(camera.image_height)

        screen_x = ((ndc[:, 0] * 0.5 + 0.5) * (width - 1)).round().long()
        screen_y = ((ndc[:, 1] * 0.5 + 0.5) * (height - 1)).round().long()

        screen_x = torch.clamp(screen_x, 0, width - 1)
        screen_y = torch.clamp(screen_y, 0, height - 1)

        mask_values = mask_tensor[screen_y, screen_x] > mask_threshold

        seen_counts[indices] += 1
        inside_counts[indices] += mask_values.to(torch.int32)

    background_counts = (seen_counts - inside_counts).clamp_min(0)

    n_mask_cams = len(mask_cameras)
    prop = float(mask_prune_min_prop)
    if prop <= 0.0:
        computed_threshold = 1
    else:
        computed_threshold = int(math.ceil(prop * n_mask_cams))
        computed_threshold = max(computed_threshold, 1)

    # prune_mask = background_counts >= computed_threshold
    # prune_mask = prune_mask | (seen_counts < computed_threshold)
    prune_mask = inside_counts < computed_threshold

    return prune_mask, computed_threshold


def _setup_prune_state(gaussians: GaussianModel, opt):
    """Init minimal optimizer/state for pruning-only runs (no exposure optimizer)."""
    device = gaussians.get_xyz.device
    gaussians.percent_dense = getattr(opt, "percent_dense", 0)
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device=device)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device=device)
    gaussians.tmp_radii = torch.zeros((gaussians.get_xyz.shape[0]), device=device)

    l = [
        {"params": [gaussians._xyz], "lr": opt.position_lr_init * getattr(gaussians, "spatial_lr_scale", 1.0), "name": "xyz"},
        {"params": [gaussians._features_dc], "lr": opt.feature_lr, "name": "f_dc"},
        {"params": [gaussians._features_rest], "lr": opt.feature_lr / 20.0, "name": "f_rest"},
        {"params": [gaussians._opacity], "lr": opt.opacity_lr, "name": "opacity"},
        {"params": [gaussians._scaling], "lr": opt.scaling_lr, "name": "scaling"},
        {"params": [gaussians._rotation], "lr": opt.rotation_lr, "name": "rotation"},
    ]
    gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


def parse_args():
    parser = ArgumentParser(description="Prune a Gaussian PLY using per-view masks")
    lp = ModelParams(parser)
    opt_group = OptimizationParams(parser)

    parser.add_argument(
        "--gaussian_ply",
        type=str,
        default=None,
        help="Path to an existing Gaussian PLY to prune. If omitted, initializes from COLMAP sparse points.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save pruned outputs (default: <model_path>/mask_pruned)",
    )
    parser.add_argument(
        "--save_removed",
        action="store_true",
        help="Also save the removed (outside-mask) points to output_path/removed as PLYs",
    )
    parser.add_argument(
        "--prune_all_masks",
        action="store_true",
        help="(default on) Prune once per available mask slot and save to separate subfolders obj_0, obj_1, ...",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=None,
        help="Optional iteration to load (not used for COLMAP init, kept for parity with training args)",
    )

    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = opt_group.extract(args)

    # Allow running without explicitly passing --model_path; fall back to source_path, or to output_path parent.
    src = getattr(dataset, "source_path", None)
    mdl = getattr(dataset, "model_path", None)

    # Resolve output_path early if provided, to reuse its parent as a last resort model_path.
    out_arg = args.output_path
    if out_arg:
        out_arg = os.path.abspath(out_arg)

    if src is None and mdl is not None:
        src = mdl
    if mdl is None and src is not None:
        mdl = src
    if mdl is None and src is None and out_arg is not None:
        mdl = os.path.dirname(out_arg)
    if src is None and mdl is not None:
        src = mdl

    if src is None or mdl is None:
        parser.error("Provide --source_path or --model_path (can fall back to output_path parent if only output_path is given)")

    dataset.source_path = os.path.abspath(src)
    dataset.model_path = os.path.abspath(mdl)

    if not os.path.isdir(dataset.source_path):
        parser.error(f"source_path missing or not a directory: {dataset.source_path}")

    os.makedirs(dataset.model_path, exist_ok=True)
    args.model_path = dataset.model_path

    if args.gaussian_ply:
        args.gaussian_ply = os.path.abspath(args.gaussian_ply)
        if not os.path.isfile(args.gaussian_ply):
            parser.error(f"gaussian_ply not found: {args.gaussian_ply}")

    if out_arg:
        args.output_path = out_arg
    else:
        args.output_path = os.path.join(dataset.model_path, "mask_pruned")
    os.makedirs(args.output_path, exist_ok=True)

    # Keep optimizer type consistent with possible checkpoints
    opt.optimizer_type = getattr(opt, "optimizer_type", "default")

    # Expose mask-related options on args for clarity
    args.mask_prune_threshold = opt.mask_prune_threshold
    args.mask_prune_min_prop = opt.mask_prune_min_prop
    args.mask_prune_blur_sigma = opt.mask_prune_blur_sigma

    # Default to prune all masks when not specified
    if not hasattr(args, "prune_all_masks") or args.prune_all_masks is False:
        args.prune_all_masks = True

    return args, dataset, opt


def build_cameras(scene_info, dataset):
    cam_infos = []
    if scene_info.train_cameras:
        cam_infos.extend(scene_info.train_cameras)
    if scene_info.test_cameras:
        cam_infos.extend(scene_info.test_cameras)
    if scene_info.finetune_cameras:
        cam_infos.extend(scene_info.finetune_cameras)

    cameras = []
    if scene_info.train_cameras:
        cameras += cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, False)
    if scene_info.test_cameras:
        cameras += cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, True)
    if scene_info.finetune_cameras:
        cameras += cameraList_from_camInfos(scene_info.finetune_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, False)

    return cam_infos, cameras


def _max_mask_slots(cameras):
    max_len = 0
    for cam in cameras:
        masks = getattr(cam, "object_masks", None)
        if isinstance(masks, list):
            max_len = max(max_len, len(masks))
        elif getattr(cam, "object_mask", None) is not None:
            max_len = max(max_len, 1)
    return max_len


def load_gaussians(args, dataset, opt, scene_info, cam_infos):
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    if args.gaussian_ply:
        gaussians.load_ply(args.gaussian_ply)
        _setup_prune_state(gaussians, opt)
    else:
        gaussians.create_from_pcd(scene_info.point_cloud, cam_infos, scene_info.nerf_normalization["radius"])
        gaussians.training_setup(opt)
    return gaussians


def prune_once(args, cameras, gaussians, output_dir, total_points, mask_label=None, mask_index=0):
    print("\n[Prune] Mask-based pruning of Gaussian PLY")
    prune_mask, used_threshold = compute_prune_mask(
        gaussians,
        cameras,
        mask_prune_min_prop=args.mask_prune_min_prop,
        mask_threshold=args.mask_prune_threshold,
        mask_blur_sigma=args.mask_prune_blur_sigma,
        mask_index=mask_index,
    )

    removed = int(prune_mask.sum().item())
    kept_mask = (~prune_mask).cpu().numpy().astype(bool)
    removed_mask = prune_mask.cpu().numpy().astype(bool)

    # Save removed (outer/cut) if requested
    if getattr(args, "save_removed", False) and removed > 0:
        cut_dir = os.path.join(output_dir, "removed")
        prefix = "point3D_removed" if mask_label is None else f"point3D_removed_{mask_label}"
        cut_colmap, cut_gauss = save_gaussian_subset(gaussians, removed_mask, cut_dir, prefix)
        print(f"[Prune] Saved removed subset: {cut_colmap}, {cut_gauss}")

    # Apply pruning to gaussians in-place for kept cloud
    if removed > 0:
        if not hasattr(gaussians, "tmp_radii") or gaussians.tmp_radii is None or gaussians.tmp_radii.shape[0] != gaussians.get_xyz.shape[0]:
            gaussians.tmp_radii = gaussians.get_xyz.new_zeros((gaussians.get_xyz.shape[0],))
        gaussians.prune_points(prune_mask)
        gaussians.tmp_radii = None

    kept = gaussians.get_xyz.shape[0]

    print(f"[Prune] Removed {removed} points (kept {kept}/{total_points}, threshold={used_threshold} mask views)")

    colmap_ply_path, gaussian_ply_path = save_outputs(gaussians, output_dir)

    print(f"Total: {total_points}, removed: {removed}, kept: {kept}, threshold_used: {used_threshold}")
    print(f"COLMAP PLY: {colmap_ply_path}")
    print(f"Gaussian PLY: {gaussian_ply_path}")


def prune_all_masks(args, dataset, opt, scene_info, cam_infos, cameras):
    max_masks = _max_mask_slots(cameras)
    if max_masks == 0:
        print("[WARN] No masks found on any camera; skipping.")
        return

    print(f"[INFO] Detected up to {max_masks} mask slots; pruning sequentially on a single Gaussian cloud.")

    # Load original gaussians once and compute per-mask removed subsets against that original
    gaussians_orig = load_gaussians(args, dataset, opt, scene_info, cam_infos)
    total_start = gaussians_orig.get_xyz.shape[0]

    # keep union of extracted object indices across masks (numpy bool array)
    import numpy as _np
    union_objects = _np.zeros((total_start,), dtype=bool)

    for idx in range(max_masks):
        sub_out = os.path.join(args.output_path, f"obj_{idx}")
        os.makedirs(sub_out, exist_ok=True)

        print(f"\n[Prune] Slot {idx}: computing removal mask against original cloud ({total_start} points)")

        prune_mask, used_threshold = compute_prune_mask(
            gaussians_orig,
            cameras,
            mask_prune_min_prop=args.mask_prune_min_prop,
            mask_threshold=args.mask_prune_threshold,
            mask_blur_sigma=args.mask_prune_blur_sigma,
            mask_index=idx,
        )

        removed = int(prune_mask.sum().item())
        # object mask = points considered inside the object (not removed by prune_mask)
        object_mask = (~prune_mask).cpu().numpy().astype(bool)
        n_object = int(object_mask.sum())

        print(f"[Prune] Slot {idx}: object points detected {n_object}, background marked {removed} (threshold={used_threshold})")

        # Save extracted object points for this slot if requested
        if n_object > 0:
            obj_dir = os.path.join(sub_out, "object")
            prefix = f"point3D_object_obj{idx}"
            obj_colmap, obj_gauss = save_gaussian_subset(gaussians_orig, object_mask, obj_dir, prefix)
            print(f"[Prune] Saved object subset for slot {idx}: {obj_colmap}, {obj_gauss}")

        # accumulate union of object points
        if n_object > 0:
            union_objects |= object_mask

    # After computing all slots, save final remaining cloud = original - union_removed
    final_dir = os.path.join(args.output_path, "final")
    os.makedirs(final_dir, exist_ok=True)

    remaining_mask = ~union_objects
    remaining_count = int(remaining_mask.sum())
    print(f"\n[Prune] Final remaining points after extracting all objects: {remaining_count} / {total_start}")

    if remaining_count > 0:
        rem_prefix = "point3D_remaining"
        rem_colmap, rem_gauss = save_gaussian_subset(gaussians_orig, remaining_mask, final_dir, rem_prefix)
        print(f"Final remaining saved: {rem_colmap}, {rem_gauss}")
    else:
        print("[Prune] Warning: no points remain after extracting all object slots.")


def report_masks(cam_infos, cameras, dataset):
    mask_paths = []
    for c in cam_infos:
        paths = getattr(c, "mask_paths", []) or []
        mask_paths.extend([p for p in paths if p])

    cams_with_masks = [c for c in cameras if any(m is not None for m in (getattr(c, "object_masks", []) or []))]

    print(f"Detected cameras: {len(cameras)}, with masks: {len(cams_with_masks)}")
    if dataset.ft_masks and not mask_paths:
        print(f"[WARN] --ft_masks={dataset.ft_masks} provided but no mask files were found.")
    elif len(cams_with_masks) == 0:
        print("[WARN] No masks loaded; pruning will be skipped.")
    elif len(cams_with_masks) < len(cameras):
        missing = [c.image_name for c in cam_infos if not any((getattr(c, "mask_paths", []) or []))]
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        print(f"[INFO] Some views lack masks, e.g., {preview}{more}")


def save_outputs(gaussians, output_dir):
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    dc = gaussians.get_features_dc.detach().squeeze(1)
    rgb = SH2RGB(dc).clamp(0.0, 1.0)
    rgb = (rgb * 255.0).round().to(torch.uint8).cpu().numpy().astype(np.uint8)

    colmap_ply_path = os.path.join(output_dir, "point3D_mask_pruned.ply")
    storePly(colmap_ply_path, xyz, rgb)

    gaussian_ply_path = os.path.join(output_dir, "point3D_mask_pruned_gaussian.ply")
    gaussians.save_ply(gaussian_ply_path)

    return colmap_ply_path, gaussian_ply_path


def save_gaussian_subset(gaussians, mask_bool, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    keep_xyz = gaussians.get_xyz.detach().cpu().numpy()[mask_bool]
    # RGB for lightweight COLMAP PLY
    dc = gaussians.get_features_dc.detach().squeeze(1)
    rgb = SH2RGB(dc).clamp(0.0, 1.0)
    rgb = (rgb * 255.0).round().to(torch.uint8).cpu().numpy().astype(np.uint8)[mask_bool]

    colmap_ply_path = os.path.join(output_dir, f"{prefix}.ply")
    storePly(colmap_ply_path, keep_xyz, rgb)


    # Gaussian-format subset: match GaussianModel.save_ply ordering and sources
    normals = np.zeros_like(keep_xyz)
    # Use the underlying parameters (not post-activation getters) to match save_ply
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[mask_bool]
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()[mask_bool]
    opacities = gaussians._opacity.detach().cpu().numpy()[mask_bool]
    scale = gaussians._scaling.detach().cpu().numpy()[mask_bool]
    rotation = gaussians._rotation.detach().cpu().numpy()[mask_bool]

    dtype_full = [(attribute, 'f4') for attribute in gaussians.construct_list_of_attributes()]
    elements = np.empty(keep_xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((keep_xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

    # Debug sanity-checks: ensure shapes align
    try:
        assert attributes.shape[1] == len(dtype_full), f"Attribute columns {attributes.shape[1]} != dtype fields {len(dtype_full)}"
    except AssertionError as e:
        print(f"[ERROR] Gaussian subset write shape mismatch: {e}")
        print(f" attributes.shape: {attributes.shape}")
        print(f" dtype_full len: {len(dtype_full)}")
        # fall back to saving only lightweight COLMAP PLY
        gaussian_ply_path = os.path.join(output_dir, f"{prefix}_gaussian_skipped.ply")
        print(f"[WARN] Skipping gaussian PLY write, wrote: {gaussian_ply_path}")
        return colmap_ply_path, gaussian_ply_path

    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    gaussian_ply_path = os.path.join(output_dir, f"{prefix}_gaussian.ply")
    PlyData([el]).write(gaussian_ply_path)

    return colmap_ply_path, gaussian_ply_path


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

    cam_infos, cameras = build_cameras(scene_info, dataset)
    if not cameras:
        print("No cameras found; check sparse/0 and images/ under source_path.")
        return

    report_masks(cam_infos, cameras, dataset)

    # Always prune all masks (default enabled); single-mask mode removed for simplicity
    prune_all_masks(args, dataset, opt, scene_info, cam_infos, cameras)


if __name__ == "__main__":
    main()
