import os
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import torchvision
from datetime import datetime
import re

# Default rotation matrix to apply to loaded object point clouds (hard-coded)
DEFAULT_MOVE_R = torch.tensor([
    [0.769, 0.242, -0.591],
    [-0.625, 0.090, -0.776],
    [-0.135, 0.966, 0.221],
], dtype=torch.float32)

# DEFAULT_MOVE_R = torch.tensor([
#     [ 0.038, -0.663,  0.748],
#     [ 0.999,  0.047, -0.009],
#     [-0.028,  0.732,  0.681],
# ], dtype=torch.float32)

# Per-object default translations (one Vec3 per object). Default to two zero translations.
DEFAULT_MOVE_T = [
    torch.tensor([0.8, 0.0, 0.0], dtype=torch.float32),
    torch.tensor([0.8, 0.0, 0.0], dtype=torch.float32),
    torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
]


def merge_gaussians(gaussians):
    if not gaussians:
        raise ValueError("merge_gaussians expects a non-empty list")
    if len(gaussians) == 1:
        return gaussians[0]

    merged_xyz = []
    merged_features_dc = []
    merged_features_rest = []
    merged_scaling = []
    merged_rotation = []
    merged_opacity = []

    for g in gaussians:
        merged_xyz.append(g._xyz)
        merged_features_dc.append(g._features_dc)
        merged_features_rest.append(g._features_rest)
        merged_scaling.append(g._scaling)
        merged_rotation.append(g._rotation)
        merged_opacity.append(g._opacity)

    merged_gaussians = GaussianModel(gaussians[0].max_sh_degree)
    merged_gaussians._xyz = torch.cat(merged_xyz, dim=0)
    merged_gaussians._features_dc = torch.cat(merged_features_dc, dim=0)
    merged_gaussians._features_rest = torch.cat(merged_features_rest, dim=0)
    merged_gaussians._scaling = torch.cat(merged_scaling, dim=0)
    merged_gaussians._rotation = torch.cat(merged_rotation, dim=0)
    merged_gaussians._opacity = torch.cat(merged_opacity, dim=0)
    merged_gaussians.active_sh_degree = gaussians[0].active_sh_degree

    return merged_gaussians


def cam_has_mask(camera, obj_idx):
    masks = getattr(camera, "object_masks", []) or []
    return obj_idx < len(masks) and masks[obj_idx] is not None


def load_object_gaussians(dataset, loaded_iter, num_objects, active_sh_degree, apply_move=False):
    iteration_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{loaded_iter}")
    obj_gaussians = []

    for obj_idx in range(num_objects):
        g = GaussianModel(dataset.sh_degree)
        suffix = "obj.ply" if obj_idx == 0 else f"obj_{obj_idx}.ply"
        ply_path = os.path.join(iteration_dir, suffix)
        fallback_path = dataset.obj_ply_path

        if os.path.exists(ply_path):
            g.load_obj_ply(ply_path)
        elif os.path.exists(fallback_path):
            print(f"[WARN] Missing {ply_path}, falling back to {fallback_path}")
            g.load_obj_ply(fallback_path)
        else:
            print(f"[WARN] No object ply found for obj {obj_idx}; creating empty gaussian set")
            g._xyz = torch.empty((0, 3), device="cuda")
            g._features_dc = torch.empty((0, 1, 1), device="cuda")
            g._features_rest = torch.empty((0, 3, (g.max_sh_degree + 1) ** 2 - 1), device="cuda")
            g._scaling = torch.empty((0, 3), device="cuda")
            g._rotation = torch.empty((0, 4), device="cuda")
            g._opacity = torch.empty((0, 1), device="cuda")

        g.active_sh_degree = active_sh_degree

        # Apply default rotation to align object (user requested hard-coded R)
        if apply_move:
            try:
                move(g, R=DEFAULT_MOVE_R, obj_idx=obj_idx)
            except Exception as e:
                print(f"[WARN] move() failed for obj {obj_idx}: {e}")

        obj_gaussians.append(g)

    return obj_gaussians


def move(gaussian: GaussianModel, R=None, translation=None, obj_idx=None):
    """
    Apply a rotation R, then a translation, then the inverse rotation to `gaussian._xyz`.

    Args:
        gaussian: GaussianModel instance whose `_xyz` will be modified in-place.
        R: rotation matrix (3x3) as torch.Tensor, numpy array, or None. If None, no rotation is applied.
        translation: translation vector (3,) as torch.Tensor, list, or None. If None, no translation applied.

    This implements: xyz' = R_inv * ( R * xyz + translation )
    If R is None, simply adds translation to `gaussian._xyz`.
    """
    # If obj_idx provided and no explicit translation passed, try per-object defaults
    if translation is None and obj_idx is not None:
        try:
            translation = DEFAULT_MOVE_T[obj_idx]
        except Exception:
            translation = None

    if translation is None and R is None:
        return

    xyz = gaussian._xyz
    if xyz is None or xyz.numel() == 0:
        return

    device = xyz.device

    # prepare translation
    if translation is None:
        t = torch.zeros(3, device=device)
    else:
        t = torch.as_tensor(translation, device=device, dtype=xyz.dtype).view(3)

    # prepare R
    if R is None:
        # simple translate
        gaussian._xyz = xyz + t.view(1, 3)
        return

    R_t = None
    R_mat = torch.as_tensor(R, device=device, dtype=xyz.dtype)
    if R_mat.shape == (3, 3):
        R_t = R_mat
    elif R_mat.shape == (4, 4):
        R_t = R_mat[:3, :3]
    else:
        raise ValueError("Rotation R must be 3x3 or 4x4")

    # compute inverse (prefer transpose for orthonormal matrices)
    try:
        det = torch.det(R_t)
    except Exception:
        det = None

    if det is not None and torch.isfinite(det) and torch.allclose(det, torch.tensor(1.0, device=device), atol=1e-4):
        R_inv = R_t.t()
    else:
        R_inv = torch.inverse(R_t)

    # apply: xyz_rot = R * xyz^T
    xyz_rot = (R_t @ xyz.t()).t()
    # translate in rotated frame
    xyz_rot = xyz_rot + t.view(1, 3)
    # bring back
    xyz_new = (R_inv @ xyz_rot.t()).t()

    gaussian._xyz = xyz_new


def render_view(camera, gaussians, pipe, background, train_test_exp, separate_sh, mask_index=None):
    if camera is None:
        return None

    render_pkg = render(camera, gaussians, pipe, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    pred = render_pkg["render"]
    gt = getattr(camera, "original_image", None)
    if gt is not None:
        gt = gt.to(pred.device)

    alpha_mask = getattr(camera, "alpha_mask", None)
    if alpha_mask is not None:
        alpha_mask = alpha_mask.to(pred.device)
        pred = pred * alpha_mask
        if gt is not None:
            gt = gt * alpha_mask

    if mask_index is not None:
        masks = getattr(camera, "object_masks", []) or []
        if mask_index >= len(masks) or masks[mask_index] is None:
            return None
        obj_mask = masks[mask_index].to(pred.device)
        pred = pred * obj_mask
        if gt is not None:
            gt = gt * obj_mask

    if train_test_exp:
        pred = pred[..., pred.shape[-1] // 2:]
        if gt is not None:
            gt = gt[..., gt.shape[-1] // 2:]

    pred = torch.clamp(pred, 0.0, 1.0)
    if gt is not None:
        gt = torch.clamp(gt, 0.0, 1.0)
    return pred, gt


def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(tensor, path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Render-only script for gaussian-splatting-joint")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test", "finetune", "all"], default="all")
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--merge_objects", action="store_true", help="When rendering finetune, also render merged objects")
    parser.add_argument("--include_objects", default=None, type=str, help="Comma-separated 0/1 flags per object to include in merged model (default all 1)")
    parser.add_argument("--mask_index", default=None, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    placeholder_obj = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, placeholder_obj, load_iteration=args.iteration, shuffle=False)

    loaded_iter = scene.loaded_iter if scene.loaded_iter is not None else args.iteration
    num_objects = max(1, getattr(scene, "num_objects", 1))

    # parse include_objects flags (string) into list of ints length num_objects
    def _parse_include_objects(s, n):
        if s is None:
            return [1] * n
        s = s.strip()
        parts = re.split('[,;\s]+', s)
        parts = [p for p in parts if p != '']
        if len(parts) == 1 and len(s) == n and all(c in '01' for c in s):
            toks = [int(c) for c in s]
        else:
            try:
                toks = [int(x) for x in parts]
            except Exception:
                toks = [1] * n
        if len(toks) < n:
            toks = toks + [1] * (n - len(toks))
        if len(toks) > n:
            toks = toks[:n]
        toks = [1 if x else 0 for x in toks]
        return toks

    include_flags = _parse_include_objects(getattr(args, "include_objects", None), num_objects)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    out_root = getattr(args, "output_root", None) or os.path.join(dataset.model_path, "render")

    splits = []
    if args.split in ("train", "all"):
        splits.append(("train", scene.getTrainCameras(), scene.gaussians))
    if args.split in ("test", "all"):
        splits.append(("test", scene.getTestCameras(), scene.gaussians))
    if args.split in ("finetune", "all"):
        splits.append(("finetune", scene.getFinetuneCameras(), scene.gaussians))

    # optionally load object gaussians for merged rendering and build merged model
    obj_gaussians = None
    merged_all = None
    if args.merge_objects:
        obj_gaussians = load_object_gaussians(dataset, loaded_iter, num_objects, scene.gaussians.active_sh_degree, apply_move=True)
        if obj_gaussians:
            included_objs = [obj_gaussians[i] for i in range(min(len(obj_gaussians), num_objects)) if include_flags[i]]
            merged_all = merge_gaussians([scene.gaussians] + included_objs)

    for split_name, cameras, gauss in splits:
        if not cameras:
            print(f"[INFO] Split '{split_name}' has no cameras; skipping.")
            continue

        split_out_dir = os.path.join(out_root, split_name)
        os.makedirs(split_out_dir, exist_ok=True)

        # prepare merged output dir for this split if requested
        merged_out = None
        if merged_all and split_name in ("train", "finetune"):
            merged_out = os.path.join(out_root, f"{split_name}_merged")
            os.makedirs(merged_out, exist_ok=True)

        for cam in tqdm(cameras, desc=f"Render {split_name}", leave=False):
            # render scene gaussians
            rendered = render_view(cam, gauss, pipe, background, dataset.train_test_exp, False, mask_index=getattr(args, "mask_index", None))
            if rendered is None:
                continue
            pred, gt = rendered
            safe_name = getattr(cam, "image_name", "view").replace("/", "_").replace("\\", "_")
            out_path = os.path.join(split_out_dir, f"{safe_name}.png")
            save_image(pred, out_path)
            # if merged output enabled for this split, also render merged gaussians and save
            if merged_out is not None and merged_all is not None:
                rendered_m = render_view(cam, merged_all, pipe, background, dataset.train_test_exp, False, mask_index=None)
                if rendered_m is not None:
                    pred_m, _ = rendered_m
                    out_path_m = os.path.join(merged_out, f"{safe_name}.png")
                    save_image(pred_m, out_path_m)

        # (merged images per-split are handled per-camera above)

    print("Rendering finished.")
