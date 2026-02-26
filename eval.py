import os
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except ImportError:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


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


def load_object_gaussians(dataset, loaded_iter, num_objects, active_sh_degree):
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
        obj_gaussians.append(g)

    return obj_gaussians


def render_view(camera, gaussians, pipe, background, train_test_exp, separate_sh, mask_index=None):
    if camera is None:
        return None

    render_pkg = render(camera, gaussians, pipe, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)
    pred = render_pkg["render"]
    gt = camera.original_image.to(pred.device)

    alpha_mask = getattr(camera, "alpha_mask", None)
    if alpha_mask is not None:
        alpha_mask = alpha_mask.to(pred.device)
        pred = pred * alpha_mask
        gt = gt * alpha_mask

    if mask_index is not None:
        masks = getattr(camera, "object_masks", []) or []
        if mask_index >= len(masks) or masks[mask_index] is None:
            return None
        obj_mask = masks[mask_index].to(pred.device)
        pred = pred * obj_mask
        gt = gt * obj_mask

    if train_test_exp:
        pred = pred[..., pred.shape[-1] // 2:]
        gt = gt[..., gt.shape[-1] // 2:]

    pred = torch.clamp(pred, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)
    return pred, gt


def evaluate_split(name, cameras, gaussians, pipe, background, train_test_exp, separate_sh, mask_index=None):
    if not cameras:
        print(f"[INFO] Split '{name}' has no cameras; skipping.")
        return {"name": name, "count": 0}

    l1_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for cam in tqdm(cameras, desc=f"Eval {name}", leave=False):
        rendered = render_view(cam, gaussians, pipe, background, train_test_exp, separate_sh, mask_index)
        if rendered is None:
            continue
        pred, gt = rendered
        l1_val = l1_loss(pred, gt).mean().double()
        psnr_val = psnr(pred, gt).mean().double()
        if FUSED_SSIM_AVAILABLE:
            ssim_val = fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        else:
            ssim_val = ssim(pred, gt)
        ssim_val = torch.clamp(ssim_val, 0.0, 1.0)

        l1_sum += l1_val.item()
        psnr_sum += psnr_val.item()
        ssim_sum += ssim_val.item()
        count += 1

    if count == 0:
        print(f"[INFO] Split '{name}' had no usable masked views; skipping metrics.")
        return {"name": name, "count": 0}

    result = {
        "name": name,
        "count": count,
        "l1": l1_sum / count,
        "psnr": psnr_sum / count,
        "ssim": ssim_sum / count,
    }
    print(f"[RESULT] {name}: views={count} L1={result['l1']:.4f} PSNR={result['psnr']:.2f} SSIM={result['ssim']:.4f}")
    return result


def evaluate(dataset, pipe, iteration, skip_train=False, skip_test=False, skip_finetune=False, skip_objects=False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        placeholder_obj = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, placeholder_obj, load_iteration=iteration, shuffle=False)

        loaded_iter = scene.loaded_iter if scene.loaded_iter is not None else iteration
        num_objects = max(1, getattr(scene, "num_objects", 1))
        obj_gaussians = load_object_gaussians(dataset, loaded_iter, num_objects, scene.gaussians.active_sh_degree)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        results = []

        if not skip_train:
            results.append(
                evaluate_split(
                    "train_scene",
                    scene.getTrainCameras(),
                    scene.gaussians,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    SPARSE_ADAM_AVAILABLE,
                )
            )

        if not skip_test:
            results.append(
                evaluate_split(
                    "test_scene",
                    scene.getTestCameras(),
                    scene.gaussians,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    SPARSE_ADAM_AVAILABLE,
                )
            )

        ft_cameras = scene.getFinetuneCameras()
        if (not skip_finetune) and ft_cameras:
            merged_all = merge_gaussians([scene.gaussians] + obj_gaussians)
            results.append(
                evaluate_split(
                    "finetune_scene_plus_objects",
                    ft_cameras,
                    merged_all,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    SPARSE_ADAM_AVAILABLE,
                )
            )

            if not skip_objects:
                for obj_idx in range(num_objects):
                    cams_for_obj = [c for c in ft_cameras if cam_has_mask(c, obj_idx)]
                    results.append(
                        evaluate_split(
                            f"finetune_obj{obj_idx}_masked",
                            cams_for_obj,
                            obj_gaussians[obj_idx],
                            pipe,
                            background,
                            dataset.train_test_exp,
                            SPARSE_ADAM_AVAILABLE,
                            mask_index=obj_idx,
                        )
                    )

        return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_finetune", action="store_true")
    parser.add_argument("--skip_objects", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)

    safe_state(args.quiet)

    evaluate(
        lp.extract(args),
        pp.extract(args),
        args.iteration,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        skip_finetune=args.skip_finetune,
        skip_objects=args.skip_objects,
    )
