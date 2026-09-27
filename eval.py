import os
import torch
from utils.loss_utils import l1_loss, ssim
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
import torchvision
import json
from datetime import datetime

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


from utils.joint_utils import merge_gaussians, load_object_gaussians, render_view, share_exposure
from utils.mask_utils import cam_has_mask, object_mask


@torch.no_grad()
def _evaluate_modes(cameras, gaussians, pipe, background, train_test_exp, separate_sh,
                    modes, output_root=None, sample_count=6):
    """Score multiple mask modes from one render per camera."""
    records = [[] for _ in modes]
    names = [[] for _ in modes]
    samples = set()
    if sample_count > 0 and cameras:
        count = min(sample_count, len(cameras))
        samples = {round(i * (len(cameras) - 1) / max(1, count - 1)) for i in range(count)}
    for view_index, cam in enumerate(tqdm(cameras, desc="Eval " + modes[0]["name"], leave=False)):
        rendered = render_view(cam, gaussians, pipe, background, train_test_exp, separate_sh)
        if rendered is None or rendered[1] is None:
            continue
        raw_pred, raw_gt = rendered
        for index, mode in enumerate(modes):
            pred, gt = raw_pred, raw_gt
            mask_index = mode.get("mask_index")
            if mask_index is not None:
                mask = object_mask(cam, mask_index)
                if mask is None:
                    continue
                mask = mask.to(pred.device)
                if train_test_exp:
                    mask = mask[..., mask.shape[-1] // 2:]
                gt = gt * mask
                if mode.get("mask_prediction", True):
                    pred = pred * mask
            ssim_value = (fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
                          if FUSED_SSIM_AVAILABLE else ssim(pred, gt))
            records[index].append(torch.stack((l1_loss(pred, gt).mean(), psnr(pred, gt).mean(),
                                               ssim_value.clamp(0, 1))).double())
            names[index].append(cam.image_name)
            if output_root and view_index in samples:
                folder = os.path.join(output_root, mode["name"])
                os.makedirs(folder, exist_ok=True)
                filename = cam.image_name.replace("/", "_").replace("\\", "_") + ".png"
                torchvision.utils.save_image(torch.cat((gt, pred, (gt-pred).abs()), dim=2),
                                             os.path.join(folder, filename))
    results = []
    for mode, values, image_names in zip(modes, records, names):
        result = {"name": mode["name"], "count": len(values)}
        if values:
            # One device-to-host transfer per split, instead of three per view.
            scores = torch.stack(values).cpu()
            result.update(dict(zip(("l1", "psnr", "ssim"), scores.mean(dim=0).tolist())))
            print(f"[RESULT] {mode['name']}: views={len(values)} L1={result['l1']:.4f} "
                  f"PSNR={result['psnr']:.2f} SSIM={result['ssim']:.4f}")
            if output_root:
                folder = os.path.join(output_root, mode["name"])
                os.makedirs(folder, exist_ok=True)
                rows = [dict(image_name=name, l1=row[0], psnr=row[1], ssim=row[2])
                        for name, row in zip(image_names, scores.tolist())]
                with open(os.path.join(folder, "per_view_metrics.json"), "w") as file:
                    json.dump(rows, file, indent=2)
        else:
            print(f"[INFO] Split '{mode['name']}' has no usable views; skipping.")
        results.append(result)
    return results


def evaluate_split(name, cameras, gaussians, pipe, background, train_test_exp, separate_sh,
                   mask_index=None, gt_mask_index=None, output_root=None, sample_count=6):
    modes = [{"name": name, "mask_index": mask_index if mask_index is not None else gt_mask_index,
              "mask_prediction": mask_index is not None}]
    return _evaluate_modes(cameras, gaussians, pipe, background, train_test_exp, separate_sh,
                           modes, output_root, sample_count)[0]


def evaluate(dataset, pipe, iteration, skip_train=False, skip_test=False, skip_finetune=False, skip_objects=False, sample_count=6):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        loaded_iter = scene.loaded_iter if scene.loaded_iter is not None else iteration
        num_objects = scene.num_objects
        obj_gaussians = load_object_gaussians(dataset, loaded_iter, num_objects, scene.gaussians.active_sh_degree)
        for model in obj_gaussians:
            share_exposure(model, scene.gaussians)
        merged_all = merge_gaussians([scene.gaussians] + obj_gaussians)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        results = []
        output_root = os.path.join(dataset.model_path, "eval")

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
                    output_root=output_root, sample_count=sample_count,
                )
            )

        if not skip_test:
            results.append(
                evaluate_split(
                    "test_scene",
                    scene.getBackgroundTestCameras(),
                    scene.gaussians,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    SPARSE_ADAM_AVAILABLE,
                    output_root=output_root, sample_count=sample_count,
                )
            )

            results.append(
                evaluate_split(
                    "test_scene_plus_objects", scene.getComposedTestCameras(), merged_all,
                    pipe, background, dataset.train_test_exp, SPARSE_ADAM_AVAILABLE,
                    output_root=output_root, sample_count=sample_count,
                )
            )

        ft_cameras = scene.getFinetuneCameras()
        if (not skip_finetune) and ft_cameras:
            results.append(
                evaluate_split(
                    "finetune_scene_plus_objects",
                    ft_cameras,
                    merged_all,
                    pipe,
                    background,
                    dataset.train_test_exp,
                    SPARSE_ADAM_AVAILABLE,
                    output_root=output_root, sample_count=sample_count,
                )
            )

            if not skip_objects:
                for obj_idx in range(num_objects):
                    cams_for_obj = [c for c in ft_cameras if cam_has_mask(c, obj_idx)]
                    modes = [
                        {"name": f"finetune_obj{obj_idx}_masked", "mask_index": obj_idx, "mask_prediction": True},
                        {"name": f"finetune_obj{obj_idx}_unmasked", "mask_index": obj_idx, "mask_prediction": False},
                    ]
                    results.extend(_evaluate_modes(cams_for_obj, obj_gaussians[obj_idx], pipe, background,
                                                   dataset.train_test_exp, SPARSE_ADAM_AVAILABLE,
                                                   modes, output_root, sample_count))

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
    parser.add_argument("--sample_count", type=int, default=6)

    args = get_combined_args(parser)
    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        args.iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    print("Evaluating " + args.model_path)

    safe_state(args.quiet)

    results = evaluate(
        lp.extract(args),
        pp.extract(args),
        args.iteration,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        skip_finetune=args.skip_finetune,
        skip_objects=args.skip_objects,
        sample_count=args.sample_count,
    )

    # Persist results to JSON under model_path/eval
    out_dir = os.path.join(args.model_path, "eval")
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, f"metrics_iter{args.iteration}.json")
    dump = {
        "model_path": args.model_path,
        "iteration": args.iteration,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": results,
    }
    with open(metrics_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Wrote metrics to {metrics_path}")
