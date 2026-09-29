#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import math
import json
import time
import torch
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace, SUPPRESS
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
    
from utils.joint_utils import merge_gaussians, load_object_gaussians, share_exposure
from utils.mask_utils import prune_gaussians_with_object_masks
from utils.training_state import (CameraSampler, capture_rng, restore_rng, run_config,
                                 save_run_config, save_checkpoint, load_checkpoint, parse_training_args)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, pruning_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    if opt.log_interval < 1:
        raise ValueError("log_interval must be positive")
    state = load_checkpoint(checkpoint) if checkpoint else None
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    num_objects = scene.num_objects
    dataset.num_objects = num_objects
    if not dataset.obj_ply_path:
        dataset.obj_ply_path = scene.initial_ply_path
    gaussians.training_setup(opt)
    obj_gaussians_list = load_object_gaussians(dataset, None, num_objects, gaussians.active_sh_degree)
    for g in obj_gaussians_list:
        g.obj_training_setup(opt)
    scene.setObjectGaussians(obj_gaussians_list)
    config = run_config(dataset, opt, pipe, {
        "test_iterations": testing_iterations, "save_iterations": saving_iterations,
        "prune_iterations": pruning_iterations, "checkpoint_iterations": checkpoint_iterations,
        "start_checkpoint": checkpoint, "debug_from": debug_from,
    })

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True) if tb_writer else None
    iter_end = torch.cuda.Event(enable_timing=True) if tb_writer else None

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    background_sampler = CameraSampler(scene.getTrainCameras())
    if not background_sampler.pool:
        raise ValueError("Training requires background cameras")
    ft_cameras_all = scene.getFinetuneCameras().copy()
    composed_sampler = CameraSampler(ft_cameras_all)
    # Build per-object finetune stacks (only cameras that have a mask for that object).
    ft_viewpoint_pools = [[] for _ in range(num_objects)]
    for cam in ft_cameras_all:
        masks = getattr(cam, "object_masks", []) or []
        for obj_idx in range(num_objects):
            mask_tensor = masks[obj_idx] if obj_idx < len(masks) else None
            if mask_tensor is not None:
                ft_viewpoint_pools[obj_idx].append(cam)

    object_samplers = [CameraSampler(pool) for pool in ft_viewpoint_pools]
    if opt.object_only_until_iter > 1 and not any(ft_viewpoint_pools):
        raise ValueError("Object-only pretraining requires masked object views")
    # Report per-object available mask camera counts and computed prune threshold
    try:
        prop = float(opt.mask_prune_min_prop)
    except Exception:
        prop = 0.5
    for obj_idx, pool in enumerate(ft_viewpoint_pools):
        n_avail = len(pool)
        computed_th = int(math.ceil(prop * n_avail)) if n_avail > 0 else 0
        computed_th = max(computed_th, 1) if n_avail > 0 else 0
        print(f"[INFO] Obj {obj_idx}: available_masks={n_avail}, prune_threshold={computed_th} (prop={prop})")
    ema_loss_for_log = torch.zeros((), device="cuda")
    ema_Ll1depth_for_log = torch.zeros((), device="cuda")
    scene_optim_initialized = False
    objects_finetuned = False
    samplers = [background_sampler, composed_sampler] + object_samplers
    if checkpoint:
        if len(state["objects"]) != num_objects or len(state["samplers"]) != len(samplers):
            raise ValueError("Checkpoint object count differs from the current mask folders")
        gaussians.restore_full(state["background"], opt)
        objects_finetuned = state["objects_finetuned"]
        for g, saved in zip(obj_gaussians_list, state["objects"]):
            g.restore_full(saved, opt, "finetune" if objects_finetuned else "object")
        for sampler, saved in zip(samplers, state["samplers"]):
            sampler.load_state_dict(saved)
        first_iter = state["iteration"]
        scene_optim_initialized = state["scene_optim_initialized"]
        ema_loss_for_log, ema_Ll1depth_for_log = state["ema"].to("cuda").unbind()
        restore_rng(state["rng"])
        del state
    save_run_config(os.path.join(scene.model_path, "training_config.json"), config)
    # Keep compatibility with existing inference scripts.
    with open(os.path.join(scene.model_path, "cfg_args"), "w") as file:
        file.write(str(Namespace(**vars(dataset))))
    for g in obj_gaussians_list:
        share_exposure(g, gaussians)
    prune_cache = {}
    metrics_file = open(os.path.join(scene.model_path, "training_metrics.jsonl"),
                        "a" if checkpoint else "w", buffering=1)
    training_start = time.perf_counter()

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        log_this_iter = iteration % opt.log_interval == 0 or iteration == opt.iterations
        if iter_start is not None and log_this_iter:
            iter_start.record()

        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Before a certain iteration, train only the object branches.
        object_only_phase = (iteration < getattr(opt, "object_only_until_iter", 0))

        # Keep the baseline RNG draw order, including background sampling
        # during object-only pretraining, so refactoring does not change views.
        viewpoint_cam = background_sampler.sample()
        ft_global_cam = composed_sampler.sample() if not object_only_phase else None
        obj_viewpoint_cams = [sampler.sample() for sampler in object_samplers]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # If we just finished the object-only phase, re-run finetuning setup for the object model
        # to reset its optimizer state and accumulators before joint training starts.
        if (getattr(opt, "object_only_until_iter", 0) > 0) and (iteration == getattr(opt, "object_only_until_iter", 0)):
            # Note: we intentionally use finetuning_setup here (not training_setup) to avoid exposure optimizer
            # which is only configured for the scene gaussians.
            for gi, obj_g in enumerate(obj_gaussians_list):
                obj_g.finetuning_setup(opt)
                print(f"[ITER {iteration}] Re-initialized object optimizer {gi} via finetuning_setup after object-only phase.")
            objects_finetuned = True
        
        render_pkg = {}
        viewspace_point_tensor = visibility_filter = radii = None
        # The global finetune render supplies both composed RGB and depth.
        ft_global_pkg = None
        ft_global_image = None
        ft_global_gt = None

        if not object_only_phase:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE,
                                retain_viewspace_grad=iteration < opt.densify_until_iter)
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

        if (not object_only_phase) and (viewpoint_cam.alpha_mask is not None):
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Global ft render: scene + all objects merged, using the sampled ft_global_cam
        if not object_only_phase and ft_global_cam is not None:
            merged_all = merge_gaussians([gaussians] + obj_gaussians_list)
            ft_global_pkg = render(ft_global_cam, merged_all, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE,
                                  retain_viewspace_grad=False)
            ft_global_image = ft_global_pkg["render"]
            if ft_global_cam.alpha_mask is not None:
                ft_global_image *= ft_global_cam.alpha_mask
            ft_global_gt = ft_global_cam.original_image.cuda()

        # Per-object renders and GTs
        obj_render_pkgs = [None] * num_objects
        obj_images = [None] * num_objects
        obj_gt_images = [None] * num_objects
        obj_masks_used = [None] * num_objects
        obj_viewspace_point_tensors = [None] * num_objects
        obj_visibility_filters = [None] * num_objects
        obj_radii_list = [None] * num_objects

        for obj_idx, cam in enumerate(obj_viewpoint_cams):
            if cam is None:
                continue

            masks = getattr(cam, "object_masks", []) or []
            obj_mask = masks[obj_idx] if obj_idx < len(masks) else None

            obj_render_pkg = render(cam, obj_gaussians_list[obj_idx], pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE,
                                   retain_viewspace_grad=iteration < opt.densify_until_iter)
            obj_img = obj_render_pkg["render"]
            if cam.alpha_mask is not None:
                obj_img *= cam.alpha_mask
            obj_gt = cam.original_image.cuda()
            if obj_mask is not None:
                mask_tensor = obj_mask.to(obj_img.device)
                obj_img = obj_img * mask_tensor
                obj_gt = obj_gt * mask_tensor

            obj_render_pkgs[obj_idx] = obj_render_pkg
            obj_images[obj_idx] = obj_img
            obj_gt_images[obj_idx] = obj_gt
            obj_masks_used[obj_idx] = obj_mask
            obj_viewspace_point_tensors[obj_idx] = obj_render_pkg.get("viewspace_points")
            obj_visibility_filters[obj_idx] = obj_render_pkg.get("visibility_filter")
            obj_radii_list[obj_idx] = obj_render_pkg.get("radii")

        # Loss
        active_obj_indices = [i for i, cam in enumerate(obj_viewpoint_cams) if cam is not None]
        scene_weight = 1.0
        composed_weight = 0.1
        object_weight = 10.0

        if object_only_phase:
            l1_terms = []
            ssim_terms = []
            for idx in active_obj_indices:
                if obj_images[idx] is None or obj_gt_images[idx] is None:
                    continue
                l1_terms.append(l1_loss(obj_images[idx], obj_gt_images[idx]))
                if FUSED_SSIM_AVAILABLE:
                    ssim_terms.append(fused_ssim(obj_images[idx].unsqueeze(0), obj_gt_images[idx].unsqueeze(0)))
                else:
                    ssim_terms.append(ssim(obj_images[idx], obj_gt_images[idx]))

            if l1_terms:
                Ll1 = sum(l1_terms) / len(l1_terms)
                ssim_value = sum(ssim_terms) / len(ssim_terms) if ssim_terms else torch.tensor(0.0, device=gaussians.get_xyz.device)
                ssim_value = torch.clamp(ssim_value, 0.0, 1.0)
            else:
                Ll1 = torch.tensor(0.0, device=gaussians.get_xyz.device)
                ssim_value = torch.tensor(0.0, device=gaussians.get_xyz.device)
        else:
            gt_image = viewpoint_cam.original_image.cuda()

            Ll1_num = scene_weight * l1_loss(image, gt_image)
            denom = scene_weight

            # Global finetune term: scene + all objects merged
            if ft_global_image is not None and ft_global_gt is not None:
                Ll1_num += composed_weight * l1_loss(ft_global_image, ft_global_gt)
                denom += composed_weight

            for idx in active_obj_indices:
                if obj_images[idx] is not None and obj_gt_images[idx] is not None:
                    Ll1_num += object_weight * l1_loss(obj_images[idx], obj_gt_images[idx])
                    denom += object_weight

            Ll1 = Ll1_num / denom

            if FUSED_SSIM_AVAILABLE:
                ssim_scene = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_scene = ssim(image, gt_image)

            ssim_value = torch.clamp(ssim_scene, 0.0, 1.0)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization follows the same three supervision levels as RGB:
        # scene-only, scene + all objects, and each object in isolation.
        depth_terms_for_log = []
        current_depth_weight = depth_l1_weight(iteration)

        # 1) Scene-only depth.
        if current_depth_weight > 0 and not object_only_phase and getattr(viewpoint_cam, "depth_reliable", False):
            invDepth = render_pkg.get("depth", None)
            if invDepth is not None and getattr(viewpoint_cam, "invdepthmap", None) is not None:
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                Ll1depth_pure = torch.abs(invDepth - mono_invdepth).mean()
                Ll1depth_w = current_depth_weight * Ll1depth_pure
                loss += Ll1depth_w
                depth_terms_for_log.append(Ll1depth_w.detach())

        # 2) Composed depth, reusing the scene + all-objects RGB render.
        if (
            current_depth_weight > 0
            and ft_global_pkg is not None
            and ft_global_cam is not None
            and getattr(ft_global_cam, "depth_reliable", False)
            and getattr(ft_global_cam, "invdepthmap", None) is not None
        ):
            invDepth = ft_global_pkg.get("depth", None)
            if invDepth is not None:
                mono_invdepth = ft_global_cam.invdepthmap.cuda()
                Ll1depth_pure = torch.abs(invDepth - mono_invdepth).mean()
                Ll1depth_w = composed_weight * current_depth_weight * Ll1depth_pure
                loss += Ll1depth_w
                depth_terms_for_log.append(Ll1depth_w.detach())

        # 3) Object-only masked depth for each object.
        if current_depth_weight > 0:
            for idx in active_obj_indices:
                cam = obj_viewpoint_cams[idx]
                if cam is None:
                    continue

                obj_mask = obj_masks_used[idx]
                if obj_render_pkgs[idx] is not None and getattr(cam, "depth_reliable", False) and (obj_mask is not None):
                    invDepth = obj_render_pkgs[idx].get("depth", None)
                    if invDepth is not None and getattr(cam, "invdepthmap", None) is not None:
                        mono_invdepth = cam.invdepthmap.cuda()
                        depth_mask_obj = cam.depth_mask.cuda() * obj_mask
                        gt_full = mono_invdepth * depth_mask_obj
                        Ll1depth_pure = torch.abs(invDepth - gt_full).mean()
                        Ll1depth_w = object_weight * current_depth_weight * Ll1depth_pure
                        loss += Ll1depth_w
                        depth_terms_for_log.append(Ll1depth_w.detach())

        loss.backward()
        Ll1depth = torch.stack(depth_terms_for_log).sum() if depth_terms_for_log else loss.detach().new_zeros(())
        elapsed = None
        if iter_end is not None and log_this_iter:
            iter_end.record()

        with torch.no_grad():
            pruned_this_iter = False
            # Progress bar
            ema_loss_for_log = 0.4 * loss.detach() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if log_this_iter:
                loss_log, depth_log = torch.stack([ema_loss_for_log, ema_Ll1depth_for_log]).cpu().tolist()
                if iter_end is not None:
                    iter_end.synchronize()
                    elapsed = iter_start.elapsed_time(iter_end)
                progress_bar.set_postfix({"Loss": f"{loss_log:.7f}", "Depth Loss": f"{depth_log:.7f}"})
                progress_bar.update(iteration - first_iter + 1 - progress_bar.n)
                metrics_file.write(json.dumps({
                    "iteration": iteration, "ema_loss": loss_log, "ema_depth_loss": depth_log,
                    "phase": "object_only" if object_only_phase else "joint",
                    "background_points": len(gaussians.get_xyz),
                    "object_points": [len(g.get_xyz) for g in obj_gaussians_list],
                    "elapsed_training_seconds": time.perf_counter() - training_start,
                    "gpu_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                }) + "\n")
            if iteration == opt.iterations:
                progress_bar.close()

            # Log, prune, and save. Pruning deliberately comes before saving so
            # coincident prune/save iterations persist the already-pruned models.
            if log_this_iter or iteration in testing_iterations:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp, log_this_iter)
            if iteration in pruning_iterations:
                ft_cams_current = scene.getFinetuneCameras()
                if ft_cams_current:
                    print("\n[ITER {}] Mask-based pruning of object Gaussians".format(iteration))
                    for obj_idx, obj_g in enumerate(obj_gaussians_list):
                        pruned, used_threshold = prune_gaussians_with_object_masks(
                            obj_g,
                            ft_cams_current,
                            mask_prune_min_prop=opt.mask_prune_min_prop,
                            mask_threshold=opt.mask_prune_threshold,
                            mask_prune_expand=opt.mask_prune_expand,
                            mask_index=obj_idx,
                            cache=prune_cache,
                        )
                        if pruned > 0:
                            pruned_this_iter = True
                            print("\n[ITER {}] Mask pruning removed {} object Gaussians for obj {} (used threshold={} views)".format(iteration, pruned, obj_idx, used_threshold))
                else:
                    print("\n[ITER {}] Skipping mask-based pruning: no finetune cameras available".format(iteration))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if (not pruned_this_iter) and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if not object_only_phase:
                    # Scene: use scene-only render stats (viewspace_point_tensor, visibility_filter, radii)
                    torch.maximum(gaussians.max_radii2D, radii, out=gaussians.max_radii2D)
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Obj: use object-only render stats for densification
                for idx in active_obj_indices:
                    obj_vpt = obj_viewspace_point_tensors[idx]
                    obj_vis = obj_visibility_filters[idx]
                    obj_r = obj_radii_list[idx]
                    obj_g = obj_gaussians_list[idx]
                    if obj_vpt is None or obj_vis is None or obj_r is None or obj_vpt.grad is None:
                        continue

                    torch.maximum(obj_g.max_radii2D, obj_r, out=obj_g.max_radii2D)
                    obj_g.add_densification_stats(
                        obj_vpt, obj_vis
                    )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    if not object_only_phase:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                    for idx in active_obj_indices:
                        obj_r = obj_radii_list[idx]
                        obj_g = obj_gaussians_list[idx]
                        if obj_r is not None:
                            obj_g.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, obj_r)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    if (not object_only_phase) and scene_optim_initialized:
                        gaussians.reset_opacity()
                    for obj_g in obj_gaussians_list:
                        obj_g.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # Exposure is shared by all branches, but learned only during
                # joint training. Do not accumulate stale pretraining gradients.
                if dataset.train_test_exp:
                    if not object_only_phase:
                        gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if not object_only_phase:
                    if use_sparse_adam:
                        visible = radii > 0
                        gaussians.optimizer.step(visible, radii.shape[0])
                        gaussians.optimizer.zero_grad(set_to_none = True)
                        scene_optim_initialized = True
                    else:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)
                        scene_optim_initialized = True
                # Always step object branches
                for obj_g in obj_gaussians_list:
                    obj_g.optimizer.step()
                    obj_g.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_checkpoint(os.path.join(scene.model_path, f"chkpnt{iteration}.pth"), {
                    "schema_version": 1, "iteration": iteration,
                    "background": gaussians.capture_full(),
                    "objects": [g.capture_full(include_exposure=False) for g in obj_gaussians_list],
                    "objects_finetuned": objects_finetuned,
                    "scene_optim_initialized": scene_optim_initialized,
                    "samplers": [sampler.state_dict() for sampler in samplers],
                    "rng": capture_rng(), "config": config,
                    "ema": torch.stack([ema_loss_for_log, ema_Ll1depth_for_log]),
                })
    if tb_writer:
        tb_writer.close()
    metrics_file.close()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, log_training=True):
    if tb_writer and log_training:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        if elapsed is not None:
            tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        composed_cameras = scene.getComposedTestCameras()
        composed = merge_gaussians([scene.gaussians] + scene.object_gaussians) if composed_cameras else None
        validation_configs = (
            {'name': 'test_scene', 'cameras': scene.getBackgroundTestCameras(), 'model': scene.gaussians},
            {'name': 'test_scene_plus_objects', 'cameras': composed_cameras, 'model': composed},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)], 'model': scene.gaussians})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, config['model'], *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
    parser.add_argument("--mask_prune_on_save", action="store_true",
                        help="Compatibility with older experiment commands: prune at every save iteration")
    # Recommended: include the final training iteration so the final save is
    # written immediately after mask-based object pruning.
    parser.add_argument(
        "--prune_iterations",
        nargs="+",
        type=int,
        default=[],
        help="Iterations for mask-based object pruning. Include the final training iteration to prune immediately before the final save.",
    )
    parser.add_argument("--quiet", action="store_true")
    # Accept historical DexMirror commands; training is now always headless.
    parser.add_argument('--disable_viewer', action='store_true', help=SUPPRESS)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parse_training_args(parser, sys.argv[1:])
    args.save_iterations = sorted(set(args.save_iterations + [args.iterations]))
    if args.mask_prune_on_save:
        args.prune_iterations = sorted(set(args.prune_iterations + args.save_iterations))
        print("mask_prune_on_save resolved to prune_iterations:", args.prune_iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training.
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.prune_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
