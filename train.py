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
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
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
    
_GAUSSIAN_KERNEL_CACHE = {}


def _get_gaussian_kernel(sigma: float, device: torch.device):
    sigma_eff = max(float(sigma), 1e-6)
    key = (sigma_eff, str(device))
    if key in _GAUSSIAN_KERNEL_CACHE:
        return _GAUSSIAN_KERNEL_CACHE[key]

    radius = max(1, int(math.ceil(3.0 * sigma_eff)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel1d = torch.exp(-0.5 * (coords / sigma_eff) ** 2)
    kernel1d /= kernel1d.sum()
    kernel2d = torch.matmul(kernel1d.unsqueeze(1), kernel1d.unsqueeze(0))
    kernel = kernel2d.unsqueeze(0).unsqueeze(0)
    _GAUSSIAN_KERNEL_CACHE[key] = kernel
    return kernel


def prune_gaussians_with_object_masks(gaussians, cameras, mask_prune_min_prop=0.5, mask_threshold=0.5, mask_prune_expand=0.0, mask_index=0):
    """Prune Gaussians that fall outside per-view object masks.

    Args:
        gaussians: GaussianModel to prune.
        cameras: list of Camera objects.
        mask_index: which object mask to use if cameras carry multiple masks.
    """

    if gaussians.get_xyz.numel() == 0:
        return 0

    mask_cameras = []
    for cam in cameras:
        mask = getattr(cam, "object_masks", None)
        if isinstance(mask, list):
            if mask_index < len(mask) and mask[mask_index] is not None:
                mask_cameras.append(cam)
        elif getattr(cam, "object_mask", None) is not None:
            mask_cameras.append(cam)
    if len(mask_cameras) == 0:
        return 0

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

        if mask_prune_expand > 0:
            radius = max(1, int(math.ceil(mask_prune_expand)))
            ksize = 2 * radius + 1
            mask_tensor = F.max_pool2d(
                (mask_tensor > 0.5).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                kernel_size=ksize,
                stride=1,
                padding=radius,
            ).squeeze(0).squeeze(0)

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

    # Proportion-only mode: compute threshold = ceil(prop * n_mask_cams), at least 1
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

    removed = int(prune_mask.sum().item())
    if removed > 0:
        if not hasattr(gaussians, "tmp_radii") or gaussians.tmp_radii is None or gaussians.tmp_radii.shape[0] != positions.shape[0]:
            gaussians.tmp_radii = positions.new_zeros((positions.shape[0],))
        gaussians.prune_points(prune_mask)
        gaussians.tmp_radii = None

    # Return also the computed threshold for reporting if caller wants it
    return removed, computed_threshold


def merge_gaussians(gaussians):
    """
    Merge the gaussians from the two models
    
    Args:
        gaussians (list): List of several GaussianModel objects to be merged.
        
    Returns:
        GaussianModel: A new GaussianModel object containing the merged data.
    """
    # merged_gaussians = GaussianModel(sh_degree)
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
    assert gaussians[0].active_sh_degree == gaussians[1].active_sh_degree
    merged_gaussians.active_sh_degree = gaussians[0].active_sh_degree

    return merged_gaussians

def training(dataset, opt, pipe, testing_iterations, saving_iterations, pruning_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    # Build scene once with a placeholder object model to load cameras/ply; we replace/extend object models below.
    placeholder_obj = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, placeholder_obj)
    num_objects = max(1, getattr(scene, "num_objects", 1))
    dataset.num_objects = num_objects
    gaussians.training_setup(opt)

    obj_gaussians_list = []
    for _ in range(num_objects):
        g = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        # Load the same initial obj ply for each object; callers can customize per-object ply via downstream hooks if needed.
        try:
            g.load_obj_ply(dataset.obj_ply_path)
        except Exception:
            pass
        # Important: set up optimizer/stats *after* loading points so buffers match tensor sizes
        g.obj_training_setup(opt)
        obj_gaussians_list.append(g)

    # Keep the first object model wired into the scene for compatibility with existing save/report hooks
    scene.obj_gaussians = obj_gaussians_list[0]

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))

    ft_cameras_all = scene.getFinetuneCameras().copy()
    ft_global_stack = ft_cameras_all.copy()
    ft_global_indices = list(range(len(ft_global_stack)))
    # Build per-object finetune stacks (only cameras that have a mask for that object).
    ft_viewpoint_pools = [[] for _ in range(num_objects)]
    for cam in ft_cameras_all:
        masks = getattr(cam, "object_masks", []) or []
        for obj_idx in range(num_objects):
            mask_tensor = masks[obj_idx] if obj_idx < len(masks) else None
            if mask_tensor is not None:
                ft_viewpoint_pools[obj_idx].append(cam)

    ft_viewpoint_stacks = [pool.copy() for pool in ft_viewpoint_pools]
    ft_viewpoint_indices = [list(range(len(pool))) for pool in ft_viewpoint_stacks]
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
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    scene_optim_initialized = False

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Before a certain iteration, train only the object branches.
        object_only_phase = (iteration < getattr(opt, "object_only_until_iter", 0))

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        # Refresh per-object finetune stacks when empty
        for obj_idx in range(num_objects):
            if not ft_viewpoint_stacks[obj_idx]:
                ft_viewpoint_stacks[obj_idx] = ft_viewpoint_pools[obj_idx].copy()
                ft_viewpoint_indices[obj_idx] = list(range(len(ft_viewpoint_stacks[obj_idx])))
        if not object_only_phase and not ft_global_stack:
            ft_global_stack = ft_cameras_all.copy()
            ft_global_indices = list(range(len(ft_global_stack)))

        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # The global finetune camera is unused during object-only pretraining.
        ft_global_cam = None
        if not object_only_phase and ft_global_indices:
            ft_global_rand = randint(0, len(ft_global_indices) - 1)
            ft_global_cam = ft_global_stack.pop(ft_global_rand)
            ft_global_indices.pop(ft_global_rand)

        obj_viewpoint_cams = []
        for obj_idx in range(num_objects):
            if not ft_viewpoint_stacks[obj_idx]:
                obj_viewpoint_cams.append(None)
                continue
            ft_rand_idx = randint(0, len(ft_viewpoint_indices[obj_idx]) - 1)
            cam_sel = ft_viewpoint_stacks[obj_idx].pop(ft_rand_idx)
            ft_viewpoint_indices[obj_idx].pop(ft_rand_idx)
            obj_viewpoint_cams.append(cam_sel)

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
        
        render_pkg = {}
        viewspace_point_tensor = visibility_filter = radii = None
        # The global finetune render supplies both composed RGB and depth.
        ft_global_pkg = None
        ft_global_image = None
        ft_global_gt = None

        if not object_only_phase:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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
            ft_global_pkg = render(ft_global_cam, merged_all, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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

            obj_render_pkg = render(cam, obj_gaussians_list[obj_idx], pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
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
        Ll1depth = torch.stack(depth_terms_for_log).sum().item() if depth_terms_for_log else 0.0

        iter_end.record()

        with torch.no_grad():
            pruned_this_iter = False
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log, prune, and save. Pruning deliberately comes before saving so
            # coincident prune/save iterations persist the already-pruned models.
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
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
                        )
                        if pruned > 0:
                            pruned_this_iter = True
                            print("\n[ITER {}] Mask pruning removed {} object Gaussians for obj {} (used threshold={} views)".format(iteration, pruned, obj_idx, used_threshold))
                else:
                    print("\n[ITER {}] Skipping mask-based pruning: no finetune cameras available".format(iteration))

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # Persist additional object gaussians (obj_0 already saved via scene.save)
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration))
                os.makedirs(point_cloud_path, exist_ok=True)
                for obj_idx, obj_g in enumerate(obj_gaussians_list):
                    suffix = "obj.ply" if obj_idx == 0 else f"obj_{obj_idx}.ply"
                    obj_g.save_ply(os.path.join(point_cloud_path, suffix))

            # Densification
            if (not pruned_this_iter) and iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                if not object_only_phase:
                    # Scene: use scene-only render stats (viewspace_point_tensor, visibility_filter, radii)
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Obj: use object-only render stats for densification
                for idx in active_obj_indices:
                    obj_vpt = obj_viewspace_point_tensors[idx]
                    obj_vis = obj_visibility_filters[idx]
                    obj_r = obj_radii_list[idx]
                    obj_g = obj_gaussians_list[idx]
                    if obj_vpt is None or obj_vis is None or obj_r is None or obj_vpt.grad is None:
                        continue

                    obj_g.max_radii2D[obj_vis] = torch.max(
                        obj_g.max_radii2D[obj_vis],
                        obj_r[obj_vis]
                    )
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
                if not object_only_phase:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none = True)
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
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 30_000])
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
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.prune_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
