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

import torch
import math
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    if separate_sh:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out


def render_debug_mask(viewpoint_camera, pc: GaussianModel, object_mask: torch.Tensor):
    """Visualize Gaussian projections relative to a binary object mask."""

    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    if object_mask is None or pc.get_xyz.numel() == 0:
        return canvas

    device = pc.get_xyz.device
    homo_positions = torch.cat(
        [pc.get_xyz, torch.ones((pc.get_xyz.shape[0], 1), device=device, dtype=pc.get_xyz.dtype)],
        dim=1,
    )

    clip_coords = torch.matmul(homo_positions, viewpoint_camera.full_proj_transform)
    clip_w = clip_coords[:, 3]
    positive_w = clip_w > 0
    if positive_w.sum() == 0:
        return canvas

    clip_coords = clip_coords[positive_w]
    ndc = clip_coords[:, :3] / clip_coords[:, 3:4]

    inside_frustum = (
        (ndc[:, 0] >= -1.0)
        & (ndc[:, 0] <= 1.0)
        & (ndc[:, 1] >= -1.0)
        & (ndc[:, 1] <= 1.0)
    )
    if inside_frustum.sum() == 0:
        return canvas

    ndc = ndc[inside_frustum]

    screen_x = ((ndc[:, 0] * 0.5 + 0.5) * (width - 1)).round().long()
    screen_y = ((ndc[:, 1] * 0.5 + 0.5) * (height - 1)).round().long()

    screen_x = torch.clamp(screen_x, 0, width - 1)
    screen_y = torch.clamp(screen_y, 0, height - 1)

    mask_cpu = object_mask.squeeze(0).detach().cpu()
    # Paint mask area green first (keep dtype uint8 canvas)
    mask_full = (mask_cpu > 0.5).numpy()
    green = np.array([0, 255, 0], dtype=np.uint8)
    canvas[mask_full] = green

    mask_values = mask_cpu[screen_y.cpu(), screen_x.cpu()] > 0.5

    coords = torch.stack([screen_y, screen_x], dim=1).cpu().numpy()
    mask_values = mask_values.numpy()

    blue = np.array([0, 0, 255], dtype=np.uint8)
    red = np.array([255, 0, 0], dtype=np.uint8)

    for (y, x), is_inside in zip(coords, mask_values):
        color = blue if is_inside else red
        y0 = max(y - 1, 0)
        y1 = min(y + 2, height)
        x0 = max(x - 1, 0)
        x1 = min(x + 2, width)
        canvas[y0:y1, x0:x1] = color

    return canvas
