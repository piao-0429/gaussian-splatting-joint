"""Shared multi-view mask pruning, with the training criterion preserved."""

import math
import torch
import torch.nn.functional as F


def object_mask(camera, index):
    masks = getattr(camera, "object_masks", None)
    if isinstance(masks, list):
        return masks[index] if index < len(masks) else None
    return getattr(camera, "object_mask", None) if index == 0 else None


def cam_has_mask(camera, index):
    return object_mask(camera, index) is not None


@torch.no_grad()
def compute_prune_mask(gaussians, cameras, mask_prune_min_prop=0.5,
                       mask_threshold=0.5, mask_expand=0.0, mask_index=0, cache=None):
    positions = gaussians.get_xyz.detach()
    empty = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    views = [(cam, object_mask(cam, mask_index)) for cam in cameras]
    views = [(cam, mask) for cam, mask in views if mask is not None]
    if not positions.shape[0] or not views:
        return empty, 0

    threshold = max(1, math.ceil(float(mask_prune_min_prop) * len(views)))
    homogeneous = torch.cat([positions, positions.new_ones((len(positions), 1))], dim=1)
    inside_counts = torch.zeros(len(positions), dtype=torch.int32, device=positions.device)
    for camera, source in views:
        key = (id(camera), mask_index, str(positions.device), mask_expand, mask_threshold)
        signature = (source.data_ptr(), source._version, tuple(source.shape))
        cached = cache.get(key) if cache is not None else None
        if cached is not None and cached[0] == signature:
            mask = cached[1]
        else:
            mask = source.squeeze(0).to(device=positions.device, dtype=torch.float32)
            if mask_expand > 0:
                radius = max(1, math.ceil(mask_expand))
                mask = F.max_pool2d((mask > 0.5).float()[None, None],
                                    2 * radius + 1, stride=1, padding=radius)[0, 0]
            mask = mask > mask_threshold
            if cache is not None:
                cache[key] = (signature, mask)

        clip = homogeneous @ camera.full_proj_transform.to(positions.device)
        positive = clip[:, 3] > 0
        denominator = torch.where(positive, clip[:, 3], torch.ones_like(clip[:, 3]))
        ndc = clip[:, :2] / denominator[:, None]
        valid = positive & (ndc[:, 0] >= -1) & (ndc[:, 0] <= 1)
        valid &= (ndc[:, 1] >= -1) & (ndc[:, 1] <= 1)
        # Clamp before indexing even for invalid projections. No per-view
        # nonzero()/tensor-to-Python conditions, which synchronize CUDA.
        x = ((ndc[:, 0] * 0.5 + 0.5) * (camera.image_width - 1)).nan_to_num(0)
        y = ((ndc[:, 1] * 0.5 + 0.5) * (camera.image_height - 1)).nan_to_num(0)
        x = x.round().clamp(0, camera.image_width - 1).long()
        y = y.round().clamp(0, camera.image_height - 1).long()
        inside_counts += (valid & mask[y, x]).to(torch.int32)
    return inside_counts < threshold, threshold


@torch.no_grad()
def prune_gaussians_with_object_masks(gaussians, cameras, mask_prune_min_prop=0.5,
                                      mask_threshold=0.5, mask_prune_expand=0.0,
                                      mask_index=0, cache=None):
    prune_mask, threshold = compute_prune_mask(
        gaussians, cameras, mask_prune_min_prop, mask_threshold,
        mask_prune_expand, mask_index, cache)
    removed = int(prune_mask.sum().item())
    if removed:
        gaussians.tmp_radii = gaussians.get_xyz.new_zeros((len(prune_mask),))
        gaussians.prune_points(prune_mask)
        gaussians.tmp_radii = None
    return removed, threshold
