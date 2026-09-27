"""Model composition and inference shared by training, evaluation and rendering."""

import os
import json
import torch
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.mask_utils import cam_has_mask, object_mask


DEFAULT_MOVE_R = [[0.769, 0.242, -0.591], [-0.625, 0.090, -0.776], [-0.135, 0.966, 0.221]]
DEFAULT_MOVE_T = [[0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.0, 0.0]]


def share_exposure(model, source):
    for name in ("_exposure", "exposure_mapping", "pretrained_exposures"):
        if hasattr(source, name):
            setattr(model, name, getattr(source, name))


def merge_gaussians(gaussians):
    if not gaussians:
        raise ValueError("merge_gaussians expects at least one model")
    if len(gaussians) == 1:
        return gaussians[0]
    reference = gaussians[0]
    if any((g.max_sh_degree, g.active_sh_degree) !=
           (reference.max_sh_degree, reference.active_sh_degree) for g in gaussians):
        raise ValueError("Composed models must use the same spherical-harmonic degrees")
    merged = GaussianModel(reference.max_sh_degree)
    for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        setattr(merged, name, torch.cat([getattr(g, name) for g in gaussians], dim=0))
    merged.active_sh_degree = reference.active_sh_degree
    share_exposure(merged, reference)
    return merged


def load_object_gaussians(dataset, loaded_iter, num_objects, active_sh_degree, apply_move=False):
    models = []
    degrees = [active_sh_degree] * num_objects
    if loaded_iter is not None:
        metadata_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{loaded_iter}", "model_meta.json")
        if os.path.isfile(metadata_path):
            with open(metadata_path) as file:
                degrees = json.load(file)["object_active_sh_degrees"]
            if len(degrees) != num_objects:
                raise ValueError("Saved object metadata disagrees with the object count")
    for index in range(num_objects):
        if loaded_iter is None:
            path = dataset.obj_ply_path
        else:
            suffix = "obj.ply" if index == 0 else f"obj_{index}.ply"
            path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{loaded_iter}", suffix)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Missing object {index} PLY: {path}")
        model = GaussianModel(dataset.sh_degree)
        model.load_obj_ply(path)
        model.active_sh_degree = degrees[index]
        if apply_move:
            move(model, DEFAULT_MOVE_R, obj_idx=index)
        models.append(model)
    return models


@torch.no_grad()
def move(gaussian, R=None, translation=None, obj_idx=None):
    """Translate in the coordinate frame given by R, then return to world space."""
    if translation is None and obj_idx is not None and obj_idx < len(DEFAULT_MOVE_T):
        translation = DEFAULT_MOVE_T[obj_idx]
    if translation is None or not gaussian.get_xyz.shape[0]:
        return
    xyz = gaussian.get_xyz
    offset = torch.as_tensor(translation, dtype=xyz.dtype, device=xyz.device).reshape(3)
    if R is not None:
        rotation = torch.as_tensor(R, dtype=xyz.dtype, device=xyz.device)
        if rotation.shape == (4, 4):
            rotation = rotation[:3, :3]
        if rotation.shape != (3, 3):
            raise ValueError("Rotation must be a 3x3 or 4x4 matrix")
        offset = torch.linalg.solve(rotation, offset)
    gaussian._xyz = xyz + offset


@torch.no_grad()
def render_view(camera, gaussians, pipe, background, train_test_exp, separate_sh, mask_index=None):
    if camera is None or (mask_index is not None and not cam_has_mask(camera, mask_index)):
        return None
    pred = render(camera, gaussians, pipe, background,
                  use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
    gt = getattr(camera, "original_image", None)
    if gt is not None:
        gt = gt.to(pred.device)
    masks = [getattr(camera, "alpha_mask", None)]
    if mask_index is not None:
        masks.append(object_mask(camera, mask_index))
    for mask in masks:
        if mask is not None:
            pred = pred * mask.to(pred.device)
            if gt is not None:
                gt = gt * mask.to(gt.device)
    if train_test_exp:
        pred = pred[..., pred.shape[-1] // 2:]
        if gt is not None:
            gt = gt[..., gt.shape[-1] // 2:]
    return pred.clamp(0, 1), gt.clamp(0, 1) if gt is not None else None
