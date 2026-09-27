"""Common CLI implementation for composed and object-only inference."""

import os
import re
from argparse import ArgumentParser

import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.joint_utils import load_object_gaussians, merge_gaussians, render_view, share_exposure


def parse_include_objects(value, count):
    if value is None:
        return [True] * count
    parts = [part for part in re.split(r"[,;\s]+", value.strip()) if part]
    if len(parts) == 1 and len(parts[0]) == count and all(c in "01" for c in parts[0]):
        parts = list(parts[0])
    if any(part not in ("0", "1") for part in parts) or len(parts) > count:
        raise ValueError("include_objects must contain one 0/1 flag per object")
    return [part == "1" for part in parts] + [True] * (count - len(parts))


def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.utils.save_image(tensor, path)


@torch.no_grad()
def main(objects_only=False):
    parser = ArgumentParser(description="Render independent or composed Gaussian models")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", choices=["train", "test", "finetune", "all"], default="all")
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--merge_objects", action="store_true")
    parser.add_argument("--include_objects", default=None)
    parser.add_argument("--mask_index", default=None, type=int)
    parser.add_argument("--no_move", action="store_true", help="Disable the existing demo object translations")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset, pipe = lp.extract(args), pp.extract(args)
    scene = Scene(dataset, GaussianModel(dataset.sh_degree), load_iteration=args.iteration, shuffle=False)
    merged = None
    if args.merge_objects or objects_only:
        models = load_object_gaussians(dataset, scene.loaded_iter, scene.num_objects,
                                       scene.gaussians.active_sh_degree, apply_move=not args.no_move)
        for model in models:
            share_exposure(model, scene.gaussians)
        include = parse_include_objects(args.include_objects, len(models))
        selected = [model for model, use in zip(models, include) if use]
        if objects_only and not selected:
            raise ValueError("Object-only rendering requires at least one selected object")
        merged = merge_gaussians(selected if objects_only else [scene.gaussians] + selected)

    background = torch.tensor([1., 1., 1.] if dataset.white_background else [0., 0., 0.], device="cuda")
    root = args.output_root or os.path.join(dataset.model_path, "render")
    splits = [("train", scene.getTrainCameras()), ("test", scene.getTestCameras()),
              ("finetune", scene.getFinetuneCameras())]
    for name, cameras in splits:
        if args.split not in (name, "all"):
            continue
        for camera in tqdm(cameras, desc=f"Render {name}", leave=False):
            model = merged if objects_only else scene.gaussians
            result = render_view(camera, model, pipe, background, dataset.train_test_exp, False, args.mask_index)
            if result is None:
                continue
            filename = camera.image_name.replace("/", "_").replace("\\", "_") + ".png"
            save_image(result[0], os.path.join(root, name, filename))
            # Object-only mode already rendered this exact model. Write one
            # output per view instead of repeating both rasterization and I/O.
            if not objects_only and merged is not None:
                combined = render_view(camera, merged, pipe, background, dataset.train_test_exp, False)
                save_image(combined[0], os.path.join(root, name + "_merged", filename))
    print("Rendering finished.")
