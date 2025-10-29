#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser
import torch
import numpy as np

from arguments import ModelParams, OptimizationParams
from scene import Scene, GaussianModel
from train import prune_gaussians_with_object_masks
from utils.sh_utils import SH2RGB
from scene.dataset_readers import storePly


def parse_args():
    parser = ArgumentParser(description="Use training pipeline masks to filter COLMAP sparse points")
    lp = ModelParams(parser)
    opt_group = OptimizationParams(parser)

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="保存过滤后点云的目录（默认保存到模型目录下）",
    )
    parser.add_argument(
        "--load_iteration",
        type=int,
        default=None,
        help="加载指定迭代的模型（默认加载latest）",
    )
    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = opt_group.extract(args)

    if not getattr(dataset, "model_path", None):
        parser.error("必须提供 --model_path，指向已有的训练输出目录。")

    dataset.model_path = os.path.abspath(dataset.model_path)
    os.makedirs(dataset.model_path, exist_ok=True)
    args.model_path = dataset.model_path

    # 确保优化器类型与加载的模型保持一致
    opt.optimizer_type = getattr(opt, "optimizer_type", "default")

    # 将与mask相关的选项同步到args，方便后续使用
    args.mask_prune_threshold = opt.mask_prune_threshold
    args.mask_prune_min_views = opt.mask_prune_min_views
    args.mask_prune_blur_sigma = opt.mask_prune_blur_sigma

    if args.output_path and not os.path.isabs(args.output_path):
        args.output_path = os.path.join(dataset.model_path, args.output_path)

    return args, dataset, opt


def main():
    args, dataset, opt = parse_args()
    torch.set_grad_enabled(False)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.load_iteration)
    gaussians.training_setup(opt)
    cameras = scene.getTrainCameras()

    if not cameras:
        print("未找到任何训练相机，确认数据配置是否正确。")
        return

    total_points = gaussians.get_xyz.shape[0]
    pruned = prune_gaussians_with_object_masks(
        scene,
        min_visible_views=args.mask_prune_min_views,
        mask_threshold=args.mask_prune_threshold,
        mask_blur_sigma=args.mask_prune_blur_sigma,
        cameras=cameras,
    )
    kept = gaussians.get_xyz.shape[0]

    # 默认保存到模型目录下
    output_dir = args.output_path if args.output_path else os.path.join(dataset.model_path, "cut_output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 保存兼容COLMAP的PLY，方便后续重新训练或SIBR查看
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    dc = gaussians.get_features_dc.detach().squeeze(1)
    rgb = SH2RGB(dc).clamp(0.0, 1.0)
    rgb = (rgb * 255.0).round().to(torch.uint8).cpu().numpy().astype(np.uint8)
    colmap_ply_path = os.path.join(output_dir, "point3D_masked.ply")
    storePly(colmap_ply_path, xyz, rgb)

    # 额外保留高斯格式，便于调试或进一步处理
    gaussian_ply_path = os.path.join(output_dir, "point3D_masked_gaussian.ply")
    gaussians.save_ply(gaussian_ply_path)

    print(f"原始点数: {total_points}, 移除点数: {pruned}, 保留点数: {kept}")
    print(f"COLMAP格式: {colmap_ply_path}")
    print(f"Gaussian格式: {gaussian_ply_path}")


if __name__ == "__main__":
    main()