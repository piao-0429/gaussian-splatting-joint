#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np

# Ensure project root is on PYTHONPATH so sibling packages (e.g. `arguments`) can be imported
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from arguments import ModelParams, OptimizationParams
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from utils.camera_utils import cameraList_from_camInfos
from train import prune_gaussians_with_object_masks
from utils.sh_utils import SH2RGB


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
        parser.error("必须提供 --model_path，指向用于输出的目录（可与source_path相同）。")

    # 若未显式提供 source_path，则默认等于 model_path（常见布局：在同一目录下包含 sparse/ images/ 等）
    if not getattr(dataset, "source_path", None):
        dataset.source_path = dataset.model_path

    if not os.path.isdir(dataset.source_path):
        parser.error(f"source_path 不存在或不可访问: {dataset.source_path}")

    dataset.model_path = os.path.abspath(dataset.model_path)
    os.makedirs(dataset.model_path, exist_ok=True)
    args.model_path = dataset.model_path

    # 确保优化器类型与加载的模型保持一致
    opt.optimizer_type = getattr(opt, "optimizer_type", "default")

    # 将与mask相关的选项同步到args，方便后续使用
    args.mask_prune_threshold = opt.mask_prune_threshold
    args.mask_prune_min_prop = opt.mask_prune_min_prop
    args.mask_prune_blur_sigma = opt.mask_prune_blur_sigma

    if args.output_path and not os.path.isabs(args.output_path):
        args.output_path = os.path.join(dataset.model_path, args.output_path)

    return args, dataset, opt


def main():
    args, dataset, opt = parse_args()
    torch.set_grad_enabled(False)

    # 直接从 COLMAP 数据生成点云和相机，不依赖已有 Gaussian 训练输出
    scene_info = sceneLoadTypeCallbacks["Colmap"](
        dataset.source_path,
        dataset.images,
        dataset.depths,
        dataset.ft_masks,
        dataset.eval,
        dataset.train_test_exp,
    )

    # 收集相机，用于 mask 可见性裁剪
    cam_infos = []
    if scene_info.train_cameras:
        cam_infos.extend(scene_info.train_cameras)
    if scene_info.test_cameras:
        cam_infos.extend(scene_info.test_cameras)
    if scene_info.finetune_cameras:
        cam_infos.extend(scene_info.finetune_cameras)

    cameras = []
    if scene_info.train_cameras:
        cameras += cameraList_from_camInfos(scene_info.train_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, False)
    if scene_info.test_cameras:
        cameras += cameraList_from_camInfos(scene_info.test_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, True)
    if scene_info.finetune_cameras:
        cameras += cameraList_from_camInfos(scene_info.finetune_cameras, 1.0, dataset, scene_info.is_nerf_synthetic, False)

    if not cameras:
        print("未找到任何相机，确认 source_path 下的 sparse/0 与 images/ 是否完整。")
        return

    # 明确告知是否加载到了掩码，方便定位 mask 配置问题
    mask_paths = []
    for c in cam_infos:
        paths = getattr(c, "mask_paths", []) or []
        mask_paths.extend([p for p in paths if p])
    cams_with_masks = [c for c in cameras if any(m is not None for m in (getattr(c, "object_masks", []) or []))]
    print(f"检测到相机 {len(cameras)} 个，其中已加载掩码 {len(cams_with_masks)} 个")
    if dataset.ft_masks and not mask_paths:
        print(f"[WARN] 指定了 --ft_masks={dataset.ft_masks}，但没有找到匹配的掩码文件。请确认目录/文件名与图像一致。")
    elif len(cams_with_masks) == 0:
        print("[WARN] 未加载到任何掩码，将不会执行基于掩码的点云裁剪。")
    elif len(cams_with_masks) < len(cameras):
        missing = [c.image_name for c in cam_infos if not any((getattr(c, "mask_paths", []) or []))]
        preview = ", ".join(missing[:5])
        more = "" if len(missing) <= 5 else f" 等 {len(missing)} 张"
        print(f"[INFO] 部分视角缺少掩码，示例: {preview}{more}")

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gaussians.create_from_pcd(scene_info.point_cloud, cam_infos, scene_info.nerf_normalization["radius"])
    gaussians.training_setup(opt)

    total_points = gaussians.get_xyz.shape[0]
    pruned, used_th = prune_gaussians_with_object_masks(
        gaussians,
        cameras,
        mask_prune_min_prop=args.mask_prune_min_prop,
        mask_threshold=args.mask_prune_threshold,
        mask_blur_sigma=args.mask_prune_blur_sigma,
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