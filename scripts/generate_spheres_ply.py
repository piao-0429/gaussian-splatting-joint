#!/usr/bin/env python3
"""
生成随机球体点云并保存为 PLY（与项目中读取 Colmap-like PLY 的格式兼容）。
用法: 直接运行脚本会生成几个示例球并保存到当前目录下的 'generated_spheres.ply'

包含函数:
- make_random_points_in_sphere(center, radius, n_points, mode='volume'|'surface', color=None)
    返回 (points, colors)

- save_point_cloud_as_ply(path, points, colors, normals=None)
    将点云保存为 PLY，属性为 x,y,z,nx,ny,nz,red,green,blue（与仓库中 `fetchPly` / `storePly` 格式一致）

"""
import numpy as np
from plyfile import PlyData, PlyElement
import os


def make_random_points_in_sphere(center, radius, n_points, mode='volume', color=None):
    """
    生成在球体内或球面上的均匀随机点。

    Args:
        center: iterable of length 3 (x,y,z)
        radius: float
        n_points: int
        mode: 'volume' 或 'surface'。'volume' 在球体内部均匀采样，'surface' 在球面上均匀采样。
        color: None 或长度为3的 RGB 值（0-255）。如果为 None，则随机颜色。

    Returns:
        points: (N,3) ndarray of float32
        colors: (N,3) ndarray of uint8
    """
    center = np.asarray(center, dtype=np.float32).reshape(3)
    if mode == 'volume':
        # 使用球体内均匀采样：立方体采样+球坐标拒绝采样效率低，但对中等点数足够
        # 更高效的方法：采样半径 r ~ U(0,1)^(1/3) * R，方向均匀
        u = np.random.rand(n_points)
        r = radius * np.cbrt(u)
        # 方向均匀采样
        phi = np.arccos(1 - 2 * np.random.rand(n_points))
        theta = 2 * np.pi * np.random.rand(n_points)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        points = np.stack([x, y, z], axis=1) + center
    elif mode == 'surface':
        # 球面均匀采样
        phi = np.arccos(1 - 2 * np.random.rand(n_points))
        theta = 2 * np.pi * np.random.rand(n_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points = np.stack([x, y, z], axis=1) + center
    else:
        raise ValueError("mode must be 'volume' or 'surface'")

    if color is None:
        # 随机颜色 0-255
        colors = (np.random.rand(n_points, 3) * 255).astype(np.uint8)
    else:
        c = np.asarray(color, dtype=np.int32).reshape(3)
        colors = np.tile(c, (n_points, 1)).astype(np.uint8)

    return points.astype(np.float32), colors


def save_point_cloud_as_ply(path, points, colors, normals=None):
    """
    保存为 PLY，属性为 x,y,z,nx,ny,nz,red,green,blue（与本工程中 storePly/fetchPly 格式兼容）
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    N = points.shape[0]
    if normals is None:
        normals = np.zeros((N, 3), dtype=np.float32)
    else:
        normals = normals.astype(np.float32)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(N, dtype=dtype)
    attrs = np.concatenate((points.astype(np.float32), normals.astype(np.float32), colors.astype(np.uint8)), axis=1)
    elements[:] = list(map(tuple, attrs))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(path)


if __name__ == '__main__':
    # 示例：生成三个球体并合并
    all_points = []
    all_colors = []
    centers = [(-1.551702, 1.743503, 1.011794), (1.5935650, 0.392071, 2.111832),]
    radii = [0.7, 1.1]
    counts = [7000, 7000]
    modes = ['volume', 'volume']
    colors = [(255, 0, 0), (0, 255, 0)]

    for c, r, n, m, col in zip(centers, radii, counts, modes, colors):
        pts, cols = make_random_points_in_sphere(c, r, n, mode=m, color=col)
        all_points.append(pts)
        all_colors.append(cols)

    all_points = np.vstack(all_points)
    all_colors = np.vstack(all_colors)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'generated_spheres.ply')
    out_path = os.path.abspath(out_path)
    save_point_cloud_as_ply(out_path, all_points, all_colors)
    print(f"Saved combined point cloud to {out_path}")
