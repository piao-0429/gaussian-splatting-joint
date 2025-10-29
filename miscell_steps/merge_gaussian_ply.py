# 合并两个 binary_little_endian PLY（高斯点云），假定每个顶点所有属性均为 float
# 用法: python merge_gaussian_ply.py a.ply b.ply -o merged.ply

import argparse
import sys
from typing import List, Tuple
import numpy as np


def read_ply_binary_floats(path: str) -> Tuple[List[str], int, List[str], np.ndarray]:
    """
    读取 binary_little_endian PLY 的 header 并把顶点属性作为 float32 矩阵返回
    返回: (header_lines, vertex_count, property_names, data_array shape=(vertex_count, n_props))
    """
    header_lines: List[str] = []
    with open(path, "rb") as f:
        # 逐行读取 header（保留换行）
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"未找到 end_header: {path}")
            try:
                line_str = line.decode("ascii")
            except UnicodeDecodeError:
                raise ValueError(f"非 ASCII header 或文件已损坏: {path}")
            header_lines.append(line_str)
            if line_str.strip() == "end_header":
                break

        # 解析 header 找到 element vertex 和 property 列表
        vertex_count = None
        prop_names: List[str] = []
        in_vertex = False
        for ln in header_lines:
            parts = ln.strip().split()
            if len(parts) == 0:
                continue
            if parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if parts[0] == "element" and parts[1] != "vertex":
                # 进入其它 element，则不再读取 vertex property
                in_vertex = False
            if in_vertex and parts[0] == "property":
                # 假定格式: property float name
                if len(parts) >= 3:
                    prop_names.append(parts[-1])

        if vertex_count is None:
            raise ValueError("header 中未找到 element vertex 行")

        # 读取顶点二进制数据：float32 little endian
        # 按理说紧接 header 到文件末尾就是顶点数据（如果文件只包含 vertex element）
        # 读取剩余全部二进制数据
        data_bytes = f.read()

    nprops = len(prop_names)
    expected_floats = vertex_count * nprops
    # interpret as little-endian float32
    try:
        floats = np.frombuffer(data_bytes, dtype="<f4", count=expected_floats)
    except Exception as e:
        raise ValueError(f"读取二进制数据失败: {e}")

    if floats.size < expected_floats:
        raise ValueError(
            f"文件 {path} 缺少数据: 期望 {expected_floats} floats，读取到 {floats.size}"
        )
    # reshape
    arr = floats.reshape((vertex_count, nprops)).copy()  # copy 保证连续可写
    return header_lines, vertex_count, prop_names, arr


def write_ply_binary_floats(
    path: str, header_lines: List[str], prop_names: List[str], data: np.ndarray
) -> None:
    """
    写出 binary_little_endian PLY。header_lines 会被修改：element vertex 的数量替换为 data.shape[0]
    data 必须是 float32 或可转换为 float32，且列数等于 prop_names 长度
    """
    vertex_count = data.shape[0]
    nprops = data.shape[1]
    if nprops != len(prop_names):
        raise ValueError("属性列数与 prop_names 长度不匹配")

    # 更新 header 中的 element vertex 行
    out_header_lines = []
    updated = False
    for ln in header_lines:
        parts = ln.strip().split()
        if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
            out_header_lines.append(f"element vertex {vertex_count}\n")
            updated = True
        else:
            out_header_lines.append(ln)
    if not updated:
        # 如果没有找到，插入一个新的 element vertex 行（放在 end_header 之前）
        for i, ln in enumerate(out_header_lines):
            if ln.strip() == "end_header":
                out_header_lines.insert(i, f"element vertex {vertex_count}\n")
                updated = True
                break
    if not updated:
        raise ValueError("无法写入 header: 未找到 end_header 位置")

    # 确保以 binary_little_endian 1.0 标明（如果 header 中没写则写入）
    has_format = any("format " in ln for ln in out_header_lines)
    if not has_format:
        out_header_lines.insert(0, "format binary_little_endian 1.0\n")

    # 写文件
    with open(path, "wb") as f:
        for ln in out_header_lines:
            f.write(ln.encode("ascii"))
        # 保证按 little-endian float32 写入
        arr = data.astype("<f4", copy=False)
        f.write(arr.tobytes())


def merge_ply_files(a: str, b: str, out: str) -> None:
    a_header, a_n, a_props, a_data = read_ply_binary_floats(a)
    b_header, b_n, b_props, b_data = read_ply_binary_floats(b)

    if a_props != b_props:
        # 更严格：顺序和名称必须一致
        raise ValueError("两个 PLY 的 property 列不一致，无法合并")

    merged = np.vstack([a_data, b_data])
    # 采用 a 的 header 作为模板（保留 comment 等其他字段），但修改 vertex 数
    write_ply_binary_floats(out, a_header, a_props, merged)


def main():
    parser = argparse.ArgumentParser(description="合并两个 binary_little_endian PLY（float 属性）")
    parser.add_argument("ply_a", help="第一个 ply 文件路径")
    parser.add_argument("ply_b", help="第二个 ply 文件路径")
    parser.add_argument("-o", "--out", required=True, help="输出合并后的 ply 文件路径")
    args = parser.parse_args()

    try:
        merge_ply_files(args.ply_a, args.ply_b, args.out)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()