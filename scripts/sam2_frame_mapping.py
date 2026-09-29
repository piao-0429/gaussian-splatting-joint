#!/usr/bin/env python3
"""Prepare numbered SAM2 frames and restore mask names without changing inputs.

Only Pillow and the Python standard library are required; this script uses no GPU.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def natural_key(name):
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", name)), name


def plain_filename(name):
    return (isinstance(name, str) and name not in {"", ".", ".."}
            and "/" not in name and "\\" not in name
            and not any(ord(char) < 32 for char in name))


def require_new(path, label):
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label}已存在，请指定新路径，避免覆盖：{path}")


def require_separate(source, target):
    if source == target or source in target.parents or target in source.parents:
        raise ValueError(f"输入与输出目录不能相同或互相包含：{source} / {target}")


def unique_json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"映射中存在重复键：{key}")
        result[key] = value
    return result


def prepare(args):
    source = Path(args.images_dir).expanduser().resolve()
    frames_dir = Path(args.frames_dir).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"图像目录不存在：{source}")
    require_separate(source, frames_dir)
    require_new(frames_dir, "数字帧目录")
    require_new(manifest, "映射文件")
    if manifest == frames_dir or frames_dir in manifest.parents or manifest in frames_dir.parents:
        raise ValueError("映射文件应放在数字帧目录之外，例如 sam2_work/frame_map.json")

    if args.frame_list:
        names = [line.strip() for line in Path(args.frame_list).expanduser().read_text(
            encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        names = sorted((path.name for path in source.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                       key=natural_key)
    if not names:
        raise ValueError("没有可处理的图片；输入应为平铺的图片目录")
    if len(names) != len(set(names)):
        raise ValueError("帧列表有重复文件名")

    stems = set()
    formats = {}
    # Validate all inputs before creating outputs, including ambiguous mask names.
    for name in names:
        if not plain_filename(name) or Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"不是支持的图片文件名：{name!r}")
        if Path(name).stem in stems:
            raise ValueError(f"去掉扩展名后有重名，无法唯一恢复掩码：{name}")
        stems.add(Path(name).stem)
        with Image.open(source / name) as image:
            formats[name] = image.format
            image.verify()

    mapping = {f"{index:05d}.jpg": name for index, name in enumerate(names)}
    frames_dir.mkdir(parents=True)
    for frame_name, name in mapping.items():
        target = frames_dir / frame_name
        if formats[name] == "JPEG":
            shutil.copy2(source / name, target)
        else:
            with Image.open(source / name) as image:
                image.convert("RGB").save(target, format="JPEG", quality=95, subsampling=0)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"已准备 {len(mapping)} 张数字帧：{frames_dir}")
    print(f"帧名映射：{manifest}")
    print("原图未修改。请将 SAM2 的 video_dir 设为数字帧目录。")


def restore(args):
    source = Path(args.masks_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"掩码目录不存在：{source}")
    require_separate(source, output)
    require_new(output, "恢复后的掩码目录")
    with Path(args.manifest).expanduser().open(encoding="utf-8-sig") as file:
        mapping = json.load(file, object_pairs_hook=unique_json_pairs)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("映射必须是非空 JSON 对象：数字帧名 -> 原图文件名")

    by_stem = {}
    original_stems = set()
    for frame, original in mapping.items():
        if not re.fullmatch(r"[0-9]+\.(?:jpg|jpeg)", frame, flags=re.IGNORECASE):
            raise ValueError(f"映射中的数字帧名不合法：{frame!r}")
        if not plain_filename(original) or Path(original).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"映射中的原图文件名不合法：{original!r}")
        stem = Path(frame).stem
        original_stem = Path(original).stem
        if stem in by_stem or original_stem in original_stems:
            raise ValueError(f"映射不是一一对应，存在重复主体名：{frame} -> {original}")
        by_stem[stem] = original_stem
        original_stems.add(original_stem)

    plan = []
    destinations = set()
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.suffix.lower() != ".png":
            raise ValueError(f"掩码应为 PNG，请使用 SAM2 的 separate 导出目录：{path}")
        if path.suffix.lower() != ".png":
            continue
        if path.stem not in by_stem:
            raise ValueError(f"掩码找不到对应数字帧，未写入任何输出：{path}")
        target = output / path.relative_to(source).parent / f"{by_stem[path.stem]}.png"
        if target in destinations:
            raise ValueError(f"多个掩码将写入同一文件：{target}")
        destinations.add(target)
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"掩码文件内容不是 PNG：{path}")
            image.verify()
        plan.append((path, target))
    if not plan:
        raise ValueError(f"没有找到 PNG 掩码：{source}")

    output.mkdir(parents=True)
    for path, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f"已恢复 {len(plan)} 个掩码文件名：{output}")
    print("物体子目录和掩码内容保持不变；没有为缺失视角生成全黑掩码。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="复制/转换为数字 JPEG 帧并保存映射")
    prep.add_argument("--images-dir", required=True, help="原始物体图片目录，例如 images_ft")
    prep.add_argument("--frames-dir", required=True, help="新的 SAM2 数字帧目录")
    prep.add_argument("--manifest", required=True, help="新的 JSON 映射文件，放在帧目录之外")
    prep.add_argument("--frame-list", help="可选：每行一个原文件名，按指定顺序选择帧；默认自然排序")
    prep.set_defaults(run=prepare)
    back = commands.add_parser("restore", help="按映射复制掩码，恢复原图主体名")
    back.add_argument("--masks-dir", required=True, help="SAM2 的 masks_bw/separate 目录，或单物体 PNG 目录")
    back.add_argument("--manifest", required=True, help="prepare 保存的 JSON 映射文件")
    back.add_argument("--output-dir", required=True, help="新的输出目录，例如 masks_ft_obj")
    back.set_defaults(run=restore)
    args = parser.parse_args()
    try:
        args.run(args)
    except (OSError, ValueError) as error:
        parser.exit(2, f"错误：{error}\n")


if __name__ == "__main__":
    main()
