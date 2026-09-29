#!/usr/bin/env python3
"""Propagate SAM2 point/box prompts and export per-object binary PNG masks.

Run in a separate environment with SAM2 installed. Frame filenames must be
numeric JPEGs; scripts/sam2_frame_mapping.py prepares and restores their names.
"""

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

from PIL import Image


def read_inputs(frames_dir, prompts_path):
    if not frames_dir.is_dir():
        raise ValueError(f"Frame directory does not exist: {frames_dir}")
    frames = [p for p in frames_dir.iterdir()
              if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    if not frames or any(not p.stem.isascii() or not p.stem.isdecimal() for p in frames):
        raise ValueError("Use numeric JPEG frame names, e.g. 00000.jpg; run sam2_frame_mapping.py prepare first")
    frames.sort(key=lambda p: int(p.stem))
    if len({int(p.stem) for p in frames}) != len(frames):
        raise ValueError("JPEG frames have duplicate numeric identifiers")
    sizes = set()
    for path in frames:
        with Image.open(path) as image:
            sizes.add(image.size)
            image.verify()
    if len(sizes) != 1:
        raise ValueError("SAM2 video frames must have the same dimensions")
    width, height = next(iter(sizes))

    prompts = json.loads(prompts_path.read_text(encoding="utf-8-sig"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompts must be a nonempty JSON list")
    frame_indices = {p.name: i for i, p in enumerate(frames)}
    validated = []
    used = set()
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("Each prompt must be a JSON object")
        if set(prompt) - {"frame", "object_id", "points", "labels", "box"}:
            raise ValueError(f"Unknown prompt fields: {set(prompt) - {'frame', 'object_id', 'points', 'labels', 'box'}}")
        frame = prompt.get("frame")
        obj_id = prompt.get("object_id")
        if not isinstance(frame, str) or frame not in frame_indices:
            raise ValueError(f"Prompt references an unknown frame: {frame!r}")
        if type(obj_id) is not int or obj_id < 1:
            raise ValueError("object_id must be a positive integer")
        key = (frame, obj_id)
        if key in used:
            raise ValueError(f"Combine all clicks for the same frame/object into one prompt: {key}")
        used.add(key)

        points, labels, box = prompt.get("points"), prompt.get("labels"), prompt.get("box")
        if points is None and box is None:
            raise ValueError(f"Provide points with labels, a box, or both: {key}")
        if (points is None) != (labels is None):
            raise ValueError(f"points and labels must be provided together: {key}")
        if points is not None:
            if (not isinstance(points, list) or not points or not isinstance(labels, list)
                    or len(points) != len(labels)):
                raise ValueError(f"points and labels must be nonempty lists of equal length: {key}")
            for point, label in zip(points, labels):
                if (not isinstance(point, list) or len(point) != 2
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)
                        or not 0 <= point[0] < width or not 0 <= point[1] < height):
                    raise ValueError(f"Point must be a finite [x, y] inside the image: {point!r}")
                if type(label) is not int or label not in (0, 1):
                    raise ValueError("Point labels must be 1 (foreground) or 0 (background)")
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= width
                    or not 0 <= box[1] < box[3] <= height):
                raise ValueError(f"Box must be [x_min, y_min, x_max, y_max] inside the image: {box!r}")
        validated.append(dict(prompt, frame_index=frame_indices[frame]))
    return frames, validated, (width, height)


def export_masks(predictor, state, frames, prompts, size, output_dir):
    """Stream PNGs to disk, covering frames before the first annotation too."""
    import numpy as np

    for prompt in prompts:
        kwargs = {"inference_state": state, "frame_idx": prompt["frame_index"],
                  "obj_id": prompt["object_id"]}
        if prompt.get("points") is not None:
            kwargs.update(points=np.asarray(prompt["points"], dtype=np.float32),
                          labels=np.asarray(prompt["labels"], dtype=np.int32))
        if prompt.get("box") is not None:
            kwargs["box"] = np.asarray(prompt["box"], dtype=np.float32)
        predictor.add_new_points_or_box(**kwargs)

    output_dir.mkdir(parents=True)
    first_prompt = min(p["frame_index"] for p in prompts)
    object_ids = {p["object_id"] for p in prompts}
    seen = set()
    empty_count = 0
    for reverse in ([False, True] if first_prompt > 0 else [False]):
        for frame_idx, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=first_prompt, reverse=reverse):
            if not 0 <= frame_idx < len(frames):
                raise RuntimeError(f"SAM2 returned an invalid frame index: {frame_idx}")
            for i, obj_id in enumerate(obj_ids):
                obj_id = int(obj_id)
                key = (frame_idx, obj_id)
                if obj_id not in object_ids:
                    raise RuntimeError(f"SAM2 returned an unexpected object: {obj_id}")
                if key in seen:
                    continue
                mask = (logits[i] > 0).cpu().numpy().squeeze()
                if mask.shape != (size[1], size[0]):
                    raise RuntimeError(f"Unexpected mask shape {mask.shape} for {frames[frame_idx].name}")
                folder = output_dir / f"obj{obj_id:02d}"
                folder.mkdir(exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(folder / f"{frames[frame_idx].stem}.png")
                empty_count += int(not mask.any())
                seen.add(key)
    expected = len(frames) * len(object_ids)
    if len(seen) != expected:
        raise RuntimeError(f"Incomplete propagation: saved {len(seen)} / {expected} masks; inspect the partial output")
    return len(seen), empty_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path, help="JSON list of frame/object point or box prompts")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for objXX/*.png")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--offload-video-to-cpu", action="store_true", help="Store input frames in CPU memory")
    args = parser.parse_args()
    try:
        frames, prompts, size = read_inputs(args.frames_dir, args.prompts)
        if args.output_dir.exists() or args.output_dir.is_symlink():
            raise ValueError(f"Output already exists; use a new directory: {args.output_dir}")
        if not args.checkpoint.is_file():
            raise ValueError(f"Checkpoint not found: {args.checkpoint}")
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; check the SAM2 environment or pass --device cpu")
        predictor = build_sam2_video_predictor(args.config, str(args.checkpoint), device=args.device)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if args.device == "cuda" and torch.cuda.is_bf16_supported() else nullcontext())
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(args.frames_dir),
                                         offload_video_to_cpu=args.offload_video_to_cpu)
            count, empty_count = export_masks(predictor, state, frames, prompts, size, args.output_dir)
        print(f"Saved {count} masks for {len(frames)} frames to {args.output_dir}")
        print(f"Empty masks: {empty_count}. Review tracking and occlusions before using masks for pruning.")
    except ImportError as error:
        parser.exit(2, f"Missing dependency: {error}. Run in the SAM2 environment described in README.md.\n")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
