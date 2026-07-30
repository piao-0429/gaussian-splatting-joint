#!/usr/bin/env python3
import argparse
from PIL import Image
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--mask', required=True)
parser.add_argument('--out', default=None)
args = parser.parse_args()

img_path = args.image
mask_path = args.mask

img = Image.open(img_path).convert('RGB')
mask = Image.open(mask_path).convert('L')

# resize mask to image if needed
if mask.size != img.size:
    mask = mask.resize(img.size, resample=Image.NEAREST)

img_np = np.array(img)
mask_np = np.array(mask)

# consider mask>127 as object (1), else background (0)
obj = mask_np > 127

# create white background
white = np.ones_like(img_np, dtype=img_np.dtype) * 255

out_np = img_np.copy()
out_np[~obj] = white[~obj]

out_img = Image.fromarray(out_np)

if args.out:
    out_path = args.out
else:
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(os.path.dirname(img_path), base + '_masked.png')

out_img.save(out_path)
print('Saved masked image to', out_path)
