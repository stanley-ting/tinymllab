#!/usr/bin/env python3
"""Create Q4 robustness variants: rotation, overexposure, blur and partial occlusion."""
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("input", type=Path)
parser.add_argument("--output-dir", type=Path, default=Path("images/q4_variants"))
args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
image = Image.open(args.input).convert("RGB")
stem = args.input.stem
variants = {
    "rotation45": image.rotate(45, expand=False, fillcolor=(0, 0, 0)),
    "overexposure": ImageEnhance.Brightness(image).enhance(1.8),
    "blur": image.filter(ImageFilter.GaussianBlur(radius=4)),
}
occluded = image.copy(); draw = ImageDraw.Draw(occluded)
w, h = image.size; draw.rectangle((w // 3, h // 3, 2 * w // 3, 2 * h // 3), fill=(0, 0, 0))
variants["partial_occlusion"] = occluded
for name, variant in variants.items():
    out = args.output_dir / f"{stem}_{name}.jpg"; variant.save(out, quality=95); print(out)
