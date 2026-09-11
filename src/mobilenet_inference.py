#!/usr/bin/env python3
"""Pretrained MobileNet V2 inference; the generic classifier does the heavy lifting."""
import argparse, json
from pathlib import Path
from classifier_core import classify

root = Path(__file__).parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--model", default=str(root / "models" / "mobilenet_v2_1.0_224_quant.tflite"))
parser.add_argument("--labels", default=str(root / "models" / "labels.txt"))
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()
print(json.dumps(classify(args.model, args.image, args.labels, args.top_k), indent=2))
