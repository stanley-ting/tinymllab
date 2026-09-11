#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from classifier_core import classify

parser = argparse.ArgumentParser(description="Classify one image with any Lab 1 .tflite model.")
parser.add_argument("--model", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--labels", default=str(Path(__file__).parents[1] / "models" / "labels.txt"))
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()
print(json.dumps(classify(args.model, args.image, args.labels, args.top_k), indent=2))
