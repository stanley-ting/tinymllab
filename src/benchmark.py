#!/usr/bin/env python3
"""Collect Q3/Q5 model size, latency and top-k confidence in CSV or JSON."""
import argparse, csv, json, sys
from pathlib import Path
from classifier_core import classify

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", required=True, help="Repeat for each model.")
parser.add_argument("--image", required=True)
parser.add_argument("--labels", default=str(Path(__file__).parents[1] / "models" / "labels.txt"))
parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--output", default="-", help="CSV output path, or - for stdout.")
args = parser.parse_args()
rows = []
for model in args.model:
    samples = [classify(model, args.image, args.labels, 3) for _ in range(max(1, args.runs))]
    result = samples[-1]
    result["inference_ms_mean"] = sum(s["inference_ms"] for s in samples) / len(samples)
    result["inference_ms_samples"] = args.runs
    result["model_size_bytes"] = Path(model).stat().st_size
    result["top1_label"] = result["predictions"][0]["label"]
    result["top1_confidence"] = result["predictions"][0]["confidence"]
    result["top3"] = " | ".join(f"{p['label']} ({p['confidence']:.4f})" for p in result["predictions"])
    rows.append(result)
fields = ["model", "input_shape", "input_dtype", "output_dtype", "model_size_bytes", "inference_ms_mean", "top1_label", "top1_confidence", "top3", "image"]
if args.output == "-":
    csv.DictWriter(sys.stdout, fieldnames=fields).writeheader(); csv.DictWriter(sys.stdout, fieldnames=fields).writerows(rows)
else:
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {args.output}")
