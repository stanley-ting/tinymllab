"""Shared preprocessing and inference for uint8, int8 and float32 TFLite models."""
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


def load_labels(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def _quantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero = detail["quantization"]
    if not scale:
        return values
    return np.round(values / scale + zero).clip(
        np.iinfo(detail["dtype"]).min, np.iinfo(detail["dtype"]).max
    ).astype(detail["dtype"])


def _dequantize(values: np.ndarray, detail: dict) -> np.ndarray:
    scale, zero = detail["quantization"]
    return (values.astype(np.float32) - zero) * scale if scale else values.astype(np.float32)


def classify(model_path: str | Path, image_path: str | Path, labels_path: str | Path | None = None, top_k: int = 5) -> dict:
    interpreter = Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    height, width = int(inp["shape"][1]), int(inp["shape"][2])
    image = Image.open(image_path).convert("RGB").resize((width, height))
    pixels = np.asarray(image, dtype=np.float32)[None, ...]
    if inp["dtype"] == np.float32:
        pixels = pixels / 127.5 - 1.0  # MobileNet-style float input range.
    else:
        pixels = _quantize(pixels, inp)
    interpreter.set_tensor(inp["index"], pixels)
    start = perf_counter()
    interpreter.invoke()
    elapsed_ms = (perf_counter() - start) * 1000
    scores = _dequantize(interpreter.get_tensor(out["index"]), out).reshape(-1)
    # Quantized classifiers often emit logits; softmax makes confidence comparable.
    exp_scores = np.exp(scores - scores.max())
    probabilities = exp_scores / exp_scores.sum()
    indices = np.argsort(probabilities)[::-1][:top_k]
    labels = load_labels(labels_path) if labels_path else []
    predictions = [{"index": int(i), "label": labels[i] if i < len(labels) else str(i), "confidence": float(probabilities[i])} for i in indices]
    return {"model": str(model_path), "image": str(image_path), "input_shape": list(inp["shape"]), "input_dtype": inp["dtype"].__name__, "output_dtype": out["dtype"].__name__, "inference_ms": elapsed_ms, "predictions": predictions}
