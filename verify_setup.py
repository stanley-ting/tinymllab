"""Check the Python packages and inspect the pretrained model, if present."""
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError as exc:
    raise SystemExit("Install requirements first: python -m pip install -r requirements.txt") from exc

MODEL = Path(__file__).parent / "models" / "mobilenet_v2_1.0_224_quant.tflite"
print("NumPy:", np.__version__)
print("Pillow: available")
if not MODEL.exists():
    print(f"Model not found yet: {MODEL}")
    raise SystemExit(0)

interpreter = Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print("Model:", MODEL.name)
print("Input:", inp["shape"], inp["dtype"].__name__, "quantization=", inp["quantization"])
print("Output:", out["shape"], out["dtype"].__name__, "quantization=", out["quantization"])
