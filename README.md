# TinyML Lab 1

Run commands from this directory on the Raspberry Pi.

## First setup

```bash
cd ~/TINYML/Lab1
chmod +x setup.sh src/*.py
./setup.sh
source .venv/bin/activate
python verify_setup.py
```

## File map

| Lab part | Files | Command / action |
|---|---|---|
| Environment | `setup.sh`, `requirements.txt`, `verify_setup.py` | `./setup.sh`, then `python verify_setup.py` |
| Pretrained MobileNet V2 | `src/mobilenet_inference.py` | `python src/mobilenet_inference.py images/test.jpg` |
| Generic custom inference | `src/classify.py` | `python src/classify.py --model models/custom_160_int8.tflite --image images/test.jpg` |
| Q3 comparison | `src/benchmark.py`, `results/q3/` | Benchmark all three exports after Edge Impulse download |
| Q4 transformations | `src/augment_images.py`, `results/q4/` | `python src/augment_images.py images/test.jpg --output-dir results/q4/variants` |
| Q5 int8 vs float32 | `src/benchmark.py`, `results/q5/` | Benchmark both exports on the same images |
| Written answers/evidence | `results/RESULTS_TEMPLATE.md` | Fill in measurements, screenshots and observations |

## Edge Impulse work

Dataset upload, impulse creation, training, testing and model export still happen in Edge Impulse Studio. Put the downloaded `.tflite` exports in `models/` with descriptive names such as `custom_160_int8.tflite`, `small_96_int8.tflite` and `custom_float32.tflite`.

The comparison names follow the lab workflow: MobileNet V2 224x224 alpha=1.0, custom MobileNet 160x160 alpha=1.0, and small MobileNet 96x96 alpha=0.1. Confirm exact export filenames in Edge Impulse before running commands.

## Useful benchmark command

```bash
python src/benchmark.py \
  --model models/mobilenet_v2_1.0_224_quant.tflite \
  --model models/custom_160_int8.tflite \
  --model models/small_96_int8.tflite \
  --image images/test.jpg --output results/q3/benchmark.csv
```

Record model size, inference time, top-1 result/confidence and top-3 results. Use the same test image set for every model.
