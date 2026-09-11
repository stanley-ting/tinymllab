# Q3 - model comparison

Run `python src/benchmark.py --model models/mobilenet_v2_1.0_224_quant.tflite --model models/custom_160_int8.tflite --model models/small_96_int8.tflite --image images/test.jpg --output results/q3/benchmark.csv` after downloading the Edge Impulse models. Record input size, model size, latency, top-1 accuracy/confidence and top-3 predictions.
