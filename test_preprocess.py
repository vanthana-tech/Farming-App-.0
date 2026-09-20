"""
Finds out how your model wants its images prepared.

Put this next to app.py, crop_disease_model.tflite and labels.json, then run:

    python test_preprocess.py healthy_tomato.jpg diseased_tomato.jpg healthy_potato.jpg

Use photos where YOU know the right answer. For each photo it prints the top
prediction under 3 different preprocessing modes. The mode where the answers
match reality is the one to put in app.py (PREPROCESS_MODE).
"""
import json
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

with open("labels.json") as f:
    raw = json.load(f)
labels = [raw[k] for k in sorted(raw, key=lambda x: int(x))] if isinstance(raw, dict) else raw

print("Model input shape :", inp["shape"], inp["dtype"])
print("Model output shape:", out["shape"], "| labels in labels.json:", len(labels))
print()

MODES = {
    "0-1   (pixel/255)": lambda a: a / 255.0,
    "-1-1  (MobileNet) ": lambda a: a / 127.5 - 1.0,
    "0-255 (raw)       ": lambda a: a,
}

for path in sys.argv[1:]:
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    print(path)
    for name, fn in MODES.items():
        interpreter.set_tensor(inp["index"], np.expand_dims(fn(arr), 0).astype(np.float32))
        interpreter.invoke()
        probs = interpreter.get_tensor(out["index"])[0]
        top = np.argsort(probs)[::-1][:2]
        summary = ", ".join(f"{labels[int(i)]} {probs[i] * 100:.0f}%" for i in top)
        print(f"   {name} -> {summary}")
    print()
