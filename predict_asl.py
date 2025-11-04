"""
predict_asl.py
--------------
Real-time or image-based ASL prediction using the trained MobileNetV2 model.
Features:
- Top-3 predictions with confidence
- Works with webcam or --image file
- Auto-brightness correction for stable predictions
"""

import cv2
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ==============================
# TensorFlow Optimizations
# ==============================
tf.get_logger().setLevel('ERROR')
tf.config.experimental.enable_tensor_float_32_execution(True)
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ==============================
# Load Model
# ==============================
model_file = None
for name in ['asl_mobilenetv2_improved.h5', 'best_model.h5', 'asl_mobilenetv2.h5']:
    if tf.io.gfile.exists(name):
        model_file = name
        break

if not model_file:
    print("❌ No trained model file found. Train first!")
    exit(1)

model = load_model(model_file)
print(f"✅ Loaded model: {model_file}")

# ==============================
# Labels
# ==============================
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

# ==============================
# CLI
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to a single image for prediction')
args = parser.parse_args()

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame) / 255.0
    return np.expand_dims(frame, axis=0)

def predict_image(frame):
    preds = model.predict(frame)
    top_indices = np.argsort(preds[0])[::-1][:3]
    return [(labels[i], float(preds[0][i])) for i in top_indices]

def auto_brightness(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

# ==============================
# Single Image Mode
# ==============================
if args.image:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"❌ Could not read image: {args.image}")
        exit(1)

    frame = auto_brightness(frame)
    img = preprocess(frame)
    preds = predict_image(img)

    print("\n🔍 Top Predictions:")
    for label, conf in preds:
        print(f"{label}: {conf * 100:.2f}%")

    cv2.putText(frame, f"Predicted: {preds[0][0]} ({preds[0][1]*100:.1f}%)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit(0)

# ==============================
# Webcam Mode
# ==============================
cap = cv2.VideoCapture(0)
print("\n🎥 Press 'q' to quit webcam mode.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Camera frame not received.")
        break

    frame = auto_brightness(frame)
    img = preprocess(frame)
    preds = predict_image(img)
    label, conf = preds[0]

    cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.imshow("ASL Translator - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed.")
