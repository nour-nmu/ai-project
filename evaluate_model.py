"""
evaluate_model.py
-----------------
Enhanced evaluation script for ASL Translator using MobileNetV2
Includes: top-5 accuracy, precision/recall/F1, normalized confusion matrix, CSV export.
"""

import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import glob

# ==============================
# TensorFlow Optimization
# ==============================
tf.get_logger().setLevel('ERROR')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Using GPU with mixed precision for evaluation.")
    except Exception as e:
        print("⚠️ GPU setup failed, using CPU:", e)
else:
    print("⚙️ Running on CPU (no GPU detected).")

# ==============================
# Load Model
# ==============================
model_name = None
for candidate in ['asl_mobilenetv2_improved.h5', 'asl_mobilenetv2.h5', 'best_model.h5']:
    if os.path.exists(candidate):
        model_name = candidate
        break

if not model_name:
    print("❌ No trained model file found.")
    exit(1)

print(f"✅ Loaded model: {model_name}")
model = load_model(model_name)

# ==============================
# Argument Parser
# ==============================
parser = argparse.ArgumentParser(description='Evaluate ASL model on test dataset')
parser.add_argument('--test-dir', dest='test_dir', default='asl_alphabet_test/asl_alphabet_test',
                    help='Path to test dataset root')
args = parser.parse_args()

test_dir = args.test_dir

if not os.path.exists(test_dir):
    print(f"❌ Test dataset not found at: {test_dir}")
    exit(1)

# ==============================
# Data Preparation
# ==============================
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    class_mode='categorical'
)

# ==============================
# Predictions
# ==============================
print("\n🔍 Evaluating model...")
predictions = model.predict(test_data, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# ==============================
# Metrics
# ==============================
acc = accuracy_score(y_true, y_pred)
top5 = top_k_accuracy_score(y_true, predictions, k=5)
print(f"\n✅ Top-1 Accuracy: {acc * 100:.2f}%")
print(f"✅ Top-5 Accuracy: {top5 * 100:.2f}%")

report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('classification_report.csv')
print("📄 Saved detailed classification report as classification_report.csv")

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(22, 20))
sns.heatmap(cm_norm, annot=False, cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix_normalized.png')
plt.close()
print("📊 Saved confusion matrix as confusion_matrix_normalized.png")

print("\n🎯 Evaluation complete.")
