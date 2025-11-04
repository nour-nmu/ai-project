"""
test_model.py
--------------
Enhanced ASL Recognition Tester GUI
Features:
- Top-3 predictions with confidence bars
- Real-time webcam mode with FPS counter
- Supports improved model (asl_mobilenetv2_improved.h5)
- Auto-brightness correction and mixed precision inference
"""

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# ==============================
# TensorFlow Optimization
# ==============================
tf.get_logger().setLevel('ERROR')
if tf.config.list_physical_devices('GPU'):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("✅ GPU and mixed precision enabled for inference.")
else:
    print("⚙️ Running on CPU.")

class ASLTesterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 ASL Recognition Tester")
        self.root.geometry("600x750")
        self.root.configure(bg="#1e1e1e")

        # Load model
        model_path = None
        for candidate in ['asl_mobilenetv2_improved.h5', 'best_model.h5', 'asl_mobilenetv2.h5']:
            if tf.io.gfile.exists(candidate):
                model_path = candidate
                break

        if not model_path:
            print("❌ No trained model found. Please train the model first.")
            return

        self.model = load_model(model_path)
        print(f"✅ Loaded model: {model_path}")

        # Labels
        self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]

        # Webcam state
        self.cap = None
        self.running = False
        self.last_time = time.time()

        self.create_widgets()

    # ==============================
    # UI Components
    # ==============================
    def create_widgets(self):
        title = tk.Label(self.root, text="ASL Recognition Tester",
                         font=('Arial', 22, 'bold'), bg="#1e1e1e", fg="white")
        title.pack(pady=15)

        # Image preview
        self.image_label = tk.Label(self.root, bg="#1e1e1e")
        self.image_label.pack(pady=10)

        # Prediction display
        self.pred_label = tk.Label(self.root, text="Prediction: -",
                                   font=('Arial', 20, 'bold'), bg="#1e1e1e", fg="#00ff88")
        self.pred_label.pack(pady=5)

        # Confidence frame
        self.conf_frame = tk.Frame(self.root, bg="#1e1e1e")
        self.conf_frame.pack(pady=10)

        # Progress bars for top-3 predictions
        self.progress_bars = []
        for i in range(3):
            lbl = tk.Label(self.conf_frame, text=f"-", fg="white", bg="#1e1e1e", font=('Arial', 14))
            lbl.grid(row=i, column=0, padx=5, sticky='w')
            bar = ttk.Progressbar(self.conf_frame, length=250, maximum=100)
            bar.grid(row=i, column=1, padx=10, pady=4)
            self.progress_bars.append((lbl, bar))

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#1e1e1e")
        btn_frame.pack(pady=15)
        tk.Button(btn_frame, text="📁 Load Image", font=('Arial', 12, 'bold'),
                  command=self.load_image, bg="#0078ff", fg="white", width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="🎥 Webcam", font=('Arial', 12, 'bold'),
                  command=self.toggle_webcam, bg="#00c853", fg="white", width=15).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.root, text="Idle", bg="#1e1e1e", fg="#aaaaaa", font=('Arial', 12))
        self.status_label.pack(pady=10)

    # ==============================
    # Image Prediction
    # ==============================
    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        arr = img_to_array(image) / 255.0
        return np.expand_dims(arr, axis=0)

    def auto_brightness(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def predict(self, frame):
        preds = self.model.predict(frame, verbose=0)[0]
        top_indices = np.argsort(preds)[::-1][:3]
        return [(self.labels[i], float(preds[i]) * 100) for i in top_indices]

    # ==============================
    # Load Image
    # ==============================
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            print("❌ Could not open image.")
            return
        image = self.auto_brightness(image)
        self.display_prediction(image)

    # ==============================
    # Webcam Mode
    # ==============================
    def toggle_webcam(self):
        if self.running:
            self.running = False
            self.status_label.config(text="Webcam stopped")
            if self.cap:
                self.cap.release()
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Cannot open webcam.")
            return

        self.running = True
        self.status_label.config(text="Webcam active")
        self.update_webcam()

    def update_webcam(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self.auto_brightness(frame)
        self.display_prediction(frame)
        self.root.after(20, self.update_webcam)

    # ==============================
    # Display Results
    # ==============================
    def display_prediction(self, frame):
        input_tensor = self.preprocess(frame)
        preds = self.predict(input_tensor)

        # Update main label
        top_label, top_conf = preds[0]
        self.pred_label.config(text=f"Prediction: {top_label} ({top_conf:.1f}%)")

        # Update top-3 progress bars
        for i, (lbl, bar) in enumerate(self.progress_bars):
            if i < len(preds):
                lbl.config(text=f"{preds[i][0]}")
                bar['value'] = preds[i][1]
            else:
                lbl.config(text="-")
                bar['value'] = 0

        # Calculate FPS (for webcam)
        now = time.time()
        fps = 1 / (now - self.last_time + 1e-5)
        self.last_time = now
        self.status_label.config(text=f"FPS: {fps:.1f}")

        # Update image in UI
        display_img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 400))
        img_tk = ImageTk.PhotoImage(Image.fromarray(display_img))
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLTesterApp(root)
    root.mainloop()
