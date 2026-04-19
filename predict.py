print("STARTED")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

print("Loading model...")
model = load_model("model/garbage_model.h5")

classes = ['glass', 'metal', 'organic', 'paper', 'plastic']

img_path = "test.jpg"

if not os.path.exists(img_path):
    print("❌ test.jpg not found")
    exit()

img = cv2.imread(img_path)

if img is None:
    print("❌ Image not loading")
    exit()

img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.reshape(img, (1, 224, 224, 3))

print("🚀 Predicting...")
prediction = model.predict(img)

print("Raw:", prediction)
print("Final:", classes[np.argmax(prediction)])
