import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

# Load trained model
model = load_model("../models/vgg16_forgery_model.keras")

def predict_frame(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    label = np.argmax(pred)

    if label == 0:
        print("Prediction: REAL")
    else:
        print("Prediction: FAKE")

# Test image path
predict_frame("./data/frames/fake/frame_303.jpg")