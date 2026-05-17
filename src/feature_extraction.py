import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

IMG_SIZE = 224

# Load pretrained VGG16
base_model = VGG16(weights='imagenet', include_top=False)

# Output from last pooling layer
model = Model(inputs=base_model.input,
              outputs=base_model.output)

def extract_features(frame):

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame / 255.0

    frame = np.expand_dims(frame, axis=0)

    features = model.predict(frame, verbose=0)

    return features.flatten()