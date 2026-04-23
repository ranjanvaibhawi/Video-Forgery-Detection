import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Paths
REAL_PATH = "./data/frames/real"
FAKE_PATH = "./data/frames/fake"

IMG_SIZE = 224

# Load images
def load_data():
    data = []
    labels = []

    # Real images
    for img in os.listdir(REAL_PATH):
        path = os.path.join(REAL_PATH, img)
        image = cv2.imread(path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(0)

    # Fake images
    for img in os.listdir(FAKE_PATH):
        path = os.path.join(FAKE_PATH, img)
        image = cv2.imread(path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        data.append(image)
        labels.append(1)

    data = np.array(data) / 255.0
    labels = to_categorical(labels, 2)

    return data, labels

# Load dataset
X, y = load_data()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Load VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Build model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=8)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# Save model
model.save("./models/vgg16_forgery_model.keras")