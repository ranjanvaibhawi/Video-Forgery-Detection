from flask import Flask, render_template, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = load_model("./models/vgg16_ffpp.keras")

IMG_SIZE = 224

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    predictions = []
    count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Sample every 15th frame
        if count % 15 == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=0)

            pred = model.predict(frame, verbose=0)[0][0]

            predictions.append(pred)

        count += 1

        if len(predictions) >= 10:
            break

    cap.release()

    avg_pred = np.mean(predictions)

    if avg_pred > 0.5:
        return "FAKE", round(avg_pred * 100, 2)
    else:
        return "REAL", round((1 - avg_pred) * 100, 2)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["video"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result, confidence = predict_video(filepath)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        filename=file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)