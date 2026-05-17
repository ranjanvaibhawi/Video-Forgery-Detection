from flask import Flask, render_template, request
from flask import send_from_directory

import os

from src.paper_pipeline import analyze_video

# Flask App Setup
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home Route
@app.route("/")
def home():

    return render_template("index.html")

# Uploaded Video Route
@app.route('/uploads/<filename>')
def uploaded_file(filename):

    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        filename
    )

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["video"]

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Final Logistic Regression Pipeline
    result, confidence = analyze_video(filepath)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        filename=file.filename,
        video_path=file.filename
    )

if __name__ == "__main__":

    app.run(debug=True)