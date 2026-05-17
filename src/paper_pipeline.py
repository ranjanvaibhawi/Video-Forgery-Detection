import cv2
import numpy as np

from src.feature_extraction import extract_features
from src.kpca_analysis import apply_kpca
from src.correlation_analysis import (
    compute_correlations,
    detect_forgery
)

def analyze_video(video_path):

    cap = cv2.VideoCapture(video_path)

    features = []

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Sample every 15th frame
        if count % 15 == 0:

            feature_vector = extract_features(frame)

            features.append(feature_vector)

        count += 1

        # Limit frames
        if len(features) >= 20:
            break

    cap.release()

    features = np.array(features)

    # KPCA
    reduced_features = apply_kpca(features)

    # Correlation analysis
    correlations = compute_correlations(reduced_features)

    # Final decision
    result = detect_forgery(correlations)

    avg_corr = np.mean(correlations)

    confidence = round(abs(avg_corr) * 100, 2)

    return result, confidence