import cv2
import numpy as np

from src.feature_extraction import extract_features
from src.kpca_analysis import apply_kpca
from src.correlation_analysis import compute_correlations



# Compute median correlation for a video

def get_video_correlation(video_path):

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


        if len(features) >= 50:
            break

    cap.release()

    if len(features) == 0:
        return 0

    features = np.array(features)

    reduced_features = apply_kpca(features)

    correlations = compute_correlations(reduced_features)

    correlations = np.array(correlations)

    correlations = correlations[
        ~np.isnan(correlations)
    ]
    median_corr = np.median(correlations)

    return median_corr


# Final prediction using calibrated threshold
def analyze_video(video_path, threshold=0.6529912604549386):

    median_corr = get_video_correlation(video_path)

    print("Median Correlation:", median_corr)

    if median_corr < threshold:
        result = "FAKE"
    else:
        result = "REAL"

    confidence = round(abs(median_corr) * 100, 2)

    return result, confidence