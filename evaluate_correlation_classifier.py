import os
import numpy as np
import joblib

from src.paper_pipeline import get_correlation_features

REAL_PATH = "data/split/test/real_videos"
FAKE_PATH = "data/split/test/fake_videos"

clf = joblib.load("models/correlation_classifier.pkl")

correct = 0
total = 0

# REAL
for vid in os.listdir(REAL_PATH):

    path = os.path.join(REAL_PATH, vid)
    print("Working on REAL " + vid)
    feats = np.array(
        get_correlation_features(path)
    ).reshape(1,-1)

    pred = clf.predict(feats)[0]

    print(vid, "->", pred)

    if pred == 0:
        correct += 1

    total += 1

# FAKE
for vid in os.listdir(FAKE_PATH):

    path = os.path.join(FAKE_PATH, vid)
    print("Working on FAKE " + vid)
    feats = np.array(
        get_correlation_features(path)
    ).reshape(1,-1)

    pred = clf.predict(feats)[0]

    print(vid, "->", pred)

    if pred == 1:
        correct += 1

    total += 1

accuracy = correct / total

print("\nAccuracy:", accuracy)