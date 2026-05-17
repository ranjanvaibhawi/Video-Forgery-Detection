import os
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression

from src.paper_pipeline import get_correlation_features

REAL_PATH = "data/split/train/real_videos"
FAKE_PATH = "data/split/train/fake_videos"

X = []
y = []

# REAL
for vid in os.listdir(REAL_PATH):
    print("Working on REAL "+vid)
    path = os.path.join(REAL_PATH, vid)

    feats = get_correlation_features(path)

    X.append(feats)

    y.append(0)

# FAKE
for vid in os.listdir(FAKE_PATH):

    print("Working on FAKE "+vid)

    path = os.path.join(FAKE_PATH, vid)

    feats = get_correlation_features(path)

    X.append(feats)

    y.append(1)

X = np.array(X)
y = np.array(y)

clf = LogisticRegression()

clf.fit(X, y)

joblib.dump(clf, "models/correlation_classifier.pkl")

print("Classifier trained and saved.")