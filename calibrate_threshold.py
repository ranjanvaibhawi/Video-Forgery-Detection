import os
import numpy as np

from src.paper_pipeline import get_video_correlation

REAL_PATH = "data/split/val/real_videos"
FAKE_PATH = "data/split/val/fake_videos"

real_corrs = []
fake_corrs = []

# REAL videos
for vid in os.listdir(REAL_PATH):

    path = os.path.join(REAL_PATH, vid)

    print(f"[Real Processing:", vid)

    corr = get_video_correlation(path)

    real_corrs.append(corr)

# FAKE videos
for vid in os.listdir(FAKE_PATH):

    path = os.path.join(FAKE_PATH, vid)

    print(f"[Fake Processing:", vid)

    corr = get_video_correlation(path)

    fake_corrs.append(corr)

real_mean = np.mean(real_corrs)
fake_mean = np.mean(fake_corrs)

threshold = (real_mean + fake_mean) / 2

print("Real Mean:", real_mean)
print("Fake Mean:", fake_mean)
print("Optimal Threshold:", threshold)