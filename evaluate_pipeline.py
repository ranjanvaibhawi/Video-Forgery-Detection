import os
from src.paper_pipeline import analyze_video

REAL_PATH = "./data/split/test/real_videos"
FAKE_PATH = "./data/split/test/fake_videos"

correct = 0
total = 0

# Test REAL videos
for vid in os.listdir(REAL_PATH):

    path = os.path.join(REAL_PATH, vid)

    result, confidence = analyze_video(path)

    print(f"{vid} -> {result}")

    if result == "REAL":
        correct += 1

    total += 1

# Test FAKE videos
for vid in os.listdir(FAKE_PATH):

    path = os.path.join(FAKE_PATH, vid)

    result, confidence = analyze_video(path)

    print(f"{vid} -> {result}")

    if result == "FAKE":
        correct += 1

    total += 1

accuracy = correct / total

print("\nAccuracy:", accuracy)