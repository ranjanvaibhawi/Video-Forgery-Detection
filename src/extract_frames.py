import cv2
import os

def extract_frames(video_path, output_folder, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            frame = cv2.resize(frame, (224, 224))
            cv2.imwrite(f"{output_folder}/frame_{saved}.jpg", frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Saved {saved} frames from {video_path}")


# Process real videos
for vid in os.listdir("./data/real_videos"):
    extract_frames(f"./data/real_videos/{vid}", "./data/frames/real")

# Process fake videos
for vid in os.listdir("./data/fake_videos"):
    extract_frames(f"./data/fake_videos/{vid}", "./data/frames/fake")