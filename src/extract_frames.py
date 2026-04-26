import cv2
import os

IMG_SIZE = 224
FRAME_SKIP = 10
MAX_FRAMES = 20

def extract_frames(video_path, output_folder, prefix):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    os.makedirs(output_folder, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_SKIP == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            filename = os.path.join(output_folder, f"{prefix}_{saved}.jpg")
            cv2.imwrite(filename, frame)
            saved += 1

        if saved >= MAX_FRAMES:
            break

        count += 1

    cap.release()


def process_folder(input_folder, output_folder):
    for vid in os.listdir(input_folder):
        video_path = os.path.join(input_folder, vid)
        video_name = os.path.splitext(vid)[0]

        extract_frames(video_path, output_folder, video_name)
        print("Done:", vid)


splits = ["train", "val", "test"]
labels = ["real_videos", "fake_videos"]

for split in splits:
    for label in labels:
        input_dir = f"./data/split/{split}/{label}"

        out_label = "real" if label == "real_videos" else "fake"

        output_dir = f"./data/frames_split/{split}/{out_label}"

        process_folder(input_dir, output_dir)

print("Frame extraction completed.")