import os
import shutil
import random

random.seed(42)

REAL_SRC = "./data/real_videos"
FAKE_SRC = "./data/fake_videos"

BASE_OUT = "./data/split"

splits = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def split_and_copy(src_folder, label_folder):
    files = os.listdir(src_folder)
    random.shuffle(files)

    n = len(files)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    split_files = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, file_list in split_files.items():
        out_dir = os.path.join(BASE_OUT, split_name, label_folder)
        os.makedirs(out_dir, exist_ok=True)

        for f in file_list:
            shutil.copy(
                os.path.join(src_folder, f),
                os.path.join(out_dir, f)
            )

split_and_copy(REAL_SRC, "real_videos")
split_and_copy(FAKE_SRC, "fake_videos")

print("Done splitting videos.")