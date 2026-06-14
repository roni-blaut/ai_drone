"""
FRED Drone Detection - Step 1: Download Dataset
================================================
Downloads the FRED (Florence RGB-Event Drone) dataset from HuggingFace.
The dataset includes synchronized RGB frames + Event camera frames,
with bounding box annotations for drone detection.

Install requirements first:
    pip install datasets huggingface_hub
"""

from datasets import load_dataset
import os

# ── Where to save the data locally ──────────────────────────────────────────
SAVE_DIR = "./fred_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Option A: Load via HuggingFace datasets library (easiest) ───────────────
def load_with_hf():
    print("Loading FRED dataset from HuggingFace...")

    # Load full dataset (train + test splits)
    ds = load_dataset("GabrieleMagrini/FRED")
    print(ds)  # Shows splits and column names

    # Load specific splits
    train_set = load_dataset("GabrieleMagrini/FRED", split="train")
    test_set  = load_dataset("GabrieleMagrini/FRED", split="test")

    print(f"\nTrain samples: {len(train_set)}")
    print(f"Test  samples: {len(test_set)}")

    # Peek at the first sample to understand the data structure
    sample = train_set[0]
    print("\nSample keys:", list(sample.keys()))

    return train_set, test_set


# ── Option B: Clone full dataset via git-lfs (gets all raw files) ────────────
def clone_with_git():
    """
    Run these commands in your terminal (requires git + git-lfs installed):

        git lfs install
        git clone https://huggingface.co/datasets/GabrieleMagrini/FRED

    This downloads the raw .zip files for every sequence.
    Each .zip contains:
        ├── rgb/          ← RGB frames (.png)
        ├── event/        ← Event frames (.png)
        ├── events.hdf5   ← Raw event stream
        ├── coordinates.txt       ← Bounding boxes (extended, recommended)
        └── coordinates_rgb.txt   ← Bounding boxes (RGB-only, no padding)
    """
    print("Run in terminal:")
    print("  git lfs install")
    print("  git clone https://huggingface.co/datasets/GabrieleMagrini/FRED")


# ── Annotation format reference ──────────────────────────────────────────────
"""
Each coordinates.txt file has one line per frame:

    time: x1, y1, x2, y2, id, class

Where:
    time  = seconds.microseconds since recording start
    x1,y1 = top-left corner of bounding box (pixels)
    x2,y2 = bottom-right corner of bounding box (pixels)
    id    = unique drone ID within the video
    class = object class (0 = drone)

Example line:
    0.000000: 512, 300, 560, 340, 1, 0
"""


if __name__ == "__main__":
    train_set, test_set = load_with_hf()
    print("\nDone! Move to step 2 to convert annotations to YOLO format.")
