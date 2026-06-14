"""
FRED Drone Detection - Step 2: Convert Annotations to YOLO Format
=================================================================
FRED annotations use: time: x1, y1, x2, y2, id, class  (absolute pixel coords)
YOLO format needs:    class cx cy w h                    (normalized 0-1 values)

Run after Step 1. Point FRED_ROOT to your downloaded dataset folder.
"""

import os
import glob
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
FRED_ROOT   = "./FRED"           # Root of the cloned FRED dataset
OUTPUT_DIR  = "./fred_yolo"      # Where YOLO-format files will be saved
IMG_W, IMG_H = 1280, 720         # FRED image resolution (from the paper)
USE_EVENT   = True               # True = use event frames (best per paper)
                                 # False = use RGB frames

# ── Helpers ──────────────────────────────────────────────────────────────────

def fred_to_yolo(x1, y1, x2, y2, img_w=IMG_W, img_h=IMG_H):
    """Convert absolute corner coords to YOLO normalized cx,cy,w,h."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to [0,1] to handle any edge cases
    cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
    w,  h  = max(0, min(1, w)),  max(0, min(1, h))
    return cx, cy, w, h


def parse_annotation_file(ann_path):
    """
    Parse a FRED coordinates.txt file.
    Returns a dict: { frame_index (int) -> list of [x1,y1,x2,y2] }
    """
    annotations = {}
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "time: x1, y1, x2, y2, id, class"
            time_part, coords_part = line.split(":")
            values = [v.strip() for v in coords_part.split(",")]
            x1, y1, x2, y2 = int(values[0]), int(values[1]), int(values[2]), int(values[3])
            # Use time as frame key (rounded to frame index at 30fps)
            time_sec = float(time_part.strip())
            frame_idx = round(time_sec * 30)  # 30 FPS
            if frame_idx not in annotations:
                annotations[frame_idx] = []
            annotations[frame_idx].append([x1, y1, x2, y2])
    return annotations


def convert_sequence(seq_dir, split, seq_id):
    """Convert one unzipped FRED sequence to YOLO folder structure."""
    frame_subdir = "event" if USE_EVENT else "rgb"
    img_dir  = os.path.join(seq_dir, frame_subdir)
    ann_file = os.path.join(seq_dir, "coordinates.txt")

    if not os.path.exists(img_dir) or not os.path.exists(ann_file):
        print(f"  Skipping {seq_id}: missing {frame_subdir}/ or coordinates.txt")
        return

    annotations = parse_annotation_file(ann_file)
    img_files   = sorted(glob.glob(os.path.join(img_dir, "*.png")))

    out_img_dir   = os.path.join(OUTPUT_DIR, "images",  split)
    out_label_dir = os.path.join(OUTPUT_DIR, "labels",  split)
    os.makedirs(out_img_dir,   exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    for frame_idx, img_path in enumerate(img_files):
        img_name  = f"seq{seq_id:04d}_frame{frame_idx:06d}.png"
        label_name = img_name.replace(".png", ".txt")

        # Copy image
        import shutil
        shutil.copy(img_path, os.path.join(out_img_dir, img_name))

        # Write YOLO label (one line per drone in frame)
        label_path = os.path.join(out_label_dir, label_name)
        boxes = annotations.get(frame_idx, [])
        with open(label_path, "w") as f:
            for (x1, y1, x2, y2) in boxes:
                cx, cy, w, h = fred_to_yolo(x1, y1, x2, y2)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            # If no drone in frame, leave file empty (YOLO handles this)


def write_yaml():
    """Write the dataset.yaml file that YOLO needs for training."""
    yaml_content = f"""# FRED Drone Detection Dataset
path: {os.path.abspath(OUTPUT_DIR)}
train: images/train
val:   images/test

nc: 1          # number of classes
names: ['drone']
"""
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Wrote {yaml_path}")
    return yaml_path


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Converting FRED → YOLO format (using {'EVENT' if USE_EVENT else 'RGB'} frames)...")

    for split in ["train", "test"]:
        split_dir = os.path.join(FRED_ROOT, split)
        if not os.path.exists(split_dir):
            print(f"WARNING: {split_dir} not found. Run Step 1 first.")
            continue

        seq_dirs = sorted(glob.glob(os.path.join(split_dir, "*")))
        print(f"\nProcessing {split}: {len(seq_dirs)} sequences...")
        for seq_dir in seq_dirs:
            seq_id = int(os.path.basename(seq_dir))
            print(f"  Sequence {seq_id}...")
            convert_sequence(seq_dir, split, seq_id)

    yaml_path = write_yaml()
    print(f"\nDone! Dataset ready at: {OUTPUT_DIR}")
    print(f"Next: use '{yaml_path}' in Step 3 to train YOLO.")
