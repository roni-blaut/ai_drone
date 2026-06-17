"""
dataset_builder.py — Build YOLO training dataset from FRED sequences.

Multi-sequence mode (default):
  Reads data_from_fred/splits.yaml to assign each zip to train/val/test.
  All generated PNGs go into dataset/images/ (flat).
  Split membership is recorded in dataset/train.txt, val.txt, test.txt.
  Run: python dataset_builder.py

Single-sequence mode (legacy):
  Randomly splits frames 80/20 within sequence 7.
  Run: python dataset_builder.py --single

For each 33ms window:
  1. Read events from events.raw (zip or folder)
  2. Apply noise filters
  3. Generate 4 channels → shape (4, 720, 1280)
  4. Find matching annotation from coordinates.txt
  5. Save 4-channel PNG (RGBA) to dataset/images/
  6. Save YOLO label to dataset/labels/
"""

import os
import glob
import numpy as np
import random
from PIL import Image
from zip_utils import seq_open_lines, seq_exists, init_sequence
from config import (
    RAW_FILE, COORDS_FILE, DATASET_DIR,
    IMG_W, IMG_H, WINDOW_US,
    TRAIN_RATIO, RANDOM_SEED,
    DEBUG_MODE, DEBUG_SAMPLES
)
from evt3_reader import EVT3Reader
from filters import fast_filter
from channels import generate_channels, channels_to_rgb_preview

_HERE          = os.path.dirname(os.path.abspath(__file__))
DATA_FROM_FRED = os.path.normpath(os.path.join(_HERE, '..', 'data_from_fred'))


# ── Annotation loader ─────────────────────────────────────────────────────────

def load_annotations(coords_file):
    """
    Load coordinates.txt into a sorted list of (time_us, x1, y1, x2, y2).

    FRED format: "time_sec: x1, y1, x2, y2, drone_id"
    We convert time to microseconds for alignment with event timestamps.
    """
    annotations = []
    for line in seq_open_lines(coords_file):
            line = line.strip()
            if not line:
                continue
            time_part, coords_part = line.split(':')
            vals = [v.strip() for v in coords_part.split(',')]
            x1, y1, x2, y2 = float(vals[0]), float(vals[1]), \
                              float(vals[2]), float(vals[3])
            t_us = int(float(time_part.strip()) * 1_000_000)
            annotations.append((t_us, x1, y1, x2, y2))

    annotations.sort(key=lambda a: a[0])
    print(f"Loaded {len(annotations)} annotations from {coords_file}")
    print(f"  Time range: {annotations[0][0]/1e6:.2f}s – "
          f"{annotations[-1][0]/1e6:.2f}s")
    return annotations


def load_removed_windows(annotations, min_gap_us=50_000):
    """
    Derive removed time windows from gaps in the annotation file.

    The FRED Removed_frames folder marks bad windows that were manually excluded.
    Those windows show up as gaps > 50ms between consecutive annotations.

    Returns list of (gap_start_us, gap_end_us) — windows to skip entirely.
    """
    removed = []
    ann_times = [a[0] for a in annotations]

    for i in range(1, len(ann_times)):
        gap = ann_times[i] - ann_times[i - 1]
        if gap > min_gap_us:
            removed.append((ann_times[i - 1], ann_times[i]))

    if removed:
        print(f"  Removed windows (annotation gaps > {min_gap_us/1000:.0f}ms):")
        total_frames = 0
        for t0, t1 in removed:
            n = round((t1 - t0) / 33_333)
            total_frames += n
            print(f"    t={t0/1e6:.3f}s – {t1/1e6:.3f}s  "
                  f"({(t1-t0)/1e6:.2f}s = ~{n} frames skipped)")
        print(f"  Total removed: {total_frames} frames across {len(removed)} gaps")

    return removed


def in_removed_window(t_start_us, t_end_us, removed_windows):
    """Return True if this window overlaps any removed region."""
    for r0, r1 in removed_windows:
        if t_start_us < r1 and t_end_us > r0:
            return True
    return False


def find_annotation(annotations, t_start_us, t_end_us, max_gap_us=100_000):
    """
    Find annotation whose timestamp falls within [t_start_us, t_end_us].
    Returns (x1, y1, x2, y2) or None if no annotation found.

    max_gap_us: maximum allowed time gap between window center and annotation.
    """
    t_center = (t_start_us + t_end_us) // 2
    ann_times = np.array([a[0] for a in annotations])

    idx  = np.searchsorted(ann_times, t_center)
    idx  = min(max(idx, 0), len(annotations) - 1)
    best = min([max(0, idx-1), min(len(annotations)-1, idx)],
               key=lambda i: abs(ann_times[i] - t_center))

    gap = abs(ann_times[best] - t_center)
    if gap > max_gap_us:
        return None   # No annotation near this window

    _, x1, y1, x2, y2 = annotations[best]
    return x1, y1, x2, y2


def bbox_to_yolo(x1, y1, x2, y2, img_w=IMG_W, img_h=IMG_H):
    """Convert absolute corner coords to YOLO normalized cx,cy,w,h."""
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    # Clamp to [0,1]
    cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
    w,  h  = max(0.001, min(1.0, w)), max(0.001, min(1.0, h))
    return cx, cy, w, h


# ── Debug visualization ───────────────────────────────────────────────────────

def _save_debug_comparison(events_raw, events_clean, t_start, t_end, idx):
    """
    Save before-filter, after-filter, and side-by-side comparison PNGs.
    Output: debug/before/window_NNNN.png
            debug/after/window_NNNN.png
            debug/compare/window_NNNN.png  ← both panels side by side with banner
    """
    try:
        import cv2
    except ImportError:
        print("  [DEBUG] Install opencv-python to save debug images: "
              "pip install opencv-python")
        return

    debug_root = './debug'
    for sub in ('before', 'after', 'compare'):
        os.makedirs(os.path.join(debug_root, sub), exist_ok=True)

    ch_raw   = generate_channels(events_raw,   t_start, t_end)
    ch_clean = generate_channels(events_clean, t_start, t_end)

    img_before = channels_to_rgb_preview(ch_raw)
    img_after  = channels_to_rgb_preview(ch_clean)

    # Banner rows for the comparison image
    def _banner(img, text):
        bar = np.zeros((44, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, text, (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
        return np.vstack([bar, img])

    removed   = len(events_raw) - len(events_clean)
    pct       = 100 * removed / max(len(events_raw), 1)
    t_sec     = t_start / 1e6
    b_titled  = _banner(img_before,
                        f"BEFORE filter  [{len(events_raw):,} events]  "
                        f"t={t_sec:.3f}s  sample {idx+1}")
    a_titled  = _banner(img_after,
                        f"AFTER filter   [{len(events_clean):,} events]  "
                        f"removed {removed:,} ({pct:.0f}%)")

    compare = np.hstack([b_titled, a_titled])

    cv2.imwrite(os.path.join(debug_root, 'before',  f'window_{idx:04d}.png'), img_before)
    cv2.imwrite(os.path.join(debug_root, 'after',   f'window_{idx:04d}.png'), img_after)
    cv2.imwrite(os.path.join(debug_root, 'compare', f'window_{idx:04d}.png'), compare)

    print(f"  [DEBUG] Saved window {idx+1:2d}: "
          f"{len(events_raw):,} raw → {len(events_clean):,} filtered  "
          f"→ debug/compare/window_{idx:04d}.png")


# ── Main builder ──────────────────────────────────────────────────────────────

def build_dataset(
    raw_file    = RAW_FILE,
    coords_file = COORDS_FILE,
    output_dir  = DATASET_DIR,
    window_us   = WINDOW_US,
    train_ratio = TRAIN_RATIO,
    seed        = RANDOM_SEED,
):
    """
    Build the full 4-channel YOLO dataset from one FRED sequence.

    Parameters
    ----------
    raw_file    : path to events.raw
    coords_file : path to coordinates.txt
    output_dir  : where to save dataset
    window_us   : window duration in microseconds
    train_ratio : fraction for training set
    seed        : random seed for split
    """
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Load annotations and derive removed windows from gaps
    annotations     = load_annotations(coords_file)
    removed_windows = load_removed_windows(annotations)

    # Init reader
    reader   = EVT3Reader(raw_file)
    ts_shift = reader.ts_shift_us   # raw timestamps are ts_shift_us ahead of coordinates.txt

    # Stats
    n_total    = 0
    n_with_ann = 0
    n_empty    = 0
    debug_count = 0   # counts how many before/after images have been saved

    if DEBUG_MODE:
        print(f"\n[DEBUG] Will save {DEBUG_SAMPLES} before/after comparison images "
              f"to ./debug/compare/")

    print(f"\nBuilding dataset → {output_dir}")
    print(f"Window: {window_us/1000:.1f}ms  "
          f"Train: {train_ratio*100:.0f}%  Val: {(1-train_ratio)*100:.0f}%\n")

    # Skip countdown/junk frames — drone annotations start at 9.87s (synchronized time).
    # Raw time = synchronized time + ts_shift_us.
    T_DRONE_START_US = 9_800_000 + ts_shift

    for t_start, events in reader.iter_windows(window_us, t_start=T_DRONE_START_US):
        n_total += 1

        if n_total % 50 == 0:
            print(f"  Processed {n_total} windows, "
                  f"{n_with_ann} with annotations, "
                  f"{n_empty} skipped (no events)")

        # Skip empty windows
        if len(events) == 0:
            n_empty += 1
            continue

        t_end = t_start + window_us

        # Convert raw timestamps → synchronized time before annotation lookup.
        # coordinates.txt and removed_windows use synchronized time (ts_shift already applied).
        t_sync_start = t_start - ts_shift
        t_sync_end   = t_end   - ts_shift

        # Skip windows that fall inside a removed/bad region (matches FRED paper)
        if in_removed_window(t_sync_start, t_sync_end, removed_windows):
            n_empty += 1
            continue

        # Check for annotation
        ann = find_annotation(annotations, t_sync_start, t_sync_end)
        # If no annotation, this window has no drone — still useful as negative
        # But for initial training, skip unannotated windows to keep it simple
        # Uncomment the next 2 lines to include negative examples:
        # if ann is None:
        #     continue

        # Apply noise filter
        events_clean = fast_filter(events)

        # DEBUG: save before/after filter comparison for the first N windows
        if DEBUG_MODE and debug_count < DEBUG_SAMPLES:
            _save_debug_comparison(events, events_clean, t_start, t_end, debug_count)
            debug_count += 1

        if len(events_clean) == 0:
            n_empty += 1
            continue

        # Generate 4 channels
        channels = generate_channels(events_clean, t_start, t_end)

        # Determine split
        split = 'train' if random.random() < train_ratio else 'val'

        # File name
        frame_name = f"seq_t{t_start:012d}"

        # Save channels as 4-channel PNG (RGBA) — required by Ultralytics YOLO
        # channels shape: (4, H, W) float32 [0,1] → (H, W, 4) uint8
        img_hwc = (channels.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        img_path = os.path.join(output_dir, 'images', split,
                                frame_name + '.png')
        Image.fromarray(img_hwc, mode='RGBA').save(img_path)

        # Save YOLO label
        label_path = os.path.join(output_dir, 'labels', split,
                                  frame_name + '.txt')  # label name matches PNG stem
        with open(label_path, 'w') as f:
            if ann is not None:
                x1, y1, x2, y2 = ann
                cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_with_ann += 1
            # If no annotation: empty label file = no drone in this frame

    print(f"\nDone!")
    print(f"  Total windows processed : {n_total}")
    print(f"  With drone annotation   : {n_with_ann}")
    print(f"  Skipped (empty)         : {n_empty}")

    # Write dataset.yaml
    yaml_path = _write_yaml(output_dir)
    print(f"  Dataset YAML: {yaml_path}")

    return output_dir


def _write_yaml(output_dir):
    """Write dataset.yaml for YOLO training."""
    abs_dir  = os.path.abspath(output_dir)
    yaml_str = f"""# FRED 4-Channel Drone Detection Dataset
# Generated by dataset_builder.py

path:  {abs_dir}
train: images/train
val:   images/val

# 4-channel input: positive polarity, negative polarity, rotor map, time surface
channels: 4

nc: 1
names: ['drone']
"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_str)
    return yaml_path


# ── Multi-sequence builder ────────────────────────────────────────────────────

def _find_coords(seq_dir):
    """Return the best annotation file for seq_dir (interpolated preferred)."""
    interp = os.path.join(seq_dir, 'interpolated_coordinates.txt')
    return interp if seq_exists(interp) else os.path.join(seq_dir, 'coordinates.txt')


def _process_sequence(seq_num, raw_file, coords_file, output_dir, window_us):
    """
    Process one sequence: generate 4-channel PNGs and labels into output_dir/images|labels/.
    Returns list of absolute image paths that were generated.
    """
    annotations     = load_annotations(coords_file)
    removed_windows = load_removed_windows(annotations)

    reader   = EVT3Reader(raw_file)
    ts_shift = reader.ts_shift_us

    n_total = n_with_ann = n_empty = 0
    img_paths = []

    T_DRONE_START_US = 9_800_000 + ts_shift

    for t_start, events in reader.iter_windows(window_us, t_start=T_DRONE_START_US):
        n_total += 1
        if n_total % 100 == 0:
            print(f"  [seq {seq_num}] {n_total} windows  "
                  f"{n_with_ann} annotated  {n_empty} skipped")

        if len(events) == 0:
            n_empty += 1
            continue

        t_end        = t_start + window_us
        t_sync_start = t_start - ts_shift
        t_sync_end   = t_end   - ts_shift

        if in_removed_window(t_sync_start, t_sync_end, removed_windows):
            n_empty += 1
            continue

        ann          = find_annotation(annotations, t_sync_start, t_sync_end)
        events_clean = fast_filter(events)

        if len(events_clean) == 0:
            n_empty += 1
            continue

        channels   = generate_channels(events_clean, t_start, t_end)
        frame_name = f"s{seq_num}_{t_start:012d}"

        img_hwc  = (channels.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        img_path = os.path.join(output_dir, 'images', frame_name + '.png')
        Image.fromarray(img_hwc, mode='RGBA').save(img_path)

        lbl_path = os.path.join(output_dir, 'labels', frame_name + '.txt')
        with open(lbl_path, 'w') as f:
            if ann is not None:
                x1, y1, x2, y2 = ann
                cx, cy, w, h = bbox_to_yolo(x1, y1, x2, y2)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                n_with_ann += 1

        img_paths.append(os.path.abspath(img_path))

    print(f"  [seq {seq_num}] Done: {n_total} processed  "
          f"{n_with_ann} annotated  {n_empty} skipped")
    return img_paths


def build_multi_sequence(
    splits_yaml = None,
    output_dir  = DATASET_DIR,
    window_us   = WINDOW_US,
):
    """
    Build 4-channel dataset from multiple sequences using splits.yaml.

    Each sequence zip is assigned entirely to one split (train/val/test).
    Generates dataset/images/ and dataset/labels/ (flat) plus
    dataset/train.txt, val.txt, test.txt index files for Ultralytics YOLO.
    """
    try:
        import yaml as _yaml
    except ImportError:
        print("ERROR: pip install pyyaml"); return

    if splits_yaml is None:
        splits_yaml = os.path.join(DATA_FROM_FRED, 'splits.yaml')

    if not os.path.isfile(splits_yaml):
        print(f"ERROR: splits.yaml not found: {splits_yaml}")
        print("Create it or run with --single for legacy single-sequence mode.")
        return

    with open(splits_yaml) as f:
        splits = _yaml.safe_load(f) or {}

    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    split_paths = {'train': [], 'val': [], 'test': []}

    for split in ('train', 'val', 'test'):
        seq_list = splits.get(split) or []
        for seq_num in seq_list:
            seq_dir = os.path.join(DATA_FROM_FRED, str(seq_num))
            print(f"\n{'='*55}")
            print(f"  Sequence {seq_num}  →  {split}")
            print(f"{'='*55}")
            init_sequence(seq_dir)
            raw_file    = os.path.join(seq_dir, 'Event', 'events.raw')
            coords_file = _find_coords(seq_dir)
            paths = _process_sequence(seq_num, raw_file, coords_file,
                                      output_dir, window_us)
            split_paths[split].extend(paths)

    # Write index txt files (absolute paths, one per line)
    for split, paths in split_paths.items():
        txt_path = os.path.join(output_dir, f'{split}.txt')
        with open(txt_path, 'w') as f:
            f.writelines(p + '\n' for p in paths)
        print(f"\n{split}.txt : {len(paths)} images")

    yaml_path = _write_yaml_multi(output_dir)

    print(f"\n{'='*55}")
    print(f"Dataset ready: {os.path.abspath(output_dir)}")
    print(f"  train : {len(split_paths['train'])} images")
    print(f"  val   : {len(split_paths['val'])} images")
    print(f"  test  : {len(split_paths['test'])} images")
    print(f"  YAML  : {yaml_path}")
    return output_dir


def _write_yaml_multi(output_dir):
    """Write dataset.yaml using txt-file split format for Ultralytics YOLO."""
    abs_dir  = os.path.abspath(output_dir)
    yaml_str = f"""# FRED 4-Channel Drone Detection Dataset
# Generated by dataset_builder.py (multi-sequence mode)
# Splits are defined in data_from_fred/splits.yaml

path:  {abs_dir}
train: train.txt
val:   val.txt
test:  test.txt

# 4-channel input: positive polarity, negative polarity, rotor map, time surface
channels: 4

nc: 1
names: ['drone']
"""
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_str)
    return yaml_path


# ── Quick stats ───────────────────────────────────────────────────────────────

def print_dataset_stats(output_dir=DATASET_DIR):
    """Print summary of a built dataset (supports both flat and split-subdir layouts)."""
    flat_img_dir   = os.path.join(output_dir, 'images')
    flat_label_dir = os.path.join(output_dir, 'labels')
    is_flat = (os.path.isdir(flat_img_dir) and
               not os.path.isdir(os.path.join(flat_img_dir, 'train')))

    if is_flat:
        # Multi-sequence flat layout: report per txt index file
        total_imgs = len(glob.glob(os.path.join(flat_img_dir, '*.png')))
        print(f"  Total images : {total_imgs}")
        for split in ['train', 'val', 'test']:
            txt = os.path.join(output_dir, f'{split}.txt')
            if os.path.isfile(txt):
                lines = [l.strip() for l in open(txt) if l.strip()]
                print(f"  {split:5s}        : {len(lines)} images")
    else:
        # Legacy split-subdir layout
        for split in ['train', 'val']:
            img_dir   = os.path.join(output_dir, 'images', split)
            label_dir = os.path.join(output_dir, 'labels', split)
            n_imgs  = len(glob.glob(os.path.join(img_dir,   '*.png')))
            n_drone = sum(1 for lf in glob.glob(os.path.join(label_dir, '*.txt'))
                          if os.path.getsize(lf) > 0)
            print(f"  {split:5s}: {n_imgs} images, "
                  f"{n_drone} with drone ({100*n_drone/max(n_imgs,1):.0f}%)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Build FRED 4-channel YOLO dataset")
    parser.add_argument('--single', action='store_true',
                        help='Legacy: single sequence mode (random 80/20 split, seq 7 only)')
    args = parser.parse_args()

    print("=" * 60)
    print("FRED 4-Channel Dataset Builder")
    print("=" * 60)

    if args.single:
        # Legacy single-sequence mode
        if not seq_exists(RAW_FILE):
            print(f"ERROR: Raw file not found: {RAW_FILE}")
            sys.exit(1)
        if not seq_exists(COORDS_FILE):
            print(f"ERROR: Annotations not found: {COORDS_FILE}")
            sys.exit(1)
        build_dataset()
    else:
        # Multi-sequence mode (reads splits.yaml)
        build_multi_sequence()

    print("\nDataset statistics:")
    print_dataset_stats()
    print("\nNext step: python train_4ch_yolo.py")
