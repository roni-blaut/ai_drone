"""
build_dataset.py — Build FRED baseline dataset from pre-extracted frames.

Two modes:
  --mode event   Use Event/Frames/ PNGs + Event_YOLO/ labels  (default)
                 → target: ~87.68 mAP50 (paper result)
  --mode rgb     Use PADDED_RGB/ JPGs + RGB_YOLO/ labels
                 → target: ~76.23 mAP50 (paper result for RGB)

Reads data_from_fred/splits.yaml to determine which sequences go to train/val/test.
Set percentages with:
    python 4channel_project/make_catalog.py --auto-split --train 70 --val 20 --test 10

Run from ai_drone/Fred/:
    python build_dataset.py              # event camera mode
    python build_dataset.py --mode rgb   # RGB camera mode
"""

import os
import sys
import shutil
import argparse

HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_FROM_FRED = os.path.normpath(os.path.join(HERE, '..', 'data_from_fred'))
SPLITS_YAML    = os.path.join(DATA_FROM_FRED, 'splits.yaml')

sys.path.insert(0, os.path.join(HERE, '..', '4channel_project'))
from zip_utils import init_sequence, seq_glob, seq_exists, seq_open_lines


def _copy_file(src, dst):
    """Copy a file from disk or zip to dst (always disk)."""
    if os.path.isfile(src):
        shutil.copy(src, dst)
    else:
        from zip_utils import _ACTIVE_SEQ
        with open(dst, 'wb') as out:
            out.write(_ACTIVE_SEQ.read_bytes(src))


def build(mode='event'):
    try:
        import yaml as _yaml
    except ImportError:
        print("ERROR: pip install pyyaml"); return

    if not os.path.isfile(SPLITS_YAML):
        print(f"ERROR: splits.yaml not found: {SPLITS_YAML}")
        print("Run: python 4channel_project/make_catalog.py --auto-split --train 70 --val 20 --test 10")
        return

    with open(SPLITS_YAML) as f:
        splits = _yaml.safe_load(f) or {}

    if mode == 'event':
        out_dir = os.path.join(HERE, 'fred_yolo')
        img_ext = '.png'
    else:
        out_dir = os.path.join(HERE, 'fred_rgb_yolo')
        img_ext = '.jpg'

    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    totals = {'train': 0, 'val': 0, 'test': 0}
    n_drone = n_empty = n_skip = 0

    for split in ('train', 'val', 'test'):
        seq_list = splits.get(split) or []
        for seq_num in seq_list:
            seq_dir = os.path.join(DATA_FROM_FRED, str(seq_num))
            init_sequence(seq_dir)

            if mode == 'event':
                img_dir   = os.path.join(seq_dir, 'Event', 'Frames')
                label_dir = os.path.join(seq_dir, 'Event_YOLO')
            else:
                img_dir   = os.path.join(seq_dir, 'PADDED_RGB')
                label_dir = os.path.join(seq_dir, 'RGB_YOLO')

            if not seq_exists(img_dir):
                print(f"  WARNING: {img_dir} not found — skipping seq {seq_num}")
                continue

            img_files = seq_glob(img_dir, f'*{img_ext}')
            if not img_files:
                print(f"  WARNING: No {img_ext} files in seq {seq_num} — skipping")
                continue

            print(f"  Seq {seq_num:>4} → {split}  ({len(img_files)} frames)")

            # Skip frames before first drone appearance (first non-empty label)
            recording = False
            for img_path in img_files:
                stem       = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, stem + '.txt')

                if not recording:
                    if seq_exists(label_path) and ''.join(seq_open_lines(label_path)).strip():
                        recording = True
                    else:
                        n_skip += 1
                        continue

                if not seq_exists(label_path):
                    n_skip += 1
                    continue

                # Prefix with seq number to avoid filename collisions across sequences
                dst_stem = f"s{seq_num}_{stem}"
                _copy_file(img_path,   os.path.join(out_dir, 'images', split, dst_stem + img_ext))
                _copy_file(label_path, os.path.join(out_dir, 'labels', split, dst_stem + '.txt'))

                label_content = ''.join(seq_open_lines(label_path)).strip()
                if label_content:
                    n_drone += 1
                else:
                    n_empty += 1

                totals[split] += 1

    total = sum(totals.values())
    print(f"\nDataset  → {out_dir}")
    print(f"  Train        : {totals['train']}")
    print(f"  Val          : {totals['val']}")
    print(f"  Test         : {totals['test']}")
    print(f"  With drone   : {n_drone}  ({100*n_drone/max(total,1):.0f}%)")
    print(f"  Empty labels : {n_empty}  ({100*n_empty/max(total,1):.0f}%)")
    print(f"  Skipped      : {n_skip}  (pre-recording + missing labels)")

    _write_yaml(out_dir, mode)
    print(f"\nNext step: python train.py --mode {mode}")


def _write_yaml(out_dir, mode):
    abs_dir = os.path.abspath(out_dir)
    desc    = "event frames (33ms)" if mode == 'event' else "RGB frames (30fps)"
    yaml    = f"""# FRED Baseline Dataset — {desc}
# Generated by Fred/build_dataset.py --mode {mode}
# Sequence splits defined in data_from_fred/splits.yaml

path:  {abs_dir}
train: images/train
val:   images/val
test:  images/test

channels: 3
nc: 1
names: ['drone']
"""
    path = os.path.join(out_dir, 'dataset.yaml')
    with open(path, 'w') as f:
        f.write(yaml)
    print(f"  dataset.yaml : {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['event', 'rgb'], default='event',
                        help='event = Event/Frames + Event_YOLO,  rgb = PADDED_RGB + RGB_YOLO')
    args = parser.parse_args()
    build(args.mode)
