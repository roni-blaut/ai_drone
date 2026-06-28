"""
build_dataset.py — Build FRED baseline dataset index (no image copying).

Images are served directly from the zip files at training time via a patched
imread in train.py — no PNG/JPG files are copied to disk.
Only the small YOLO label .txt files are written to disk.

Two modes:
  --mode event   Event/Frames/ PNGs + Event_YOLO/ labels  (default, ~87.68 mAP50)
  --mode rgb     PADDED_RGB/ JPGs  + RGB_YOLO/ labels     (~76.23 mAP50)

Reads data_from_fred/splits.yaml for train/val/test assignment.
Set percentages with:
    python 4channel_project/make_catalog.py --auto-split --train 70 --val 20 --test 10

Run from ai_drone/Fred/:
    python build_dataset.py              # event mode
    python build_dataset.py --mode rgb   # RGB mode
"""

import os
import sys
import argparse

HERE           = os.path.dirname(os.path.abspath(__file__))
DATA_FROM_FRED = os.path.normpath(os.path.join(HERE, '..', 'data_from_fred'))
SPLITS_YAML    = os.path.join(DATA_FROM_FRED, 'splits.yaml')

sys.path.insert(0, os.path.join(HERE, '..', '4channel_project'))
from zip_utils import init_sequence, seq_glob, seq_exists, seq_open_lines


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

    # Only label directories need to exist on disk — images are read from zip
    for split in ('train', 'val', 'test'):
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)

    split_img_paths = {'train': [], 'val': [], 'test': []}
    totals  = {'train': 0, 'val': 0, 'test': 0}
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

            recording = False
            for img_path in img_files:
                stem       = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, stem + '.txt')

                # Skip frames before first drone appearance
                if not recording:
                    if seq_exists(label_path) and ''.join(seq_open_lines(label_path)).strip():
                        recording = True
                    else:
                        n_skip += 1
                        continue

                if not seq_exists(label_path):
                    n_skip += 1
                    continue

                # Write label to disk (small txt file)
                dst_stem  = f"s{seq_num}_{stem}"
                lbl_dst   = os.path.join(out_dir, 'labels', split, dst_stem + '.txt')
                label_content = ''.join(seq_open_lines(label_path))
                with open(lbl_dst, 'w') as f:
                    f.write(label_content)

                # Image stays in zip — record the real zip-internal path for train.txt
                # multi_seq_imread in train.py will serve it directly from the zip
                split_img_paths[split].append(os.path.abspath(img_path))

                if label_content.strip():
                    n_drone += 1
                else:
                    n_empty += 1
                totals[split] += 1

    # Write split index txt files (absolute paths to zip-internal images)
    for split, paths in split_img_paths.items():
        txt_path = os.path.join(out_dir, f'{split}.txt')
        with open(txt_path, 'w') as f:
            f.writelines(p + '\n' for p in paths)

    # Write mode so train.py patch knows which folder to read from
    with open(os.path.join(out_dir, 'mode.txt'), 'w') as f:
        f.write(mode)

    total = sum(totals.values())
    print(f"\nDataset  → {out_dir}")
    print(f"  Train        : {totals['train']}  (labels on disk, images from zip)")
    print(f"  Val          : {totals['val']}")
    print(f"  Test         : {totals['test']}")
    print(f"  With drone   : {n_drone}  ({100*n_drone/max(total,1):.0f}%)")
    print(f"  Empty labels : {n_empty}  ({100*n_empty/max(total,1):.0f}%)")
    print(f"  Skipped      : {n_skip}  (pre-recording + missing labels)")
    print(f"  Images copied: 0  (served from zip at training time)")

    _write_yaml(out_dir, mode)
    print(f"\nNext step: python train.py --mode {mode}")


def _write_yaml(out_dir, mode):
    abs_dir = os.path.abspath(out_dir)
    desc    = "event frames (33ms)" if mode == 'event' else "RGB frames (30fps)"
    content = f"""# FRED Baseline Dataset — {desc}
# Generated by Fred/build_dataset.py --mode {mode}
# Images served from zip at training time (no copies on disk).
# Sequence splits defined in data_from_fred/splits.yaml

path:  {abs_dir}
train: train.txt
val:   val.txt
test:  test.txt

channels: 3
nc: 1
names: ['drone']
"""
    path = os.path.join(out_dir, 'dataset.yaml')
    with open(path, 'w') as f:
        f.write(content)
    print(f"  dataset.yaml : {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['event', 'rgb'], default='event',
                        help='event = Event/Frames + Event_YOLO,  rgb = PADDED_RGB + RGB_YOLO')
    args = parser.parse_args()
    build(args.mode)
