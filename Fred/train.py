"""
train.py — Train YOLO v11n on FRED dataset (paper baseline).

Images are loaded directly from zip files via a patched imread — no image
files need to be on disk (only the small label .txt files are written).

Two modes:
  --mode event   Train on event frames  → target ~87.68 mAP50
  --mode rgb     Train on RGB frames    → target ~76.23 mAP50

Run from ai_drone/Fred/:
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    python train.py              # event camera baseline
    python train.py --mode rgb   # RGB baseline
"""

import os
import sys
import argparse

if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['event', 'rgb'], default='event')
    args = parser.parse_args()

    YAML_PATH = os.path.join(HERE,
        'fred_yolo' if args.mode == 'event' else 'fred_rgb_yolo',
        'dataset.yaml')
    RUNS_DIR  = os.path.join(HERE, 'runs')
    RUN_NAME  = f'fred_baseline_{args.mode}'

    PAPER = {'event': 87.68, 'rgb': 76.23}

    if not os.path.exists(YAML_PATH):
        raise FileNotFoundError(
            f"dataset.yaml not found — run build_index.py --mode {args.mode} first\n{YAML_PATH}"
        )

    # ── Patch Ultralytics imread to serve images from zip ─────────────────────
    # Images are stored in zip files under data_from_fred/{seq}/...
    # build_index.py wrote paths pointing into the zip (virtual paths on disk).
    # multi_seq_imread transparently reads from the correct zip based on the path.

    sys.path.insert(0, os.path.join(HERE, '..', 'common'))
    from config import DEVICE, BATCH, EPOCHS
    from zip_utils import multi_seq_imread

    import cv2 as _cv2

    def _patched_imread(path, flags=_cv2.IMREAD_COLOR):
        return multi_seq_imread(path, flags)

    import ultralytics.utils.patches as _patches
    import ultralytics.data.base as _base
    _patches.imread = _patched_imread
    _base.imread    = _patched_imread

    # ─────────────────────────────────────────────────────────────────────────

    from ultralytics import YOLO

    last_pt = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'last.pt')

    if os.path.exists(last_pt):
        print(f"Checkpoint found: {last_pt} — resuming")
        model  = YOLO(last_pt)
        resume = True
    else:
        print("No checkpoint — starting fresh")
        model  = YOLO('yolo11n.pt')
        resume = False

    results = model.train(
        data     = YAML_PATH,
        epochs   = EPOCHS,
        batch    = BATCH,
        imgsz    = 640,
        device   = DEVICE,
        project  = RUNS_DIR,
        name     = RUN_NAME,
        exist_ok = True,
        patience = 20,
        resume   = resume,
        verbose  = True,
    )

    print(f"\nTraining complete.")
    print(f"Best model : {os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')}")
    print(f"Paper ref  : {PAPER[args.mode]:.2f} mAP50")
    print(f"\nNext step  : python evaluate.py --mode {args.mode}")
