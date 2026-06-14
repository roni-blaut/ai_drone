"""
evaluate.py — Evaluate FRED baseline vs paper results.

Run from ai_drone/Fred/:
    python evaluate.py              # event camera baseline
    python evaluate.py --mode rgb   # RGB baseline
"""

import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['event', 'rgb'], default='event')
args = parser.parse_args()

YAML_PATH  = os.path.join(HERE,
    'fred_yolo' if args.mode == 'event' else 'fred_rgb_yolo',
    'dataset.yaml')
MODEL_PATH = os.path.join(HERE, 'runs', f'fred_baseline_{args.mode}', 'weights', 'best.pt')
PAPER      = {'event': 87.68, 'rgb': 76.23}

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found — run train.py --mode {args.mode} first\n{MODEL_PATH}"
    )

from ultralytics import YOLO

model   = YOLO(MODEL_PATH)
metrics = model.val(data=YAML_PATH, verbose=False)

map50   = metrics.box.map50 * 100
map5095 = metrics.box.map   * 100
prec    = metrics.box.mp    * 100
rec     = metrics.box.mr    * 100

paper   = PAPER[args.mode]
delta   = map50 - paper
sign    = '+' if delta >= 0 else ''

print("\n" + "=" * 50)
print(f"FRED BASELINE — {args.mode.upper()}")
print("=" * 50)
print(f"  mAP50          : {map50:.2f}%")
print(f"  mAP50:95       : {map5095:.2f}%")
print(f"  Precision      : {prec:.2f}%")
print(f"  Recall         : {rec:.2f}%")
print(f"\n  Paper baseline : {paper:.2f}%")
print(f"  Difference     : {sign}{delta:.2f}%")
print("=" * 50)
