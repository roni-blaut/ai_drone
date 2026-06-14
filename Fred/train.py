"""
train.py — Train YOLO v11n on FRED dataset (paper baseline).

Two modes:
  --mode event   Train on event frames  → target ~87.68 mAP50
  --mode rgb     Train on RGB frames    → target ~76.23 mAP50

Run from ai_drone/Fred/:
    $env:KMP_DUPLICATE_LIB_OK="TRUE"
    python train.py              # event camera baseline
    python train.py --mode rgb   # RGB baseline
"""

import os
import argparse

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
        f"dataset.yaml not found — run build_dataset.py --mode {args.mode} first\n{YAML_PATH}"
    )

try:
    import torch
    GPU    = torch.cuda.is_available()
    DEVICE = 0 if GPU else 'cpu'
    BATCH  = 16 if GPU else 4
    print(f"Device: {'GPU — ' + torch.cuda.get_device_name(0) if GPU else 'CPU'}")
except ImportError:
    DEVICE, BATCH = 'cpu', 4

from ultralytics import YOLO

last_pt = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'last.pt')
best_pt = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')

if os.path.exists(last_pt):
    print(f"Checkpoint found: {last_pt}")
    print("Resuming training from last checkpoint...")
    model = YOLO(last_pt)
    resume = True
else:
    print("No checkpoint found — starting fresh training...")
    model = YOLO('yolo11n.pt')
    resume = False

results = model.train(
    data     = YAML_PATH,
    epochs   = 100,
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
