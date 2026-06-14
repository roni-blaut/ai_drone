"""
evaluate.py — Compare 4-channel model vs FRED paper baseline.

Runs evaluation on the validation split and prints:
  - mAP50  (main metric — matches FRED paper)
  - mAP50:95
  - Precision, Recall
  - Comparison against paper's 87.68 mAP50

Also runs ablation: evaluates each channel combination to show
which channel contributes the most.

Usage:
    python evaluate.py                        # evaluate 4-channel model
    python evaluate.py --ablation             # run full ablation study
    python evaluate.py --model path/to/best.pt
"""

import os
import sys
import argparse
import numpy as np
import torch

from config import (
    DATASET_DIR, RUNS_DIR, RUN_NAME,
    IMG_SIZE, DEVICE, N_CHANNELS,
    DEBUG_MODE
)

PAPER_MAP50 = 87.68   # FRED paper baseline (YOLO v11, 1-channel event frame)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(model_path=None):
    """Evaluate trained model on validation split."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run train_4ch_yolo.py first")
        return

    print(f"Evaluating: {model_path}")
    if DEBUG_MODE:
        print(f"  [DEBUG] Dataset : {os.path.join(DATASET_DIR, 'dataset.yaml')}")
        print(f"  [DEBUG] Device  : {DEVICE}")
        print(f"  [DEBUG] img_size: {IMG_SIZE}")
    model = YOLO(model_path)

    metrics = model.val(
        data   = os.path.join(DATASET_DIR, 'dataset.yaml'),
        imgsz  = IMG_SIZE,
        device = DEVICE,
        split  = 'val',
    )

    map50    = metrics.box.map50 * 100
    map5095  = metrics.box.map   * 100
    prec     = metrics.box.mp    * 100
    recall   = metrics.box.mr    * 100

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  mAP50      : {map50:.2f}%")
    print(f"  mAP50:95   : {map5095:.2f}%")
    print(f"  Precision  : {prec:.2f}%")
    print(f"  Recall     : {recall:.2f}%")
    print()
    print(f"  Paper baseline (1-channel): {PAPER_MAP50:.2f}%")
    diff = map50 - PAPER_MAP50
    if diff > 0:
        print(f"  Improvement: +{diff:.2f}% ✓")
    else:
        print(f"  Difference : {diff:.2f}%")
    print("=" * 50)

    return metrics


# ── Ablation study ────────────────────────────────────────────────────────────

def run_ablation():
    """
    Train and evaluate each channel combination.

    Combinations tested:
      A: [Ch1] only — positive polarity
      B: [Ch1, Ch2] — polarity pair
      C: [Ch1, Ch2, Ch3] — polarity + rotor
      D: [Ch1, Ch2, Ch3, Ch4] — full 4-channel (our proposal)
      E: [Ch3] only — rotor map alone
      F: [Ch4] only — time surface alone
    """
    print("=" * 50)
    print("ABLATION STUDY")
    print("=" * 50)
    print("This trains separate models for each channel combination.")
    print("Each training run takes significant time.")
    print()

    combinations = [
        ("A", [0],          "Ch1 only (positive polarity)"),
        ("B", [0, 1],       "Ch1+Ch2 (polarity pair)"),
        ("C", [0, 1, 2],    "Ch1+Ch2+Ch3 (+ rotor map)"),
        ("D", [0, 1, 2, 3], "Ch1+Ch2+Ch3+Ch4 (full proposal)"),
        ("E", [2],          "Ch3 only (rotor map)"),
        ("F", [3],          "Ch4 only (time surface)"),
    ]

    results = {}

    for label, channel_indices, description in combinations:
        print(f"\nRunning ablation {label}: {description}")
        print(f"  Channels: {[i+1 for i in channel_indices]}")

        map50 = _train_ablation(label, channel_indices)
        results[label] = (description, map50)

    print("\n" + "=" * 50)
    print("ABLATION RESULTS")
    print("=" * 50)
    print(f"  {'Label':4s}  {'mAP50':6s}  Description")
    print(f"  {'─'*4}  {'─'*6}  {'─'*40}")
    print(f"  {'BASE':4s}  {PAPER_MAP50:6.2f}  FRED paper (1-channel accumulated)")
    for label, (desc, map50) in sorted(results.items()):
        marker = " ← best" if map50 == max(v[1] for v in results.values()) else ""
        print(f"  {label:4s}  {map50:6.2f}  {desc}{marker}")
    print("=" * 50)

    return results


def _train_ablation(label, channel_indices):
    """Train one ablation model and return its mAP50."""
    from ultralytics import YOLO
    from train_4ch_yolo import patch_yolo_input_channels

    n_ch      = len(channel_indices)
    run_name  = f"{RUN_NAME}_ablation_{label}"

    print(f"  Training {n_ch}-channel model...")

    model = YOLO("yolo11n.pt")
    model = patch_yolo_input_channels(model, n_ch)

    # TODO: need to build a dataset with only the selected channels
    # For now, train on the full dataset and note which channels are used
    # A proper ablation would rebuild the dataset with only selected channels
    # This is left as an exercise — see dataset_builder.py

    print(f"  NOTE: Full ablation requires rebuilding dataset with "
          f"channels {[i+1 for i in channel_indices]} only.")
    print(f"  Skipping training for this demo — returning placeholder.")
    return 0.0


# ── Visualize predictions ─────────────────────────────────────────────────────

def visualize_predictions(model_path=None, n_samples=5):
    """
    Show 4-channel input + YOLO prediction for a few validation samples.
    Saves images to ./predictions/
    """
    import cv2
    import glob

    if model_path is None:
        model_path = os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')

    from ultralytics import YOLO
    model = YOLO(model_path)

    val_dir   = os.path.join(DATASET_DIR, 'images', 'val')
    png_files = sorted(glob.glob(os.path.join(val_dir, '*.png')))[:n_samples]

    os.makedirs('./predictions', exist_ok=True)

    from PIL import Image as _PIL

    for png_path in png_files:
        channels = np.array(_PIL.open(png_path)).transpose(2, 0, 1).astype(np.float32) / 255.0

        results = model.predict(
            source  = png_path,
            conf    = 0.25,
            verbose = False,
        )

        # Create 4-panel visualization
        h, w    = channels.shape[1], channels.shape[2]
        panel   = np.zeros((h*2, w*2), dtype=np.uint8)
        names   = ["Positive", "Negative", "Rotor", "TimeSurface"]

        for i in range(4):
            r, c  = divmod(i, 2)
            ch_img = (channels[i] * 255).astype(np.uint8)
            panel[r*h:(r+1)*h, c*w:(c+1)*w] = ch_img

            cv2.putText(panel, names[i],
                        (c*w + 10, r*h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

        # Draw prediction box on each panel
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = box.conf[0].item()
                for r in range(2):
                    for c in range(2):
                        cv2.rectangle(panel,
                                      (c*w + x1, r*h + y1),
                                      (c*w + x2, r*h + y2),
                                      (255, 255, 255), 2)
                        cv2.putText(panel,
                                    f"{conf:.2f}",
                                    (c*w + x1, r*h + y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1)

        out_path = os.path.join('./predictions',
                                os.path.basename(npy_path).replace('.npy', '.png'))
        cv2.imwrite(out_path, panel)
        print(f"  Saved: {out_path}")
        if DEBUG_MODE:
            for result in results:
                for box in result.boxes:
                    conf = box.conf[0].item()
                    cls  = int(box.cls[0].item())
                    print(f"  [DEBUG]   box conf={conf:.3f}  class={cls}  "
                          f"xyxy={[round(v,1) for v in box.xyxy[0].tolist()]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    args = parser.parse_args()

    if args.ablation:
        run_ablation()
    elif args.visualize:
        visualize_predictions(model_path=args.model)
    else:
        evaluate(model_path=args.model)
