"""
infer.py — Run a trained physics-channel model on a new/unseen sequence zip,
showing the prediction side by side with that sequence's RGB video.

Works for either the 3-channel or 4-channel model — the weights' actual
input channel count is detected and must match the current DRONE_CHANNELS
(config.N_CHANNELS), so the right channels are generated from events.raw.

For each 33ms window: read + filter events the same way build_dataset.py
does (fast_filter → generate_channels), run the model directly on the
in-memory channel array (no PNGs written to disk), draw the predicted box
on a colorized event view (positive=green, negative=red) and on the nearest
PADDED_RGB/ frame, and write both side by side to an .mp4. If the zip has
coordinates.txt, the ground-truth box is overlaid too (best effort).

Usage:
    cd ai_drone
    DRONE_CHANNELS=3 python 4channel_project/infer.py --zip data_from_fred/50.zip
    python 4channel_project/infer.py --zip data_from_fred/50.zip --start 0 --end 20
    python 4channel_project/infer.py --zip C:\\path\\to\\any_sequence.zip --show
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))

from config import (
    IMG_W, IMG_H, WINDOW_US, N_CHANNELS,
    RUNS_DIR, RUN_NAME, DEVICE, IMG_SIZE,
)
from zip_utils import init_sequence, seq_exists, seq_glob, seq_imread
from evt3_reader import EVT3Reader
from filters import fast_filter
from channels import generate_channels
from build_dataset import (
    _find_coords, load_annotations, load_removed_windows,
    in_removed_window, find_annotation,
)
from train_4ch_yolo import _first_conv_in_channels

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python"); sys.exit(1)

from ultralytics import YOLO


# ── Args ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Run a trained physics-channel model on a new sequence zip, "
                "side by side with its RGB video.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument('--zip',    required=True, help='Path to any sequence zip')
parser.add_argument('--weights', default=None,
                    help=f'Model weights (default: {RUNS_DIR}/{RUN_NAME}/weights/best.pt, '
                         f'matching the current DRONE_CHANNELS={N_CHANNELS})')
parser.add_argument('--conf',   type=float, default=0.25)
parser.add_argument('--iou',    type=float, default=0.45)
parser.add_argument('--device', default=DEVICE)
parser.add_argument('--start',  type=float, default=9.8,
                    help='Start time in synced seconds (default: 9.8, matches '
                         "build_dataset.py's drone-start convention)")
parser.add_argument('--end',    type=float, default=None,
                    help='End time in synced seconds (default: end of file)')
parser.add_argument('--save',   default=None,
                    help='Output video path (default: '
                         'runs/infer/<zip_stem>_<N>ch.mp4)')
parser.add_argument('--save-frames', default=None,
                    help='Optional directory to also save per-frame PNGs')
parser.add_argument('--show',   action='store_true', help='Live cv2 preview')
parser.add_argument('--no-video', action='store_true', help='Skip writing the .mp4')
args = parser.parse_args()

zip_stem = os.path.splitext(os.path.basename(args.zip))[0]
seq_dir  = args.zip[:-4] if args.zip.lower().endswith('.zip') else args.zip

weights = args.weights or os.path.join(RUNS_DIR, RUN_NAME, 'weights', 'best.pt')
if not os.path.isfile(weights):
    print(f"ERROR: weights not found: {weights}")
    print("Train first (train_4ch_yolo.py) or pass --weights explicitly.")
    sys.exit(1)

save_path = args.save or os.path.join(RUNS_DIR, 'infer', f'{zip_stem}_{N_CHANNELS}ch.mp4')


# ── Init sequence + model ─────────────────────────────────────────────────────

init_sequence(seq_dir)

RAW_FILE       = os.path.join(seq_dir, 'Event', 'events.raw')
PADDED_RGB_DIR = os.path.join(seq_dir, 'PADDED_RGB')

if not seq_exists(RAW_FILE):
    print(f"ERROR: {RAW_FILE} not found in {args.zip}"); sys.exit(1)

print(f"Loading model: {weights}")
model = YOLO(weights)
model_channels = _first_conv_in_channels(model)
if model_channels != N_CHANNELS:
    print(f"ERROR: weights expect {model_channels} input channels, but "
          f"DRONE_CHANNELS={N_CHANNELS} (config.N_CHANNELS) would generate "
          f"{N_CHANNELS}-channel data from events.raw — these must match.")
    print(f"  Fix: set $env:DRONE_CHANNELS=\"{model_channels}\" and re-run.")
    sys.exit(1)
print(f"  Channels: {N_CHANNELS}  (matches weights)")

reader   = EVT3Reader(RAW_FILE)
ts_shift = reader.ts_shift_us
print(f"  ts_shift_us: {ts_shift:,}")


# ── Ground truth (best effort) ────────────────────────────────────────────────

annotations, removed_windows = [], []
coords_file = _find_coords(seq_dir)
if seq_exists(coords_file):
    annotations     = load_annotations(coords_file)
    removed_windows = load_removed_windows(annotations)
else:
    print("  (no coordinates.txt found — ground truth overlay disabled)")


# ── Index PADDED_RGB/ frames chronologically ──────────────────────────────────

def _rgb_wall_seconds(path):
    """Video_50_19_14_02.028561.jpg → seconds since midnight."""
    name  = os.path.splitext(os.path.basename(path))[0]
    parts = name.split('_')   # ['Video', '50', '19', '14', '02.028561']
    hh, mm, ss = int(parts[-3]), int(parts[-2]), float(parts[-1])
    return hh * 3600 + mm * 60 + ss

_rgb_files = seq_glob(PADDED_RGB_DIR, '*.jpg')
if _rgb_files:
    _rgb_pairs = sorted((_rgb_wall_seconds(p), p) for p in _rgb_files)
    _rgb_t0    = _rgb_pairs[0][0]
    _rgb_rel   = np.array([t - _rgb_t0 for t, _ in _rgb_pairs])
    _rgb_paths = [p for _, p in _rgb_pairs]
    print(f"  PADDED_RGB/: {len(_rgb_files)} frames  "
          f"t=0.000s – {_rgb_rel[-1]:.3f}s (relative)")
else:
    _rgb_rel, _rgb_paths = np.array([]), []
    print("  (no PADDED_RGB/ frames found — right panel will be blank)")


def nearest_rgb(t_sync_sec):
    """Return path of the PADDED_RGB/ frame nearest to synced time t_sync_sec, or None."""
    if len(_rgb_rel) == 0:
        return None
    idx = int(np.argmin(np.abs(_rgb_rel - t_sync_sec)))
    return _rgb_paths[idx]


# ── Drawing helpers ────────────────────────────────────────────────────────────

PRED_COLOR = (0, 255, 255)   # yellow — model prediction
GT_COLOR   = (255, 200, 0)   # cyan   — ground truth

def draw_xyxy(img, x1, y1, x2, y2, color, label=None):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    if label:
        cv2.putText(img, label, (int(x1), max(0, int(y1) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def event_view_bgr(channels):
    """Colorize positive polarity (ch0) → green, negative (ch1) → red."""
    view = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    view[..., 1] = (channels[0] * 255).astype(np.uint8)   # green = positive
    view[..., 2] = (channels[1] * 255).astype(np.uint8)   # red   = negative
    return view


# ── Output setup ──────────────────────────────────────────────────────────────

PANEL_W, PANEL_H = IMG_W // 2, IMG_H // 2
BANNER_H = 40
DIV_W    = 4
FRAME_W  = PANEL_W * 2 + DIV_W
FRAME_H  = BANNER_H + PANEL_H

writer = None
if not args.no_video:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, 30, (FRAME_W, FRAME_H))
    print(f"Saving to: {os.path.abspath(save_path)}")

if args.save_frames:
    os.makedirs(args.save_frames, exist_ok=True)

WIN = f"infer — {zip_stem} ({N_CHANNELS}ch)  SPACE=pause  Q/ESC=quit"
if args.show:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, FRAME_W, FRAME_H)


# ── Main loop ──────────────────────────────────────────────────────────────────

t_start_us = int(args.start * 1e6) + ts_shift
t_end_us   = int(args.end   * 1e6) + ts_shift if args.end is not None else None

print(f"\nInferring t={args.start:.3f}s – "
      f"{'end' if args.end is None else f'{args.end:.3f}s'} (synced time)\n")

n_frames = n_detections = 0
paused = False

for t_start, events in reader.iter_windows(WINDOW_US, t_start=t_start_us, t_end=t_end_us):
    t_end        = t_start + WINDOW_US
    t_sync_start = t_start - ts_shift
    t_sync_end   = t_end   - ts_shift
    t_sync_sec   = t_sync_start / 1e6

    if in_removed_window(t_sync_start, t_sync_end, removed_windows):
        continue

    events_clean = fast_filter(events)
    channels      = generate_channels(events_clean, t_start, t_end)
    img_hwc       = (channels.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    results = model.predict(source=img_hwc, conf=args.conf, iou=args.iou,
                            imgsz=IMG_SIZE, device=args.device, verbose=False)
    boxes = results[0].boxes

    gt_box = find_annotation(annotations, t_sync_start, t_sync_end) if annotations else None

    # ── Left panel: colorized event view ──
    ev_bgr = event_view_bgr(channels)
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        draw_xyxy(ev_bgr, x1, y1, x2, y2, PRED_COLOR, f"{b.conf.item():.2f}")
        n_detections += 1
    if gt_box:
        draw_xyxy(ev_bgr, *gt_box, GT_COLOR, "GT")
    ev_panel = cv2.resize(ev_bgr, (PANEL_W, PANEL_H))
    cv2.putText(ev_panel, "Event view (green=pos red=neg)",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # ── Right panel: nearest RGB frame ──
    rgb_path = nearest_rgb(t_sync_sec)
    if rgb_path is not None:
        rgb_bgr = seq_imread(rgb_path)
        if rgb_bgr is None:
            rgb_bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    else:
        rgb_bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        draw_xyxy(rgb_bgr, x1, y1, x2, y2, PRED_COLOR, f"{b.conf.item():.2f}")
    if gt_box:
        draw_xyxy(rgb_bgr, *gt_box, GT_COLOR, "GT")
    rgb_panel = cv2.resize(rgb_bgr, (PANEL_W, PANEL_H))
    cv2.putText(rgb_panel, "PADDED_RGB/",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    div    = np.full((PANEL_H, DIV_W, 3), 40, dtype=np.uint8)
    panels = np.hstack([ev_panel, div, rgb_panel])

    banner = np.zeros((BANNER_H, FRAME_W, 3), dtype=np.uint8)
    cv2.putText(banner,
                f"t={t_sync_sec:.3f}s  events={len(events):,}  "
                f"detections={len(boxes)}  [yellow=prediction cyan=ground truth]",
                (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    frame_out = np.vstack([banner, panels])
    n_frames += 1

    if writer:
        writer.write(frame_out)
    if args.save_frames:
        cv2.imwrite(os.path.join(args.save_frames, f'frame_{n_frames:05d}.png'), frame_out)

    if args.show:
        cv2.imshow(WIN, frame_out)
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                cv2.waitKey(0)

if writer:
    writer.release()
if args.show:
    cv2.destroyAllWindows()

print(f"\nDone.  {n_frames} windows processed, {n_detections} detection(s) drawn.")
if not args.no_video:
    print(f"Saved: {os.path.abspath(save_path)}")
