"""
view_raw_events.py — Live OpenCV viewer of raw event camera data.

Shows accumulated event frames (like the Frames/ PNGs) as a real-time movie.
Highlights GREEN = drone annotated, RED = removed/gap window, YELLOW = pre/post flight.

Controls:
    SPACE      — pause / resume
    → / D      — step one frame forward
    ← / A      — step one frame back
    Q / ESC    — quit

Usage:
    cd 4channel_project
    python view_raw_events.py                        # sequence 7 (default)
    python view_raw_events.py --seq 4                # sequence 4
    python view_raw_events.py --seq 4 --delay 100   # slow down
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from evt3_reader import EVT3Reader
from dataset_builder import load_annotations, load_removed_windows
from config import WINDOW_US, IMG_W, IMG_H
from zip_utils import init_sequence, seq_exists

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seq',   type=str, default='7',
                    help='Sequence folder number (default: 7)')
parser.add_argument('--delay', type=int, default=33,
                    help='ms per frame during playback (default: 33)')
parser.add_argument('--start', type=float, default=None,
                    help='Start time in seconds (default: beginning of file)')
parser.add_argument('--end',   type=float, default=None,
                    help='End time in seconds (default: end of file)')
args = parser.parse_args()

BASE = os.path.join(os.path.dirname(__file__), '..', 'data_from_fred', args.seq)
RAW  = os.path.join(BASE, 'Event', 'events.raw')
ANN  = os.path.join(BASE, 'coordinates.txt')

init_sequence(BASE)

if not seq_exists(RAW):
    print(f"ERROR: {RAW} not found")
    sys.exit(1)

# ── Load annotation info ──────────────────────────────────────────────────────

annotations     = load_annotations(ANN)
removed_windows = load_removed_windows(annotations)
ann_times_us    = np.array([a[0] for a in annotations])
ann_start_us    = int(ann_times_us[0])
ann_end_us      = int(ann_times_us[-1])

print(f"\nSequence {args.seq}")
print(f"  Annotated region : {ann_start_us/1e6:.2f}s – {ann_end_us/1e6:.2f}s")
print(f"  Removed windows  : {len(removed_windows)}")
print(f"\nLoading frames... (press Ctrl+C to stop loading early)\n")

# ── Frame status classifier ───────────────────────────────────────────────────

def classify_window(t_start, t_end):
    """Return (label, color_bgr) for this time window."""
    # Pre-flight
    if t_end < ann_start_us:
        return "PRE-FLIGHT (no drone)", (0, 200, 255)   # yellow
    # Post-flight
    if t_start > ann_end_us:
        return "POST-FLIGHT (no drone)", (0, 200, 255)
    # Removed window
    for r0, r1 in removed_windows:
        if t_start < r1 and t_end > r0:
            return "REMOVED FRAME (drone out of frame)", (0, 0, 220)   # red
    # Has annotation?
    idx  = np.searchsorted(ann_times_us, (t_start + t_end) // 2)
    idx  = min(max(idx, 0), len(ann_times_us) - 1)
    gap  = abs(ann_times_us[idx] - (t_start + t_end) // 2)
    if gap < 100_000:
        return "DRONE ANNOTATED", (0, 200, 50)   # green
    return "no annotation", (120, 120, 120)       # gray

# ── Build frames ──────────────────────────────────────────────────────────────

reader  = EVT3Reader(RAW)
frames  = []
t_start_arg = int(args.start * 1e6) if args.start else None
t_end_arg   = int(args.end   * 1e6) if args.end   else None

try:
    for t_start, events in reader.iter_windows(WINDOW_US,
                                               t_start=t_start_arg,
                                               t_end=t_end_arg):
        t_end = t_start + WINDOW_US

        # Accumulate events into grayscale frame
        frame_gray = np.zeros((IMG_H, IMG_W), dtype=np.float32)
        if len(events) > 0:
            np.add.at(frame_gray, (events['y'], events['x']), 1.0)
            max_val = frame_gray.max()
            if max_val > 0:
                frame_gray /= max_val

        img = (frame_gray * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Positive events = white, negative = blue tint
        if len(events) > 0:
            pos = events[events['p'] == 1]
            neg = events[events['p'] == 0]
            pos_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            neg_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            if len(pos) > 0:
                np.add.at(pos_map, (pos['y'], pos['x']), 1.0)
            if len(neg) > 0:
                np.add.at(neg_map, (neg['y'], neg['x']), 1.0)

            # White = positive, blue = negative
            pmax = pos_map.max() or 1
            nmax = neg_map.max() or 1
            img_bgr[:, :, 2] = np.clip(pos_map / pmax * 255, 0, 255).astype(np.uint8)
            img_bgr[:, :, 1] = np.clip(pos_map / pmax * 255, 0, 255).astype(np.uint8)
            img_bgr[:, :, 0] = np.clip(pos_map / pmax * 128 + neg_map / nmax * 255, 0, 255).astype(np.uint8)

        # Status banner
        label, color = classify_window(t_start, t_end)
        banner = np.zeros((48, IMG_W, 3), dtype=np.uint8)
        banner[:] = (30, 30, 30)

        # Colored status indicator bar (left strip)
        banner[:, :6] = color

        cv2.putText(banner,
                    f"t={t_start/1e6:.3f}s  |  {len(events):,} events  |  {label}",
                    (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

        frame_out = np.vstack([banner, img_bgr])

        # Scale to fit screen
        scale = 0.6
        h, w  = frame_out.shape[:2]
        frame_out = cv2.resize(frame_out, (int(w * scale), int(h * scale)))
        frames.append((t_start, frame_out))

        if len(frames) % 100 == 0:
            print(f"  {len(frames)} frames loaded  t={t_start/1e6:.1f}s  "
                  f"[{label}]")

except KeyboardInterrupt:
    print(f"\nStopped early — loaded {len(frames)} frames")

print(f"\n{len(frames)} frames loaded. Opening viewer...")
print("Controls:  SPACE=pause   → =step fwd   ← =step back   Q=quit\n")

# ── Legend ────────────────────────────────────────────────────────────────────

print("Color legend:")
print("  GREEN  strip = drone annotated (drone visible)")
print("  RED    strip = removed frame   (drone out of frame)")
print("  YELLOW strip = pre/post flight (no drone)")
print("  GRAY   strip = unannotated but not removed")
print()
print("  White pixels = positive polarity events")
print("  Blue  pixels = negative polarity events")

# ── Playback ──────────────────────────────────────────────────────────────────

cv2.namedWindow(f"Raw Events — Sequence {args.seq}", cv2.WINDOW_NORMAL)

i      = 0
paused = False

while True:
    if i >= len(frames):
        i = 0

    t_sec, frame = frames[i]
    cv2.imshow(f"Raw Events — Sequence {args.seq}", frame)

    key = cv2.waitKey(1 if paused else args.delay) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Playing")
    elif key in (83, ord('d')):   # → or D
        i += 1
    elif key in (81, ord('a')):   # ← or A
        i = max(0, i - 1)
    elif not paused:
        i += 1

cv2.destroyAllWindows()
print("Done.")
