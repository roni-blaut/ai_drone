"""
make_filter_movie.py — Live OpenCV viewer: before vs after filter.

Each frame = one 33ms event window shown as a 4-channel 2x2 grid.
Left panel = raw noisy events.  Right panel = after refractory filter.

Controls:
    SPACE  — pause / resume
    →      — step one frame forward (while paused)
    Q/ESC  — quit

Usage:
    cd 4channel_project
    python make_filter_movie.py                      # drone segment 9.87s–35s
    python make_filter_movie.py --start 0 --end 10   # first 10 seconds
    python make_filter_movie.py --delay 50           # ms per frame (default 33)
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evt3_reader import EVT3Reader
from filters import fast_filter
from channels import generate_channels
from config import RAW_FILE, WINDOW_US, IMG_W, IMG_H

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, default=9.87,
                    help='Start time in seconds (default: 9.87 — drone appears)')
parser.add_argument('--end',   type=float, default=35.0,
                    help='End time in seconds (default: 35.0)')
parser.add_argument('--delay', type=int,   default=33,
                    help='Milliseconds per frame (default: 33 = ~30fps). '
                         'Increase to slow down.')
args = parser.parse_args()

T_START_US = int(args.start * 1_000_000)
T_END_US   = int(args.end   * 1_000_000)
DELAY_MS   = args.delay

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_banner(img, title, subtitle=""):
    bar = np.zeros((50, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, title,    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(bar, subtitle, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return np.vstack([bar, img])


CHANNEL_NAMES = [
    "Ch1  Positive polarity",
    "Ch2  Negative polarity",
    "Ch3  Rotor map",
    "Ch4  Time surface",
]

def build_frame(raw_events, clean_events, t_start):
    t_end   = t_start + WINDOW_US
    removed = len(raw_events) - len(clean_events)
    pct     = 100 * removed / max(len(raw_events), 1)
    t_sec   = t_start / 1e6

    ch_raw   = generate_channels(raw_events,   t_start, t_end)
    ch_clean = generate_channels(clean_events, t_start, t_end)

    rows = []
    for i, name in enumerate(CHANNEL_NAMES):
        before_gray = (ch_raw[i]   * 255).astype(np.uint8)
        after_gray  = (ch_clean[i] * 255).astype(np.uint8)

        before_bgr  = cv2.cvtColor(before_gray, cv2.COLOR_GRAY2BGR)
        after_bgr   = cv2.cvtColor(after_gray,  cv2.COLOR_GRAY2BGR)

        # Label each panel
        cv2.putText(before_bgr, f"{name}  BEFORE",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)
        cv2.putText(after_bgr,  f"{name}  AFTER",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)

        # Draw divider line between panels
        divider = np.zeros((IMG_H, 3, 3), dtype=np.uint8)
        divider[:] = (80, 80, 80)

        row = np.hstack([before_bgr, divider, after_bgr])
        rows.append(row)

        # Thin separator between channel rows
        sep = np.zeros((2, row.shape[1], 3), dtype=np.uint8)
        sep[:] = (60, 60, 60)
        rows.append(sep)

    frame = np.vstack(rows[:-1])  # drop last separator

    # Top banner with timestamp + stats
    banner = np.zeros((44, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner,
                f"t={t_sec:.3f}s    raw={len(raw_events):,}  "
                f"clean={len(clean_events):,}  removed={pct:.0f}%",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    frame = np.vstack([banner, frame])

    scale = 0.4
    h, w  = frame.shape[:2]
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

# ── Pre-load all frames ───────────────────────────────────────────────────────

print(f"Loading {args.start:.2f}s – {args.end:.2f}s ...")
print("(building frames — this may take ~30s for a long segment)\n")

reader = EVT3Reader(RAW_FILE)
frames = []

for t_start, raw_events in reader.iter_windows(WINDOW_US,
                                                t_start=T_START_US,
                                                t_end=T_END_US):
    if len(raw_events) == 0:
        continue

    clean_events = fast_filter(raw_events)
    frame        = build_frame(raw_events, clean_events, t_start)
    frames.append(frame)

    if len(frames) % 30 == 0:
        print(f"  {len(frames)} frames ready  "
              f"t={t_start/1e6:.2f}s  "
              f"{len(raw_events):,} → {len(clean_events):,} events")

print(f"\n{len(frames)} frames loaded. Opening viewer...")
print("Controls:  SPACE=pause/resume   →=step frame   Q/ESC=quit\n")

# ── Playback ──────────────────────────────────────────────────────────────────

cv2.namedWindow("Before vs After Filter", cv2.WINDOW_NORMAL)

i      = 0
paused = False

while True:
    if i >= len(frames):
        i = 0   # loop

    cv2.imshow("Before vs After Filter", frames[i])

    key = cv2.waitKey(1 if paused else DELAY_MS) & 0xFF

    if key == ord('q') or key == 27:       # Q or ESC — quit
        break
    elif key == ord(' '):                  # SPACE — pause/resume
        paused = not paused
        print("Paused" if paused else "Playing")
    elif key == 83 or key == ord('d'):     # → arrow or D — step forward
        i += 1
    elif key == 81 or key == ord('a'):     # ← arrow or A — step back
        i = max(0, i - 1)
    elif not paused:
        i += 1

cv2.destroyAllWindows()
print("Done.")
