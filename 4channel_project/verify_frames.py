"""
verify_frames.py — Measure pixel-level match between events.raw and Event/Frames/.

For N evenly-spaced frames, reconstructs each from events.raw and computes
Mean Absolute Error vs the original Frames/ PNG.

If aligned: MAE < 5  (nearly identical)
If off by 1 frame: MAE ~20-40
If badly misaligned: MAE ~60-80 (like comparing random images)

Usage:
    cd 4channel_project
    python verify_frames.py              # 20 frames from drone segment
    python verify_frames.py --n 50       # check 50 frames
    python verify_frames.py --start 0    # include countdown region
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evt3_reader import EVT3Reader
from config import IMG_W, IMG_H, WINDOW_US
from zip_utils import init_sequence, seq_glob, seq_imread

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python"); sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seq',   default='7')
parser.add_argument('--n',     type=int,   default=20,   help='frames to check')
parser.add_argument('--start', type=float, default=9.8,  help='start time in Frames/ seconds')
parser.add_argument('--end',   type=float, default=60.0, help='end time in Frames/ seconds')
args = parser.parse_args()

BASE       = os.path.join(os.path.dirname(__file__), '..', 'data_from_fred', args.seq)
RAW_FILE   = os.path.join(BASE, 'Event', 'events.raw')
FRAMES_DIR = os.path.join(BASE, 'Event', 'Frames')

init_sequence(BASE)

# ── Load ts_shift_us ──────────────────────────────────────────────────────────

reader  = EVT3Reader(RAW_FILE)
SKIP_US = reader.ts_shift_us
print(f"ts_shift_us = {SKIP_US:,} µs ({SKIP_US/1e6:.3f} s)")
print(f"Raw window:   raw_t = frames_t + {SKIP_US/1e6:.3f}s\n")

# ── Load Frames/ index (numeric sort) ─────────────────────────────────────────

all_pairs  = sorted([(int(os.path.basename(p).split('_frame_')[1][:-4]), p)
                      for p in seq_glob(FRAMES_DIR, '*.png')])
frame_ts   = np.array([t for t, _ in all_pairs], dtype=np.int64)
frame_paths = [p for _, p in all_pairs]

start_us = int(args.start * 1e6)
end_us   = int(args.end   * 1e6)

# Pick N evenly-spaced frames in the requested range
in_range = [(t, p) for t, p in zip(frame_ts, frame_paths) if start_us <= t <= end_us]
step     = max(1, len(in_range) // args.n)
samples  = in_range[::step][:args.n]

print(f"Checking {len(samples)} frames  t={args.start:.1f}s – {args.end:.1f}s\n")
print(f"{'Frame time':>12}  {'Events':>8}  {'MAE':>7}  Result")
print("-" * 50)

# ── Reconstruct and compare ───────────────────────────────────────────────────

def reconstruct(evs):
    pos = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    neg = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    if len(evs):
        p1 = evs[evs['p'] == 1]; p0 = evs[evs['p'] == 0]
        if len(p1): np.add.at(pos, (p1['y'], p1['x']), 1.0)
        if len(p0): np.add.at(neg, (p0['y'], p0['x']), 1.0)
        m = pos.max(); pos = pos / m if m else pos
        m = neg.max(); neg = neg / m if m else neg
    return np.clip(pos * 255 + neg * 120, 0, 255).astype(np.uint8)

maes   = []
panels = []

for frames_t, frames_path in samples:
    # Raw window corresponding to this Frames/ timestamp
    raw_t0 = frames_t - WINDOW_US + SKIP_US   # window that ends at frames_t (raw time)
    raw_t1 = raw_t0 + WINDOW_US

    evs   = reader.read_window(raw_t0, raw_t1)
    recon = reconstruct(evs)
    orig  = seq_imread(frames_path, cv2.IMREAD_GRAYSCALE)
    mae   = float(np.mean(np.abs(recon.astype(int) - orig.astype(int))))
    maes.append(mae)

    grade = "✓ good" if mae < 10 else ("~ ok" if mae < 25 else "✗ mismatch")
    print(f"{frames_t/1e6:>12.3f}s  {len(evs):>8,}  {mae:>7.1f}  {grade}")

    # thumbnail for contact sheet
    tw = 200
    th = int(IMG_H * tw / IMG_W)
    o  = cv2.resize(orig,  (tw, th))
    r  = cv2.resize(recon, (tw, th))
    diff = cv2.resize(np.abs(orig.astype(int) - recon.astype(int)).clip(0,255).astype(np.uint8), (tw, th))

    bar = np.zeros((18, tw * 3, 3), dtype=np.uint8)
    col = (0,220,80) if mae < 10 else (0,200,255) if mae < 25 else (0,80,255)
    cv2.putText(bar, f"t={frames_t/1e6:.1f}s  MAE={mae:.0f}",
                (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)
    panel = np.vstack([bar, np.hstack([
        cv2.cvtColor(o, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(r, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR) * 3  # amplify diff
    ])])
    panels.append(panel)

# ── Summary ───────────────────────────────────────────────────────────────────

print("-" * 50)
print(f"Mean MAE : {np.mean(maes):.1f}   Median: {np.median(maes):.1f}   "
      f"Max: {np.max(maes):.1f}")
print()
if np.mean(maes) < 10:
    print("✓  ALIGNED  — raw reconstruction matches Frames/ well")
elif np.mean(maes) < 25:
    print("~  CLOSE    — minor differences (normalization or hot pixels)")
else:
    print("✗  MISMATCH — timestamps are still off")

# ── Contact sheet: original | reconstructed | diff (×3) ──────────────────────

cols = 5
rows_n = (len(panels) + cols - 1) // cols
blank  = np.zeros_like(panels[0])
while len(panels) % cols:
    panels.append(blank)

sheet = np.vstack([np.hstack(panels[i*cols:(i+1)*cols]) for i in range(rows_n)])

out = "./verify_frames_output.png"
cv2.imwrite(out, sheet)
print(f"\nContact sheet saved: {os.path.abspath(out)}")
print("Columns per frame: original | reconstructed | diff (×3 brightness)")

cv2.namedWindow("Verify Frames", cv2.WINDOW_NORMAL)
cv2.imshow("Verify Frames", sheet)
cv2.waitKey(0)
cv2.destroyAllWindows()
