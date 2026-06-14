"""
raw_to_movie.py — Compare events.raw reconstruction vs Event/Frames/ PNGs.

Shows a side-by-side OpenCV movie:
  LEFT  = original PNG from Event/Frames/ (or blank if no matching file)
  RIGHT = frame reconstructed from events.raw

Also saves raw_events.mp4.

Controls:  SPACE=pause   Q/ESC=quit

Usage:
    cd 4channel_project
    python raw_to_movie.py                         # full sequence 7
    python raw_to_movie.py --start 9.8 --end 20   # drone segment only
    python raw_to_movie.py --seq 4
"""

import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from evt3_reader import EVT3Reader
from config import IMG_W, IMG_H, WINDOW_US

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python"); sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seq',   type=str,   default='7')
parser.add_argument('--start', type=float, default=None,
                    help='Start time in seconds (default: beginning of file)')
parser.add_argument('--end',   type=float, default=None,
                    help='End time in seconds   (default: end of file)')
parser.add_argument('--fps',   type=int,   default=30)
parser.add_argument('--save',  type=str,   default='raw_events.mp4',
                    help='Output video filename')
args = parser.parse_args()

BASE       = os.path.join(os.path.dirname(__file__), '..', 'data_from_fred', args.seq)
RAW_FILE   = os.path.join(BASE, 'Event', 'events.raw')
FRAMES_DIR = os.path.join(BASE, 'Event', 'Frames')

if not os.path.exists(RAW_FILE):
    print(f"ERROR: {RAW_FILE} not found"); sys.exit(1)

t_start_us = int(args.start * 1e6) if args.start is not None else None
t_end_us   = int(args.end   * 1e6) if args.end   is not None else None

# ── Video writer setup ────────────────────────────────────────────────────────

PANEL_W  = IMG_W // 2          # each panel scaled to half width
PANEL_H  = IMG_H // 2
BANNER_H = 36
FRAME_W  = PANEL_W * 2 + 4    # left panel + divider + right panel
FRAME_H  = BANNER_H + PANEL_H

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(args.save, fourcc, args.fps, (FRAME_W, FRAME_H))
print(f"Saving to: {os.path.abspath(args.save)}")

# ── OpenCV window ─────────────────────────────────────────────────────────────

WIN = f"events.raw vs Frames/  —  seq {args.seq}"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, FRAME_W, FRAME_H)

# ── Helper: accumulate events → uint8 grayscale ───────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def events_to_bgr(evs):
    # Accumulate all events per pixel, normalize by 99th percentile,
    # then apply CLAHE to boost local contrast — matching Frames/ appearance.
    img = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    if len(evs) > 0:
        np.add.at(img, (evs['y'], evs['x']), 1.0)
        nonzero = img[img > 0]
        if len(nonzero) > 0:
            p99 = np.percentile(nonzero, 99)
            if p99 > 0:
                img = np.clip(img / p99, 0, 1)
    gray = _clahe.apply((img * 255).astype(np.uint8))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ── Main loop ─────────────────────────────────────────────────────────────────

print(f"\nStreaming events.raw ...")
print(f"Controls:  SPACE=pause   Q/ESC=quit\n")
print(f"{'Time':>8}  {'Events':>8}  {'Frames/ match'}")
print("-" * 40)

# Pre-index Frames/ timestamps — sort NUMERICALLY (not lexicographically)
# Python sorted() compares strings char-by-char: '1' < '3' so "100000" < "33333"
# We extract the number first, then sort by its integer value.
import glob as _glob
_all_pairs   = sorted([(int(os.path.basename(p).split('_frame_')[1][:-4]), p)
                        for p in _glob.glob(os.path.join(FRAMES_DIR, '*.png'))])
_frame_ts    = np.array([t for t, _ in _all_pairs], dtype=np.int64)
_frame_files = [p for _, p in _all_pairs]
print(f"Frames/ index: {len(_frame_ts)} files  "
      f"t={_frame_ts[0]/1e6:.3f}s – {_frame_ts[-1]/1e6:.3f}s")

def nearest_frame(t_us):
    """Return (path, timestamp) of the Frames/ PNG nearest to t_us."""
    if len(_frame_ts) == 0:
        return None, None
    idx = int(np.argmin(np.abs(_frame_ts - t_us)))
    return _frame_files[idx], int(_frame_ts[idx])

reader = EVT3Reader(RAW_FILE)

# Skip the pre-recording portion of raw data (ts_shift_us worth of junk at the start).
# The Frames/ folder was built starting from raw t=ts_shift_us, so that is our t=0.
SKIP_US = reader.ts_shift_us   # 1,163,264 µs for seq 7
if t_start_us is None:
    t_start_us = SKIP_US        # default: begin where Frames/ begins
if t_end_us is not None:
    t_end_us += SKIP_US         # user's --end is in Frames/ time → convert to raw time

print(f"  Skipping first {SKIP_US/1e6:.3f}s of raw data (ts_shift_us)")
print(f"  Reading raw t={t_start_us/1e6:.3f}s – "
      f"{'end' if t_end_us is None else f'{t_end_us/1e6:.3f}s'}\n")

paused = False
n_frames_written = 0

for t_start, evs in reader.iter_windows(WINDOW_US,
                                         t_start=t_start_us,
                                         t_end=t_end_us):
    t_sec      = t_start / 1e6
    frames_t   = t_start - SKIP_US          # convert raw time → Frames/ time
    recon      = events_to_bgr(evs)

    # Frames/ filenames are window-END timestamps: file named T holds events [T-33ms, T]
    # Our window starts at frames_t, so the matching Frames/ file is at frames_t + WINDOW_US
    orig_path, orig_ts = nearest_frame(frames_t + WINDOW_US)
    delta_ms = abs(orig_ts - frames_t) / 1000 if orig_ts else 9999
    if orig_path and delta_ms < 50:   # within 50ms = same frame
        orig_full = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
        match_label = f"MATCH  Δ={delta_ms:.0f}ms"
        match_color = (0, 220, 80)
    else:
        orig_full   = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
        match_label = "no match"
        match_color = (80, 80, 80)

    print(f"{frames_t/1e6:8.3f}s  {len(evs):8,}  {match_label}")

    # Scale both panels to half size
    orig_small  = cv2.resize(orig_full, (PANEL_W, PANEL_H))
    recon_small = cv2.resize(recon,    (PANEL_W, PANEL_H))

    orig_bgr    = cv2.cvtColor(orig_small, cv2.COLOR_GRAY2BGR)
    recon_bgr   = recon_small   # already BGR

    # Label each panel
    cv2.putText(orig_bgr,  "Frames/ (original)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    cv2.putText(recon_bgr, "events.raw (reconstructed)",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    # Divider
    div = np.full((PANEL_H, 4, 3), 60, dtype=np.uint8)

    panels = np.hstack([orig_bgr, div, recon_bgr])

    # Banner
    banner = np.zeros((BANNER_H, FRAME_W, 3), dtype=np.uint8)
    cv2.putText(banner,
                f"t={frames_t/1e6:.3f}s   events={len(evs):,}   {match_label}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, match_color, 1)

    frame_out = np.vstack([banner, panels])

    # Write to video
    writer.write(frame_out)
    n_frames_written += 1

    # Show in window
    cv2.imshow(WIN, frame_out)
    key = cv2.waitKey(1 if not paused else 0) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
        print("  [PAUSED — press SPACE to resume]" if paused else "  [PLAYING]")
        if paused:
            cv2.waitKey(0)

writer.release()
cv2.destroyAllWindows()

print(f"\nDone.  {n_frames_written} frames written to {args.save}")
