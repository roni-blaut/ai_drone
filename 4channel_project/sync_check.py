"""
sync_check.py — Verify synchronization between event frames and RGB frames.

Draws bounding boxes from Event_YOLO/ on Event/Frames/ images and
bounding boxes from RGB_YOLO/ on PADDED_RGB/ images, side by side.

If data is in sync: the drone rectangle should be in the same relative
position in both panels at the same sequential frame index.

Printed table columns:
  #        — frame pair index
  ev_t_s   — event frame time in seconds (from sequence start)
  rgb_t_s  — RGB frame time in seconds (relative, from first RGB frame)
  ev_cx/cy — drone bbox centre in event frame (normalised 0-1)
  rgb_cx/cy— drone bbox centre in RGB frame   (normalised 0-1)
  Δcx/Δcy  — absolute difference → near 0 = good sync

Controls:  SPACE = next   LEFT/A = prev   RIGHT/D = +10   Q/ESC = quit

Usage:
    cd 4channel_project
    python sync_check.py
    python sync_check.py --start 9.8      # jump to t=9.8s in event stream
    python sync_check.py --n 50           # show 50 frames then stop
    python sync_check.py --save sync.mp4  # also save video
"""

import sys, os, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    IMG_W, IMG_H,
    SEQUENCE_DIR,
    FRAMES_DIR,
    EVENT_YOLO_DIR,
    PADDED_RGB_DIR,
    RGB_YOLO_DIR,
)

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python"); sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=float, default=None,
                    help='Start time in event-stream seconds (default: first labeled frame)')
parser.add_argument('--n',     type=int,   default=None,
                    help='Max frame pairs to show (default: all)')
parser.add_argument('--fps',   type=int,   default=5,
                    help='Auto-play speed in fps (default: 5)')
parser.add_argument('--save',  type=str,   default=None,
                    help='Save output video (e.g. --save sync.mp4)')
args = parser.parse_args()

for d, name in [(FRAMES_DIR,     'Event/Frames/'),
                (EVENT_YOLO_DIR, 'Event_YOLO/'),
                (PADDED_RGB_DIR, 'PADDED_RGB/'),
                (RGB_YOLO_DIR,   'RGB_YOLO/')]:
    if not os.path.exists(d):
        print(f"ERROR: {name} not found at:\n  {d}")
        print(f"\nCheck SEQUENCE_DIR in config.py — currently: {SEQUENCE_DIR}")
        sys.exit(1)


# ── Helper: parse timestamps from filenames ───────────────────────────────────

def _ev_ts_us(path):
    """Video_7_frame_9999000.txt  →  9999000 (µs)"""
    return int(os.path.basename(path).split('_frame_')[1].split('.')[0])

def _rgb_sec(path):
    """Video_7_17_01_52.636134.txt  →  seconds since midnight"""
    name  = os.path.splitext(os.path.basename(path))[0]
    parts = name.split('_')    # ['Video', '7', '17', '01', '52.636134']
    hh, mm, ss = int(parts[-3]), int(parts[-2]), float(parts[-1])
    return hh * 3600 + mm * 60 + ss


# ── Helper: load / draw YOLO labels ──────────────────────────────────────────

def load_boxes(lbl_path):
    """Return list of (cx, cy, w, h) from a YOLO label file."""
    boxes = []
    if os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0:
        with open(lbl_path) as f:
            for line in f:
                v = line.strip().split()
                if len(v) == 5:
                    boxes.append(tuple(float(x) for x in v[1:]))
    return boxes

def draw_boxes(img, boxes, color, thickness=2):
    """Draw YOLO normalised boxes and centre dots on img in-place."""
    H, W = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * W)
        y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W)
        y2 = int((cy + bh / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.circle(img, (int(cx * W), int(cy * H)), 5, color, -1)
    return img


# ── Index Event frames — NUMERIC sort by µs timestamp ────────────────────────
# Event_YOLO filenames: Video_7_frame_{ts_us}.txt
# Matching Frames/ PNG has the same stem with .png extension

ev_lbl_files = glob.glob(os.path.join(EVENT_YOLO_DIR, '*.txt'))
ev_pairs = []
for lbl in ev_lbl_files:
    img = os.path.join(FRAMES_DIR, os.path.basename(lbl).replace('.txt', '.png'))
    if os.path.exists(img):
        ev_pairs.append((_ev_ts_us(lbl), img, lbl))
ev_pairs.sort(key=lambda x: x[0])   # numeric sort

if not ev_pairs:
    print(f"ERROR: no matching Event_YOLO + Frames/ pairs found.\n"
          f"  Frames/    : {FRAMES_DIR}\n"
          f"  Event_YOLO : {EVENT_YOLO_DIR}"); sys.exit(1)

print(f"Event   : {len(ev_pairs)} frames   "
      f"t={ev_pairs[0][0]/1e6:.3f}s – {ev_pairs[-1][0]/1e6:.3f}s")

# apply --start
if args.start is not None:
    start_us = int(args.start * 1e6)
    ev_pairs = [(ts, img, lbl) for ts, img, lbl in ev_pairs if ts >= start_us]
    print(f"  after --start {args.start}s: {len(ev_pairs)} frames remain")


# ── Index RGB frames — alphabetical = chronological ───────────────────────────
# RGB_YOLO filenames: Video_7_HH_MM_SS.ffffff.txt
# Matching PADDED_RGB/ JPG has the same stem with .jpg extension

rgb_lbl_files = glob.glob(os.path.join(RGB_YOLO_DIR, '*.txt'))
rgb_pairs = []
for lbl in rgb_lbl_files:
    img = os.path.join(PADDED_RGB_DIR, os.path.basename(lbl).replace('.txt', '.jpg'))
    if os.path.exists(img):
        rgb_pairs.append((_rgb_sec(lbl), img, lbl))
rgb_pairs.sort(key=lambda x: x[0])   # chronological

if not rgb_pairs:
    print(f"ERROR: no matching RGB_YOLO + PADDED_RGB/ pairs found.\n"
          f"  PADDED_RGB : {PADDED_RGB_DIR}\n"
          f"  RGB_YOLO   : {RGB_YOLO_DIR}"); sys.exit(1)

rgb_t0 = rgb_pairs[0][0]   # wall-clock seconds at frame 0
print(f"RGB     : {len(rgb_pairs)} frames   "
      f"t=0.000s – {rgb_pairs[-1][0] - rgb_t0:.3f}s (relative)")
print()


# ── Pair by sequential index from start ───────────────────────────────────────
# Both streams are ~30fps. Frame 0 = first frame in each stream (after --start).
# If synchronized, frame N in event ≅ frame N in RGB.

n_pairs = min(len(ev_pairs), len(rgb_pairs))
if args.n:
    n_pairs = min(n_pairs, args.n)

print(f"Showing {n_pairs} frame pairs  (event[i] ↔ rgb[i])")
print()


# ── Layout constants ──────────────────────────────────────────────────────────

PANEL_W  = IMG_W // 2
PANEL_H  = IMG_H // 2
BANNER_H = 56
DIV_W    = 4
FRAME_W  = PANEL_W * 2 + DIV_W
FRAME_H  = BANNER_H + PANEL_H

EV_COLOR  = (0, 255, 255)    # cyan   — Event_YOLO boxes
RGB_COLOR = (0, 165, 255)    # orange — RGB_YOLO boxes


# ── Optional video writer ─────────────────────────────────────────────────────

writer = None
if args.save:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.save, fourcc, args.fps, (FRAME_W, FRAME_H))
    print(f"Saving to: {os.path.abspath(args.save)}")


# ── OpenCV window ─────────────────────────────────────────────────────────────

WIN = "Sync check — SPACE=next  ←/A=prev  →/D=+10  Q=quit"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, FRAME_W, FRAME_H)


# ── Console table header ──────────────────────────────────────────────────────

print(f"{'#':>5}  {'ev_t':>8}  {'rgb_t':>8}  "
      f"{'ev_cx':>6} {'ev_cy':>6}  {'rgb_cx':>6} {'rgb_cy':>6}  "
      f"{'Δcx':>6} {'Δcy':>6}  status")
print("─" * 82)


# ── Main loop ─────────────────────────────────────────────────────────────────

delay = max(1, int(1000 / args.fps))
idx   = 0

while 0 <= idx < n_pairs:
    ev_ts_us, ev_img_path, ev_lbl_path   = ev_pairs[idx]
    rgb_t_abs, rgb_img_path, rgb_lbl_path = rgb_pairs[idx]

    ev_t_s  = ev_ts_us / 1e6
    rgb_t_s = rgb_t_abs - rgb_t0

    ev_boxes  = load_boxes(ev_lbl_path)
    rgb_boxes = load_boxes(rgb_lbl_path)

    # ── Event panel ───────────────────────────────────────────────────────────
    ev_raw = cv2.imread(ev_img_path, cv2.IMREAD_GRAYSCALE)
    if ev_raw is None:
        ev_raw = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    ev_bgr = cv2.cvtColor(ev_raw, cv2.COLOR_GRAY2BGR)
    draw_boxes(ev_bgr, ev_boxes, EV_COLOR)
    ev_panel = cv2.resize(ev_bgr, (PANEL_W, PANEL_H))
    cv2.putText(ev_panel, f"Event/Frames/   t={ev_t_s:.3f}s",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, EV_COLOR, 1)
    cv2.putText(ev_panel, f"Event_YOLO  {len(ev_boxes)} box(es)",
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, EV_COLOR, 1)

    # ── RGB panel ─────────────────────────────────────────────────────────────
    rgb_raw = cv2.imread(rgb_img_path)
    if rgb_raw is None:
        rgb_raw = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    draw_boxes(rgb_raw, rgb_boxes, RGB_COLOR)
    rgb_panel = cv2.resize(rgb_raw, (PANEL_W, PANEL_H))
    cv2.putText(rgb_panel, f"PADDED_RGB/   t={rgb_t_s:.3f}s",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RGB_COLOR, 1)
    cv2.putText(rgb_panel, f"RGB_YOLO  {len(rgb_boxes)} box(es)",
                (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, RGB_COLOR, 1)

    # ── Divider ───────────────────────────────────────────────────────────────
    div = np.full((PANEL_H, DIV_W, 3), 40, dtype=np.uint8)
    panels = np.hstack([ev_panel, div, rgb_panel])

    # ── Sync assessment ───────────────────────────────────────────────────────
    ev_cx  = ev_boxes[0][0]  if ev_boxes  else None
    ev_cy  = ev_boxes[0][1]  if ev_boxes  else None
    rgb_cx = rgb_boxes[0][0] if rgb_boxes else None
    rgb_cy = rgb_boxes[0][1] if rgb_boxes else None

    if ev_cx is not None and rgb_cx is not None:
        dcx, dcy   = abs(ev_cx - rgb_cx), abs(ev_cy - rgb_cy)
        is_synced  = dcx < 0.15 and dcy < 0.15
        status     = "SYNC OK" if is_synced else "OFFSET?"
        status_col = (0, 220, 80) if is_synced else (0, 80, 255)
        banner_txt = (f"ev=({ev_cx:.3f},{ev_cy:.3f})  "
                      f"rgb=({rgb_cx:.3f},{rgb_cy:.3f})  "
                      f"Δ=({dcx:.3f},{dcy:.3f})  {status}")
        print(f"{idx:>5}  {ev_t_s:>8.3f}  {rgb_t_s:>8.3f}  "
              f"{ev_cx:>6.3f} {ev_cy:>6.3f}  {rgb_cx:>6.3f} {rgb_cy:>6.3f}  "
              f"{dcx:>6.3f} {dcy:>6.3f}  {status}")
    else:
        status_col = (120, 120, 120)
        banner_txt = f"ev:{len(ev_boxes)} box(es)   rgb:{len(rgb_boxes)} box(es)   (no bbox to compare)"
        print(f"{idx:>5}  {ev_t_s:>8.3f}  {rgb_t_s:>8.3f}  "
              f"{'—':>6} {'—':>6}  {'—':>6} {'—':>6}  "
              f"{'—':>6} {'—':>6}  "
              f"ev:{len(ev_boxes)} rgb:{len(rgb_boxes)}")

    # ── Banner ────────────────────────────────────────────────────────────────
    banner = np.zeros((BANNER_H, FRAME_W, 3), dtype=np.uint8)
    cv2.putText(banner, f"Frame {idx}/{n_pairs-1}   {banner_txt}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_col, 1)
    cv2.putText(banner,
                "SPACE/D=next   A=prev   RIGHT=+10   Q=quit   "
                "[cyan=Event_YOLO  orange=RGB_YOLO]",
                (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)

    frame_out = np.vstack([banner, panels])

    if writer:
        writer.write(frame_out)

    cv2.imshow(WIN, frame_out)
    key = cv2.waitKey(delay) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key in (ord('a'), 81):          # A or LEFT arrow
        idx = max(0, idx - 1)
    elif key in (ord('d'), 83):          # D or RIGHT arrow → jump +10
        idx = min(n_pairs - 1, idx + 10)
    else:                                # SPACE or anything else → next
        idx += 1

print("─" * 82)
print(f"\nDone.  Showed {min(idx+1, n_pairs)} frame pairs.")

if writer:
    writer.release()
    print(f"Saved: {os.path.abspath(args.save)}")

cv2.destroyAllWindows()
