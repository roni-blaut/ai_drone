"""
raw_label_check.py — Verify events.raw is in sync with Event_YOLO/ labels.

Renders frames directly from events.raw (bypassing Event/Frames/ PNGs) and
overlays the bounding box from the matching Event_YOLO label file.

If sync is correct: the CYAN box should tightly wrap the drone in the
raw event frame at every annotated timestamp.

Banner colours:
  GREEN  = label matched and box drawn
  ORANGE = label file exists but is empty (drone out of frame gap)
  RED    = no Event_YOLO file near this timestamp
  GREY   = window outside annotated range

Console prints one row per frame:
  #  t(s)  events  cx  cy  w  h  status

Controls:  SPACE=pause/resume  A/←=prev  D/→=+10  Q/ESC=quit

Usage:
    cd 4channel_project
    python raw_label_check.py                  # sequence 7, full run
    python raw_label_check.py --start 9.8      # jump to PID segment
    python raw_label_check.py --seq 4          # sequence 4
    python raw_label_check.py --save out.mp4   # also save video
"""

import sys
import os
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from evt3_reader import EVT3Reader
from config import WINDOW_US, IMG_W, IMG_H, SEQUENCE_DIR, EVENT_YOLO_DIR

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--seq',   type=str,   default='7',
                    help='Sequence number (default: 7)')
parser.add_argument('--start', type=float, default=None,
                    help='Start time in seconds')
parser.add_argument('--end',   type=float, default=None,
                    help='End time in seconds')
parser.add_argument('--delay', type=int,   default=33,
                    help='ms per frame during playback (default: 33 = 30fps)')
parser.add_argument('--save',  type=str,   default=None,
                    help='Save to video file (e.g. --save out.mp4)')
args = parser.parse_args()

# Allow --seq to override paths
if args.seq != '7':
    _base       = os.path.join(os.path.dirname(__file__), '..', 'data_from_fred', args.seq)
    RAW_FILE    = os.path.join(_base, 'Event', 'events.raw')
    EV_YOLO_DIR = os.path.join(_base, 'Event_YOLO')
else:
    RAW_FILE    = os.path.join(SEQUENCE_DIR, 'Event', 'events.raw')
    EV_YOLO_DIR = EVENT_YOLO_DIR

for path, label in [(RAW_FILE, 'events.raw'), (EV_YOLO_DIR, 'Event_YOLO/')]:
    if not os.path.exists(path):
        print(f"ERROR: {label} not found:\n  {path}")
        sys.exit(1)

# ── Index Event_YOLO labels by timestamp ──────────────────────────────────────
# Filename: Video_7_frame_{ts_us}.txt  →  ts_us (µs) = window-END in shifted clock
# (same convention as Event/Frames/ PNGs — shifted clock = raw_t - ts_shift_us)

def _parse_ts(path):
    return int(os.path.basename(path).split('_frame_')[1].split('.')[0])

def load_boxes(path):
    """Return list of (cx, cy, w, h) from a YOLO label file."""
    boxes = []
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path) as f:
            for line in f:
                v = line.strip().split()
                if len(v) == 5:
                    boxes.append(tuple(float(x) for x in v[1:]))
    return boxes

yolo_index = {}   # ts_us (int) → file path
for f in glob.glob(os.path.join(EV_YOLO_DIR, '*.txt')):
    try:
        yolo_index[_parse_ts(f)] = f
    except (IndexError, ValueError):
        pass

yolo_ts  = np.array(sorted(yolo_index.keys()), dtype=np.int64)
HALF_WIN = WINDOW_US // 2   # tolerance: half a frame (~16.7ms)

print(f"\nSequence {args.seq}")
print(f"  Event_YOLO files  : {len(yolo_ts)}")
if len(yolo_ts):
    print(f"  Label range       : {yolo_ts[0]/1e6:.3f}s – {yolo_ts[-1]/1e6:.3f}s (shifted clock)")

# ── Draw helper ───────────────────────────────────────────────────────────────

BOX_COLOR = (0, 255, 255)   # cyan

def draw_boxes(img, boxes):
    H, W = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * W);  y1 = int((cy - bh / 2) * H)
        x2 = int((cx + bw / 2) * W);  y2 = int((cy + bh / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.circle(img, (int(cx * W), int(cy * H)), 5, BOX_COLOR, -1)

# ── Build all frames from events.raw ─────────────────────────────────────────

print(f"\nLoading frames from events.raw ... (Ctrl+C to stop early)\n")

reader  = EVT3Reader(RAW_FILE)
SKIP_US = reader.ts_shift_us   # Fix 2: clock offset between raw and Event_YOLO/Frames/
                                # Event_YOLO timestamps = raw_t - SKIP_US (shifted clock)
if SKIP_US:
    print(f"  ts_shift_us : {SKIP_US:,} µs ({SKIP_US/1e6:.3f}s) — applied for Event_YOLO lookup")

frames = []   # (display_t_us, img_bgr, status, banner_color, boxes, n_events)

# User's --start/--end are in displayed (shifted) time; convert to raw time for EVT3Reader.
# Default start = SKIP_US so we skip the pre-recording junk at the head of the raw file.
if args.start is not None:
    t_start_arg = int(args.start * 1e6) + SKIP_US
else:
    t_start_arg = SKIP_US if SKIP_US else None

t_end_arg = (int(args.end * 1e6) + SKIP_US) if args.end is not None else None

try:
    for t_start, events in reader.iter_windows(WINDOW_US,
                                               t_start=t_start_arg,
                                               t_end=t_end_arg):
        # ── Render event frame (white=positive, blue=negative) ────────────────
        img_bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        if len(events) > 0:
            pos = events[events['p'] == 1]
            neg = events[events['p'] == 0]
            pos_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            neg_map = np.zeros((IMG_H, IMG_W), dtype=np.float32)
            if len(pos): np.add.at(pos_map, (pos['y'], pos['x']), 1.0)
            if len(neg): np.add.at(neg_map, (neg['y'], neg['x']), 1.0)
            pmax = pos_map.max() or 1
            nmax = neg_map.max() or 1
            img_bgr[:, :, 2] = np.clip(pos_map / pmax * 255, 0, 255).astype(np.uint8)
            img_bgr[:, :, 1] = np.clip(pos_map / pmax * 255, 0, 255).astype(np.uint8)
            img_bgr[:, :, 0] = np.clip(
                pos_map / pmax * 128 + neg_map / nmax * 255, 0, 255).astype(np.uint8)

        # ── Convert raw time → shifted (Event_YOLO) time ──────────────────────
        # Event_YOLO filenames encode the window-END in the shifted clock.
        # raw window starts at t_start → shifted window end = t_start - SKIP_US + WINDOW_US
        display_t  = t_start - SKIP_US                  # user-visible time (seconds 0…N)
        ev_yolo_t  = display_t + WINDOW_US              # window-end in shifted clock

        # ── Match to nearest Event_YOLO label ─────────────────────────────────
        boxes        = []
        status       = "NO LABEL"
        banner_color = (60, 60, 60)   # dark grey

        if len(yolo_ts):
            idx        = np.searchsorted(yolo_ts, ev_yolo_t)
            candidates = []
            if idx < len(yolo_ts):
                candidates.append((abs(int(yolo_ts[idx]) - ev_yolo_t), int(yolo_ts[idx])))
            if idx > 0:
                candidates.append((abs(int(yolo_ts[idx-1]) - ev_yolo_t), int(yolo_ts[idx-1])))
            delta_us, nearest_ts = min(candidates)

            if delta_us <= HALF_WIN:
                boxes = load_boxes(yolo_index[nearest_ts])
                sign  = '+' if (nearest_ts - ev_yolo_t) >= 0 else ''
                d_str = f"Δ={sign}{nearest_ts - ev_yolo_t}µs"
                if boxes:
                    status       = f"MATCH  {d_str}"
                    banner_color = (0, 200, 60)     # green
                else:
                    status       = f"EMPTY LABEL  {d_str}  (drone out of frame)"
                    banner_color = (0, 140, 255)    # orange
            else:
                status       = f"NO LABEL  (nearest {delta_us/1e3:.1f}ms away)"
                banner_color = (40, 40, 180)        # red

        # ── Overlay bbox ──────────────────────────────────────────────────────
        draw_boxes(img_bgr, boxes)

        frames.append((display_t, img_bgr, status, banner_color, boxes, len(events)))

        if len(frames) % 200 == 0:
            print(f"  {len(frames)} frames  t={display_t/1e6:.2f}s  [{status}]")

except KeyboardInterrupt:
    print(f"\nStopped early — {len(frames)} frames loaded")

if not frames:
    print("No frames generated. Check RAW_FILE path and --start/--end range.")
    sys.exit(1)

print(f"\n{len(frames)} frames ready.  Opening viewer...")
print("Controls: SPACE=pause/resume  A/←=prev  D/→=+10  Q=quit")
print("  CYAN box = Event_YOLO label  |  GREEN=matched  ORANGE=empty  RED=no label\n")

# ── Console table header ──────────────────────────────────────────────────────

print(f"{'#':>5}  {'t(s)':>8}  {'events':>7}  {'cx':>6} {'cy':>6} {'w':>6} {'h':>6}  status")
print("─" * 82)

# ── Optional video writer ─────────────────────────────────────────────────────

BANNER_H = 52
SCALE    = 0.65
OUT_W    = int(IMG_W * SCALE)
OUT_H    = int((IMG_H + BANNER_H) * SCALE)

writer = None
if args.save:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.save, fourcc, max(1, 1000 // args.delay), (OUT_W, OUT_H))
    print(f"Saving to: {os.path.abspath(args.save)}")

# ── Viewer ────────────────────────────────────────────────────────────────────

WIN    = f"raw_label_check — seq {args.seq}  |  SPACE=pause  A/D=prev/+10  Q=quit"
paused = False
i      = 0

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

while 0 <= i < len(frames):
    t_us, img_bgr, status, banner_color, boxes, n_ev = frames[i]

    # Banner
    banner = np.zeros((BANNER_H, IMG_W, 3), dtype=np.uint8)
    banner[:, :6] = banner_color
    cv2.putText(banner,
                f"#{i}  t={t_us/1e6:.3f}s  {n_ev:,} events  |  {status}",
                (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 1)
    if boxes:
        cx, cy, bw, bh = boxes[0]
        box_str = f"bbox: cx={cx:.3f}  cy={cy:.3f}  w={bw:.3f}  h={bh:.3f}"
    else:
        box_str = "no bbox"
    cv2.putText(banner, box_str,
                (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1)

    combined = np.vstack([banner, img_bgr])
    out_frame = cv2.resize(combined, (OUT_W, OUT_H))
    cv2.imshow(WIN, out_frame)

    if writer:
        writer.write(out_frame)

    # Console row (only print on forward steps to avoid spam on prev)
    if boxes:
        cx, cy, bw, bh = boxes[0]
        print(f"{i:>5}  {t_us/1e6:>8.3f}  {n_ev:>7,}  "
              f"{cx:>6.3f} {cy:>6.3f} {bw:>6.3f} {bh:>6.3f}  {status}")
    else:
        print(f"{i:>5}  {t_us/1e6:>8.3f}  {n_ev:>7,}  "
              f"{'—':>6} {'—':>6} {'—':>6} {'—':>6}  {status}")

    key = cv2.waitKey(1 if paused else args.delay) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        paused = not paused
        print("  [PAUSED]" if paused else "  [PLAYING]")
    elif key in (ord('a'), 81):           # A or ← arrow
        i = max(0, i - 1)
    elif key in (ord('d'), 83):           # D or → arrow → +10
        i = min(len(frames) - 1, i + 10)
    elif not paused:
        i += 1

print("─" * 82)
print(f"\nDone.  {min(i + 1, len(frames))} frames shown.")

if writer:
    writer.release()
    print(f"Saved: {os.path.abspath(args.save)}")

cv2.destroyAllWindows()
