"""
infer.py — Run the trained FRED baseline YOLO model on a new sequence zip.

Reads frames directly from a zip file (no extraction needed) via
common/zip_utils.py, runs the trained detector on each frame, overlays the
prediction (and the original Event_YOLO/RGB_YOLO ground-truth box, if the
zip has one) and saves an annotated video.

Run from ai_drone/Fred/:
    python infer.py --zip ../data_from_fred/49.zip
    python infer.py --zip ../data_from_fred/49.zip --mode rgb
    python infer.py --zip ../data_from_fred/49.zip --save-frames out_frames
    python infer.py --zip ../data_from_fred/49.zip --show
"""

import os
import re
import sys
import argparse
import zipfile
import collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))

from zip_utils import init_sequence, seq_glob, seq_imread, seq_open_lines, seq_exists

GT_COLOR = (0, 255, 255)  # cyan — matches tools/raw_label_check.py convention


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', required=True, help='Path to the new sequence zip (or its extracted folder)')
    parser.add_argument('--mode', choices=['event', 'rgb'], default='event')
    parser.add_argument('--weights', default=None, help='Override weights path (default: Fred/runs/fred_baseline_{mode}/weights/best.pt)')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default=None, help="Override device ('cpu', '0', ...); default: auto-detect")
    parser.add_argument('--save-frames', default=None, help='Directory to save annotated PNG frames (opt-in)')
    parser.add_argument('--save-video', default=None, help='Explicit output video path (default: Fred/runs/infer/<zip_stem>_<mode>.mp4)')
    parser.add_argument('--no-video', action='store_true', help='Disable video saving')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--show', action='store_true', help='Live cv2 preview (SPACE=pause A/D=step Q=quit)')
    args = parser.parse_args()

    if args.no_video and args.save_video:
        parser.error('--save-video and --no-video are mutually exclusive')

    return args


def resolve_zip_stem(zip_arg):
    """Return (seq_dir, zip_stem) — seq_dir is what init_sequence() expects (no .zip suffix)."""
    abs_path = os.path.abspath(zip_arg)
    seq_dir = re.sub(r'\.zip$', '', abs_path, flags=re.IGNORECASE)
    zip_stem = os.path.splitext(os.path.basename(abs_path))[0]
    return seq_dir, zip_stem


def frame_timestamp(path):
    """Parse the integer timestamp out of 'Video_{seq}_frame_{ts}.ext' / similar; 0 if unparsable."""
    base = os.path.basename(path)
    for sep in ('_frame_', '_'):
        if sep in base:
            tail = base.split(sep)[-1]
            digits = re.match(r'\d+', tail)
            if digits:
                try:
                    return int(digits.group())
                except ValueError:
                    pass
    return 0


def load_gt_boxes(label_path):
    """Return [] if the label file is missing/empty, else a list of normalized (cx, cy, w, h)."""
    if not seq_exists(label_path):
        return []
    boxes = []
    for line in seq_open_lines(label_path):
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append(tuple(float(x) for x in parts[1:]))
    return boxes


def draw_gt(img, boxes):
    import cv2
    h, w = img.shape[:2]
    for cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), GT_COLOR, 2)


def select_device(args):
    if args.device is not None:
        return args.device
    import torch
    return 0 if torch.cuda.is_available() else 'cpu'


def show_frame(win, buffer, key_wait):
    """Interactive display of the ring buffer's last frame. Returns True if the user quit."""
    import cv2
    pos = len(buffer) - 1
    paused = False
    while True:
        cv2.imshow(win, buffer[pos])
        key = cv2.waitKey(1 if paused else key_wait) & 0xFF
        if key in (ord('q'), 27):
            return True
        elif key == ord(' '):
            paused = not paused
        elif paused and key in (ord('a'), 81) and pos > 0:
            pos -= 1
        elif paused and key in (ord('d'), 83):
            return False
        elif not paused:
            return False


def main():
    args = parse_args()

    # ── Fail-fast weights check (before touching the zip or importing ultralytics/torch) ──
    weights_path = args.weights or os.path.join(HERE, 'runs', f'fred_baseline_{args.mode}', 'weights', 'best.pt')
    if not os.path.exists(weights_path):
        msg = (f"Trained model not found — run train.py --mode {args.mode} first\n{weights_path}")
        if args.mode == 'rgb':
            msg += "\nNote: no RGB baseline has been trained yet in this repo."
        raise FileNotFoundError(msg)

    # ── Resolve default output paths ──
    seq_dir, zip_stem = resolve_zip_stem(args.zip)

    video_path = None
    if not args.no_video:
        video_path = args.save_video or os.path.join(HERE, 'runs', 'infer', f'{zip_stem}_{args.mode}.mp4')
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # ── Open the sequence (zip or folder) ──
    try:
        init_sequence(seq_dir)
    except FileNotFoundError as e:
        print(f"ERROR: could not open sequence data.\n{e}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print(f"ERROR: {seq_dir}.zip exists but is not a valid zip file (corrupt/incomplete download?)")
        sys.exit(1)

    if args.mode == 'event':
        img_dir, label_dir, img_ext = (os.path.join(seq_dir, 'Event', 'Frames'),
                                        os.path.join(seq_dir, 'Event_YOLO'), '.png')
    else:
        img_dir, label_dir, img_ext = (os.path.join(seq_dir, 'PADDED_RGB'),
                                        os.path.join(seq_dir, 'RGB_YOLO'), '.jpg')

    if not seq_exists(img_dir):
        top = seq_glob(seq_dir, '*')
        print(f"ERROR: {img_dir} not found in this zip/folder.")
        print(f"Top-level contents found: {[os.path.basename(p) for p in top]}")
        sys.exit(1)

    frame_paths = seq_glob(img_dir, f'*{img_ext}')
    if not frame_paths:
        print(f"ERROR: no {img_ext} frames found under {img_dir}")
        sys.exit(1)

    frame_paths.sort(key=frame_timestamp)

    has_gt = seq_exists(label_dir)

    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    # ── Load model ──
    from ultralytics import YOLO
    import cv2

    device = select_device(args)
    model = YOLO(weights_path)
    print(f"[infer] device={device}  weights={weights_path}  frames={len(frame_paths)}  gt={'available' if has_gt else 'not available'}")

    writer = None
    win = f"FRED infer — {zip_stem} ({args.mode})  SPACE=pause  A/D=step  Q=quit"
    buffer = collections.deque(maxlen=300)
    n_frames = n_skipped = n_with_det = 0
    confidences = []
    stopped = False

    try:
        for i, frame_path in enumerate(frame_paths):
            if stopped:
                break

            img = seq_imread(frame_path)
            if img is None:
                n_skipped += 1
                print(f"  WARNING: unreadable frame, skipping: {frame_path}")
                continue

            results = model.predict(source=img, conf=args.conf, iou=args.iou,
                                     imgsz=args.imgsz, device=device, verbose=False)
            r = results[0]
            annotated = r.plot()

            if has_gt:
                stem = os.path.splitext(os.path.basename(frame_path))[0]
                boxes = load_gt_boxes(os.path.join(label_dir, stem + '.txt'))
                if boxes:
                    draw_gt(annotated, boxes)

            n_frames += 1
            n_det = len(r.boxes)
            if n_det > 0:
                n_with_det += 1
                confidences.extend(r.boxes.conf.tolist())

            if i % 100 == 0:
                print(f"  {i + 1}/{len(frame_paths)}  det={n_det}")

            if args.save_frames:
                cv2.imwrite(os.path.join(args.save_frames, os.path.basename(frame_path)), annotated)

            if video_path:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (w, h))
                writer.write(annotated)

            if args.show:
                buffer.append(annotated)
                stopped = show_frame(win, buffer, key_wait=33)
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    rate = 100 * n_with_det / n_frames if n_frames else 0.0
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print("\n" + "=" * 50)
    print(f"FRED INFERENCE — {args.mode.upper()} mode")
    print("=" * 50)
    print(f"  Source zip       : {args.zip}")
    print(f"  Weights          : {weights_path}")
    print(f"  Device           : {device}")
    print(f"  Frames found     : {len(frame_paths)}")
    print(f"  Frames processed : {n_frames}   ({n_skipped} unreadable, skipped)")
    print(f"  Frames w/ detect : {n_with_det}  ({rate:.1f}%)")
    print(f"  Mean confidence  : {mean_conf:.3f}   (over {len(confidences)} boxes)")
    print(f"  Ground truth     : {'available (' + label_dir + ')' if has_gt else 'not available in this zip'}")
    print(f"  Video saved      : {video_path if video_path else 'not saved (--no-video)'}")
    print(f"  Frames saved     : {args.save_frames if args.save_frames else 'not saved'}")
    print("=" * 50)


if __name__ == '__main__':
    main()
