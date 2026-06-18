"""
config.py — All project settings in one place.

Supports 4 environments — auto-detected, no manual changes needed:

  1. Local PC (Windows)   : your VS Code setup
  2. Google Colab         : free T4 GPU via Google Drive
  3. NVIDIA GPU server    : set ENV=nvidia or pass --env nvidia
  4. CPU only             : fallback if no GPU found

To force a specific environment (overrides auto-detect):
  Set environment variable before running:
    set DRONE_ENV=local       (Windows)
    set DRONE_ENV=colab
    set DRONE_ENV=nvidia
    set DRONE_ENV=cpu
"""

import os

# ── Torch — optional at config load time ─────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = True
    GPU_AVAILABLE   = torch.cuda.is_available()
    GPU_NAME        = torch.cuda.get_device_name(0) if GPU_AVAILABLE else "none"
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE   = False
    GPU_NAME        = "torch not installed"

# ── Environment detection ─────────────────────────────────────────────────────

def _detect_env():
    # 1. Manual override via environment variable
    forced = os.environ.get('DRONE_ENV', '').lower()
    if forced in ('local', 'colab', 'nvidia', 'cpu'):
        print(f"[config] Environment forced: {forced}")
        return forced

    # 2. Auto-detect Colab
    try:
        import google.colab  # noqa
        return 'colab'
    except ImportError:
        pass

    # 3. Auto-detect NVIDIA server (Linux + GPU + not Colab)
    if os.name == 'posix' and GPU_AVAILABLE:
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            return 'nvidia'

    # 4. Local PC (Windows or Mac with or without GPU)
    return 'local'

ENV      = _detect_env()
IN_COLAB = (ENV == 'colab')

print(f"[config] Running on: {ENV.upper()}")
print(f"[config] GPU: {GPU_NAME}")

# ── Paths — per environment ───────────────────────────────────────────────────

if ENV == 'colab':
    # Google Colab — data on Drive, outputs on fast local SSD
    DRIVE_ROOT   = "/content/drive/MyDrive/ai_drone"
    SEQUENCE_DIR = os.path.join(DRIVE_ROOT, "data_from_fred", "7")
    DATASET_DIR  = "/content/dataset"        # fast SSD — survives session
    RUNS_DIR     = "/content/runs"

elif ENV == 'nvidia':
    # NVIDIA GPU server — adjust DATA_ROOT to your server's data path
    DATA_ROOT    = os.environ.get('DRONE_DATA', '/data/fred')
    SEQUENCE_DIR = os.path.join(DATA_ROOT, "7")
    DATASET_DIR  = os.path.join(DATA_ROOT, "dataset")
    RUNS_DIR     = os.path.join(DATA_ROOT, "runs")

elif ENV == 'cpu':
    # CPU only — same paths as local but slower settings applied below
    SEQUENCE_DIR = "../data_from_fred/7"
    DATASET_DIR  = "./dataset"
    RUNS_DIR     = "./runs"

else:
    # Local PC — Windows VS Code
    SEQUENCE_DIR = "../data_from_fred/7"
    DATASET_DIR  = "./dataset"
    RUNS_DIR     = "./runs"

# Initialise zip or real-folder access for SEQUENCE_DIR
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or '.')
from zip_utils import init_sequence as _init_sequence, seq_exists as _seq_exists
_init_sequence(SEQUENCE_DIR)

# Raw event file
RAW_FILE    = os.path.join(SEQUENCE_DIR, "Event", "events.raw")

# Ground-truth annotations
# interpolated_coordinates.txt has smoother (float) bboxes vs integer coords in coordinates.txt
_interp = os.path.join(SEQUENCE_DIR, "interpolated_coordinates.txt")
COORDS_FILE = _interp if _seq_exists(_interp) else os.path.join(SEQUENCE_DIR, "coordinates.txt")

# Event data
FRAMES_DIR     = os.path.join(SEQUENCE_DIR, "Event", "Frames")
EVENT_YOLO_DIR = os.path.join(SEQUENCE_DIR, "Event_YOLO")

# RGB data
RGB_DIR        = os.path.join(SEQUENCE_DIR, "RGB")
PADDED_RGB_DIR = os.path.join(SEQUENCE_DIR, "PADDED_RGB")
RGB_YOLO_DIR   = os.path.join(SEQUENCE_DIR, "RGB_YOLO")

RUN_NAME    = "fred_4channel"

# ── Sensor ────────────────────────────────────────────────────────────────────

IMG_W = 1280
IMG_H = 720

# ── EVT3 Reader ───────────────────────────────────────────────────────────────

HEADER_BYTES = 289   # fallback if auto-detection fails

# ── Noise Filters ─────────────────────────────────────────────────────────────

REFRACTORY_US = 1000    # 1ms  — refractory period per pixel
BAF_RADIUS_PX = 3       # pixels — BAF neighbourhood radius
BAF_DELTA_US  = 10000   # 10ms — BAF time window

# ── Channel Generation ────────────────────────────────────────────────────────

WINDOW_US       = 33333   # ~33ms = 30fps (matches FRED paper)
ROTOR_THRESHOLD = 5       # min events/pixel to count as rotor

# ── Dataset Split ─────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ── Debug Mode ────────────────────────────────────────────────────────────────
# Set DEBUG_MODE=True (or env var DEBUG_MODE=true) to enable verbose output
# and save 10 before/after filter comparison images to 4channel_project/debug/

DEBUG_MODE    = bool(os.getenv('DEBUG_MODE',   'False').lower() == 'true')
DEBUG_SAMPLES = int(os.getenv('DEBUG_SAMPLES', '10'))

# ── YOLO Training — tuned per environment ────────────────────────────────────

YOLO_MODEL = "yolo11n.pt"
IMG_SIZE   = 640
PATIENCE   = 20
N_CHANNELS = 4

if ENV == 'colab':
    # Colab T4 — 15GB VRAM
    EPOCHS  = 100
    BATCH   = 16
    DEVICE  = 0

elif ENV == 'nvidia':
    # NVIDIA server — may have large VRAM, use bigger batch
    EPOCHS  = 150
    BATCH   = int(os.environ.get('DRONE_BATCH', 32))
    DEVICE  = int(os.environ.get('DRONE_GPU',   0))

elif ENV == 'cpu':
    # CPU only — reduce everything for reasonable speed
    EPOCHS  = 20      # fewer epochs — CPU training is slow
    BATCH   = 4
    DEVICE  = 'cpu'

else:
    # Local PC — conservative defaults
    # Adjust BATCH down if you get out-of-memory errors
    EPOCHS  = 100
    BATCH   = 8
    DEVICE  = 0 if GPU_AVAILABLE else 'cpu'

# ── Print summary ─────────────────────────────────────────────────────────────

print(f"[config] Sequence dir : {SEQUENCE_DIR}")
print(f"[config] Dataset dir  : {DATASET_DIR}")
print(f"[config] Device       : {DEVICE}")
print(f"[config] Batch size   : {BATCH}")
print(f"[config] Epochs       : {EPOCHS}")
print(f"[config] Debug mode   : {'ON (saving ' + str(DEBUG_SAMPLES) + ' before/after images)' if DEBUG_MODE else 'off'}")
