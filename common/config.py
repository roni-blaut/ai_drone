"""
config.py — All project settings in one place.

Supports 6 environments — auto-detected, no manual changes needed:

  1. Local PC (Windows)   : your VS Code / pip setup
  2. Intel Arc (Windows)  : Intel Arc GPU via Microsoft DirectML backend
  3. WSL2                 : Windows Subsystem for Linux with NVIDIA GPU
  4. Google Colab         : free T4 GPU via Google Drive
  5. NVIDIA GPU server    : dedicated Linux GPU server
  6. CPU only             : fallback if no GPU found

To force a specific environment (overrides auto-detect):
  Set environment variable before running:
    set DRONE_ENV=local       (Windows cmd)
    set DRONE_ENV=intel       (Intel Arc / DirectML)
    export DRONE_ENV=wsl      (WSL / Linux)
    set DRONE_ENV=colab
    set DRONE_ENV=nvidia
    set DRONE_ENV=cpu
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # common/

# ── Shared paths — used by all pipelines ─────────────────────────────────────

DATA_FROM_FRED = os.path.normpath(os.path.join(_HERE, '..', 'data_from_fred'))
SPLITS_YAML    = os.path.join(DATA_FROM_FRED, 'splits.yaml')
CATALOG_YAML   = os.path.join(DATA_FROM_FRED, 'catalog.yaml')

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

# ── Intel Arc / DirectML — optional ──────────────────────────────────────────

try:
    import torch_directml as _dml
    DML_AVAILABLE = _dml.device_count() > 0
    DML_NAME      = _dml.device_name(0) if DML_AVAILABLE else "none"
except (ImportError, Exception):
    DML_AVAILABLE = False
    DML_NAME      = "torch-directml not installed"

# ── Environment detection ─────────────────────────────────────────────────────

def _detect_env():
    # 1. Manual override via environment variable
    forced = os.environ.get('DRONE_ENV', '').lower()
    if forced in ('local', 'intel', 'wsl', 'colab', 'nvidia', 'cpu'):
        print(f"[config] Environment forced: {forced}")
        return forced

    # 2. Auto-detect Colab
    try:
        import google.colab  # noqa
        return 'colab'
    except ImportError:
        pass

    # 3. Auto-detect Intel Arc via DirectML (Windows, no CUDA)
    if not GPU_AVAILABLE and DML_AVAILABLE:
        return 'intel'

    # 4. Auto-detect WSL2 (Linux kernel built by Microsoft)
    try:
        with open('/proc/version') as _f:
            if 'microsoft' in _f.read().lower():
                return 'wsl'
    except OSError:
        pass

    # 5. Auto-detect NVIDIA server (Linux + GPU + no display, not WSL)
    if os.name == 'posix' and GPU_AVAILABLE:
        if not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
            return 'nvidia'

    # 6. Local PC (Windows or Mac with or without GPU)
    return 'local'

ENV      = _detect_env()
IN_COLAB = (ENV == 'colab')

print(f"[config] Running on: {ENV.upper()}")
if ENV == 'intel':
    print(f"[config] GPU: {DML_NAME}  (DirectML)")
else:
    print(f"[config] GPU: {GPU_NAME}")

# ── Paths — per environment ───────────────────────────────────────────────────

if ENV == 'intel':
    # Intel Arc GPU (Windows) — DirectML backend via torch-directml
    SEQUENCE_DIR = os.path.join(_HERE, '..', 'data_from_fred', '7')
    DATASET_DIR  = os.path.join(_HERE, '..', '4channel_project', 'dataset')
    RUNS_DIR     = os.path.join(_HERE, '..', '4channel_project', 'runs')

elif ENV == 'wsl':
    # WSL2 — project cloned into WSL filesystem; GPU via CUDA WSL2 driver
    SEQUENCE_DIR = os.path.join(_HERE, '..', 'data_from_fred', '7')
    DATASET_DIR  = os.path.join(_HERE, '..', '4channel_project', 'dataset')
    RUNS_DIR     = os.path.join(_HERE, '..', '4channel_project', 'runs')

elif ENV == 'colab':
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
    SEQUENCE_DIR = os.path.join(_HERE, '..', 'data_from_fred', '7')
    DATASET_DIR  = os.path.join(_HERE, '..', '4channel_project', 'dataset')
    RUNS_DIR     = os.path.join(_HERE, '..', '4channel_project', 'runs')

else:
    # Local PC — Windows VS Code
    SEQUENCE_DIR = os.path.join(_HERE, '..', 'data_from_fred', '7')
    DATASET_DIR  = os.path.join(_HERE, '..', '4channel_project', 'dataset')
    RUNS_DIR     = os.path.join(_HERE, '..', '4channel_project', 'runs')

# Single-sequence path helpers — these are plain strings used by tools/ and
# build_dataset.py --single.  No zip is opened here; each tool calls
# zip_utils.init_sequence(seq_dir) itself after parsing its --seq argument.

RAW_FILE       = os.path.join(SEQUENCE_DIR, "Event", "events.raw")
COORDS_FILE    = os.path.join(SEQUENCE_DIR, "interpolated_coordinates.txt")
FRAMES_DIR     = os.path.join(SEQUENCE_DIR, "Event", "Frames")
EVENT_YOLO_DIR = os.path.join(SEQUENCE_DIR, "Event_YOLO")
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

if ENV == 'intel':
    # Intel Arc GPU — DirectML backend; train scripts patch Ultralytics to accept 'dml'
    EPOCHS  = 100
    BATCH   = 8     # start conservative; Arc 140T has 16 GB shared — can increase
    DEVICE  = 'dml'

elif ENV == 'wsl':
    # WSL2 — same GPU settings as local but confirmed CUDA available
    EPOCHS  = 100
    BATCH   = 16
    DEVICE  = 0 if GPU_AVAILABLE else 'cpu'

elif ENV == 'colab':
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
