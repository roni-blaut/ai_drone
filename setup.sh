#!/usr/bin/env bash
# Create drone_detect venv and install all dependencies.
# Auto-detects CUDA version and installs the matching GPU PyTorch build.
#
# Usage:
#   bash setup.sh
#   source drone_detect/bin/activate

set -e
cd "$(dirname "$0")"

echo "=== Drone Detect — environment setup ==="
echo ""

# ── Check Python ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VER"

# ── Check venv module ─────────────────────────────────────────────────────────
if ! python3 -m venv --help &>/dev/null; then
    echo ""
    echo "ERROR: python3-venv is not installed."
    echo "Fix (Ubuntu/Debian/WSL):"
    echo "    sudo apt install python3-venv python3-pip"
    echo "Then re-run:  bash setup.sh"
    exit 1
fi

# Create venv
python3 -m venv drone_detect
source drone_detect/bin/activate

pip install --upgrade pip --quiet

# ── PyTorch — GPU or CPU ──────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    RAW=$(nvidia-smi | grep -oP "CUDA Version: \K[\d.]+")
    MAJOR=$(echo "$RAW" | cut -d. -f1)
    MINOR=$(echo "$RAW" | cut -d. -f2)
    CU="cu${MAJOR}${MINOR}"
    echo "GPU detected: CUDA $RAW  →  torch index: $CU"
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CU" --quiet
else
    echo "No GPU detected — installing CPU-only PyTorch"
    pip install torch torchvision --quiet
fi

# ── Everything else ───────────────────────────────────────────────────────────
echo "Installing project dependencies..."
pip install -r requirements.txt --quiet

echo ""
echo "=== Done ==="
echo "Activate:  source drone_detect/bin/activate"
echo "Run:       python common/make_catalog.py --help"
