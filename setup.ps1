# Create drone_detect venv and install all dependencies.
# Auto-detects CUDA version and installs the matching GPU PyTorch build.
#
# Usage (from ai_drone/):
#   .\setup.ps1
#   .\drone_detect\Scripts\Activate.ps1

Set-Location $PSScriptRoot

Write-Host "=== Drone Detect - environment setup ===" -ForegroundColor Cyan
Write-Host ""

# ── Check Python ──────────────────────────────────────────────────────────────
$pyExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyExe) {
    Write-Host "ERROR: python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVer = (python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
Write-Host "Python: $pyVer"

# ── Check venv module ─────────────────────────────────────────────────────────
$venvCheck = python -m venv --help 2>$null
if (-not $?) {
    Write-Host ""
    Write-Host "ERROR: venv module not available." -ForegroundColor Red
    Write-Host "Install Python from python.org (venv is included) or reinstall Python."
    exit 1
}

# Create venv
python -m venv drone_detect
& ".\drone_detect\Scripts\Activate.ps1"

pip install --upgrade pip --quiet

# ── PyTorch — GPU or CPU ──────────────────────────────────────────────────────
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $nvsmi  = (nvidia-smi) -join " "
    $match  = [regex]::Match($nvsmi, "CUDA Version: (\d+)\.(\d+)")
    $major  = $match.Groups[1].Value
    $minor  = $match.Groups[2].Value
    $cu     = "cu${major}${minor}"
    Write-Host "GPU detected: CUDA $major.$minor  ->  torch index: $cu" -ForegroundColor Green
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu" --quiet
} else {
    Write-Host "No GPU detected - installing CPU-only PyTorch" -ForegroundColor Yellow
    pip install torch torchvision --quiet
}

# ── Everything else ───────────────────────────────────────────────────────────
Write-Host "Installing project dependencies..."
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "Activate:  .\drone_detect\Scripts\Activate.ps1"
Write-Host "Run:       python common\make_catalog.py --help"
