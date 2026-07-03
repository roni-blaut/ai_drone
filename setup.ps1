# Create drone_detect venv and install all dependencies.
# Auto-detects CUDA version and installs the matching GPU PyTorch build.
#
# Usage (from ai_drone/):
#   .\setup.ps1
#   .\drone_detect\Scripts\Activate.ps1

Set-Location $PSScriptRoot

Write-Host "=== Drone Detect - environment setup ===" -ForegroundColor Cyan
Write-Host ""

# --- Find real Python (skip Windows Store stub) ---
$pyExe = $null
$candidates = @(
    "C:\ProgramData\anaconda3\envs\drone_detect\python.exe",
    "C:\ProgramData\anaconda3\python.exe",
    "$env:USERPROFILE\anaconda3\envs\drone_detect\python.exe",
    "$env:USERPROFILE\anaconda3\python.exe",
    "$env:USERPROFILE\miniconda3\envs\drone_detect\python.exe",
    "$env:USERPROFILE\miniconda3\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
)
foreach ($c in $candidates) {
    if (Test-Path $c) { $pyExe = $c; break }
}
# Fall back to PATH python if it is not the Windows Store stub
if (-not $pyExe) {
    $cmdPython = Get-Command python -ErrorAction SilentlyContinue
    if ($cmdPython) {
        $fromPath = $cmdPython.Source
        if ($fromPath -and $fromPath -notlike "*WindowsApps*") {
            $pyExe = $fromPath
        }
    }
}
if (-not $pyExe) {
    Write-Host "ERROR: No real Python found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVer = (& $pyExe -c "import sys; print(str(sys.version_info.major) + '.' + str(sys.version_info.minor))").Trim()
Write-Host "Python: $pyVer  ($pyExe)"

# --- Ensure venv is available, then create venv ---
$null = & $pyExe -c "import venv" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "venv not found - installing virtualenv as fallback..." -ForegroundColor Yellow
    & $pyExe -m pip install virtualenv --quiet
    & $pyExe -m virtualenv drone_detect
} else {
    & $pyExe -m venv drone_detect
}

# Explicitly target the new environment's pip path
$venvPip = ".\drone_detect\Scripts\pip.exe"

Write-Host "Upgrading pip inside environment..."
& $venvPip install --upgrade pip --quiet

# --- PyTorch: GPU or CPU ---
# Search for nvidia-smi in common locations if not on PATH
$nvSmi = $null
$cmdNvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($cmdNvidia) {
    $nvSmi = $cmdNvidia.Source
}

if (-not $nvSmi) {
    $nvPaths = @(
        "C:\Windows\System32\nvidia-smi.exe",
        "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    )
    foreach ($p in $nvPaths) { if (Test-Path $p) { $nvSmi = $p; break } }
}

if ($nvSmi) {
    $nvsmiOut = (& $nvSmi) -join " "
    $match = [regex]::Match($nvsmiOut, "CUDA Version: (\d+)\.(\d+)")
    $major = $match.Groups[1].Value
    $minor = $match.Groups[2].Value
    $cu = "cu" + $major + $minor
    Write-Host "GPU detected: CUDA $major.$minor -> torch index: $cu" -ForegroundColor Green
    & $venvPip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu" --quiet
} else {
    Write-Host "No GPU detected - installing CPU-only PyTorch" -ForegroundColor Yellow
    & $venvPip install torch torchvision --quiet
}

# --- Project dependencies ---
Write-Host "Installing project dependencies..."
& $venvPip install -r requirements.txt --quiet

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "Activate:  .\drone_detect\Scripts\Activate.ps1"
Write-Host "Run:       python common\make_catalog.py --help"
