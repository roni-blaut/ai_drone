# Create drone_detect venv and install all dependencies.
# Auto-detects GPU type and installs the matching PyTorch build:
#   NVIDIA GPU   -> pip install torch ... --index-url .../whl/cuXXX
#   Intel Arc    -> pip install torch-directml  (Microsoft DirectML)
#   No GPU       -> pip install torch torchvision  (CPU-only)
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
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "C:\Python310\python.exe",
    "C:\Python311\python.exe",
    "C:\Python312\python.exe"
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

# --- Detect GPU type ---
# Priority: NVIDIA (CUDA) > Intel Arc (DirectML) > CPU

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
    # --- NVIDIA GPU: install CUDA PyTorch ---
    $nvsmiOut = (& $nvSmi) -join " "
    $match = [regex]::Match($nvsmiOut, "CUDA Version: (\d+)\.(\d+)")
    $major = $match.Groups[1].Value
    $minor = $match.Groups[2].Value
    $cu = "cu" + $major + $minor
    Write-Host "GPU: NVIDIA CUDA $major.$minor -> torch index: $cu" -ForegroundColor Green
    & $venvPip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu" --quiet

} else {
    # --- Check for Intel Arc GPU via WMI ---
    $gpuList = Get-CimInstance -ClassName Win32_VideoController -ErrorAction SilentlyContinue |
               Select-Object -ExpandProperty Name
    $gpuNames = ($gpuList -join "|")
    $isIntelArc = $gpuNames -match "Intel.*(Arc|Xe)"

    if ($isIntelArc) {
        # --- Intel Arc GPU: install DirectML backend ---
        $arcName = ($gpuList | Where-Object { $_ -match "Intel.*(Arc|Xe)" }) -join ", "
        Write-Host "GPU: Intel Arc detected ($arcName)" -ForegroundColor Cyan
        Write-Host "     Installing torch + torch-directml (Microsoft DirectML for Intel Arc)" -ForegroundColor Cyan
        # DML works on top of standard CPU-build torch (no CUDA index needed)
        & $venvPip install torch torchvision --quiet
        & $venvPip install torch-directml --quiet
        Write-Host ""
        Write-Host "NOTE: Training will use device='dml'" -ForegroundColor Cyan
        Write-Host "      config.py auto-detects torch-directml -- no manual settings needed." -ForegroundColor Cyan

    } else {
        # --- No supported GPU: CPU-only PyTorch ---
        if ($gpuNames) {
            Write-Host "GPU: $gpuNames (no CUDA or DirectML support detected)" -ForegroundColor Yellow
        } else {
            Write-Host "GPU: None detected" -ForegroundColor Yellow
        }
        Write-Host "Installing CPU-only PyTorch"
        & $venvPip install torch torchvision --quiet
    }
}

# --- Project dependencies ---
Write-Host "Installing project dependencies..."
& $venvPip install -r requirements.txt --quiet

Write-Host ""
Write-Host "=== Done ===" -ForegroundColor Cyan
Write-Host "Activate:  .\drone_detect\Scripts\Activate.ps1"
Write-Host "Run:       python common\make_catalog.py --help"
