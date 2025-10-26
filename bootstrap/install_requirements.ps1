<#
PowerShell helper to create a virtual environment and install requirements.

Usage (PowerShell):
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\bootstrap\install_requirements.ps1

This will create a `.venv` in the `bootstrap/` folder and install packages from requirements.txt.
#>

$ErrorActionPreference = 'Stop'

Write-Host "Bootstrap installer starting..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python not found on PATH. Please install Python 3.8+ and try again."
    exit 1
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment at $venv..."
    python -m venv $venv
}

Write-Host "Activating virtual environment..."
$activatePs = Join-Path $venv "Scripts\Activate.ps1"
$activateSh = Join-Path $venv "bin\activate"
if (Test-Path $activatePs) {
  # PowerShell activation (Windows layout)
  . $activatePs
} elseif (Test-Path $activateSh) {
  # POSIX layout detected (no PowerShell activation script available)
  Write-Host "No PowerShell activation script found. Detected POSIX venv layout."
  Write-Host "To activate this environment in a bash-style shell run:`n  source $venv/bin/activate"
} else {
  Write-Warning "Activation script not found at '$venv\\Scripts\\Activate.ps1' or '$venv/bin/activate'. Skipping activation."
}

Write-Host "Upgrading pip and installing requirements..."
# Try to upgrade pip; if pip isn't installed, attempt to bootstrap it with ensurepip
try {
  python -m pip install --upgrade pip
} catch {
  Write-Host "pip not available for this Python. Attempting to bootstrap pip using ensurepip..."
  try {
    python -m ensurepip --upgrade
  } catch {
    Write-Warning "Could not bootstrap pip with ensurepip. You may need to install pip or use a different Python distribution."
  }
}

# Install requirements (pip may still be missing; this will show a clear error if so)
python -m pip install -r (Join-Path $root "requirements.txt")

Write-Host "Installation complete. To activate the venv later run:`n  .\bootstrap\.venv\Scripts\Activate.ps1"
