param(
    [switch]$InstallOnly,
    [switch]$VerifyOnly,
    [switch]$RunTests
)

Write-Host "Running project helper script..."

# Ensure we are in the repository root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

if (-not (Test-Path .\battery_soc_env\Scripts\Activate.ps1)) {
    Write-Host "Virtual environment not found. Creating battery_soc_env..."
    python -m venv battery_soc_env
}

Write-Host "Activating virtual environment..."
& .\battery_soc_env\Scripts\Activate.ps1

Write-Host "Upgrading pip and installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

if ($InstallOnly) {
    Write-Host "Install-only mode complete. Exiting."
    return
}

Write-Host "Verifying Python basic setup..."
python -c "import sys, torch; print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__)"

Write-Host "Running SimpleMLP sanity check..."
python .\src\models\simple_mlp.py

if ($VerifyOnly) {
    Write-Host "Verify-only mode complete. Exiting."
    return
}

Write-Host "Running comprehensive project verification..."
python .\scripts\verify_project.py

if ($RunTests) {
    Write-Host "Running pytest suite..."
    python -m pytest tests/ -q
}

Write-Host "Helper script finished."
Write-Host "If you want to train models, run: python train.py --model cnn" \
           "and after training, run: python .\scripts\evaluate_cnn_ukf.py"