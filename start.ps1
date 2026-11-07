$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $appPython)) {
    throw 'Application environment not found. Follow the installation steps in README.md.'
}
if (-not $env:MEDSAM2_CHECKPOINT) {
    $env:MEDSAM2_CHECKPOINT = Join-Path $PSScriptRoot 'checkpoints\MedSAM2_latest.pt'
}
if (-not $env:MEDSAM2_CONFIG) {
    $env:MEDSAM2_CONFIG = 'configs/sam2.1_hiera_t512.yaml'
}
if (-not $env:YOLO_CONFIG_DIR) {
    $env:YOLO_CONFIG_DIR = Join-Path $PSScriptRoot '.local\ultralytics'
}
if (-not $env:MASKVISION_DEVICE) {
    $env:MASKVISION_DEVICE = 'cpu'
}
if ($env:MASKVISION_DEVICE -eq 'cpu') {
    $env:CUDA_VISIBLE_DEVICES = '-1'
    $env:OMP_NUM_THREADS = '2'
    $env:MKL_NUM_THREADS = '2'
}
& $appPython (Join-Path $PSScriptRoot 'app.py')
exit $LASTEXITCODE
