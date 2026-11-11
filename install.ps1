# Whispa installer for Windows. Usage:  powershell -ExecutionPolicy Bypass -File install.ps1 [-Runtime cuda|directml|cpu]
param([ValidateSet("auto", "cuda", "directml", "cpu")][string]$Runtime = "auto")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Runtime -eq "auto") {
    $Runtime = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $Runtime = "cuda" }
}
Write-Host "Installing Whispa with the '$Runtime' ONNX Runtime..."

if (Get-Command uv -ErrorAction SilentlyContinue) {
    if (-not (Test-Path .venv)) { uv venv .venv --python 3.13 }
    uv pip install --python .venv -e ".[$Runtime]"
} else {
    if (-not (Test-Path .venv)) { py -3 -m venv .venv }
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -e ".[$Runtime]"
}

Write-Host "Downloading the Parakeet model (first run only, ~650 MB)..."
.venv\Scripts\python.exe -c "from whispa.asr import Transcriber; from whispa import config; c = config.load(); t = Transcriber(c.model, c.device, c.quantization); t.load(); print('Model ready on', t.backend)"

.venv\Scripts\python.exe -m whispa --install-shortcuts

Write-Host ""
Write-Host "Done. Start Whispa from the Start Menu or the Desktop shortcut (no console window)."
Write-Host "Turn on 'Launch at login' in Whispa > Settings."
