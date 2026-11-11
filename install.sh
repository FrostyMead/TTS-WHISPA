#!/usr/bin/env bash
# Whispa installer for macOS and Linux. Usage: ./install.sh [cuda|cpu]
set -euo pipefail
cd "$(dirname "$0")"

runtime="${1:-auto}"
if [[ "$runtime" == "auto" ]]; then
  runtime="cpu"
  if [[ "$(uname)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then runtime="cuda"; fi
fi
echo "Installing Whispa with the '$runtime' ONNX Runtime..."

if command -v uv >/dev/null 2>&1; then
  [[ -d .venv ]] || uv venv .venv --python 3.12
  uv pip install --python .venv -e ".[$runtime]"
else
  [[ -d .venv ]] || python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e ".[$runtime]"
fi

echo "Downloading the Parakeet model (first run only, ~650 MB)..."
.venv/bin/python -c "from whispa.asr import Transcriber; from whispa import config; c = config.load(); t = Transcriber(c.model, c.device, c.quantization); t.load(); print('Model ready on', t.backend)"

if [[ "$(uname)" == "Linux" ]]; then
  for tool in xclip wl-copy; do command -v "$tool" >/dev/null || echo "Tip: install '$tool' for clipboard support."; done
  [[ -n "${WAYLAND_DISPLAY:-}" ]] && echo "Wayland detected: see README 'Linux' notes (wtype/ydotool, input group)."
fi

.venv/bin/python -m whispa --install-shortcuts || true

echo
echo "Done. Start Whispa with:   .venv/bin/whispa   (Linux: also in your app launcher)"
echo "Launch at login:           .venv/bin/whispa-cli --autostart on"
