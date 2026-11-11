#!/usr/bin/env bash
# One-time setup (Linux / WSL2, NVIDIA GPU) for training a Piper voice on your recordings.
# Everything goes in ~/murmr-voice: no system packages, no sudo. About 4 GB.
set -euo pipefail
R=~/murmr-voice
PIPER_TAG="${PIPER_TAG:-v1.8.0}"
mkdir -p "$R/bin" && cd "$R"
if [[ ! -x bin/uv ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$R/bin" UV_NO_MODIFY_PATH=1 sh
fi
command -v gcc >/dev/null || { echo "Needs a C compiler (gcc) to build Piper's training extension."; exit 1; }
[[ -d .venv ]] || bin/uv venv --python 3.11 .venv
. .venv/bin/activate
bin/uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
bin/uv pip install cmake ninja cython setuptools wheel scikit-build "huggingface_hub" "onnx-asr" onnxruntime
if [[ ! -d src ]]; then
  git clone --depth 1 --branch "$PIPER_TAG" https://github.com/OHF-voice/piper1-gpl.git src
fi
cd src
bin=../bin
$bin/uv pip install -e '.[train]'
./build_monotonic_align.sh
python setup.py build_ext --inplace
python -c "import torch; print('GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))"
python -c "import piper.train; print('Piper training ready')"
