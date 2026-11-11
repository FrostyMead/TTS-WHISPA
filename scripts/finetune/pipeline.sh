#!/usr/bin/env bash
# The whole pipeline: prepare -> general data -> train -> export -> evaluate -> install.
# Run inside WSL2 (or Linux) with the venv from README.md activated. Extra arguments go to
# train.py, e.g.:  ./pipeline.sh --train-layers 8 --epochs 6
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WHISPA_FT_WORK:-$HOME/whispa-finetune/work}"
NAME="tuned-$(date +%Y%m%d-%H%M)"

python prepare_data.py
python get_general_data.py
python train.py --name "$NAME" "$@"
python export_onnx.py "$WORK/models/$NAME"
python evaluate.py "$WORK/models/$NAME"
python install_model.py "$WORK/models/$NAME" || echo "Not installed (see the evaluation above)."
