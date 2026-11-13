#!/usr/bin/env bash
# The whole pipeline: prepare -> general data -> train -> export -> evaluate -> install.
# Run inside WSL2 (or Linux) with the venv from README.md activated. Extra arguments go to
# train.py, e.g.:  ./pipeline.sh --train-layers 8 --epochs 6
# Lines starting with "== " mark the steps (Murmr's Voice training page shows them).
set -euo pipefail
cd "$(dirname "$0")"
WORK="${WHISPA_FT_WORK:-$HOME/whispa-finetune/work}"
NAME="tuned-$(date +%Y%m%d-%H%M)"

echo "== 1/6 Preparing your recordings"
python prepare_data.py ${MURMR_INCLUDE_UNEDITED:+--include-unedited}
echo "== 2/6 Getting general speech for comparison"
python get_general_data.py
echo "== 3/6 Training on your voice"
# Out of graphics memory (exit 75): again with half the batch and twice the accumulation.
batch=4; accumulate=4
until python train.py --name "$NAME" "$@" --batch "$batch" --accumulate "$accumulate"; do
  code=$?
  if [ "$code" -ne 75 ] || [ "$batch" -le 1 ]; then exit "$code"; fi
  batch=$((batch / 2)); accumulate=$((accumulate * 2))
done
echo "== 4/6 Exporting the model"
python export_onnx.py "$WORK/models/$NAME"
echo "== 5/6 Comparing with the standard model"
python evaluate.py "$WORK/models/$NAME"
echo "== 6/6 Installing"
python install_model.py "$WORK/models/$NAME" || echo "Not installed (see the evaluation above)."
echo "== done $NAME"
