#!/usr/bin/env bash
# One full round: prepare your latest reads, train (at most HOURS), export, check, install.
#     ./pipeline.sh            # 6 h round
#     HOURS=2 ./pipeline.sh
# Lines starting with "== " mark the steps (Murmr's Voice training page shows them).
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x ~/murmr-voice/.venv/bin/python ]]; then
  echo "== 0/3 Setting up voice training (one time, about 4 GB)"
  bash setup.sh
fi
. ~/murmr-voice/.venv/bin/activate
echo "== 1/3 Preparing your reading"
python prepare_voice.py
echo "== 2/3 Training your voice (at most ${HOURS:-6} h this round)"
python train_voice.py --hours "${HOURS:-6}"
echo "== 3/3 Checking and installing"
python export_voice.py
echo "== done"
