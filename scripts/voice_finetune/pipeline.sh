#!/usr/bin/env bash
# One full round: prepare your latest reads, train (at most HOURS), export, check, install.
#     ./pipeline.sh            # 6 h round
#     HOURS=2 ./pipeline.sh
set -euo pipefail
cd "$(dirname "$0")"
. ~/murmr-voice/.venv/bin/activate
python prepare_voice.py
python train_voice.py --hours "${HOURS:-6}"
python export_voice.py
