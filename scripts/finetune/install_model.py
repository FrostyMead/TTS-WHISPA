"""Step 6: copy an exported, evaluated model into Murmr's models folder.

    python install_model.py WORK/models/<name>

Copies <name>/onnx to %APPDATA%\\Murmr\\models\\<name>. It then shows up in Murmr under
Settings → Speech model, with its evaluation numbers. Switching to it (and back to the base
model) happens there; this script doesn't change your config.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import whispa_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--force", action="store_true", help="install even if evaluation said no")
    args = ap.parse_args()
    src = args.model_dir / "onnx"
    report = src / "eval.json"
    if not report.exists():
        raise SystemExit("Run evaluate.py first, so Murmr can show how the model did.")
    verdict = json.loads(report.read_text())
    if not verdict.get("recommend") and not args.force:
        raise SystemExit(f"Evaluation didn't recommend this model ({verdict.get('reason')}). "
                         "Use --force to install it anyway.")
    dst = whispa_dir() / "models" / args.model_dir.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"installed to {dst}\nIn Murmr: Settings → Speech model → Use.")


if __name__ == "__main__":
    main()
