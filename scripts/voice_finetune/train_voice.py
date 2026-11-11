"""Step 2: one training round (e.g. one night) of a Piper voice on your recordings.

The first round starts from a finished public voice (Lessac, medium quality), which makes
fine-tuning far faster than training from scratch. Every later round resumes where the last
one stopped, so the voice keeps getting closer to yours, night after night, until the
target is reached. A round stops by itself after --hours.

    python train_voice.py                      # one round, at most 6 h
    python train_voice.py --hours 0.2          # a quick test round
    python train_voice.py --espeak-voice en-gb-x-rp    # British / Australian / NZ accent

Murmr and the local AI should be closed while it runs (they use GPU memory).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

WORK = Path.home() / "murmr-voice" / "work"
BASE_REPO = "rhasspy/piper-checkpoints"
BASE_PREFIX = "en/en_US/lessac/medium/"


def base_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(BASE_REPO, repo_type="dataset")
             if f.startswith(BASE_PREFIX) and f.endswith(".ckpt")]
    if not files:
        raise SystemExit(f"No base checkpoint found under {BASE_REPO}/{BASE_PREFIX}")
    return Path(hf_hub_download(BASE_REPO, sorted(files)[-1], repo_type="dataset"))


def latest_checkpoint(root: Path) -> Path | None:
    ckpts = sorted(root.glob("lightning_logs/version_*/checkpoints/last.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def checkpoint_epoch(ckpt: Path) -> int:
    import torch

    return int(torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", 0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="me")
    ap.add_argument("--hours", type=float, default=6.0, help="stop this round after this long")
    ap.add_argument("--target-epochs", type=int, default=2000,
                    help="epochs on your data in total (Piper's advice for < 5 h of audio)")
    ap.add_argument("--batch-size", type=int, default=16, help="16 fits 8 GB of VRAM")
    ap.add_argument("--espeak-voice", default="en-us")
    args = ap.parse_args()

    root = WORK / args.name
    data = root / "dataset"
    if not (data / "metadata.csv").exists():
        raise SystemExit("Run prepare_voice.py first.")
    state_path = root / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    resume = latest_checkpoint(root)
    if resume is None:
        resume = base_checkpoint()
        state = {"base_epoch": checkpoint_epoch(resume), "rounds": 0, "espeak_voice": args.espeak_voice}
        print(f"First round: starting from the Lessac voice (epoch {state['base_epoch']})")
    start = checkpoint_epoch(resume)
    trained = start - state["base_epoch"]
    if trained >= args.target_epochs:
        print(f"Done: {trained} epochs on your data already (target {args.target_epochs}).")
        return
    end = state["base_epoch"] + args.target_epochs
    h = args.hours
    max_time = f"00:{int(h):02d}:{int(h * 60) % 60:02d}:00"
    cmd = [sys.executable, "-m", "piper.train", "fit",
           "--data.voice_name", args.name,
           "--data.csv_path", str(data / "metadata.csv"),
           "--data.audio_dir", str(data / "wavs"),
           "--model.sample_rate", "22050",
           "--data.espeak_voice", state.get("espeak_voice", args.espeak_voice),
           "--data.cache_dir", str(root / "cache"),
           "--data.config_path", str(root / "voice.onnx.json"),
           "--data.batch_size", str(args.batch_size),
           "--data.num_test_examples", "2",
           "--model.mos_metric", "none",
           "--trainer.max_epochs", str(end),
           "--trainer.max_time", max_time,
           "--trainer.default_root_dir", str(root),
           "--ckpt_path", str(resume)]
    print(f"Round {state['rounds'] + 1}: epoch {start} -> up to {end}, at most {h:g} h")
    t0 = time.time()
    code = subprocess.call(cmd)
    state["rounds"] += 1
    state["last_round_hours"] = round((time.time() - t0) / 3600, 2)
    last = latest_checkpoint(root)
    if last:
        state["epochs_on_your_data"] = checkpoint_epoch(last) - state["base_epoch"]
    state_path.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
