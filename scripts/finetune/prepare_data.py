"""Step 1: turn Murmr's voice samples into NeMo manifests with a fixed held-out test set.

    python prepare_data.py                      # your clips from %APPDATA%\\Murmr\\training
    python prepare_data.py --synthetic DIR      # proof-of-pipeline only: TTS clips (see README)

Writes WORK/data/user/{train,val,test}.json (one JSON object per line: audio_filepath,
duration, text). Clips are copied into WORK/data/user/audio, so training never touches
your Murmr folder. Which clips are used:
  read       reading-session sentences (exact labels)             always
  corrected  dictations you fixed (Parakeet's words + your fixes)  always
  unedited   dictations you left as they were                      --include-unedited
  unknown    nobody checked the label: never used (training on the model's own mistakes
             would only reinforce them)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import wave
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import WORK, read_manifest, read_whispa_manifest, split_of, whispa_dir, write_manifest  # noqa: E402


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def from_whispa(src: Path, statuses: set[str]) -> list[dict]:
    clips = read_whispa_manifest(src / "manifest.jsonl")
    print(f"{len(clips)} clips in {src}: {dict(Counter(c.get('status') for c in clips))}")
    out = []
    for c in clips:
        wav = src / c["audio"]
        if c.get("status") in statuses and c.get("text", "").strip() and wav.exists():
            out.append({"id": c["id"], "src": wav, "text": c["text"].strip()})
    return out


def from_synthetic(src: Path) -> list[dict]:
    """TTS clips made by make_synthetic.ps1: NNNN.wav + prompts.txt (line N = clip N)."""
    prompts = (src / "prompts.txt").read_text(encoding="utf-8").splitlines()
    return [{"id": f"synth-{i:04d}", "src": src / f"{i:04d}.wav", "text": t}
            for i, t in enumerate(prompts) if (src / f"{i:04d}.wav").exists()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, help="Murmr training folder (default: auto)")
    ap.add_argument("--synthetic", type=Path, help="use TTS clips instead (pipeline test only)")
    ap.add_argument("--include-unedited", action="store_true")
    ap.add_argument("--test-pct", type=int, default=15)
    ap.add_argument("--val-pct", type=int, default=10)
    ap.add_argument("--max-seconds", type=float, default=25.0, help="skip longer clips (VRAM)")
    ap.add_argument("--out", type=Path, default=WORK / "data" / "user")
    args = ap.parse_args()

    if args.synthetic:
        items = from_synthetic(args.synthetic)
    else:
        cfg_path = whispa_dir() / "config.json"
        try:
            consent = json.loads(cfg_path.read_text(encoding="utf-8")).get("save_training_audio", False)
        except (OSError, ValueError):
            consent = False
        if not consent:
            raise SystemExit("Turn on Settings > Your voice > \"Improve recognition with my voice\" "
                             "in Murmr first: recordings are only used for training with your OK.")
        statuses = {"read", "corrected"} | ({"unedited"} if args.include_unedited else set())
        items = from_whispa(args.source or whispa_dir() / "training", statuses)

    audio_dir = args.out / "audio"
    if args.out.exists():  # start clean so deleted/re-recorded clips disappear
        shutil.rmtree(args.out)
    audio_dir.mkdir(parents=True)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    skipped = 0
    for it in items:
        dur = wav_seconds(it["src"])
        if dur < 0.5 or dur > args.max_seconds:
            skipped += 1
            continue
        dst = audio_dir / f"{it['id']}.wav"
        shutil.copyfile(it["src"], dst)
        splits[split_of(it["id"], args.test_pct, args.val_pct)].append(
            {"audio_filepath": str(dst.resolve()), "duration": round(dur, 3), "text": it["text"]})
    for name, rows in splits.items():
        write_manifest(args.out / f"{name}.json", rows)
        mins = sum(r["duration"] for r in rows) / 60
        print(f"  {name:5s} {len(rows):5d} clips  {mins:6.1f} min  -> {args.out / f'{name}.json'}")
    if skipped:
        print(f"  skipped {skipped} clips shorter than 0.5 s or longer than {args.max_seconds:g} s")
    train_min = sum(r["duration"] for r in splits["train"]) / 60
    if train_min < 20 and not args.synthetic:
        print(f"\nOnly {train_min:.0f} min to train on. Record at least 30 minutes on the Voice "
              "training page for a useful tune (60+ is better).")
    assert read_manifest(args.out / "test.json") == splits["test"]


if __name__ == "__main__":
    main()
