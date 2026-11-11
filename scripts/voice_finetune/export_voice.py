"""Step 3: export the latest round to ONNX, check it, and install it into Murmr.

The check reads a few sentences with the new voice and transcribes them with Parakeet: if
the words come back right, the voice is intelligible. The voice then appears in Murmr under
Settings > Read aloud > Voice as "My voice".

    python export_voice.py                 # export + check + install
    python export_voice.py --no-install    # just export and check
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
from common import whispa_dir  # noqa: E402
from train_voice import WORK, latest_checkpoint  # noqa: E402

CHECK = [
    "Thanks for the update, I'll get back to you this afternoon.",
    "Could we move Thursday's meeting to Friday morning?",
    "The quarterly numbers look much better than last year.",
    "Please remember to bring the signed forms with you.",
]


def words(t: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", t.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(1, len(r))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="me")
    ap.add_argument("--label", default="My voice")
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    root = WORK / args.name
    ckpt = latest_checkpoint(root)
    if ckpt is None:
        raise SystemExit("No trained checkpoint yet: run train_voice.py first.")
    out = root / "export"
    out.mkdir(exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt),
                           "--output-file", str(out / "voice.onnx")])
    shutil.copyfile(root / "voice.onnx.json", out / "voice.onnx.json")

    # Intelligibility check: synthesize, then transcribe with Parakeet.
    from piper import PiperVoice
    import onnx_asr

    voice = PiperVoice.load(str(out / "voice.onnx"), config_path=str(out / "voice.onnx.json"))
    asr = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", quantization="int8",
                              providers=["CPUExecutionProvider"])
    scores, sample = [], None
    for text in CHECK:
        audio = np.concatenate([c.audio_float_array for c in voice.synthesize(text)]).astype(np.float32)
        sample = audio if sample is None else sample
        heard = asr.recognize(audio, sample_rate=voice.config.sample_rate)
        scores.append(wer(text, heard))
        print(f"  said:  {text}\n  heard: {heard}")
    mean_wer = sum(scores) / len(scores)
    print(f"Word error rate when Parakeet listens to it: {mean_wer:.1%}")
    with wave.open(str(out / "sample.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(voice.config.sample_rate)
        w.writeframes((np.clip(sample, -1, 1) * 32767).astype(np.int16).tobytes())

    state = json.loads((root / "state.json").read_text()) if (root / "state.json").exists() else {}
    meta = root / "dataset" / "metadata.csv"
    minutes = 0.0
    if meta.exists():
        for line in meta.read_text(encoding="utf-8").splitlines():
            wav = root / "dataset" / "wavs" / line.split("|")[0]
            if wav.exists():
                minutes += wav.stat().st_size / 2 / 22050 / 60
    info = {"label": args.label, "minutes": round(minutes, 1),
            "epochs": state.get("epochs_on_your_data", 0), "rounds": state.get("rounds", 0),
            "intelligibility_wer": round(mean_wer, 4), "date": time.strftime("%Y-%m-%d")}
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if mean_wer > 0.25:
        print("Hard to understand yet (more than 1 word in 4 misheard): not installing. "
              "Train another round.")
        return
    if args.no_install:
        return
    dst = whispa_dir() / "voices" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("voice.onnx", "voice.onnx.json", "info.json", "sample.wav"):
        shutil.copyfile(out / f, dst / f)
    print(f"Installed to {dst}. In Murmr: Settings > Read aloud > Voice > {args.label}.")


if __name__ == "__main__":
    main()
