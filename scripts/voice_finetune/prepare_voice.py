"""Step 1: turn your Voice training reads into a Piper dataset (wavs + metadata.csv).

Uses only what you allowed: Settings > Your voice > "Create a voice that sounds like me".
Only sentences you read live on the Voice training page are used (never dictations), and
only clean takes: no clipping, not too quiet, little background noise. Audio is the 24 kHz
copy Murmr keeps of each read, resampled to Piper's 22.05 kHz.

    python prepare_voice.py                 # -> ~/murmr-voice/work/<name>/dataset
    python prepare_voice.py --name me --max-noise-db -45
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "finetune"))
from common import read_whispa_manifest, whispa_dir  # noqa: E402

WORK = Path.home() / "murmr-voice" / "work"
SR = 22050


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    spec = np.fft.rfft(audio.astype(np.float64))
    spec = spec[: n_out // 2 + 1] if n_out < len(audio) else np.pad(spec, (0, n_out // 2 + 1 - len(spec)))
    return (np.fft.irfft(spec, n_out) * (n_out / len(audio))).astype(np.float32)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768, sr


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def trim_silence(audio: np.ndarray, sr: int, keep_s: float = 0.15) -> np.ndarray:
    """Cut long leading/trailing silence (button press, breath before speaking)."""
    win = sr // 50
    n = len(audio) // win
    if n < 3:
        return audio
    rms = np.sqrt(np.mean(audio[: n * win].reshape(n, win) ** 2, axis=1))
    loud = np.where(rms > max(0.01, rms.max() * 0.05))[0]
    if not len(loud):
        return audio
    keep = int(keep_s * sr)
    return audio[max(0, loud[0] * win - keep): min(len(audio), (loud[-1] + 1) * win + keep)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="me", help="voice name (folder name)")
    ap.add_argument("--source", type=Path, help="Murmr data folder (default: auto)")
    ap.add_argument("--max-noise-db", type=float, default=-45.0, help="noisier takes are skipped")
    ap.add_argument("--min-peak", type=float, default=0.05, help="quieter takes are skipped")
    ap.add_argument("--max-clipped", type=float, default=0.001, help="fraction of clipped samples")
    args = ap.parse_args()

    data = args.source or whispa_dir()
    try:
        consent = json.loads((data / "config.json").read_text(encoding="utf-8")).get("voice_clone", False)
    except (OSError, ValueError):
        consent = False
    if not consent:
        raise SystemExit('Turn on Settings > Your voice > "Create a voice that sounds like me" in '
                         "Murmr first: your recordings are only used for a voice with your OK.")

    rows = read_whispa_manifest(data / "training" / "manifest.jsonl")
    out = WORK / args.name / "dataset"
    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    for old in wavs.glob("*.wav"):
        old.unlink()
    lines, seconds, skipped = [], 0.0, {"no high-quality copy": 0, "noisy": 0, "too quiet": 0,
                                         "clipped": 0, "missing file": 0}
    for r in rows:
        if r.get("status") != "read":
            continue  # only sentences you read live
        if not r.get("hq"):
            skipped["no high-quality copy"] += 1
            continue
        src = data / "training" / r["hq"]
        if not src.exists():
            skipped["missing file"] += 1
            continue
        if r.get("noise_db", 0) > args.max_noise_db:
            skipped["noisy"] += 1
            continue
        if r.get("peak", 1) < args.min_peak:
            skipped["too quiet"] += 1
            continue
        if r.get("clipped", 0) > args.max_clipped:
            skipped["clipped"] += 1
            continue
        audio, sr = read_wav(src)
        audio = trim_silence(resample(audio, sr, SR), SR)
        audio = audio * (0.9 / max(1e-3, float(np.abs(audio).max())))  # even loudness
        write_wav(wavs / f"{r['id']}.wav", audio, SR)
        text = " ".join(r["text"].split()).replace("|", " ")
        lines.append(f"{r['id']}.wav|{text}")
        seconds += len(audio) / SR
    (out / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} clips, {seconds / 60:.1f} min -> {out}")
    for why, n in skipped.items():
        if n:
            print(f"  skipped {n}: {why}")
    if seconds < 15 * 60:
        print("  Under 15 minutes: training works, but the voice will be rough. 30-60 min is "
              "where it starts to sound natural.")


if __name__ == "__main__":
    main()
