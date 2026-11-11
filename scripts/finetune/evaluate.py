"""Step 5: compare the base and tuned models on held-out data, through onnx-asr (what Murmr runs).

    python evaluate.py WORK/models/<name>

Two test sets, neither ever used for training:
  yours    WORK/data/user/test.json     about 15% of your clips, chosen by a hash of the clip
                                        id, so it stays the same as you add data
  general  WORK/data/general/test.json  200 LibriSpeech-PC test-clean utterances

WER ignores case, punctuation and number style (like the Open ASR Leaderboard).
"PnC WER" also counts punctuation and capitals. The verdict (recommend: true/false) and all
numbers are saved to <model>/onnx/eval.json, which Murmr's Settings page shows.

A switch is recommended only when your WER improves by at least 5% relative AND general WER
gets no worse than +0.5 points (absolute) or +5% relative, whichever is larger.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import WORK, corpus_wer, pnc_wer, read_manifest  # noqa: E402

MIN_GAIN = 0.05  # relative improvement needed on your voice
MAX_LOSS_ABS, MAX_LOSS_REL = 0.005, 0.05  # allowed general regression


def default_base() -> str:
    """The base ONNX model Murmr already downloaded (Windows HF cache), else download it."""
    roots = [Path.home() / ".cache" / "huggingface" / "hub"]
    roots += [Path(p) for p in glob.glob("/mnt/c/Users/*/.cache/huggingface/hub")]
    for r in roots:
        hits = sorted(r.glob("models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/*/config.json"))
        if hits:
            return str(hits[-1].parent)
    return "nemo-parakeet-tdt-0.6b-v3"


def load(model: str, providers: list[str]):
    import onnx_asr

    if os.path.isdir(model):
        return onnx_asr.load_model("nemo-conformer-tdt", model, providers=providers)
    return onnx_asr.load_model(model, providers=providers)


def transcribe_all(asr, rows: list[dict]) -> tuple[list[str], float]:
    import soundfile as sf

    out, t0 = [], time.perf_counter()
    for r in rows:
        audio, sr = sf.read(r["audio_filepath"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out.append(str(asr.recognize(audio, sample_rate=sr)).strip())
    return out, time.perf_counter() - t0


def evaluate(model: str, sets: dict[str, list[dict]], providers: list[str]) -> dict:
    t0 = time.perf_counter()
    asr = load(model, providers)
    res = {"load_seconds": round(time.perf_counter() - t0, 1)}
    for name, rows in sets.items():
        if not rows:
            continue
        hyps, secs = transcribe_all(asr, rows)
        refs = [r["text"] for r in rows]
        audio = sum(r["duration"] for r in rows)
        res[name] = {"clips": len(rows), "minutes": round(audio / 60, 2),
                     "wer": round(corpus_wer(refs, hyps), 4), "pnc_wer": round(pnc_wer(refs, hyps), 4),
                     "rtf": round(secs / audio, 4)}
        res[name + "_hyps"] = hyps
    return res


def verdict(base: dict, tuned: dict) -> tuple[bool, str]:
    if "yours" not in base:
        return False, "no held-out clips of your voice"
    b, t = base["yours"]["wer"], tuned["yours"]["wer"]
    if t > b * (1 - MIN_GAIN):
        return False, f"your WER didn't improve enough ({b:.1%} -> {t:.1%})"
    if "general" in base:
        gb, gt = base["general"]["wer"], tuned["general"]["wer"]
        if gt > gb + max(MAX_LOSS_ABS, gb * MAX_LOSS_REL):
            return False, f"general WER got worse ({gb:.1%} -> {gt:.1%})"
    return True, "better on your voice, general accuracy kept"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--base", default=None, help="base ONNX dir or onnx-asr name (default: auto)")
    ap.add_argument("--data", type=Path, default=WORK / "data")
    ap.add_argument("--general-size", type=int, default=200)
    ap.add_argument("--gpu", action="store_true", help="use CUDAExecutionProvider")
    ap.add_argument("--show", type=int, default=8, help="print this many changed transcripts")
    args = ap.parse_args()

    providers = (["CUDAExecutionProvider"] if args.gpu else []) + ["CPUExecutionProvider"]
    sets = {"yours": read_manifest(args.data / "user" / "test.json")}
    gen = args.data / "general" / "test.json"
    if gen.exists():
        sets["general"] = read_manifest(gen)[: args.general_size]
    base_model = args.base or default_base()
    onnx_dir = args.model_dir / "onnx"
    print(f"base:  {base_model}\ntuned: {onnx_dir}\n" +
          ", ".join(f"{k}: {len(v)} clips" for k, v in sets.items()))

    base = evaluate(base_model, sets, providers)
    tuned = evaluate(str(onnx_dir), sets, providers)
    ok, why = verdict(base, tuned)

    print(f"\n{'':10s}{'base WER':>10s}{'tuned WER':>11s}{'base PnC':>10s}{'tuned PnC':>11s}")
    for name in sets:
        if name in base:
            print(f"{name:10s}{base[name]['wer']:>10.2%}{tuned[name]['wer']:>11.2%}"
                  f"{base[name]['pnc_wer']:>10.2%}{tuned[name]['pnc_wer']:>11.2%}")
    print(f"\nrecommend switching: {'YES' if ok else 'no'} ({why})")
    if args.show and "yours" in base:
        shown = 0
        for r, hb, ht in zip(sets["yours"], base["yours_hyps"], tuned["yours_hyps"]):
            if hb != ht and shown < args.show:
                print(f"\n  ref:   {r['text']}\n  base:  {hb}\n  tuned: {ht}")
                shown += 1

    info_path = args.model_dir / "train_info.json"
    report = {
        "recommend": ok, "reason": why, "evaluated": time.strftime("%Y-%m-%d %H:%M"),
        "base": {k: v for k, v in base.items() if not k.endswith("_hyps")},
        "tuned": {k: v for k, v in tuned.items() if not k.endswith("_hyps")},
        "training": json.loads(info_path.read_text()) if info_path.exists() else {},
    }
    (onnx_dir / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {onnx_dir / 'eval.json'}\nnext: python install_model.py {args.model_dir}")


if __name__ == "__main__":
    main()
