"""Shared helpers for the fine-tuning scripts (standard library only)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

WORK = Path(os.environ.get("WHISPA_FT_WORK", Path.home() / "whispa-finetune" / "work"))
BASE_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


def whispa_dir() -> Path:
    """Whispa's data folder (%APPDATA%\\Whispa), also when running inside WSL."""
    if os.environ.get("WHISPA_DATA"):
        return Path(os.environ["WHISPA_DATA"])
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Whispa"
    if "microsoft" in os.uname().release.lower():  # WSL: ask Windows where %APPDATA% is
        win = subprocess.run(["cmd.exe", "/c", "echo %APPDATA%"], capture_output=True, text=True,
                             cwd="/mnt/c").stdout.strip()
        return Path(subprocess.run(["wslpath", win], capture_output=True, text=True).stdout.strip()) / "Whispa"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Whispa"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "whispa"


def read_whispa_manifest(path: Path) -> list[dict]:
    """Fold Whispa's append-only manifest into one record per clip (later rows update)."""
    clips: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = row.get("id")
        if not sid:
            continue
        if "audio" in row:
            clips[sid] = dict(row)
        elif sid in clips:
            clips[sid].update({k: v for k, v in row.items() if k != "id"})
    return [c for c in clips.values() if c.get("status") != "deleted"]


def split_of(sample_id: str, test_pct: int = 15, val_pct: int = 10) -> str:
    """Stable train/val/test assignment from the clip id alone.

    A clip lands in the same split forever, so the held-out test clips are never trained on,
    however many times you add data and re-run.
    """
    h = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16) % 100
    if h < test_pct:
        return "test"
    if h < test_pct + val_pct:
        return "val"
    return "train"


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# --- WER ----------------------------------------------------------------------------------
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}


def normalize(text: str) -> list[str]:
    """Open-ASR-leaderboard-style normalization: case, punctuation and number style ignored."""
    t = text.lower().replace("’", "'")
    t = re.sub(r"\b([ap])\.\s?m\.", r"\1m", t)
    t = re.sub(r"(?<=\d)[,:.](?=\d)", "", t)
    t = t.replace("%", " percent ")
    t = re.sub(r"[^\w\s]", " ", t.replace("'", ""))
    return [_NUMBER_WORDS.get(w, w) for w in t.split()]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def corpus_wer(refs: list[str], hyps: list[str], norm=normalize) -> float:
    """Total word errors / total reference words (not an average of per-clip WERs)."""
    errors = words = 0
    for r, h in zip(refs, hyps):
        rw, hw = norm(r), norm(h)
        errors += edit_distance(rw, hw)
        words += len(rw)
    return errors / max(1, words)


def pnc_wer(refs: list[str], hyps: list[str]) -> float:
    """WER that also counts capitalization and punctuation (punctuation as separate tokens)."""
    def tok(t: str) -> list[str]:
        return re.findall(r"[\w'’]+|[.,?!:;]", t)
    return corpus_wer(refs, hyps, norm=tok)
