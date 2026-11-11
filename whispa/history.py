"""Dictation history (``history.jsonl``): how long it's kept, and reading it efficiently.

* One JSON line per dictation: time, seconds, raw and final text, app.
* Kept for ``Config.history_days`` (default a year; 0 = forever). Older lines are removed
  at start and once a day. Their word count and speaking time are added to
  ``history_totals.json`` first, so the totals on Home don't drop: only the text goes.
* "Paste last" reads just the end of the file, however long it is.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)

RETENTION_CHOICES = [(0, "Forever"), (365, "1 year"), (90, "90 days"), (30, "30 days"), (7, "7 days")]


def totals_path(history: Path) -> Path:
    return history.with_name("history_totals.json")


def totals(history: Path) -> dict:
    """Words and seconds of dictations already removed from the history."""
    try:
        data = json.loads(totals_path(history).read_text(encoding="utf-8"))
        return {"words": int(data.get("words", 0)), "seconds": float(data.get("seconds", 0.0))}
    except (OSError, ValueError, TypeError):
        return {"words": 0, "seconds": 0.0}


def recent_texts(history: Path, n: int = 10, block: int = 64 << 10) -> list[str]:
    """The last ``n`` dictated texts, reading only the end of the file."""
    try:
        size = history.stat().st_size
    except OSError:
        return []
    with history.open("rb") as f:
        while True:
            start = max(0, size - block)
            f.seek(start)
            lines = f.read().splitlines()
            if start > 0:
                lines = lines[1:]  # probably cut in the middle
            if len(lines) > n or start == 0:
                break
            block *= 4
    out = []
    for raw in lines[-n * 2:]:
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("text"):
            out.append(row["text"])
    return out[-n:]


def prune(history: Path, days: int, now: float | None = None) -> int:
    """Remove dictations older than ``days`` (0: keep everything). Returns how many."""
    if days <= 0 or not history.exists():
        return 0
    cutoff = (now or time.time()) - days * 86400
    keep, old_words, old_seconds, removed = [], 0, 0.0, 0
    with history.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                keep.append(line)
                continue
            if isinstance(row, dict) and row.get("t", now or time.time()) < cutoff:
                removed += 1
                if not row.get("command") and row.get("text"):
                    old_words += len(str(row["text"]).split())
                    old_seconds += float(row.get("seconds", 0) or 0)
            else:
                keep.append(line if line.endswith("\n") else line + "\n")
    if not removed:
        return 0
    t = totals(history)
    tp = totals_path(history)
    tp.write_text(json.dumps({"words": t["words"] + old_words,
                              "seconds": round(t["seconds"] + old_seconds, 1)}), encoding="utf-8")
    tmp = history.with_suffix(".tmp")
    tmp.write_text("".join(keep), encoding="utf-8")
    os.replace(tmp, history)
    log.info("History: removed %d dictations older than %d days", removed, days)
    return removed


def delete_all(history: Path) -> None:
    for p in (history, totals_path(history)):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
