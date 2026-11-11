"""Dictation history (``history.jsonl``): how long it's kept, and reading it efficiently.

* One JSON line per dictation: time, seconds, raw and final text, app.
* Kept for ``Config.history_days`` (default a year; 0 = forever). Older lines are removed
  at start and once a day. Their word count and speaking time are added to
  ``history_totals.json``, so the totals on Home don't drop: only the text goes.
* Every change (adding, pruning, deleting) holds one lock, so a dictation that arrives
  while old ones are being removed is never lost. Rewrites stream line by line.
* Home reads only the end of the file for the list, and caches its totals until the file
  changes; "paste last" reads only the end too.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

RETENTION_CHOICES = [(0, "Forever"), (365, "1 year"), (90, "90 days"), (30, "30 days"), (7, "7 days")]
LOCK = threading.RLock()  # one Murmr per user (single instance): an in-process lock is enough


def totals_path(history: Path) -> Path:
    return history.with_name("history_totals.json")


def totals(history: Path) -> dict:
    """Words and seconds of dictations already removed from the history."""
    try:
        data = json.loads(totals_path(history).read_text(encoding="utf-8"))
        return {"words": int(data.get("words", 0)), "seconds": float(data.get("seconds", 0.0))}
    except (OSError, ValueError, TypeError):
        return {"words": 0, "seconds": 0.0}


def append(history: Path, row: dict) -> None:
    with LOCK, history.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rows(lines) -> list[dict]:
    out = []
    for raw in lines:
        try:
            row = json.loads(raw)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("text"):
            out.append(row)
    return out


def recent_rows(history: Path, n: int, block: int = 64 << 10) -> list[dict]:
    """The last ``n`` dictations, oldest first, reading only the end of the file."""
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
    return _rows(lines[-n * 2:])[-n:]


def recent_texts(history: Path, n: int = 10, block: int = 64 << 10) -> list[str]:
    return [r["text"] for r in recent_rows(history, n, block)]


def scan(history: Path, keep: Callable[[dict], bool]) -> list[dict]:
    """Every dictation ``keep`` accepts, newest first (search), streaming the file."""
    out = []
    try:
        with history.open(encoding="utf-8") as f:
            out = [r for r in _rows(f) if keep(r)]
    except OSError:
        return []
    out.reverse()
    return out


_stats_cache: dict = {}


def stats(history: Path) -> dict:
    """Totals for Home (words, seconds, days dictated, words in the last 7 days), cached
    until the file changes."""
    try:
        st = history.stat()
        key = (str(history), st.st_size, st.st_mtime_ns)
    except OSError:
        key = (str(history), 0, 0)
    week_ago = time.time() - 7 * 86400
    if _stats_cache.get("key") != key or _stats_cache.get("week_from", 0) < week_ago - 3600:
        words = seconds = week = 0
        days: set[dt.date] = set()
        try:
            with history.open(encoding="utf-8") as f:
                for r in _rows(f):
                    days.add(dt.date.fromtimestamp(r.get("t", 0)))
                    if r.get("command"):
                        continue
                    n = len(str(r["text"]).split())
                    words += n
                    seconds += float(r.get("seconds", 0) or 0)
                    if r.get("t", 0) >= week_ago:
                        week += n
        except OSError:
            pass
        _stats_cache.update(key=key, week_from=week_ago,
                            value={"words": words, "seconds": seconds, "days": days, "week": week})
    value = dict(_stats_cache["value"])
    archived = totals(history)
    value["words"] += archived["words"]
    value["seconds"] += archived["seconds"]
    return value


def _rewrite(history: Path, keep: Callable[[dict | None, str], bool]) -> list[dict]:
    """Stream the file into a new one with the lines ``keep`` accepts. Returns the rows
    dropped. Call with LOCK held."""
    dropped = []
    tmp = history.with_suffix(".tmp")
    with history.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                row = None
            if keep(row if isinstance(row, dict) else None, line):
                dst.write(line if line.endswith("\n") else line + "\n")
            else:
                dropped.append(row)
    if dropped:
        os.replace(tmp, history)
    else:
        tmp.unlink()
    return dropped


def prune(history: Path, days: int, now: float | None = None) -> int:
    """Remove dictations older than ``days`` (0: keep everything). Returns how many."""
    if days <= 0:
        return 0
    cutoff = (now or time.time()) - days * 86400
    with LOCK:
        if not history.exists():
            return 0
        old = _rewrite(history, lambda row, line: row is None or row.get("t", cutoff) >= cutoff)
        if not old:
            return 0
        # The history is already rewritten: an interruption here undercounts, never doubles.
        t = totals(history)
        for row in old:
            if not row.get("command") and row.get("text"):
                t["words"] += len(str(row["text"]).split())
                t["seconds"] += float(row.get("seconds", 0) or 0)
        totals_path(history).write_text(json.dumps({"words": t["words"],
                                                    "seconds": round(t["seconds"], 1)}),
                                        encoding="utf-8")
    log.info("History: removed %d dictations older than %d days", len(old), days)
    return len(old)


def delete_entry(history: Path, t: float) -> None:
    with LOCK:
        if history.exists():
            _rewrite(history, lambda row, line: row is None or row.get("t") != t)


def delete_all(history: Path) -> None:
    with LOCK:
        for p in (history, totals_path(history)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
