"""Personalization: learn from the user's edits, like Wispr Flow's auto-dictionary.

After Whispa pastes text it keeps watching that text field for a short while.
If you fix a misheard word ("Sara" -> "Sarah", "cooper netties" -> "Kubernetes"),
the fix is learned and applied to every future dictation. Your corrected text
plus the audio are also saved as training pairs for fine-tuning Parakeet on
your voice later (see README: Personalization).
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .focus import Focus

log = logging.getLogger(__name__)

_WORD = re.compile(r"[\w][\w'’\-.]*[\w]|[\w]", re.UNICODE)

# Fixing these is almost always a change of mind, not a misheard word.
_NEVER_LEARN = set("""
monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday tonight
january february march april may june july august september october november december
yes no not never always maybe a an the this that these those it its i you he she we they me him
her us them my your his our their and or but if so then than to of in on at by for with from
is are was were be been am do does did have has had will would can could should might must
one two three four five six seven eight nine ten first second third last next all some any
more most less few many much very really just also only even still already again here there
now good great bad ok okay please thanks thank hi hello hey
""".split())


def _letters(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower())


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _letters(a), _letters(b)).ratio()


def _common_only(phrase: str) -> bool:
    return all(w.lower() in _NEVER_LEARN for w in phrase.split())


def _is_distinctive(phrase: str) -> bool:
    """Names and jargon (capitals, inner caps, digits) that aren't everyday words."""
    for w in phrase.split():
        if w.lower() in _NEVER_LEARN:
            continue
        if w[:1].isupper() or any(c.isdigit() for c in w) or any(c.isupper() for c in w[1:]):
            return True
    return False


def needs_confirmation(c: "Correction") -> bool:
    """One-off fixes of names/jargon are trusted; anything else must happen twice."""
    if c.heard.lower() == c.meant.lower():
        # Capitalising only the first letter is often just a sentence start.
        return not any(ch.isupper() for w in c.meant.split() for ch in w[1:])
    return not _is_distinctive(c.meant)


def edited_region(before: str, after: str) -> str:
    """The part of ``after`` that differs from ``before`` (what the dictation became)."""
    n = min(len(before), len(after))
    p = 0
    while p < n and before[p] == after[p]:
        p += 1
    s = 0
    while s < n - p and before[-1 - s] == after[-1 - s]:
        s += 1
    return after[p:len(after) - s]


@dataclass
class Correction:
    heard: str
    meant: str


def find_corrections(inserted: str, final: str) -> list[Correction]:
    """Word-level replacements that look like fixes of misrecognized words."""
    a = _WORD.findall(inserted)
    b = _WORD.findall(final)
    if not a or not b:
        return []
    overall = difflib.SequenceMatcher(None, [w.lower() for w in a], [w.lower() for w in b]).ratio()
    if overall < 0.5:
        return []  # rewritten, not corrected
    out = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or i2 - i1 > 3 or j2 - j1 > 3:
            continue
        heard, meant = " ".join(a[i1:i2]), " ".join(b[j1:j2])
        if heard == meant or not _letters(meant):
            continue
        if _common_only(meant) and _common_only(heard):
            continue  # "their" -> "there", "Tuesday" -> "Wednesday": edits, not mishearings
        case_only = heard.lower() == meant.lower()
        if not case_only and _letters(heard) == _letters(meant):
            continue  # punctuation / apostrophe style only
        if case_only or similarity(heard, meant) >= 0.55:
            out.append(Correction(heard, meant))
    return out


class Dictionary:
    """``dictionary.json``: terms you use, plus heard->meant corrections."""

    def __init__(self, path: Path):
        self.path = path
        self.terms: list[str] = []
        self.corrections: dict[str, str] = {}
        self.pending: dict[str, dict] = {}  # seen once, needs confirmation
        self._mtime = 0.0
        self._lock = threading.RLock()
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.save()
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.error("Could not read %s: %s", self.path, exc)
            return
        self.terms = list(data.get("terms", []))
        self.corrections = {k.lower(): v for k, v in data.get("corrections", {}).items()}
        self.pending = dict(data.get("pending", {}))
        self._mtime = self.path.stat().st_mtime

    def save(self) -> None:
        data = {
            "_help": "terms: words/names you use (helps spelling). corrections: heard -> meant, "
                     "applied to every dictation. Whispa adds to both when you fix its mistakes. "
                     "pending: seen once, learned after the second time.",
            "terms": self.terms, "corrections": self.corrections, "pending": self.pending,
        }
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._mtime = self.path.stat().st_mtime

    def snapshot(self) -> tuple[list[str], dict[str, str]]:
        with self._lock:
            try:  # pick up hand edits to dictionary.json
                mtime = self.path.stat().st_mtime
                if mtime != self._mtime:
                    self.load()
            except OSError:
                pass
            return list(self.terms), dict(self.corrections)

    def learn(self, c: Correction) -> bool:
        """Record a correction. Returns True when it became active."""
        key = c.heard.lower()
        with self._lock:
            if self.corrections.get(key) == c.meant:
                return False
            entry = self.pending.get(key)
            confirmed = entry is not None and entry.get("meant") == c.meant
            if not confirmed and needs_confirmation(c):
                self.pending[key] = {"meant": c.meant, "t": time.time()}
                self.save()
                return False
            self.pending.pop(key, None)
            self.corrections[key] = c.meant
            if _is_distinctive(c.meant) and c.meant not in self.terms:
                self.terms.append(c.meant)
            self.save()
            return True

    # -- edits from the settings window ---------------------------------
    def add_term(self, term: str) -> None:
        term = term.strip()
        with self._lock:
            if term and term not in self.terms:
                self.terms.append(term)
                self.save()

    def remove_term(self, term: str) -> None:
        with self._lock:
            if term in self.terms:
                self.terms.remove(term)
                self.save()

    def set_correction(self, heard: str, meant: str) -> None:
        heard, meant = heard.strip().lower(), meant.strip()
        if not heard or not meant:
            return
        with self._lock:
            self.corrections[heard] = meant
            self.pending.pop(heard, None)
            self.save()

    def remove_correction(self, heard: str) -> None:
        with self._lock:
            if self.corrections.pop(heard.lower(), None) is not None:
                self.save()


class TrainingStore:
    """Audio + final text pairs for fine-tuning on your voice."""

    def __init__(self, root: Path, max_mb: int = 2000):
        self.root = root
        self.max_bytes = max_mb * 1_000_000
        self.manifest = root / "manifest.jsonl"
        self._lock = threading.Lock()

    def add(self, audio: np.ndarray, sr: int, asr_text: str) -> str | None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self._size() > self.max_bytes:
                return None
            sample_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
            pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            with wave.open(str(self.root / f"{sample_id}.wav"), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            self._append({"id": sample_id, "audio": f"{sample_id}.wav", "duration": round(len(audio) / sr, 2),
                          "asr": asr_text, "text": asr_text, "verified": False})
            return sample_id
        except OSError as exc:
            log.warning("Could not save training sample: %s", exc)
            return None

    def update_text(self, sample_id: str, text: str) -> None:
        """Record what the user ended up with after editing (the better label)."""
        self._append({"id": sample_id, "text": text, "verified": True})

    def _append(self, row: dict) -> None:
        with self._lock, self.manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _size(self) -> int:
        return sum(p.stat().st_size for p in self.root.glob("*.wav"))


class CorrectionWatcher:
    """Polls the field we pasted into and diffs what the user changed."""

    POLL_S = 0.3
    WATCH_S = 45.0

    def __init__(self, dictionary: Dictionary, on_learned: Callable[[list[Correction]], None],
                 on_final: Callable[[str, str], None] | None = None):
        self.dictionary = dictionary
        self.on_learned = on_learned
        self.on_final = on_final  # (sample_id, final_text)
        self._session: dict | None = None
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def watch(self, focus: Focus, before: str, inserted: str, sample_id: str | None) -> None:
        self.finish()  # a new dictation ends the previous watch
        with self._lock:
            self._session = {"focus": focus, "before": before, "inserted": inserted.strip(),
                             "last": None, "sample": sample_id,
                             "deadline": time.monotonic() + self.WATCH_S}
        self._schedule()

    def _schedule(self) -> None:
        self._timer = threading.Timer(self.POLL_S, self._poll)
        self._timer.daemon = True
        self._timer.start()

    def _poll(self) -> None:
        with self._lock:
            s = self._session
        if s is None:
            return
        text = s["focus"].read_text()
        if text is not None:
            region = edited_region(s["before"], text).strip()
            if region and similarity(region, s["inserted"]) >= 0.4:
                s["last"] = region
            elif not text.strip() and s["last"] is not None:
                self.finish()  # field cleared: message was sent
                return
        if text is None or time.monotonic() > s["deadline"]:
            self.finish()
            return
        self._schedule()

    def finish(self) -> None:
        with self._lock:
            s, self._session = self._session, None
        if self._timer:
            self._timer.cancel()
        if not s or not s["last"]:
            return
        final, inserted = s["last"], s["inserted"]
        if final == inserted:
            return
        if s["sample"] and self.on_final:
            self.on_final(s["sample"], final)
        learned = [c for c in find_corrections(inserted, final) if self.dictionary.learn(c)]
        if learned:
            log.info("Learned: %s", ", ".join(f"{c.heard!r}->{c.meant!r}" for c in learned))
            self.on_learned(learned)
