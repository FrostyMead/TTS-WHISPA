"""Personalization: learn from the user's edits, like Wispr Flow's auto-dictionary.

After Murmr pastes text it keeps watching that text field for a short while.
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


def align_to_inserted(inserted: str, region: str) -> str:
    """The part of ``region`` that corresponds to the dictated text.

    The edited region can also contain text typed before/after the dictation, or the next
    dictation pasted right behind it. Leading and trailing words that don't match the
    dictation are dropped; edits *inside* it are kept.
    """
    a = [w.lower() for w in _WORD.findall(inserted)]
    spans = list(_WORD.finditer(region))
    b = [m.group().lower() for m in spans]
    if not a or not b:
        return ""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    while ops and ops[0][0] in ("insert", "delete"):
        ops.pop(0)
    while ops and ops[-1][0] in ("insert", "delete"):
        ops.pop()
    if not ops or ops[-1][4] <= ops[0][3]:
        return ""
    start, end = spans[ops[0][3]].start(), spans[ops[-1][4] - 1].end()
    end += len(re.match(r"[.!?,;:]*", region[end:]).group())
    return region[start:end]


def apply_corrections(text: str, corrections: list["Correction"]) -> str:
    for c in corrections:
        text = re.sub(rf"(?<!\w){re.escape(c.heard)}(?!\w)", lambda _m, m=c.meant: m, text,
                      flags=re.IGNORECASE)
    return text


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
        self.snippets: dict[str, str] = {}  # spoken trigger -> text to insert
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
        self.snippets = {k.strip().lower(): v for k, v in data.get("snippets", {}).items() if k.strip()}
        self._mtime = self.path.stat().st_mtime

    def save(self) -> None:
        data = {
            "_help": "terms: words/names you use (helps spelling). corrections: heard -> meant, "
                     "applied to every dictation. Murmr adds to both when you fix its mistakes. "
                     "pending: seen once, learned after the second time. snippets: say the "
                     "trigger phrase and the text is inserted instead.",
            "terms": self.terms, "corrections": self.corrections, "snippets": self.snippets,
            "pending": self.pending,
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

    def snippets_snapshot(self) -> dict[str, str]:
        with self._lock:
            self.snapshot()  # reloads on hand edits
            return dict(self.snippets)

    def set_snippet(self, trigger: str, text: str) -> None:
        trigger = " ".join(trigger.lower().split())
        if not trigger or not text.strip():
            return
        with self._lock:
            self.snippets[trigger] = text
            self.save()

    def remove_snippet(self, trigger: str) -> None:
        with self._lock:
            if self.snippets.pop(trigger, None) is not None:
                self.save()

    def remove_correction(self, heard: str) -> None:
        with self._lock:
            if self.corrections.pop(heard.lower(), None) is not None:
                self.save()


HQ_SUFFIX = ".24k.wav"


class TrainingStore:
    """Audio + transcript pairs for fine-tuning the speech model on your voice.

    The label is what Parakeet heard (before AI polish or snippets, which the audio doesn't
    contain) plus any word fixes you made afterwards. Each sample gets a status:
    ``read`` (a reading-session sentence: the label is exactly the text you read),
    ``corrected`` (you fixed words: the most valuable dictations), ``unedited`` (you left it
    as is: probably right) or ``unknown`` (couldn't watch the field). Re-recorded reading
    clips are marked ``deleted``. When the folder reaches its size limit, the oldest
    clips that aren't ``read`` or ``corrected`` are deleted first.
    """

    def __init__(self, root: Path, max_mb: int = 2000):
        self.root = root
        self.max_bytes = max_mb * 1_000_000
        self.manifest = root / "manifest.jsonl"
        self._lock = threading.Lock()
        self._raw: dict[str, str] = {}
        self._corrected: set[str] = set()
        self._read: dict[str, tuple[str, float]] = {}  # id -> (prompt, seconds)
        self._load()

    def _load(self) -> None:
        if not self.manifest.exists():
            return
        rows = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if any("verified" in r for r in rows):
            # v1 labels could include text typed after the dictation: keep the audio,
            # reset those labels to what was transcribed.
            base = [dict(r, status="read" if r.get("status") == "read" else "unknown")
                    for r in rows if "audio" in r]
            for r in base:
                r.pop("verified", None)
            self.manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base),
                                     encoding="utf-8")
            rows = base
            log.info("Reset %d voice-sample labels from the old format", len(base))
        for r in rows:
            status, sid = r.get("status"), r.get("id")
            if status == "corrected":
                self._corrected.add(sid)
            elif status == "read" and "audio" in r:
                self._read[sid] = (r.get("text", ""), float(r.get("duration", 0)))
            elif status == "deleted":
                self._corrected.discard(sid)
                self._read.pop(sid, None)

    def _clips(self) -> list[Path]:
        """The 16 kHz clips (one per recording; high-quality copies not counted)."""
        if not self.root.exists():
            return []
        return [p for p in self.root.glob("*.wav") if not p.name.endswith(HQ_SUFFIX)]

    def stats(self) -> dict:
        wavs = self._clips()
        size = sum(p.stat().st_size for p in wavs)
        return {"clips": len(wavs), "bytes": size, "seconds": size / 32000,
                "corrected": len(self._corrected), "read": len(self._read),
                "read_seconds": self.read_seconds()}

    def read_seconds(self) -> float:
        """Reading-session speech still on disk, in seconds."""
        return sum(sec for sid, (_, sec) in self._read.items() if (self.root / f"{sid}.wav").exists())

    def read_prompts(self) -> set[str]:
        """Sentences already read (and still on disk)."""
        return {text for sid, (text, _) in self._read.items() if (self.root / f"{sid}.wav").exists()}

    def delete_all(self) -> None:
        with self._lock:
            for p in self.root.glob("*.wav"):
                p.unlink(missing_ok=True)
            self.manifest.unlink(missing_ok=True)
            self._raw.clear()
            self._corrected.clear()
            self._read.clear()

    def _new_id(self) -> str:
        sample_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        while (self.root / f"{sample_id}.wav").exists():  # two saves in the same millisecond
            sample_id = sample_id[:-3] + f"{(int(sample_id[-3:]) + 1) % 1000:03d}"
        return sample_id

    @staticmethod
    def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())

    def _write(self, audio: np.ndarray, sr: int, row: dict, keep_hq: bool = False) -> str:
        """Save a clip. The main file is 16 kHz (what the speech model trains on); with
        ``keep_hq`` and a better source, a 24 kHz copy is kept too (for a voice like yours)."""
        from .audio import TARGET_SR, VOICE_SR, quality, resample

        self.root.mkdir(parents=True, exist_ok=True)
        self._enforce_cap()
        sample_id = self._new_id()
        self._save_wav(self.root / f"{sample_id}.wav", resample(audio, sr, TARGET_SR), TARGET_SR)
        extra = {}
        if keep_hq and sr >= VOICE_SR:
            self._save_wav(self.root / f"{sample_id}{HQ_SUFFIX}", resample(audio, sr, VOICE_SR), VOICE_SR)
            extra = {"hq": f"{sample_id}{HQ_SUFFIX}", "source_sr": sr, **quality(audio, sr)}
        self._append({"id": sample_id, "audio": f"{sample_id}.wav",
                      "duration": round(len(audio) / sr, 2), **row, **extra})
        return sample_id

    def add(self, audio: np.ndarray, sr: int, asr_text: str) -> str | None:
        try:
            sample_id = self._write(audio, sr, {"asr": asr_text, "text": asr_text, "status": "unknown"})
        except OSError as exc:
            log.warning("Could not save training sample: %s", exc)
            return None
        self._raw[sample_id] = asr_text
        return sample_id

    def add_read(self, audio: np.ndarray, sr: int, prompt: str, asr_text: str) -> str | None:
        """A reading-session clip: the label is the sentence that was read, word for word."""
        try:
            sample_id = self._write(audio, sr, {"asr": asr_text, "text": prompt, "status": "read"},
                                    keep_hq=True)
        except OSError as exc:
            log.warning("Could not save reading sample: %s", exc)
            return None
        self._read[sample_id] = (prompt, round(len(audio) / sr, 2))
        return sample_id

    def remove(self, sample_id: str) -> None:
        """Drop one clip (e.g. you re-recorded that sentence)."""
        (self.root / f"{sample_id}.wav").unlink(missing_ok=True)
        (self.root / f"{sample_id}{HQ_SUFFIX}").unlink(missing_ok=True)
        self._read.pop(sample_id, None)
        self._corrected.discard(sample_id)
        self._raw.pop(sample_id, None)
        self._append({"id": sample_id, "status": "deleted"})

    def record_review(self, sample_id: str, corrections: list[Correction]) -> None:
        """You had a chance to fix the text: apply your word fixes to the transcript label."""
        raw = self._raw.get(sample_id)
        if raw is None:
            return
        label = apply_corrections(raw, corrections)
        if label != raw:
            self._corrected.add(sample_id)
            self._append({"id": sample_id, "text": label, "status": "corrected"})
        else:
            self._append({"id": sample_id, "status": "unedited"})

    def _enforce_cap(self) -> None:
        """Over the limit: delete the oldest clips, read and corrected ones last."""
        wavs = sorted(self._clips())  # ids are timestamps: oldest first
        total = sum(p.stat().st_size for p in self.root.glob("*.wav"))
        if total <= self.max_bytes:
            return
        target = self.max_bytes * 0.9
        keep = self._corrected | set(self._read)
        order = [p for p in wavs if p.stem not in keep] + [p for p in wavs if p.stem in keep]
        for p in order:
            if total <= target:
                break
            total -= p.stat().st_size
            p.unlink(missing_ok=True)
            hq = p.with_name(p.stem + HQ_SUFFIX)
            if hq.exists():
                total -= hq.stat().st_size
                hq.unlink(missing_ok=True)

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
                 on_final: Callable[[str, list[Correction]], None] | None = None):
        self.dictionary = dictionary
        self.on_learned = on_learned
        self.on_final = on_final  # (sample_id, corrections the user made; [] = left as is)
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
        inserted = s["inserted"]
        # Only the words that were dictated: drop anything typed or dictated around them.
        final = align_to_inserted(inserted, s["last"])
        corrections = find_corrections(inserted, final) if final else []
        if s["sample"] and self.on_final and final:
            self.on_final(s["sample"], corrections)
        learned = [c for c in corrections if self.dictionary.learn(c)]
        if learned:
            log.info("Learned: %s", ", ".join(f"{c.heard!r}->{c.meant!r}" for c in learned))
            self.on_learned(learned)
