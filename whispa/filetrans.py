"""Transcribe an audio or video file (Meetings > Transcribe a file, or drop one on the page).

* Decoded with PyAV (its own FFmpeg, nothing else to install): mp3, m4a, wav, flac, ogg,
  opus, mp4, mov, mkv, webm... Read as a stream, so a long recording never sits in memory.
* Split at pauses like live meeting notes (meetings.find_pause), each piece transcribed with
  the speech model on this computer. Optionally voices are told apart: Person 1, Person 2...
* Saved as a meeting (title: the file's name), so it can be renamed, summarized, copied and
  exported like one. Nothing leaves the computer and the audio isn't kept.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .meetings import MAX_LINE_S, PAUSE_S, SILENT_RMS, Line, _frames, is_silent

log = logging.getLogger(__name__)

SR = 16000
EXTENSIONS = (".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff",
              ".aif", ".caf", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".3gp")


MIN_PIECE_S = 1.0  # a turn can be short ("Yes.")


class Cancelled(Exception):
    pass


def first_pause(audio: np.ndarray, sr: int, min_s: float = MIN_PIECE_S) -> int | None:
    """Sample index in the middle of the first pause after ``min_s`` of audio. A file has
    every turn already: cutting at each pause keeps one speaker per piece."""
    rms = _frames(audio, sr)
    if len(rms) < 4:
        return None
    loud = float(np.percentile(rms, 90))
    quiet = rms < max(SILENT_RMS, 0.12 * loud)
    need = max(1, int(PAUSE_S / 0.05))
    run = 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        start = i - run + 1
        if run >= need and start * 0.05 >= min_s:
            return int((start + run // 2) * 0.05 * sr)
    return None


def duration_s(path: Path) -> float:
    try:
        import av

        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / 1_000_000
            stream = next(s for s in c.streams if s.type == "audio")
            return float(stream.duration * stream.time_base) if stream.duration else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def decode(path: Path, block_s: float = 10.0) -> Iterator[np.ndarray]:
    """The file's audio as float32 mono 16 kHz, a block at a time."""
    try:
        import av
    except ImportError:
        yield from _decode_wav(path, block_s)
        return
    with av.open(str(path)) as container:
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            raise ValueError("This file has no sound")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        pending: list[np.ndarray] = []
        have = 0
        for frame in container.decode(streams[0]):
            for out in resampler.resample(frame):
                a = out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                pending.append(a)
                have += len(a)
            if have >= block_s * SR:
                yield np.concatenate(pending)
                pending, have = [], 0
        for out in resampler.resample(None):  # flush
            pending.append(out.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        if pending:
            yield np.concatenate(pending)


def _decode_wav(path: Path, block_s: float) -> Iterator[np.ndarray]:
    import wave

    from .audio import resample

    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Only 16-bit WAV files can be read without the media decoder")
        while True:
            raw = w.readframes(int(block_s * sr))
            if not raw:
                return
            a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
            if ch > 1:
                a = a.reshape(-1, ch).mean(axis=1)
            yield resample(a, sr, SR)


def transcribe_file(path: Path, transcribe: Callable[[np.ndarray, int], str],
                    progress: Callable[[float, str], None] = lambda f, s: None,
                    cancel: threading.Event | None = None,
                    embedder=None) -> list[Line]:
    """Lines of the file's transcript, with times and (if ``embedder``) speakers."""
    from .speakers import Tracker

    total = duration_s(path)
    tracker = Tracker("Person") if embedder is not None else None
    lines: list[Line] = []
    buf = np.zeros(0, np.float32)
    pos = 0  # samples of the file before ``buf``
    max_n = int(MAX_LINE_S * SR)

    def emit(piece: np.ndarray, start: int) -> None:
        if len(piece) < 0.3 * SR or is_silent(piece, SR):
            return
        text = transcribe(piece, SR).strip()
        if not text:
            return
        who = "Speaker"
        if tracker is not None:
            try:
                who = tracker.assign(embedder(piece, SR), "Speaker")
            except Exception:  # noqa: BLE001
                pass
        t, end = start / SR, (start + len(piece)) / SR
        if lines and lines[-1].who == who and t - lines[-1].end < 1.0 and \
                not lines[-1].text.rstrip().endswith((".", "?", "!")):
            lines[-1].text += " " + text  # the same sentence, cut by a short pause
            lines[-1].end = end
        else:
            lines.append(Line(t, who, text, end=end, channel="File"))

    for block in decode(path):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        buf = np.concatenate([buf, block]) if len(buf) else block
        while len(buf) >= max_n:
            cut = first_pause(buf[:max_n], SR) or max_n
            emit(buf[:cut], pos)
            buf, pos = buf[cut:], pos + cut
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            done = pos / SR
            progress(min(0.99, done / total) if total else -1.0,
                     f"{int(done // 60)}:{int(done % 60):02d} of {int(total // 60)}:{int(total % 60):02d}"
                     if total else f"{int(done // 60)} min")
    while len(buf):  # the end of the file: still one turn per piece
        cut = first_pause(buf[:max_n], SR) or min(len(buf), max_n)
        emit(buf[:cut], pos)
        buf, pos = buf[cut:], pos + cut
    progress(1.0, "Done")
    return lines


def meeting_from_file(path: Path, lines: list[Line], took_s: float) -> dict:
    return {"started": path.stat().st_mtime if path.exists() else time.time(),
            "duration": round(lines[-1].end if lines else 0.0, 1),
            "sources": ["File"], "lines": [line.__dict__ for line in lines],
            "title": path.stem[:80], "summary": "", "file": path.name,
            "transcribed_in": round(took_s, 1)}
