"""Transcribe long dictations while you're still talking.

Every couple of seconds the app calls ``LiveSession.poll()``. Once 20+ seconds are
waiting, the audio is cut at a pause (the quietest moment of the last few seconds) and
that piece is transcribed in the background. When you let go, only the last piece is
left, so a two-minute ramble is ready about as fast as a short one. One background
thread, one piece at a time: no extra CPU or memory compared with doing it at the end.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

MIN_CHUNK_S = 20.0  # don't cut before this: short pieces lose context
MAX_CHUNK_S = 45.0  # cut at the quietest moment even if there's no clear pause
SEARCH_S = 6.0  # look for a pause in the last few seconds of the waiting audio
WINDOW_S = 0.25

_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-asr")


def find_cut(audio: np.ndarray, sr: int, force: bool = False) -> int | None:
    """Sample index in the middle of a pause near the end of ``audio``, or None."""
    win = max(1, int(WINDOW_S * sr))
    start = max(0, len(audio) - int(SEARCH_S * sr))
    tail = audio[start:]
    n = len(tail) // win
    if n < 3:
        return None
    frames = tail[: n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    # Speech level of the whole piece; a pause is far below it.
    loud = float(np.percentile(np.sqrt(np.mean(
        audio[: len(audio) // win * win].reshape(-1, win) ** 2, axis=1)), 90))
    i = int(np.argmin(rms[1:-1])) + 1  # not the very edge: the next word may be starting
    if not force and rms[i] > max(0.15 * loud, 1e-4):
        return None
    return start + i * win + win // 2


class LiveSession:
    """One recording's background transcription."""

    def __init__(self, transcribe: Callable[[np.ndarray, int], str]):
        self._transcribe = transcribe
        self.cut = 0  # samples already sent for transcription
        self.parts: list[Future] = []

    def poll(self, samples_since: Callable[[int], np.ndarray], sr: int) -> None:
        if self.parts and not self.parts[-1].done():
            return  # the last piece isn't done (a slow computer): never queue up more work
        waiting = samples_since(self.cut)
        if len(waiting) < MIN_CHUNK_S * sr:
            return
        cut = find_cut(waiting, sr, force=len(waiting) >= MAX_CHUNK_S * sr)
        if cut is None:
            return
        piece = waiting[:cut]
        self.cut += cut
        log.info("Transcribing %.1fs while you talk", len(piece) / sr)
        self.parts.append(_pool.submit(self._transcribe, piece, sr))

    def finish(self, audio: np.ndarray, sr: int) -> str:
        """The whole transcript: the pieces done while recording, then the rest."""
        texts = [f.result() for f in self.parts]
        rest = audio[self.cut:]
        if len(rest) >= 0.25 * sr or not texts:
            texts.append(self._transcribe(rest, sr))
        return " ".join(t.strip() for t in texts if t and t.strip())
