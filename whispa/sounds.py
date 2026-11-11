"""Tiny synthesized UI chimes (no audio files to ship)."""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)
SR = 44100


def _note(freq: float, dur: float) -> np.ndarray:
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1.0, t / 0.006) * np.exp(-t * 18)
    wave = np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(4 * np.pi * freq * t)
    return (wave * env).astype(np.float32)


def _seq(*freqs: float, step: float = 0.07, dur: float = 0.16) -> np.ndarray:
    out = np.zeros(int(SR * (step * (len(freqs) - 1) + dur)), np.float32)
    for i, f in enumerate(freqs):
        n = _note(f, dur)
        s = int(i * step * SR)
        out[s:s + len(n)] += n
    return out / max(1e-6, float(np.abs(out).max()))


class Sounds:
    def __init__(self, enabled: bool = True, volume: float = 0.12):
        self.enabled = enabled
        self.volume = volume
        self._clips = {
            "start": _seq(784, 1175),
            "stop": _seq(1175, 784),
            "cancel": _seq(523, 392),
            "error": _seq(330, 247, step=0.11),
        }

    def play(self, name: str) -> None:
        if not self.enabled or name not in self._clips:
            return
        clip = self._clips[name] * self.volume
        threading.Thread(target=self._play, args=(clip,), daemon=True).start()

    @staticmethod
    def _play(clip: np.ndarray) -> None:
        try:
            from .audio import output_stream

            with output_stream(samplerate=SR, channels=1, dtype="float32") as out:
                out.write(clip.reshape(-1, 1))
        except Exception as exc:  # noqa: BLE001
            log.debug("sound failed: %s", exc)
