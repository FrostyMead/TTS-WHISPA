"""Who's talking: voiceprints for meeting lines, grouped into Person 1, Person 2, ...

* Voiceprints come from WeSpeaker's ResNet34 model (VoxCeleb, Apache-2.0, ~26 MB ONNX,
  downloaded on first use), run on the CPU: a few tens of milliseconds per line.
* Features are Kaldi-style 80-band log-mel filterbanks, computed here in numpy exactly as
  WeSpeaker computes them for training (25 ms Hamming frames every 10 ms, no dither), then
  mean-normalized.
* Each finished line's voiceprint is compared with the people heard so far; close enough
  joins that person, otherwise it's someone new. Very short lines don't carry enough voice
  and go to whoever spoke last on that channel.

Privacy: voiceprints only live in memory during a meeting and are never saved; only the
names you give people end up in the notes.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

REPO = "Wespeaker/wespeaker-voxceleb-resnet34"
MODEL_FILE = "voxceleb_resnet34.onnx"
SR = 16000
MIN_SPEECH_S = 1.2  # shorter lines: not enough voice for a reliable voiceprint
SAME_PERSON = 0.45  # cosine similarity: same voice ~0.65-0.85 (lower over calls), others < 0.3


def _mel_banks(n_mels: int = 80, n_fft: int = 512, sr: int = SR, low: float = 20.0) -> np.ndarray:
    """Kaldi's triangular mel filters (mel = 1127 ln(1 + f/700)), over FFT bins 0..n_fft/2-1."""
    def mel(f):
        return 1127.0 * np.log(1.0 + f / 700.0)

    high = sr / 2
    mlo, mhi = mel(low), mel(high)
    delta = (mhi - mlo) / (n_mels + 1)
    bins = mel(np.arange(n_fft // 2) * sr / n_fft)
    banks = np.zeros((n_mels, n_fft // 2))
    for m in range(n_mels):
        left, center, right = mlo + m * delta, mlo + (m + 1) * delta, mlo + (m + 2) * delta
        up = (bins - left) / (center - left)
        down = (right - bins) / (right - center)
        banks[m] = np.maximum(0.0, np.minimum(up, down))
    return banks


_BANKS = None


def fbank(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """80-dim log-mel filterbank features [frames, 80], mean-normalized (CMN)."""
    global _BANKS
    if sr != SR:
        from .audio import resample

        audio = resample(audio, sr, SR)
    x = np.asarray(audio, np.float64) * 32768.0
    flen, shift, n_fft = 400, 160, 512
    if len(x) < flen:
        return np.zeros((0, 80), np.float32)
    n = 1 + (len(x) - flen) // shift
    idx = np.arange(flen)[None, :] + shift * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)  # remove DC offset
    frames[:, 1:] -= 0.97 * frames[:, :-1]  # pre-emphasis (Kaldi: the first sample too)
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= np.hamming(flen)
    power = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    if _BANKS is None:
        _BANKS = _mel_banks()
    mel = power[:, : n_fft // 2] @ _BANKS.T
    feats = np.log(np.maximum(mel, np.finfo(np.float32).eps))
    return (feats - feats.mean(axis=0, keepdims=True)).astype(np.float32)


class Embedder:
    """Voiceprints (256-dim, unit length). Loads the model on first use."""

    def __init__(self):
        self._session = None
        self._lock = threading.Lock()
        self.failed = False

    def _load(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.enable_cpu_mem_arena = False
        path = hf_hub_download(REPO, MODEL_FILE)
        return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray | None:
        if self.failed:
            return None
        if len(audio) < MIN_SPEECH_S * sr:
            return None
        try:
            with self._lock:
                if self._session is None:
                    self._session = self._load()
                feats = fbank(audio[: int(20 * sr)], sr)  # 20 s is plenty for a voiceprint
                if len(feats) < 50:
                    return None
                emb = self._session.run(None, {"feats": feats[None]})[0][0]
        except Exception as exc:  # noqa: BLE001 - offline on first use, etc.: no speaker labels
            log.warning("Speaker voiceprints unavailable: %s", exc)
            self.failed = True
            return None
        norm = float(np.linalg.norm(emb))
        return emb / norm if norm else None


class Tracker:
    """Groups voiceprints into people as they come in (one tracker per audio channel)."""

    def __init__(self, prefix: str = "Person", known: dict[str, np.ndarray] | None = None,
                 threshold: float = SAME_PERSON, start_at: int = 1, counter: list[int] | None = None):
        self.prefix, self.threshold = prefix, threshold
        self.known = dict(known or {})  # e.g. {"You": your voiceprint}
        self.people: list[tuple[str, np.ndarray, int]] = []  # (label, sum of voiceprints, count)
        self.last: str | None = None
        # Shared by the trackers of one meeting (room mic + call audio): one Person 1.
        self._counter = counter if counter is not None else [start_at]

    def rename(self, old: str, new: str) -> None:
        """You named a voice: new lines from it use the name. Two labels given the same name
        become one person (their voiceprints are combined)."""
        merged: dict[str, tuple[np.ndarray, int]] = {}
        order: list[str] = []
        for label, total, n in self.people:
            label = new if label == old else label
            if label in merged:
                t, c = merged[label]
                merged[label] = (t + total, c + n)
            else:
                merged[label] = (total, n)
                order.append(label)
        self.people = [(label, *merged[label]) for label in order]
        if old in self.known:
            self.known[new] = self.known.pop(old)
        if self.last == old:
            self.last = new

    def assign(self, emb: np.ndarray | None, fallback: str) -> str:
        if emb is None:
            return self.last or fallback
        for label, ref in self.known.items():
            if float(emb @ ref) >= self.threshold + 0.05:
                self.last = label
                return label
        best, best_i = -1.0, -1
        for i, (_, total, _) in enumerate(self.people):
            sim = float(emb @ (total / np.linalg.norm(total)))
            if sim > best:
                best, best_i = sim, i
        if best >= self.threshold:
            label, total, n = self.people[best_i]
            self.people[best_i] = (label, total + emb, n + 1)
        else:
            label = f"{self.prefix} {self._counter[0]}"
            self._counter[0] += 1
            self.people.append((label, emb.copy(), 1))
        self.last = label
        return label


def voice_profile(embedder: Embedder, training_dir, max_clips: int = 12) -> np.ndarray | None:
    """Your own voiceprint, from recordings Murmr already has of you (Voice training reads
    first). Kept in memory only. None if there aren't any."""
    import wave
    from pathlib import Path

    root = Path(training_dir)
    if not root.exists():
        return None
    clips = sorted((p for p in root.glob("*.wav") if ".24k." not in p.name),
                   key=lambda p: p.stat().st_size, reverse=True)[:max_clips]
    embs = []
    for p in clips:
        try:
            with wave.open(str(p)) as w:
                sr = w.getframerate()
                audio = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        except (OSError, wave.Error):
            continue
        e = embedder(audio, sr)
        if e is not None:
            embs.append(e)
    if len(embs) < 2:
        return None
    m = np.mean(embs, axis=0)
    return m / np.linalg.norm(m)
