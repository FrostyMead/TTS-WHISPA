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
        # Don't keep threads spinning on the CPU after each run, waiting for the next one: an
        # app that's mostly idle would burn battery for a few milliseconds of latency.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
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


REGROUP_SAME = 0.5  # average similarity for two groups of lines to be one person (after the meeting)
SHORT_LINE_S = 3.0  # shorter lines give weak voiceprints: they join the nearest group instead
JOIN_SHORT = 0.25  # ...if they're at least this close to it,
JOIN_MARGIN = 0.05  # ...and clearly closer to it than to anyone else


def regroup_lines(embs: list[np.ndarray], seconds: list[float]) -> tuple[list[int], dict]:
    """Groups for every line: the longer lines decide who's who (``regroup``); each short
    line joins the group it's clearly closest to, else is someone new. Also returns the
    numbers behind it (for the log: no text or audio)."""
    unit = [e / (np.linalg.norm(e) or 1.0) for e in embs]
    strong = [i for i, s in enumerate(seconds) if s >= SHORT_LINE_S]
    if len(strong) < 2:
        strong = list(range(len(unit)))
    groups = dict(zip(strong, regroup([unit[i] for i in strong])))
    centroids = {}
    for i, g in groups.items():
        centroids.setdefault(g, []).append(unit[i])
    cent = {g: np.mean(v, axis=0) / (np.linalg.norm(np.mean(v, axis=0)) or 1.0) for g, v in centroids.items()}
    next_g = max(groups.values(), default=-1) + 1
    for i in range(len(unit)):
        if i in groups:
            continue
        sims = sorted(((float(unit[i] @ c), g) for g, c in cent.items()), reverse=True)
        if sims and sims[0][0] >= JOIN_SHORT and (len(sims) == 1 or sims[0][0] - sims[1][0] >= JOIN_MARGIN):
            groups[i] = sims[0][1]
        else:
            groups[i] = next_g
            next_g += 1
    out = [groups[i] for i in range(len(unit))]
    within = [float(unit[a] @ unit[b]) for a in range(len(unit)) for b in range(a + 1, len(unit))
              if out[a] == out[b]]
    between = [float(cent[a] @ cent[b]) for a in cent for b in cent if a < b]
    stats = {"loosest_same": round(min(within), 2) if within else None,
             "closest_different": round(max(between), 2) if between else None}
    return out, stats


def regroup(embs: list[np.ndarray], threshold: float = REGROUP_SAME) -> list[int]:
    """Group voiceprints into people using all of them at once (average linkage): far
    steadier than deciding line by line. Returns a group number per voiceprint.

    The table of average similarities is updated in place after each merge (the
    Lance-Williams formula), so a long meeting regroups in a fraction of a second."""
    n = len(embs)
    if n == 0:
        return []
    x = np.stack([e / (np.linalg.norm(e) or 1.0) for e in embs]).astype(np.float64)
    avg = x @ x.T
    np.fill_diagonal(avg, -np.inf)
    sizes = np.ones(n)
    alive = np.ones(n, bool)
    parent = list(range(n))
    while alive.sum() > 1:
        flat = int(np.argmax(avg))
        i, j = divmod(flat, n)
        if avg[i, j] < threshold:
            break
        # merge j into i: the average to every other group, weighted by size
        row = (sizes[i] * avg[i] + sizes[j] * avg[j]) / (sizes[i] + sizes[j])
        avg[i, :] = row
        avg[:, i] = row
        avg[i, i] = -np.inf
        avg[j, :] = -np.inf
        avg[:, j] = -np.inf
        sizes[i] += sizes[j]
        alive[j] = False
        parent[j] = i
    def root(k: int) -> int:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    order: dict[int, int] = {}
    return [order.setdefault(root(k), len(order)) for k in range(n)]


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
