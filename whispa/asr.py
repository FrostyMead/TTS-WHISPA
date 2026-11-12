"""Local speech recognition with NVIDIA Parakeet via onnx-asr / ONNX Runtime."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")  # nothing about you leaves this machine

_PROVIDERS = {
    "cuda": ("CUDAExecutionProvider", "CUDA"),
    "directml": ("DmlExecutionProvider", "DirectML"),
    "coreml": ("CoreMLExecutionProvider", "CoreML"),
    "cpu": ("CPUExecutionProvider", "CPU"),
}

LONG_AUDIO_S = 60.0  # beyond this, split on silence with Silero VAD
BASE_MODEL = "nemo-parakeet-tdt-0.6b-v3"  # English + 24 European languages
# Chinese, Japanese, Korean and ~95 more languages (language detected by itself). Slower
# than Parakeet, so it's a choice, not the default. Measured on a desktop CPU for a 5 s
# dictation: small 1.9 s (~250 MB, int8), large-v3 turbo 5.4 s (~1.1 GB) but more accurate.
WHISPER_MODEL = "onnx-community/whisper-small"
WHISPER_BEST = "onnx-community/whisper-large-v3-turbo"
LANGUAGE_MODELS = {"european": BASE_MODEL, "world": WHISPER_MODEL, "world_best": WHISPER_BEST}
# Whisper makes these up from silence or noise (subtitle credits from its training data).
_WHISPER_GHOSTS = ("字幕由amara.org社区提供", "amara.org", "请不吝点赞", "明镜与点点栏目",
                   "subtitles by the amara.org community", "ご視聴ありがとうございました")
WHISPER_MAX_S = 28.0  # Whisper hears at most 30 s at a time: longer audio is split first


def is_whisper(model: str) -> bool:
    return "whisper" in str(model or "").lower()


def local_model_dir(model: str) -> Path | None:
    """``model`` is a folder with an exported model (config.json + ONNX files), not a name."""
    try:
        p = Path(model).expanduser()
        return p if p.is_dir() and (p / "config.json").is_file() else None
    except (OSError, ValueError):
        return None


def _load(model: str, quantization: str | None, providers: list, sess_options=None):
    """onnx_asr.load_model for a model name or a local folder (e.g. your tuned model)."""
    import onnx_asr

    folder = local_model_dir(model)
    if folder is None:
        return onnx_asr.load_model(model, quantization=quantization, providers=providers,
                                   sess_options=sess_options)
    model_type = json.loads((folder / "config.json").read_text(encoding="utf-8")).get(
        "model_type", "nemo-conformer-tdt")
    if quantization and not any(folder.glob(f"*.{quantization}.onnx")):
        log.info("%s has no %s files; using full precision", folder.name, quantization)
        quantization = None
    return onnx_asr.load_model(model_type, folder, quantization=quantization, providers=providers,
                               sess_options=sess_options)


def select_providers(device: str) -> tuple[list[str], str]:
    import onnxruntime as ort

    ort.set_default_logger_severity(3)  # errors only; ORT is chatty about graph rewrites
    available = ort.get_available_providers()
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in available:
        try:  # load CUDA/cuDNN DLLs shipped as pip packages (nvidia-*)
            ort.preload_dlls()
        except Exception as exc:  # noqa: BLE001
            log.debug("preload_dlls: %s", exc)
    # CoreML is opt-in: int8 on the Apple Silicon CPU is fast and more reliable.
    order = ["cuda", "directml", "cpu"] if device == "auto" else [device, "cpu"]
    for name in order:
        if name not in _PROVIDERS:
            log.warning("Unknown device %r, ignoring", name)
            continue
        provider, label = _PROVIDERS[name]
        if provider in available:
            providers = [provider] if name == "cpu" else [provider, "CPUExecutionProvider"]
            return providers, label
        if device != "auto":
            log.warning("%s requested but not available (have: %s)", provider, available)
    return ["CPUExecutionProvider"], "CPU"


def _cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x3000 <= o <= 0x30FF or 0x3400 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF or \
        0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF


def drop_ghosts(text: str) -> str:
    """Whisper's made-up subtitle credits: never something you said."""
    return "" if any(g in text.lower() for g in _WHISPER_GHOSTS) else text


def join_pieces(parts: list[str]) -> str:
    """Join recognized pieces: with a space, except between Chinese/Japanese/Korean text."""
    out = ""
    for p in parts:
        if out and not (_cjk(out[-1]) or _cjk(p[0])):
            out += " "
        out += p
    return out


class _TurnTaking:
    """One recognition at a time, dictation first: a meeting or file waiting for its turn
    lets a dictation that arrives go ahead (it can't interrupt one that's running, but
    those pieces are short)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._busy = False
        self._urgent = 0

    def acquire(self, urgent: bool) -> None:
        with self._cv:
            if urgent:
                self._urgent += 1
            try:
                while self._busy or (not urgent and self._urgent):
                    self._cv.wait()
            finally:
                if urgent:
                    self._urgent -= 1
            self._busy = True

    def release(self) -> None:
        with self._cv:
            self._busy = False
            self._cv.notify_all()


class Transcriber:
    def __init__(self, model: str = "nemo-parakeet-tdt-0.6b-v3", device: str = "auto",
                 quantization: str = "auto"):
        self.model_name = model
        self.device = device
        self.quantization = quantization
        self.backend = "?"
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.fallback_reason = ""  # set when a local model failed and the base model was used
        self._model = None
        self._vad_model = None
        self._lock = threading.Lock()
        self._turns = _TurnTaking()
        self.last_wait_ms = 0.0

    def _quant(self, label: str) -> str | None:
        if is_whisper(self.model_name):
            return "int8"  # full precision is ~4x the download for little gain
        q = (self.quantization or "auto").lower()
        if q == "auto":
            return "int8" if label == "CPU" else None
        return None if q in ("none", "fp32", "") else q

    @staticmethod
    def _lean_options(providers: list):
        """Memory-lean ONNX Runtime settings: weights go straight to the GPU instead of also
        being kept in RAM, no pre-grown memory pools. Measured on an RTX 3070 Ti: ~0.9 GB
        less committed memory, same speed, faster load."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        # Don't keep threads spinning on the CPU after each run, waiting for the next one: an
        # app that's mostly idle would burn battery for a few milliseconds of latency.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.add_session_config_entry("session.inter_op.allow_spinning", "0")
        so.enable_mem_pattern = False  # also required by DirectML
        so.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        tuned = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"})
                 if p == "CUDAExecutionProvider" else p for p in providers]
        return so, tuned

    def switch_device(self, device: str) -> None:
        """Reload on another device (e.g. the CPU while training uses the GPU). Blocking."""
        self._turns.acquire(urgent=True)
        try:
            with self._lock:
                self.ready.clear()
                self.device = device
                self._model = None
                self.load()
        finally:
            self._turns.release()

    def load(self) -> None:
        try:
            self._load_model()
        except Exception as exc:  # noqa: BLE001
            if local_model_dir(self.model_name) is None and Path(self.model_name).anchor == "":
                raise
            # A tuned/local model that won't load must never leave you without dictation.
            log.error("Model %s failed to load (%s); using the base model", self.model_name, exc)
            self.fallback_reason = str(exc) or exc.__class__.__name__
            self.model_name = BASE_MODEL
            self.error = None
            self._load_model()

    def _load_model(self) -> None:
        t0 = time.perf_counter()
        providers, label = select_providers(self.device)
        try:
            so, tuned = self._lean_options(providers)
            self._model = _load(self.model_name, self._quant(label), tuned, so)
            self._warmup()
        except Exception as exc:  # noqa: BLE001
            if label == "CPU":
                self.error = exc
                raise
            log.warning("Loading on %s failed (%s); falling back to CPU", label, exc)
            label = "CPU"
            self._model = _load(self.model_name, self._quant(label), ["CPUExecutionProvider"])
            self._warmup()
        self.backend = label
        log.info("Loaded %s on %s in %.1fs", self.model_name, label, time.perf_counter() - t0)
        self.ready.set()

    def _warmup(self) -> None:
        self._model.recognize(np.zeros(16000, np.float32), sample_rate=16000)

    def transcribe(self, audio: np.ndarray, sample_rate: int, urgent: bool = True) -> str:
        """Dictation (``urgent``) goes before meeting notes and file transcription."""
        if not self.ready.is_set():
            raise RuntimeError("Model is not loaded")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return ""
        waited = time.perf_counter()
        self._turns.acquire(urgent)
        if urgent:
            self.last_wait_ms = (time.perf_counter() - waited) * 1000  # behind a meeting / file piece
        try:
            if not self.ready.is_set() or self._model is None:
                raise RuntimeError("Model is not loaded")
            whisper = is_whisper(self.model_name)
            if whisper and float(np.sqrt(np.mean(audio * audio))) < 0.002:
                return ""  # silence: Whisper would invent something ("Thank you.")
            limit = WHISPER_MAX_S if whisper else LONG_AUDIO_S
            if audio.size / sample_rate > limit:
                text = self._transcribe_long(audio, sample_rate)
            else:
                text = str(self._model.recognize(audio, sample_rate=sample_rate)).strip()
            return drop_ghosts(text) if whisper else text
        finally:
            self._turns.release()

    def transcribe_background(self, audio: np.ndarray, sample_rate: int) -> str:
        """Meeting notes, captions and files: wait for any dictation."""
        return self.transcribe(audio, sample_rate, urgent=False)

    def _transcribe_long(self, audio: np.ndarray, sample_rate: int) -> str:
        import onnx_asr

        if self._vad_model is None:
            self._vad_model = onnx_asr.load_vad("silero")
        segments = self._model.with_vad(
            self._vad_model, max_speech_duration_s=WHISPER_MAX_S if is_whisper(self.model_name) else 30,
            min_silence_duration_ms=300,
            speech_pad_ms=200).recognize(audio, sample_rate=sample_rate)
        parts = [s.text.strip() for s in segments if s.text.strip()]
        return join_pieces(parts)
