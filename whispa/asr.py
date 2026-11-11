"""Local speech recognition with NVIDIA Parakeet via onnx-asr / ONNX Runtime."""

from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger(__name__)

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_PROVIDERS = {
    "cuda": ("CUDAExecutionProvider", "CUDA"),
    "directml": ("DmlExecutionProvider", "DirectML"),
    "coreml": ("CoreMLExecutionProvider", "CoreML"),
    "cpu": ("CPUExecutionProvider", "CPU"),
}

LONG_AUDIO_S = 60.0  # beyond this, split on silence with Silero VAD


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


class Transcriber:
    def __init__(self, model: str = "nemo-parakeet-tdt-0.6b-v3", device: str = "auto",
                 quantization: str = "auto"):
        self.model_name = model
        self.device = device
        self.quantization = quantization
        self.backend = "?"
        self.ready = threading.Event()
        self.error: Exception | None = None
        self._model = None
        self._vad_model = None
        self._lock = threading.Lock()

    def _quant(self, label: str) -> str | None:
        q = (self.quantization or "auto").lower()
        if q == "auto":
            return "int8" if label == "CPU" else None
        return None if q in ("none", "fp32", "") else q

    def load(self) -> None:
        import onnx_asr

        t0 = time.perf_counter()
        providers, label = select_providers(self.device)
        try:
            self._model = onnx_asr.load_model(
                self.model_name, quantization=self._quant(label), providers=providers)
            self._warmup()
        except Exception as exc:  # noqa: BLE001
            if label == "CPU":
                self.error = exc
                raise
            log.warning("Loading on %s failed (%s); falling back to CPU", label, exc)
            label = "CPU"
            self._model = onnx_asr.load_model(
                self.model_name, quantization=self._quant(label), providers=["CPUExecutionProvider"])
            self._warmup()
        self.backend = label
        log.info("Loaded %s on %s in %.1fs", self.model_name, label, time.perf_counter() - t0)
        self.ready.set()

    def _warmup(self) -> None:
        self._model.recognize(np.zeros(16000, np.float32), sample_rate=16000)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if not self.ready.is_set():
            raise RuntimeError("Model is not loaded")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return ""
        with self._lock:
            if audio.size / sample_rate > LONG_AUDIO_S:
                return self._transcribe_long(audio, sample_rate)
            return str(self._model.recognize(audio, sample_rate=sample_rate)).strip()

    def _transcribe_long(self, audio: np.ndarray, sample_rate: int) -> str:
        import onnx_asr

        if self._vad_model is None:
            self._vad_model = onnx_asr.load_vad("silero")
        segments = self._model.with_vad(
            self._vad_model, max_speech_duration_s=30, min_silence_duration_ms=300,
            speech_pad_ms=200).recognize(audio, sample_rate=sample_rate)
        return " ".join(s.text.strip() for s in segments if s.text.strip())
