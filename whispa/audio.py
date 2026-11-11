"""Microphone capture."""

from __future__ import annotations

import collections
import logging
import threading

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

TARGET_SR = 16000


class Recorder:
    """Captures mono float32 audio. ``level`` is the live RMS for the UI meter."""

    def __init__(self, device: str | int | None = None, keep_open: bool = False,
                 preroll_s: float = 0.3):
        self.device = device
        self.keep_open = keep_open
        self.preroll_s = preroll_s
        self.samplerate = TARGET_SR
        self.active_device: int | None = None
        self.level = 0.0
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._samples = 0
        self._preroll: collections.deque[np.ndarray] = collections.deque()
        self._preroll_samples = 0
        self._recording = False
        self._lock = threading.Lock()
        if keep_open:
            self._open()

    @property
    def duration(self) -> float:
        return self._samples / self.samplerate

    def _open(self) -> None:
        chosen = resolve_device(self.device)
        if self.device not in (None, "") and chosen is None:
            log.warning("Microphone %r not found; using the system default", self.device)
        # Chosen mic, then the system default, then anything that looks like a mic
        # (handles unplugged headsets and broken virtual default devices).
        candidates: list[int | None] = [chosen] + ([None] if chosen is not None else [])
        candidates += [i for i in fallback_input_devices() if i != chosen]
        first_exc: Exception | None = None
        for i, dev in enumerate(candidates):
            try:
                self._open_device(dev)
            except Exception as exc:  # noqa: BLE001 - PortAudio raises many types
                first_exc = first_exc or exc
                continue
            self.active_device = dev
            if i:
                log.warning("Preferred mic failed (%s); using %s", first_exc, device_name(dev))
            return
        raise RuntimeError(f"Could not open microphone: {first_exc}")

    def _open_device(self, device) -> None:
        last_exc: Exception | None = None
        for sr in (TARGET_SR, None):
            try:
                if sr is None:  # device refuses 16 kHz: use its native rate, resample later
                    sr = int(sd.query_devices(device, "input")["default_samplerate"])
                stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                        device=device, latency="low", callback=self._callback)
                stream.start()
                self._stream, self.samplerate = stream, sr
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.debug("Opening mic %r at %s Hz failed: %s", device, sr, exc)
        raise last_exc  # type: ignore[misc]

    def _close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
                self.level = 0.0

    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0].copy()
        self.level = float(np.sqrt(np.mean(mono * mono))) if frames else 0.0
        with self._lock:
            if self._recording:
                self._chunks.append(mono)
                self._samples += len(mono)
            elif self.keep_open:
                self._preroll.append(mono)
                self._preroll_samples += len(mono)
                limit = int(self.preroll_s * self.samplerate)
                while self._preroll and self._preroll_samples - len(self._preroll[0]) >= limit:
                    self._preroll_samples -= len(self._preroll.popleft())

    def start(self) -> None:
        with self._lock:
            self._chunks = list(self._preroll)
            self._samples = sum(len(c) for c in self._chunks)
            self._preroll.clear()
            self._preroll_samples = 0
            self._recording = True
        if self._stream is None:
            try:
                self._open()
            except Exception:
                self._recording = False
                raise

    def stop(self) -> tuple[np.ndarray, int]:
        if not self.keep_open:
            self._close()  # waits for the last callback
        with self._lock:
            self._recording = False
            chunks, self._chunks = self._chunks, []
            self._samples = 0
        audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        return audio, self.samplerate

    def cancel(self) -> None:
        self.stop()

    def shutdown(self) -> None:
        self._recording = False
        self._close()


_HOSTAPI_RANK = ["Windows WASAPI", "MME", "Windows DirectSound", "Core Audio", "PulseAudio", "ALSA"]
_NOT_A_MIC = ("stereo mix", "what u hear", "loopback", "monitor of", "line in", "output")


def fallback_input_devices() -> list[int]:
    """Input devices ordered by how likely they are to be a working microphone."""
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        name = dev["name"].lower()
        if dev["max_input_channels"] < 1 or any(s in name for s in _NOT_A_MIC):
            continue
        api = sd.query_hostapis(dev["hostapi"])["name"]
        rank = _HOSTAPI_RANK.index(api) if api in _HOSTAPI_RANK else len(_HOSTAPI_RANK)
        devices.append((rank, "mic" not in name, i))
    return [i for *_, i in sorted(devices)]


def resolve_device(spec) -> int | None:
    """Config value (index, name, or None) -> device index; None = system default."""
    if spec in (None, ""):
        return None
    devices = sd.query_devices()
    if isinstance(spec, int):
        return spec if 0 <= spec < len(devices) and devices[spec]["max_input_channels"] > 0 else None
    spec_l = str(spec).lower()
    matches = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"].lower()
        if name == spec_l or spec_l in name:
            api = sd.query_hostapis(dev["hostapi"])["name"]
            rank = _HOSTAPI_RANK.index(api) if api in _HOSTAPI_RANK else len(_HOSTAPI_RANK)
            matches.append((name != spec_l, rank, i))
    return min(matches)[2] if matches else None


def device_name(index: int | None) -> str:
    try:
        if index is None:
            return sd.query_devices(kind="input")["name"]
        return sd.query_devices(index)["name"]
    except Exception:  # noqa: BLE001
        return "unknown"


_PSEUDO = ("sound mapper", "primary sound capture")


def _pretty(name: str) -> str:
    """Tidy Windows Bluetooth names: 'Headset (@System32...;(Buds3 Pro))' -> 'Headset (Buds3 Pro)'."""
    if ";(" in name:
        head = name.split("(", 1)[0].strip()
        tail = name.rsplit(";(", 1)[1].rstrip(")")
        return f"{head} ({tail})"
    return " ".join(name.split())


def input_devices() -> list[tuple[str, str]]:
    """(label, config value) for each microphone, most likely real mics first."""
    names: list[str] = []
    for i in fallback_input_devices():
        name = sd.query_devices(i)["name"]
        if any(p in name.lower() for p in _PSEUDO) or name in names:
            continue
        names.append(name)
    # MME truncates names to 31 characters; drop those when the full name exists.
    names = [n for n in names if not any(o != n and o.startswith(n) for o in names)]
    return [(_pretty(n), n) for n in names]


def test_mic(device=None, seconds: float = 3.0) -> str:
    import time

    rec = Recorder(device)
    rec.start()
    peak = 0.0
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        peak = max(peak, rec.level)
        time.sleep(0.02)
    audio, sr = rec.stop()
    name = sd.query_devices(rec.device, "input")["name"] if rec.device is not None else \
        sd.query_devices(kind="input")["name"]
    verdict = "OK" if peak > 0.01 else ("very quiet" if peak > 0.001 else "SILENT - wrong device or muted?")
    return (f"device: {name}\nrecorded {len(audio) / sr:.2f}s at {sr} Hz, "
            f"peak level {peak:.4f} -> {verdict}")


def list_input_devices() -> list[str]:
    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            api = sd.query_hostapis(dev["hostapi"])["name"]
            out.append(f"{i:3d}  {dev['name']}  ({api}, {int(dev['default_samplerate'])} Hz)")
    return out
