"""macOS: the microphone through Apple's AVAudioEngine instead of PortAudio.

PortAudio's CoreAudio streams sometimes hung while stopping (often right after headphones or
another app switched the audio device), and once that happened no stream in the process
could be opened again: dictation sat on "transcribing" until Murmr restarted. AVAudioEngine
is the API Apple's own apps record with; it copes with device changes instead of hanging.

Used for the system default microphone. A microphone you picked by name, or anything going
wrong here (the engine won't start, no audio arrives within a second), falls back to
PortAudio for that recording, so dictation always works.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from .audio import CLOSE_TIMEOUT_S, Recorder

log = logging.getLogger(__name__)

FIRST_AUDIO_TIMEOUT_S = 1.0  # a working engine delivers its first buffer within ~20 ms
TAP_FRAMES = 1024


def available() -> bool:
    try:
        import AVFoundation  # noqa: F401
        return True
    except ImportError:
        return False


def buffer_samples(buffer) -> np.ndarray | None:
    """The first channel of an AVAudioPCMBuffer (float32) as a numpy array (a copy)."""
    n = int(buffer.frameLength())
    if n <= 0:
        return np.zeros(0, np.float32)
    channels = buffer.floatChannelData()
    if channels is None:
        return None
    first = channels[0]
    try:
        data = np.frombuffer(first.as_buffer(n), dtype=np.float32, count=n)
    except (AttributeError, TypeError, ValueError):
        data = np.asarray(first.as_tuple(n), dtype=np.float32)
    return data.copy()


def default_mic_name() -> str:
    try:
        import AVFoundation

        dev = AVFoundation.AVCaptureDevice.defaultDeviceWithMediaType_(AVFoundation.AVMediaTypeAudio)
        return str(dev.localizedName()) if dev is not None else "default microphone"
    except Exception:  # noqa: BLE001
        return "default microphone"


def run_bounded(fn, timeout: float, what: str) -> bool:
    """Run ``fn`` on a throwaway thread; False if it was still running after ``timeout``."""
    t = threading.Thread(target=fn, name=f"stop-{what}", daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        log.warning("The %s didn't stop within %.0f s (an audio driver hang); left it behind and "
                    "carried on", what, timeout)
        return False
    return True


class MacRecorder(Recorder):
    """A Recorder whose default microphone goes through AVAudioEngine."""

    def __init__(self, *args, native: bool = True, **kwargs):
        self._engine = None
        self.native = native and available()
        self.backend = ""
        self._last_used = ""
        super().__init__(*args, **kwargs)

    # -- opening ------------------------------------------------------------------------
    def _open(self) -> None:
        if self.native and self.device in (None, ""):
            try:
                self._open_native()
                self._note("AVAudioEngine", default_mic_name())
                return
            except Exception as exc:  # noqa: BLE001
                log.warning("Apple's audio engine couldn't record (%s); using PortAudio", exc)
        super()._open()
        from .audio import device_name

        try:
            name = device_name(self.active_device)
        except Exception:  # noqa: BLE001
            name = str(self.device or "default microphone")
        self._note("PortAudio", name)

    def _note(self, backend: str, name: str) -> None:
        self.backend = backend
        used = f"{name} via {backend}"
        if used != self._last_used:  # once per change, not once per dictation
            self._last_used = used
            log.info("Microphone: %s", used)

    def _open_native(self) -> None:
        import AVFoundation

        self._gen += 1
        gen = self._gen
        engine = AVFoundation.AVAudioEngine.alloc().init()
        node = engine.inputNode()
        fmt = node.outputFormatForBus_(0)
        sr = int(fmt.sampleRate())
        if sr <= 0 or int(fmt.channelCount()) < 1:
            raise RuntimeError("no input device")
        first = threading.Event()
        problem: list[str] = []

        def tap(buffer, when) -> None:
            if gen != self._gen:
                return  # an engine left behind
            try:
                mono = buffer_samples(buffer)
            except Exception as exc:  # noqa: BLE001
                if not problem:
                    problem.append(f"unreadable audio buffer: {exc}")
                return
            if mono is None:
                if not problem:
                    problem.append("audio isn't 32-bit float")
                return
            first.set()
            if len(mono):
                self._callback(mono[:, None], len(mono), None, None)

        node.installTapOnBus_bufferSize_format_block_(0, TAP_FRAMES, fmt, tap)
        engine.prepare()
        ok, err = engine.startAndReturnError_(None)
        if not ok:
            self._stop_engine(engine)
            raise RuntimeError(f"start failed: {err}")
        if not first.wait(FIRST_AUDIO_TIMEOUT_S) or problem:
            self._gen += 1
            self._stop_engine(engine)
            raise RuntimeError(problem[0] if problem else "no audio arrived")
        self._engine = self._stream = engine  # _stream: Recorder.start() checks it
        self.samplerate, self.active_device = sr, None

    # -- closing ------------------------------------------------------------------------
    @staticmethod
    def _stop_engine(engine) -> bool:
        def work() -> None:
            try:
                engine.inputNode().removeTapOnBus_(0)
            except Exception:  # noqa: BLE001
                pass
            engine.stop()

        return run_bounded(work, CLOSE_TIMEOUT_S, "microphone")

    def _close(self) -> None:
        engine = self._engine
        if engine is None:
            super()._close()
            return
        self._engine = self._stream = None
        self._gen += 1
        self.level = 0.0
        if not self._stop_engine(engine):
            self.wedged = True

    def start(self) -> None:
        # Kept open, the engine stops by itself when the audio device changes (headphones
        # plugged in, AirPods connecting): start it again instead of recording silence.
        if self._engine is not None and not self._engine.isRunning():
            self._close()
        super().start()
