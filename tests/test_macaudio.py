"""macOS recording through AVAudioEngine, against a stand-in for Apple's AVFoundation (these
run on any OS): it records, falls back to PortAudio whenever the engine misbehaves, and a
stop that hangs never blocks Murmr."""
import sys
import threading
import time
import types

import numpy as np
import pytest

from whispa import audio, macaudio


class Varlist:
    """Like pyobjc's objc.varlist for a float* (no length of its own)."""

    def __init__(self, data):
        self.data = data

    def as_buffer(self, n):
        return memoryview(self.data[:n]).cast("B")


class TupleOnly(Varlist):
    as_buffer = None  # older pyobjc

    def as_tuple(self, n):
        return tuple(float(x) for x in self.data[:n])


class Buffer:
    def __init__(self, data, varlist=Varlist, float_data=True):
        self.data, self.varlist, self.float_data = data, varlist, float_data

    def frameLength(self):
        return len(self.data)

    def floatChannelData(self):
        return [self.varlist(self.data)] if self.float_data else None


class Format:
    def __init__(self, sr=48000, channels=1):
        self.sr, self.channels = sr, channels

    def sampleRate(self):
        return float(self.sr)

    def channelCount(self):
        return self.channels


class Engine:
    """Delivers 1024-frame buffers of 0.1 every 10 ms while running."""
    behaviour = {}

    def __init__(self):
        self.tap = None
        self.running = False
        self.node = types.SimpleNamespace(
            outputFormatForBus_=lambda bus: Format(**self.behaviour.get("format", {})),
            installTapOnBus_bufferSize_format_block_=self._install,
            removeTapOnBus_=lambda bus: None)

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        Engine.last = self
        return self

    def inputNode(self):
        return self.node

    def _install(self, bus, size, fmt, block):
        self.tap = block

    def prepare(self):
        pass

    def startAndReturnError_(self, err):
        if self.behaviour.get("start_fails"):
            return False, "no permission"
        self.running = True
        if not self.behaviour.get("silent"):
            threading.Thread(target=self._feed, daemon=True).start()
        return True, None

    def _feed(self):
        while self.running:
            buf = self.behaviour.get("buffer") or Buffer(np.full(1024, 0.1, np.float32))
            self.tap(buf, None)
            time.sleep(0.01)

    def isRunning(self):
        return self.running

    def stop(self):
        if self.behaviour.get("stop_hangs"):
            time.sleep(5)
        self.running = False


@pytest.fixture
def avf(monkeypatch):
    mod = types.ModuleType("AVFoundation")
    mod.AVAudioEngine = Engine
    mod.AVMediaTypeAudio = "soun"
    mod.AVCaptureDevice = types.SimpleNamespace(defaultDeviceWithMediaType_=lambda t: types.SimpleNamespace(
        localizedName=lambda: "MacBook Air Microphone"))
    monkeypatch.setitem(sys.modules, "AVFoundation", mod)
    Engine.behaviour = {}
    return Engine


@pytest.fixture
def portaudio(monkeypatch):
    """PortAudio's side, so a fallback can be seen without a real microphone."""
    opened = []

    def fake_open(self):
        opened.append(self.device)
        self._stream, self.samplerate, self.active_device = object(), 16000, None

    monkeypatch.setattr(audio.Recorder, "_open", fake_open)
    monkeypatch.setattr(audio, "close_stream", lambda *a, **k: True)
    monkeypatch.setattr(audio, "device_name", lambda i: "USB Mic")
    return opened


def _record(rec, seconds=0.1):
    rec.start()
    time.sleep(seconds)
    return rec.stop()


def test_records_through_apples_engine(avf, portaudio):
    rec = macaudio.MacRecorder()
    got, sr = _record(rec)
    assert rec.backend == "AVAudioEngine" and not portaudio
    assert sr == 48000 and len(got) >= 1024 and abs(float(got.mean()) - 0.1) < 1e-6
    assert not Engine.last.running  # stopped after the dictation


def test_logs_which_mic_once(avf, portaudio, caplog):
    import logging

    caplog.set_level(logging.INFO, "whispa.macaudio")
    rec = macaudio.MacRecorder()
    _record(rec)
    _record(rec)
    assert caplog.text.count("Microphone: MacBook Air Microphone via AVAudioEngine") == 1


@pytest.mark.parametrize("behaviour", [{"start_fails": True}, {"silent": True},
                                       {"format": {"sr": 0}},
                                       {"buffer": Buffer(np.zeros(8, np.float32), float_data=False)}])
def test_falls_back_to_portaudio(avf, portaudio, behaviour):
    Engine.behaviour = behaviour
    rec = macaudio.MacRecorder()
    t0 = time.perf_counter()
    rec.start()
    assert time.perf_counter() - t0 < macaudio.FIRST_AUDIO_TIMEOUT_S + 1.0
    assert rec.backend == "PortAudio" and portaudio == [None]
    rec.stop()


def test_a_chosen_microphone_uses_portaudio(avf, portaudio):
    rec = macaudio.MacRecorder("USB Mic")
    rec.start()
    assert rec.backend == "PortAudio" and portaudio == ["USB Mic"]


def test_switched_off_uses_portaudio(avf, portaudio):
    rec = macaudio.MacRecorder(native=False)
    rec.start()
    assert rec.backend == "PortAudio"


def test_a_stop_that_hangs_never_blocks(avf, portaudio, monkeypatch):
    monkeypatch.setattr(macaudio, "CLOSE_TIMEOUT_S", 0.3)
    rec = macaudio.MacRecorder()
    rec.start()
    time.sleep(0.05)
    Engine.behaviour = {"stop_hangs": True}
    t0 = time.perf_counter()
    got, sr = rec.stop()
    assert time.perf_counter() - t0 < 1.5 and len(got) > 0
    assert rec.wedged  # Murmr then restarts itself to get the mic back


def test_an_engine_stopped_by_a_device_change_is_started_again(avf, portaudio):
    rec = macaudio.MacRecorder(keep_open=True)
    first = Engine.last
    first.running = False  # AirPods connected: macOS stopped the engine
    rec.start()
    assert Engine.last is not first and Engine.last.running
    rec.stop()
    rec.shutdown()


def test_buffers_from_older_pyobjc():
    data = np.arange(4, dtype=np.float32)
    assert list(macaudio.buffer_samples(Buffer(data, varlist=TupleOnly))) == [0, 1, 2, 3]
    assert macaudio.buffer_samples(Buffer(data, float_data=False)) is None
    assert len(macaudio.buffer_samples(Buffer(np.zeros(0, np.float32)))) == 0


def test_the_mac_gets_the_native_recorder(monkeypatch, avf):
    monkeypatch.setattr(audio.sys, "platform", "darwin")
    assert isinstance(audio.make_recorder(None), macaudio.MacRecorder)
    monkeypatch.setattr(audio.sys, "platform", "win32")
    assert type(audio.make_recorder(None)) is audio.Recorder
