"""The microphone opens and closes off the UI thread, and Esc after letting go cancels
what's still being transcribed."""
import os
import threading
import time
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication  # noqa: E402

_qapp = QApplication.instance() or QApplication([])  # stopping draws the tray icon

from whispa.app import MurmrApp, SerialThread
from whispa.hotkey import Hotkey, PTTStateMachine


class SlowRecorder:
    """A mic whose driver takes a while, like a Bluetooth headset waking up."""

    def __init__(self, delay=0.3, fail=False):
        self.delay, self.fail = delay, fail
        self.threads = []

    def start(self):
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Invalid device")

    def stop(self):
        self.threads.append(threading.current_thread().name)
        time.sleep(self.delay)
        return np.ones(16000, np.float32) * 0.1, 16000


def _app(recorder):
    m = mock.MagicMock()
    m.recording = False
    m._speaker = None
    m.pending = 0
    m._rec_id = 0
    m._job_seq = 0
    m._cancelled_upto = 0
    m.live = None
    m._rec_target = None
    m.mode = "dictate"
    m.recorder = recorder
    m._audio = SerialThread("audio-test")
    m._end_recording = lambda then=None: MurmrApp._end_recording(m, then)
    m._cancelled = lambda job: MurmrApp._cancelled(m, job)
    events = []
    m.bridge.mic_opened.emit = lambda rid: events.append(("opened", rid))
    m.bridge.mic_failed.emit = lambda rid, err: events.append(("failed", rid, err))
    m.bridge.recorded.emit = lambda audio, sr, d: events.append(("recorded", len(audio), d["job"]))
    return m, events


def _wait(events, n, timeout=3.0):
    end = time.monotonic() + timeout
    while len(events) < n and time.monotonic() < end:
        time.sleep(0.01)


def test_start_and_stop_never_wait_for_the_driver():
    rec = SlowRecorder(delay=0.3)
    m, events = _app(rec)
    t0 = time.perf_counter()
    MurmrApp._on_start(m, "dictate")
    assert time.perf_counter() - t0 < 0.1  # the UI thread didn't wait 0.3 s
    assert m.recording and not events
    _wait(events, 1)
    assert events == [("opened", 1)]
    t0 = time.perf_counter()
    MurmrApp._on_stop(m)
    assert time.perf_counter() - t0 < 0.1
    assert m.pending == 1 and not m.recording
    _wait(events, 2)
    assert events[1] == ("recorded", 16000, 1)
    assert set(rec.threads) == {"audio-test"}  # both on the audio thread


def test_a_mic_that_fails_is_reported():
    m, events = _app(SlowRecorder(delay=0.01, fail=True))
    MurmrApp._on_start(m, "dictate")
    _wait(events, 1)
    assert events[0][0] == "failed" and "Invalid device" in events[0][2]


def test_late_mic_open_after_letting_go_is_ignored():
    m, _ = _app(SlowRecorder())
    m.recording, m._rec_id = False, 2  # already let go of recording 2
    MurmrApp._on_mic_opened(m, 2)
    m.overlay.set_state.assert_not_called()
    m.sounds.play.assert_not_called()


# --- cancel after letting go --------------------------------------------------------------
def test_esc_cancels_what_is_being_transcribed():
    m, _ = _app(SlowRecorder())
    m.pending, m._job_seq = 1, 4
    MurmrApp.cancel_pending(m)
    assert m._cancelled_upto == 4
    m.sounds.play.assert_called_with("cancel")
    m.overlay.set_state.assert_called_with("idle")
    m.sounds.play.reset_mock()
    MurmrApp.cancel_pending(m)  # nothing new to cancel
    m.sounds.play.assert_not_called()


def test_nothing_to_cancel_does_nothing():
    m, _ = _app(SlowRecorder())
    MurmrApp.cancel_pending(m)
    m.sounds.play.assert_not_called()


def test_a_cancelled_dictation_is_never_pasted_or_kept():
    m, _ = _app(SlowRecorder())
    m._cancelled_upto = 3
    MurmrApp._process(m, np.zeros(16000, np.float32), 16000, None, None, 3)
    m.bridge.discarded.emit.assert_called_once()
    m.transcriber.transcribe.assert_not_called()
    m._deliver.assert_not_called()
    m._remember.assert_not_called()


def test_cancel_while_the_ai_polishes(monkeypatch):
    m, _ = _app(SlowRecorder())
    m.transcriber.ready.wait.return_value = True
    m.transcriber.transcribe.return_value = "hello there"
    m.transcriber.last_wait_ms = 0.0
    m.transcriber.backend = "CPU"

    def slow_clean(raw):
        m._cancelled_upto = 7  # Esc pressed while this ran
        return "Hello there."

    m._clean = slow_clean
    m.polisher.last_issue = ""
    MurmrApp._process(m, np.ones(32000, np.float32) * 0.1, 16000, None, None, 7)
    m.bridge.discarded.emit.assert_called_once()
    m._deliver.assert_not_called()
    m._remember.assert_not_called()


def test_a_later_dictation_is_not_cancelled():
    m, _ = _app(SlowRecorder())
    m._cancelled_upto = 3
    assert MurmrApp._cancelled(m, 3) and not MurmrApp._cancelled(m, 4)
    assert not MurmrApp._cancelled(m, None)


def test_esc_when_idle_reaches_the_app():
    pressed = []
    sm = PTTStateMachine(Hotkey.parse("ctrl+alt"), on_start=lambda: None, on_stop=lambda: None,
                         on_cancel=lambda s: pressed.append("cancel"), on_lock=lambda: None)
    sm.on_idle_esc = lambda: pressed.append("idle-esc")
    sm.press({"esc"})
    sm.release({"esc"})
    assert pressed == ["idle-esc"]
    sm.press({"ctrl", "alt"})  # recording: Esc cancels the recording, as before
    sm.press({"esc"})
    assert pressed == ["idle-esc", "cancel"]
