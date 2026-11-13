"""A microphone driver that hangs while closing (seen on macOS) must not freeze dictation,
quitting or updating; and an update that's still starting isn't rolled back by a second
launch."""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from whispa import _boot, audio
from test_updater import runtime, make_app, NEW, OLD  # noqa: F401


class HangingStream:
    """A stream whose abort() never returns, like CoreAudio after a device switch."""

    def __init__(self):
        self.release = threading.Event()
        self.aborted = self.closed = False

    def abort(self):
        self.aborted = True
        self.release.wait(10)

    def stop(self):
        raise AssertionError("stop() waits for buffers to drain; abort() is used instead")

    def close(self):
        self.closed = True


def test_a_hanging_close_is_left_behind(monkeypatch):
    s = HangingStream()
    t0 = time.perf_counter()
    assert audio.close_stream(s, timeout=0.2) is False
    assert time.perf_counter() - t0 < 1.0 and s.aborted
    s.release.set()


def test_a_normal_close():
    s = HangingStream()
    s.release.set()
    assert audio.close_stream(s, timeout=1.0) is True and s.closed


def test_stopping_returns_the_audio_even_if_the_driver_hangs(monkeypatch):
    monkeypatch.setattr(audio, "CLOSE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(audio.close_stream, "__defaults__", (0.2, "microphone"))
    r = audio.Recorder()
    stream = HangingStream()
    r._stream, r._recording = stream, True
    r._chunks, r._samples = [np.ones(1600, np.float32)], 1600
    t0 = time.perf_counter()
    got, sr = r.stop()
    assert time.perf_counter() - t0 < 1.0
    assert len(got) == 1600 and sr == audio.TARGET_SR
    assert r._stream is None  # the next recording opens a new stream
    stream.release.set()


def test_a_stream_left_behind_cant_add_to_the_next_recording(monkeypatch):
    callbacks = []

    class FakeStream:
        def __init__(self, **kw):
            callbacks.append(kw["callback"])

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(audio.sd, "InputStream", FakeStream)
    r = audio.Recorder()
    r._open_device(None)
    old = callbacks[-1]
    r._close()
    r._open_device(None)
    r._recording = True
    old(np.ones((160, 1), np.float32), 160, None, None)  # the abandoned stream wakes up
    assert r._samples == 0
    callbacks[-1](np.ones((160, 1), np.float32), 160, None, None)
    assert r._samples == 160


# --- an update that's still starting -------------------------------------------------------
@pytest.fixture
def trial(runtime, monkeypatch):  # noqa: F811
    for name, value in {"RUNTIME": runtime, "APP": runtime / "app",
                        "CURRENT": runtime / "app/current", "NEXT": runtime / "app/next",
                        "PREVIOUS": runtime / "app/previous", "TRIAL": runtime / "trial",
                        "LOG": runtime / "boot.log"}.items():
        monkeypatch.setattr(_boot, name, str(value))
    monkeypatch.setattr(_boot, "refresh_self", lambda: None)
    make_app(runtime / "app/next")
    (runtime / "app/next/version.json").write_text(json.dumps({"sha": NEW}))
    _boot.apply_update()
    assert (runtime / "trial").exists()
    return runtime


def _starting(runtime, pid, age=0.0, sha=NEW):
    (runtime / "starting").write_text(json.dumps({"sha": sha, "pid": pid,
                                                  "time": time.time() - age}))


@pytest.fixture
def other_process():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    yield p
    p.kill()
    p.wait()


def test_still_starting_is_left_alone(trial, other_process):
    _starting(trial, other_process.pid)
    assert _boot.trial_state() == "starting"


def test_a_dead_start_is_a_failed_start(trial):
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    _starting(trial, p.pid)
    assert _boot.trial_state() == "failed"


def test_no_record_is_a_failed_start(trial):
    assert _boot.trial_state() == "failed"


def test_a_start_of_another_version_doesnt_count(trial, other_process):
    _starting(trial, other_process.pid, sha=OLD)
    assert _boot.trial_state() == "failed"


def test_stuck_starting_for_minutes_is_ended(trial, other_process):
    _starting(trial, other_process.pid, age=_boot.STARTING_GRACE_S + 5)
    assert _boot.trial_state() == "stuck"
    other_process.wait(timeout=10)  # it was ended


def test_this_process_is_never_counted(trial):
    _starting(trial, os.getpid())
    assert _boot.trial_state() == "failed"


# --- once the driver has hung, Murmr restarts itself to get the mic back -------------------
def test_a_hang_marks_the_recorder_wedged(monkeypatch):
    monkeypatch.setattr(audio.close_stream, "__defaults__", (0.2, "microphone"))
    r = audio.Recorder()
    stream = HangingStream()
    r._stream = stream
    r.stop()
    assert r.wedged
    stream.release.set()


def _wedge_app(monkeypatch, tmp_path):
    from unittest import mock

    from whispa import app as app_mod
    monkeypatch.setattr("whispa.config.config_dir", lambda: tmp_path)
    m = mock.MagicMock()
    m.recording = False
    m.pending = 0
    m._rec_id = 0
    m._wedge_restart = False
    m.recorder.wedged = True
    m._on_mic_wedged = lambda: app_mod.MurmrApp._on_mic_wedged(m)
    return app_mod, m


def test_a_new_dictation_on_a_hung_driver_restarts_murmr(monkeypatch, tmp_path):
    app_mod, m = _wedge_app(monkeypatch, tmp_path)
    timers = []
    monkeypatch.setattr(app_mod.QTimer, "singleShot", lambda ms, fn: timers.append(fn))
    app_mod.MurmrApp._on_start(m, "dictate")
    m._audio.submit.assert_not_called()  # opening would hang as well
    assert m._wedge_restart
    monkeypatch.setattr(app_mod.threading, "Timer", lambda *a, **k: _NoTimer())
    timers.pop()()  # nothing pending: restart now
    m.restart.assert_called_once()
    assert not app_mod.mic_restart_allowed()  # and not again for a couple of minutes


class _NoTimer:
    daemon = False

    def start(self):
        pass


def test_no_restart_loop(monkeypatch, tmp_path):
    app_mod, m = _wedge_app(monkeypatch, tmp_path)
    app_mod.note_mic_restart()
    app_mod.MurmrApp._on_mic_wedged(m)
    m.restart.assert_not_called()
    assert "isn't responding" in m.overlay.flash.call_args.args[0]


def test_a_mic_that_never_opens_counts_as_hung(monkeypatch, tmp_path):
    app_mod, m = _wedge_app(monkeypatch, tmp_path)
    m.recorder.wedged = False
    m.recording, m._rec_id, m._mic_open_id = True, 3, 2
    called = []
    m._on_mic_wedged = lambda: called.append(1)
    app_mod.MurmrApp._check_mic_opened(m, 3)
    assert m.recorder.wedged and called
    m.recorder.wedged, called[:] = False, []
    m._mic_open_id = 3  # it did open
    app_mod.MurmrApp._check_mic_opened(m, 3)
    assert not called


def test_the_audio_thread_reports_a_hang(monkeypatch, tmp_path):
    app_mod, m = _wedge_app(monkeypatch, tmp_path)
    m.recorder.stop.return_value = (np.zeros(10, np.float32), 16000)
    jobs = []
    m._audio.submit = jobs.append
    got = []
    app_mod.MurmrApp._end_recording(m, lambda a, sr: got.append(len(a)))
    jobs[0]()
    assert got == [10]  # what you said is still delivered
    m.bridge.mic_wedged.emit.assert_called_once()
