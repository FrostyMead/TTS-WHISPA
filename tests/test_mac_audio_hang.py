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
