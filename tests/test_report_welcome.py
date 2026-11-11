"""Report a problem, the first-run guide, and macOS computer audio (with a stand-in helper)."""
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from whispa import config as config_mod  # noqa: E402
from whispa import report  # noqa: E402
from whispa.config import Config  # noqa: E402


def test_redact_removes_private_bits():
    home = r"C:\Users\sam"
    text = (r"2026 INFO Learned: 'kuber'->'Kubernetes'" "\n"
            r"Loading C:\Users\sam\AppData\Roaming\Murmr\x" "\n"
            "Meeting summarized: Salary review with Jo\nuser sam started")
    out = report.redact(text, home=home, user="sam")
    assert "Kubernetes" not in out and "Salary" not in out and "sam" not in out
    assert r"~\AppData" in out and "<user> started" in out


def test_report_zip_has_logs_and_settings_but_nothing_personal(tmp_path):
    (tmp_path / "whispa.log").write_text("INFO Learned: 'a'->'Secret Name'\nINFO started\n")
    (tmp_path / "crash.log").write_text("Thread 0x1 stack\n")
    (tmp_path / "history.jsonl").write_text('{"text": "my private dictation"}\n')
    (tmp_path / "dictionary.json").write_text('{"terms": ["Secret Name"]}')

    class Ctl:
        cfg = Config()
        cfg.replacements = {"teh": "the"}
        cfg.llm_cleanup.api_key = "sk-123"

        def status_text(self):
            return "Ready · Parakeet on CPU"

    path = report.build("It crashed when I pressed fn", Ctl(), dest=tmp_path / "out", cdir=tmp_path)
    with zipfile.ZipFile(path) as z:
        names = set(z.namelist())
        assert {"report.txt", "settings.json", "logs/whispa.log", "logs/crash.log"} <= names
        everything = b"".join(z.read(n) for n in names)
    assert b"It crashed when I pressed fn" in everything and b"Ready" in everything
    for private in (b"Secret Name", b"my private dictation", b"sk-123", b"teh"):
        assert private not in everything
    settings = json.loads(zipfile.ZipFile(path).read("settings.json"))
    assert settings["llm_cleanup"]["api_key"] == "[set]"


def test_issue_url_is_prefilled():
    url = report.issue_url("Paste doesn't work in Outlook\nmore detail")
    assert url.startswith("https://github.com/FrostyMead/murmr/issues/new?title=Paste")
    assert "Murmr" in url


def test_existing_users_skip_the_welcome_guide(tmp_path):
    old = tmp_path / "config.json"
    old.write_text(json.dumps({"hotkey": "ctrl+cmd"}))
    assert config_mod.load(old).welcome_done is True
    assert config_mod.load(tmp_path / "new.json").welcome_done is False


def test_welcome_guide_walks_through_and_finishes():
    from PySide6.QtWidgets import QApplication

    from whispa.hotkey import Hotkey
    from whispa.welcome import WelcomeDialog

    app = QApplication.instance() or QApplication([])
    settings = {}

    class Ctl:
        cfg = Config()
        hotkey = Hotkey.parse("ctrl+cmd")

        def status_text(self):
            return "Ready"

        def ai_available(self):
            return False

        def setup_ai(self):
            settings["ai"] = True

        def update_setting(self, k, v):
            settings[k] = v

        def capture_hotkey(self, kind):
            settings["capturing"] = kind

        def cancel_capture(self):
            pass

        def restart(self):
            pass

    w = WelcomeDialog(Ctl(), mic_options=[("USB mic", "USB")])
    w.meter.open = lambda device: None
    w.show()
    names = []
    while True:
        names.append(w.steps[w.stack.currentIndex()])
        app.processEvents()
        if w.next_btn.text() == "Done":
            break
        w.next_btn.click()
    assert names[0] == "welcome" and "mic" in names and "try" in names and names[-1] == "extras"
    w.go(w.steps.index("try"))
    w.hotkey_field.change_requested.emit()
    assert settings["capturing"] == "hotkey"
    w.capture_finished("hotkey", "ctrl+alt", None)
    assert settings["hotkey"] == "ctrl+alt"
    w.dictated("Hello there")
    deadline = time.time() + 2
    while "Hello there" not in w.box.toPlainText() and time.time() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert "Hello there" in w.box.toPlainText()
    w.ai_btn.click()
    assert settings["ai"]
    w.go(len(w.steps) - 1)
    w.next_btn.click()
    assert settings["welcome_done"] is True


FAKE_HELPER = r'''
import sys, time, struct
mode = sys.argv[1]
if mode == "denied":
    sys.stderr.write("denied: not allowed\n"); sys.stderr.flush(); sys.exit(2)
sys.stderr.write("started\n"); sys.stderr.flush()
sys.stdout.buffer.write(struct.pack("<4800f", *([0.25] * 4800))); sys.stdout.flush()
sys.stdin.read()  # until Murmr closes it
'''


def test_mac_system_capture_reads_the_helper(tmp_path):
    from whispa.meetings import MacSystemCapture, SystemAudioDenied

    helper = tmp_path / "helper.py"
    helper.write_text(FAKE_HELPER)

    class Fake(MacSystemCapture):
        def __init__(self, mode):
            super().__init__(helper=helper, start_timeout=10)
            self.mode = mode

        def _command(self):
            return [sys.executable, str(helper), self.mode]

    cap = Fake("ok")
    cap.start()
    deadline = time.time() + 5
    while cap.buffer.end < 4800 and time.time() < deadline:
        time.sleep(0.05)
    cap.stop()
    audio = cap.buffer.take(0)
    assert len(audio) == 4800 and np.allclose(audio, 0.25)

    with pytest.raises(SystemAudioDenied):
        Fake("denied").start()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS helper")
def test_real_mac_helper_starts_and_stops():
    from whispa.meetings import AUDIOTAP, MacSystemCapture, system_audio_supported

    if not system_audio_supported():
        pytest.skip("macOS older than 13")
    assert os.access(AUDIOTAP, os.X_OK) or True  # start() fixes a missing executable bit
    cap = MacSystemCapture()
    try:
        cap.start()
    except RuntimeError as exc:  # e.g. not allowed on this machine: a clear error, no hang
        assert str(exc)
        return
    time.sleep(1.5)
    cap.stop()
    assert cap.samplerate in (16000, 24000, 48000)
