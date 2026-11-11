"""The window only shows features that work on this computer."""
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from whispa import capabilities  # noqa: E402
from whispa.config import Config  # noqa: E402
from whispa.learn import Dictionary  # noqa: E402


class Ctl:
    def __init__(self, tmp: Path, ai: bool):
        self.cfg = Config()
        self.dictionary = Dictionary(tmp / "dict.json")
        self.history_path = tmp / "history.jsonl"
        self.history_path.write_text("")
        self._ai = ai

    def __getattr__(self, name):  # every other controller call: harmless defaults
        defaults = {"status_text": "Ready", "llm_status": (True, "Active"), "update_state": {
            "installed": False, "sha": "", "date": "", "available": False, "changes": [], "error": "",
            "checked": True, "busy": "", "has_previous": False, "whats_new": [], "staged": False},
            "voice_storage": {"clips": 0, "seconds": 0, "models": 0, "voices": 0, "bytes": 0, "folder": ""},
            "managed_reason": "", "meetings": [], "meeting_running": False, "captions_on": False,
            "meeting_status": "", "training_stats": None}
        if name in defaults:
            return lambda *a, **k: defaults[name]
        return lambda *a, **k: None

    def ai_available(self):
        return self._ai


def _window(monkeypatch, mac: bool, ai: bool):
    from whispa import window as window_mod

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(capabilities, "can_train", lambda: not mac)
    monkeypatch.setattr(capabilities, "gpu_choice", lambda: not mac)
    monkeypatch.setattr(window_mod, "config_mod_dir", lambda: tmp)
    app = QApplication.instance() or QApplication([])
    w = window_mod.MainWindow(Ctl(tmp, ai), mic_options=[])
    w.show()
    w.show_page(w.pages.indexOf(w.settings))
    for _ in range(5):
        app.processEvents()
        time.sleep(0.01)
    nav = [b.text().strip() for b in w.nav.buttons()]
    titles = {lab.text() for lab in w.settings.widget().findChildren(QLabel)
              if lab.objectName() in ("rowTitle", "h2") and lab.isVisibleTo(w.settings.widget())}
    return w, nav, titles


@pytest.mark.parametrize("mac,ai", [(True, False), (False, True)])
def test_only_what_works_is_shown(monkeypatch, mac, ai):
    w, nav, titles = _window(monkeypatch, mac, ai)
    assert ("Voice training" in nav) == (not mac)
    assert ("Improve recognition with my voice" in titles) == (not mac)
    assert ("Create a voice that sounds like me" in titles) == (not mac)
    assert ("Speech model runs on" in titles) == (not mac)
    assert ("Speech model" in titles) == (not mac)
    assert ("Command Mode" in titles) == ai
    assert ("AI clean-up" in titles) == ai
    assert ("Local AI" in titles) == (not ai)  # the one-click setup, only when missing
    assert "Read aloud" in titles and "Your voice" in titles  # these work everywhere
    w.close()
