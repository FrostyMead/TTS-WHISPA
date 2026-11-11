"""The window works at any size: nothing is cut off when it's narrow, and Settings can
jump straight to a section (e.g. Updates from the tray menu)."""
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea  # noqa: E402

from tests.test_capabilities_ui import Ctl  # noqa: E402


class FullCtl(Ctl):
    def __getattr__(self, name):
        extra = {"recorded_minutes": {"recognition": 3.0, "voice": 2.0}, "tuning_blocker": "",
                 "tuning_active": False}
        if name in extra:
            return lambda *a, **k: extra[name]
        if name == "training_store":
            from whispa.learn import TrainingStore

            return lambda *a, **k: TrainingStore(Path(tempfile.mkdtemp()))
        return super().__getattr__(name)


def settle(app, n=20):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(monkeypatch):
    from whispa import capabilities
    from whispa import window as window_mod

    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(capabilities, "can_train", lambda: True)
    monkeypatch.setattr(window_mod, "config_mod_dir", lambda: tmp)
    app = QApplication.instance() or QApplication([])
    w = window_mod.MainWindow(FullCtl(tmp, ai=True), mic_options=[("Microphone Array (Realtek(R) Audio)", "x")])
    w.show()
    yield app, w
    w.hide()


@pytest.mark.parametrize("width", [520, 640, 760, 980])
def test_no_page_is_wider_than_the_window(window, width):
    app, w = window
    w.resize(width, 600)
    settle(app)
    assert w.minimumWidth() <= 520
    for i in range(w.pages.count()):
        w.show_page(i)
        settle(app)
        page = w.pages.widget(i)
        inner = page.widget() if isinstance(page, QScrollArea) else page
        need = inner.layout().minimumSize().width()
        assert need <= page.viewport().width(), (w.nav.button(i).property("label"), need)
    assert (w.side.width() < 100) == (width < w.COMPACT_BELOW)  # icons-only sidebar when narrow


def test_settings_jumps_to_updates(window):
    app, w = window
    w.resize(900, 600)
    w.show_page(w.pages.indexOf(w.settings))
    settle(app)
    w.settings.show_section("Updates")
    settle(app, 60)
    bar = w.settings.verticalScrollBar()
    assert bar.value() > bar.maximum() // 2
