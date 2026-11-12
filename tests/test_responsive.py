"""The window works at any size: nothing is cut off when it's narrow, and Settings can
jump straight to a section (e.g. Updates from the tray menu)."""
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QScrollArea, QWidget  # noqa: E402

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


def test_scrolling_past_a_dropdown_never_changes_it(window):
    """The mouse wheel over Settings scrolls the page; it doesn't flip the microphone or the
    dictation mode the pointer passes over (that switched people to Email mode by accident)."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QComboBox

    app, w = window
    w.resize(900, 600)
    w.show_page(w.pages.indexOf(w.settings))
    settle(app)
    combos = [c for c in w.settings.findChildren(QComboBox) if c.count() > 1]
    assert combos
    bar = w.settings.verticalScrollBar()
    for combo in combos:
        before, top = combo.currentIndex(), bar.value()
        ev = QWheelEvent(QPointF(5, 5), QPointF(combo.mapToGlobal(QPoint(5, 5))), QPoint(0, 0),
                         QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(combo, ev)
        assert combo.currentIndex() == before, combo
    assert bar.value() >= top  # the page scrolled instead (or was already at the bottom)


def test_wheel_over_every_page_and_scrollbar_is_safe(window):
    """Scrolling over any page, its scrollbar, or a control on it never recurses (0.8.11
    guarded the scrollbars themselves, which forwarded the wheel to themselves forever)."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QAbstractScrollArea, QAbstractSlider, QComboBox, QScrollBar

    def wheel(target):
        ev = QWheelEvent(QPointF(3, 3), QPointF(target.mapToGlobal(QPoint(3, 3))), QPoint(0, 0),
                         QPoint(0, -120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        app.sendEvent(target, ev)

    app, w = window
    w.resize(700, 420)  # small, so most pages scroll
    for i in range(w.pages.count()):
        w.show_page(i)
        settle(app, 5)
        page = w.pages.widget(i)
        targets = [x for x in page.findChildren(QWidget)
                   if isinstance(x, (QScrollBar, QComboBox, QAbstractSlider, QAbstractScrollArea))
                   and x.isVisible()]
        for t in targets:
            wheel(t)
    assert not any(isinstance(x, QScrollBar) and x.property("_wheel_guarded")
                   for x in w.findChildren(QWidget))
