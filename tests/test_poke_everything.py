"""Exercise the real Qt event paths, with hardware and external actions fenced off.

No sleeps per control: process queued layout/deletion callbacks after each gesture.
Keep the shared responsive window fixture so new application pages join this sweep.
"""
import builtins
import io
import os
import re
import socket
import subprocess
import sys
import tempfile
import traceback
import webbrowser
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt, qInstallMessageHandler
from PySide6.QtGui import QColor, QDesktopServices, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (QAbstractButton, QApplication, QColorDialog, QComboBox,
                               QDialog, QFileDialog, QInputDialog, QMessageBox,
                               QScrollArea, QScrollBar, QWidget)
from shiboken6 import isValid

from tests.test_responsive import FullCtl, window  # noqa: F401 (shared pytest fixture)


@pytest.fixture(autouse=True)
def sandbox(monkeypatch, tmp_path, caplog, capfd, request):
    """Installed before window construction; restored only after window teardown."""
    errors = []
    context = {"gesture": "constructing windows", "cleanup": []}
    monkeypatch.setattr(sys, "excepthook", lambda *exc: errors.append(
        context["gesture"] + "\n" + "".join(traceback.format_exception(*exc))))
    messages = []
    previous = qInstallMessageHandler(lambda kind, ctx, msg: messages.append(msg))

    def forbidden(*args, **kwargs):
        errors.append(f"{context['gesture']}: external action attempted: {args!r}")
        raise AssertionError("External action attempted by UI robustness test")

    # Redirect configuration before importing/constructing any UI. Also replace already
    # imported aliases (other test modules may have imported them during collection).
    from whispa import ai_choice, autostart, config, welcome, voice_training
    from whispa.learn import TrainingStore

    original_appdata = os.environ.get("APPDATA")
    for key in ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "HOME", "USERPROFILE"):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("whispa"):
            for name in ("config_dir", "config_mod_dir"):
                if hasattr(module, name):
                    monkeypatch.setattr(module, name, lambda: tmp_path)
    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "config.json")
    monkeypatch.setattr(autostart, "is_enabled", lambda: False)
    monkeypatch.setattr(autostart, "set_enabled", forbidden)
    monkeypatch.setattr(ai_choice, "detect", lambda: ai_choice.Hardware())
    store = TrainingStore(tmp_path / "training")
    monkeypatch.setattr(FullCtl, "training_store", lambda self: store, raising=False)
    monkeypatch.setattr(FullCtl, "hotkey", SimpleNamespace(label=lambda: "Ctrl+Win"), raising=False)
    monkeypatch.setattr(FullCtl, "ai_available", lambda self: request.node.callspec.params["ai"])
    monkeypatch.setattr(welcome.LevelMeter, "open", lambda *a: None)
    monkeypatch.setattr(voice_training.VoiceTrainingPage, "_default_recorder", lambda self:
                        SimpleNamespace(start=lambda: None, stop=lambda: ([], 16000), level=0.0))

    # Permit fixture data writes only, and explicitly reject even reads of private data.
    def guarded_open(original):
        def open_file(file, mode="r", *args, **kwargs):
            if not isinstance(file, int):
                path = Path(file).resolve()
                private = Path(original_appdata) / "Whispa" if original_appdata else None
                assert private is None or not path.is_relative_to(private), "Private voice data access"
                if any(c in mode for c in "wax+"):
                    assert path.is_relative_to(tmp_path), f"Write outside sandbox: {path}"
            return original(file, mode, *args, **kwargs)
        return open_file

    monkeypatch.setattr(builtins, "open", guarded_open(builtins.open))
    monkeypatch.setattr(io, "open", guarded_open(io.open))
    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
    monkeypatch.setattr(webbrowser, "open_new", lambda *a, **k: False)
    monkeypatch.setattr(webbrowser, "open_new_tab", lambda *a, **k: False)
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda *a: False)
    for name in ("Popen", "run", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)
    for name in ("startfile", "system", "popen", "spawnl", "spawnv", "execv"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    monkeypatch.setattr(QDialog, "exec", lambda *a: QDialog.DialogCode.Rejected)
    monkeypatch.setattr(QMessageBox, "exec", lambda *a: QMessageBox.StandardButton.Cancel)
    for name in ("question", "warning", "information", "critical", "about", "aboutQt"):
        monkeypatch.setattr(QMessageBox, name, lambda *a, **k: QMessageBox.StandardButton.No)
    for cls in (QFileDialog, QInputDialog, QColorDialog):
        monkeypatch.setattr(cls, "exec", lambda *a: QDialog.DialogCode.Rejected)
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, lambda *a, **k: ("", ""))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: "")
    for name, value in (("getText", ""), ("getMultiLineText", ""), ("getItem", ""),
                        ("getInt", 0), ("getDouble", 0.0)):
        monkeypatch.setattr(QInputDialog, name, lambda *a, _value=value, **k: (_value, False))
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: QColor())
    try:
        yield context
    finally:
        app = QApplication.instance()
        if app:
            for widget in context["cleanup"]:
                if isValid(widget):
                    widget.deleteLater()
            drain(app)
        qInstallMessageHandler(previous)
        output = capfd.readouterr()
        diagnostics = "\n".join(messages + [caplog.text, output.out, output.err])
        assert not errors, "\n".join(errors)
        assert not re.search(r"RecursionError|maximum recursion|(?:C\+\+|Internal C\+\+).*deleted|"
                             r"deleted.*(?:C\+\+|object)", diagnostics, re.I), diagnostics


def drain(app):
    for _ in range(3):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def wheel_visible(app, root, context, counts):
    for target in [root, *root.findChildren(QWidget)]:
        if not isValid(target) or not target.isVisible():
            continue
        context["gesture"] = f"{root!r} at {root.size()}: wheel {target!r}"
        for delta in (-120, 120):
            if not isValid(target):
                break
            pos = target.rect().center()
            app.sendEvent(target, QWheelEvent(QPointF(pos), QPointF(target.mapToGlobal(pos)),
                          QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                          Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))
            counts["wheels"] += 1
            if isinstance(target, QScrollBar):
                counts["scrollbar_wheels"] += 1
            drain(app)


def sweep(app, root, page, restore, size, context, counts):
    """Re-query after mutations: refreshes can delete and rebuild whole widget trees."""
    restore()
    root.resize(*size)
    drain(app)
    wheel_visible(app, root, context, counts)

    # Include root navigation, and controls newly exposed by switches. Hold references
    # in seen so Python cannot reuse ids of deleted controls. Bound dynamic rebuilds.
    seen = set()
    for _ in range(500):
        restore()
        drain(app)
        candidates = [x for x in root.findChildren(QWidget)
                      if isinstance(x, (QAbstractButton, QComboBox)) and isValid(x)
                      and x.isVisible() and x.isEnabled() and x not in seen]
        if not candidates:
            break
        target = candidates[0]
        seen.add(target)
        context["gesture"] = f"{page!r} at {size}: activate {target!r}"
        # Scroll clipped controls into view before synthesising a real mouse click.
        ancestor = target.parentWidget()
        while ancestor is not None:
            if isinstance(ancestor, QScrollArea):
                ancestor.ensureWidgetVisible(target)
            ancestor = ancestor.parentWidget()
        drain(app)
        root.resize(size[0] + len(seen) % 2, size[1] + len(seen) % 2)
        if isinstance(target, QComboBox):
            target.showPopup()
            drain(app)
            popup = app.activePopupWidget()
            assert popup is not None, f"Combo popup did not open: {target!r}"
            wheel_visible(app, popup, context, counts)
            QTest.keyClick(target, Qt.Key.Key_Escape)
            target.hidePopup()
            counts["combos"] += 1
        else:
            QTest.mouseClick(target, Qt.MouseButton.LeftButton, pos=target.rect().center())
            counts["clicks"] += 1
        drain(app)
    else:
        pytest.fail(f"Unbounded control rebuilding on {page!r}")

    restore()
    root.activateWindow()
    root.setFocus()
    context["gesture"] = f"{page!r} at {size}: Tab / Escape"
    # Traverse at least one complete focus chain, with a cap against malformed chains.
    focus_seen = set()
    for _ in range(256):
        QTest.keyClick(app.focusWidget() or root, Qt.Key.Key_Tab)
        drain(app)
        focus = app.focusWidget()
        if focus in focus_seen:
            break
        focus_seen.add(focus)
    else:
        pytest.fail(f"Focus chain did not cycle on {page!r}")
    QTest.keyClick(app.focusWidget() or root, Qt.Key.Key_Escape)
    drain(app)


@pytest.mark.parametrize("ai", [False, True], ids=["without-ai", "with-ai"])
def test_poke_everything(window, sandbox, record_property, ai):
    from whispa.welcome import WelcomeDialog

    app, w = window
    sandbox["cleanup"].append(w)
    counts = dict(wheels=0, scrollbar_wheels=0, clicks=0, combos=0)
    guide = WelcomeDialog(w.ctl, mic_options=[("Test microphone", "test")])
    try:
        for size in ((520, 420), (900, 700), (1400, 900)):
            for index in range(w.pages.count()):
                def restore_main(i=index):
                    w.show()
                    w.show_page(i)
                sweep(app, w, w.pages.widget(index), restore_main, size, sandbox, counts)
            w.hide()
            for index in range(guide.stack.count()):
                def restore_guide(i=index):
                    guide.go(i)
                    guide.show()
                sweep(app, guide, guide.stack.widget(index), restore_guide, size, sandbox, counts)
            guide.hide()
        assert counts["scrollbar_wheels"] > 0, "Must exercise the 0.8.11 regression"
        assert counts["clicks"] > 0 and counts["combos"] > 0
        record_property("interactions", str(counts))
    finally:
        guide._timer.stop()
        guide.close()
        guide.deleteLater()
        w.close()
        drain(app)
