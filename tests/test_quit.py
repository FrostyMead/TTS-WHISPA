"""Quitting or restarting always ends Murmr, and a new copy ends any old one that didn't."""
import os
import sys
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_qapp = QApplication.instance() or QApplication([])

from whispa import app as app_mod  # noqa: E402
from whispa.app import MurmrApp  # noqa: E402


class StaysInTray(QWidget):
    """Like Murmr's window: closing it only hides it."""

    def closeEvent(self, event):
        event.ignore()
        self.hide()


def _app(monkeypatch):
    backstops = []
    monkeypatch.setattr(app_mod, "exit_backstop", lambda delay=5.0: backstops.append(delay))
    m = mock.MagicMock()
    m.qapp = _qapp
    m._exit = lambda: MurmrApp._exit(m)
    return m, backstops


def _run_loop(action):
    QTimer.singleShot(0, action)
    QTimer.singleShot(2000, lambda: _qapp.exit(99))  # the quit didn't happen
    return _qapp.exec()


def test_quit_ends_the_event_loop_with_the_window_open(monkeypatch):
    m, backstops = _app(monkeypatch)
    w = StaysInTray()
    w.show()
    try:
        assert _run_loop(lambda: MurmrApp.quit(m)) == 0
    finally:
        w.deleteLater()
    m.shutdown.assert_called_once()
    m.overlay.hide.assert_called_once()  # the pill goes at once
    assert backstops == [5.0]  # and the process ends even if something holds it up


def test_quit_still_exits_when_shutting_down_fails(monkeypatch):
    m, backstops = _app(monkeypatch)
    m.shutdown.side_effect = RuntimeError("audio driver")
    code = []

    def go():
        try:
            MurmrApp.quit(m)
        except RuntimeError:
            code.append("raised")

    assert _run_loop(go) == 0
    assert code == ["raised"] and backstops


def test_restart_exits_even_if_the_new_copy_cant_start(monkeypatch):
    m, backstops = _app(monkeypatch)
    m.window = None
    with mock.patch("whispa.shortcuts.mac_app_usable", return_value=False), \
            mock.patch("whispa.shortcuts.launcher", return_value=("pythonw.exe", ["boot.py"])), \
            mock.patch("subprocess.Popen", side_effect=OSError("no such file")) as popen:
        assert _run_loop(lambda: MurmrApp.restart(m)) == 0
    popen.assert_called_once()
    m.instance.release.assert_called_once()
    assert backstops


def test_restart_starts_the_new_copy_then_exits(monkeypatch):
    m, _ = _app(monkeypatch)
    m.window = None
    with mock.patch("whispa.shortcuts.mac_app_usable", return_value=False), \
            mock.patch("whispa.shortcuts.launcher", return_value=("pythonw.exe", ["boot.py"])), \
            mock.patch("subprocess.Popen") as popen:
        assert _run_loop(lambda: MurmrApp.restart(m)) == 0
    assert popen.call_args.args[0] == ["pythonw.exe", "boot.py", "--background"]


# --- leftover copies -----------------------------------------------------------------------
BOOT = r"C:\Users\me\AppData\Local\Whispa\boot.py" if sys.platform == "win32" else \
    "/Users/me/Library/Application Support/Murmr/boot.py"


def test_only_other_copies_of_this_install_are_leftovers():
    procs = [
        (10, f'"C:\\py\\pythonw.exe" -I -S {BOOT}'),  # an old copy
        (11, f"pythonw.exe -I -S {BOOT.upper() if sys.platform == 'win32' else BOOT}"),
        (20, f"pythonw.exe -I -S {BOOT}"),  # this one
        (30, "python.exe -m pytest"),  # something else
        (31, "python.exe C:\\other\\boot.py"),  # another app's bootstrap
        (32, ""),  # no command line visible
    ]
    assert app_mod.leftover_pids(procs, BOOT, {20}) == [10, 11]


def test_development_runs_end_nothing(monkeypatch):
    monkeypatch.setattr("whispa.updater.runtime_dir", lambda: None)
    with mock.patch.object(app_mod, "_python_processes") as listing, \
            mock.patch("os.kill") as kill:
        assert app_mod.end_leftover_copies() == 0
    listing.assert_not_called()
    kill.assert_not_called()


def test_leftover_copies_are_ended(monkeypatch, tmp_path):
    monkeypatch.setattr("whispa.updater.runtime_dir", lambda: tmp_path)
    boot = str(tmp_path / "boot.py")
    procs = [(os.getpid(), f"pythonw -I -S {boot}"), (999_999, f"pythonw -I -S {boot}"),
             (999_998, "python something_else.py")]
    with mock.patch.object(app_mod, "_python_processes", return_value=procs), \
            mock.patch("os.kill") as kill:
        assert app_mod.end_leftover_copies() == 1
    assert kill.call_args.args[0] == 999_999


def test_listing_processes_works_here():
    rows = app_mod._python_processes()
    assert any(pid == os.getpid() for pid, _ in rows)  # pytest itself is a Python process
