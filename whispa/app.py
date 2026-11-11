"""Wires hotkey -> microphone -> Parakeet -> text clean-up -> paste, with the overlay and tray."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PySide6.QtCore import QLockFile, QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QGuiApplication
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from . import autostart, focus, textproc
from . import config as config_mod
from .asr import Transcriber
from .audio import Recorder
from .card import CopyCard
from .learn import CorrectionWatcher, Dictionary, TrainingStore
from .llm import Polisher
from .hotkey import Hotkey, HotkeyService, PTTStateMachine
from .icons import app_icon, tray_icon
from .inject import Injector
from .overlay import Overlay
from .sounds import Sounds

log = logging.getLogger("whispa")


class Bridge(QObject):
    """Signals let the keyboard-hook and worker threads talk to the Qt main thread."""

    start = Signal()
    stop = Signal()
    cancel = Signal(bool)
    lock = Signal()
    done = Signal(str)
    failed = Signal(str)
    model_ready = Signal(str)
    model_failed = Signal(str)
    show_card = Signal(str)
    learned = Signal(str)
    captured = Signal(str, object, object)  # kind, spec, error
    history_changed = Signal()


class WhispaApp:
    def __init__(self, qapp: QApplication, cfg: config_mod.Config):
        self.qapp = qapp
        self.cfg = cfg
        self.bridge = Bridge()
        self.recording = False
        self.pending = 0
        self.history: list[str] = []
        self._focus = focus.Focus()
        self._before: str | None = None
        self.window = None
        self._status_text = "Loading speech model…"
        self.history_path = config_mod.config_dir() / "history.jsonl"

        try:
            hotkey = Hotkey.parse(cfg.hotkey)
        except ValueError as exc:
            log.error("%s Falling back to %s", exc, config_mod.default_hotkey())
            hotkey = Hotkey.parse(config_mod.default_hotkey())
        self.hotkey = hotkey

        self.sounds = Sounds(cfg.sounds, cfg.sound_volume)
        self.recorder = Recorder(cfg.input_device, keep_open=cfg.keep_mic_open)
        self.transcriber = Transcriber(cfg.model, cfg.device, cfg.quantization)
        self.sm = PTTStateMachine(
            hotkey, hands_free_key=cfg.hands_free_key or None, double_tap=cfg.double_tap_to_lock,
            on_start=self.bridge.start.emit, on_stop=self.bridge.stop.emit,
            on_cancel=self.bridge.cancel.emit, on_lock=self.bridge.lock.emit)
        self.hotkeys = HotkeyService(self.sm)
        self.injector = Injector(cfg.paste_method, cfg.paste_shortcut, cfg.restore_clipboard,
                                 wait_for_release=self.hotkeys.wait_for_modifiers_released)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.polisher = Polisher(cfg.llm_cleanup)
        cdir = config_mod.config_dir()
        self.dictionary = Dictionary(cdir / "dictionary.json")
        self.training = TrainingStore(cdir / "training", cfg.training_max_mb)             if cfg.save_training_audio else None
        self.watcher = CorrectionWatcher(
            self.dictionary,
            on_learned=lambda cs: self.bridge.learned.emit(", ".join(c.meant for c in cs)),
            on_final=self.training.update_text if self.training else None)
        self.card = CopyCard()

        hint = f"Click or hold {hotkey.label()} to dictate"
        self.overlay = Overlay(lambda: self.recorder.level, hint, cfg.show_idle_pill)
        self.overlay.idle_clicked.connect(self.sm.toggle_hands_free)
        self.overlay.stop_clicked.connect(self.sm.finish)
        self.overlay.cancel_clicked.connect(self.sm.cancel)

        b = self.bridge
        b.start.connect(self._on_start)
        b.stop.connect(self._on_stop)
        b.cancel.connect(self._on_cancel)
        b.lock.connect(lambda: self.overlay.set_locked(True))
        b.done.connect(self._on_done)
        b.failed.connect(self._on_failed)
        b.model_ready.connect(self._on_model_ready)
        b.model_failed.connect(self._on_model_failed)
        b.show_card.connect(self.card.show_text)
        b.captured.connect(self._on_captured)
        b.history_changed.connect(lambda: self.window and self.window.history_changed())
        b.learned.connect(lambda words: self.overlay.flash(f"Learned: {words}", seconds=2.5))

        self.max_timer = QTimer()
        self.max_timer.setInterval(250)
        self.max_timer.timeout.connect(self._check_max_duration)

        self._build_tray()
        self.hotkeys.start()
        threading.Thread(target=self._load_model, name="model-load", daemon=True).start()

    # -- model -----------------------------------------------------------
    def _load_model(self) -> None:
        focus.inspect()  # warm up the accessibility client (first call is slow)
        try:
            self.transcriber.load()
            self.bridge.model_ready.emit(self.transcriber.backend)
        except Exception as exc:  # noqa: BLE001
            log.exception("Model failed to load")
            self.bridge.model_failed.emit(str(exc))
            return
        try:
            self.polisher.warm_up()
            log.info("LLM polish: %s", "on (" + self.cfg.llm_cleanup.model + ")"
                     if self.polisher.active() else "off (no local LLM found)")
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM warm-up failed: %s", exc)

    def _on_model_ready(self, backend: str) -> None:
        self._status(f"Ready · Parakeet on {backend}")
        self.tray.showMessage("Whispa is ready",
                              f"Hold {self.hotkey.label()} and speak. Running locally on {backend}.",
                              tray_icon(), 4000)

    def _on_model_failed(self, err: str) -> None:
        self._status("Model failed to load")
        self.overlay.flash("Speech model failed to load", error=True, seconds=4)
        self.tray.showMessage("Whispa: model failed to load", err[:300],
                              QSystemTrayIcon.MessageIcon.Critical, 8000)

    # -- recording lifecycle ----------------------------------------------
    def _on_start(self) -> None:
        if self.recording:
            return
        try:
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001
            log.error("Mic start failed: %s", exc)
            self.sm.reset()
            self.sounds.play("error")
            self.overlay.flash("Microphone unavailable", error=True)
            return
        self.recording = True
        self.overlay.set_state("listening")
        self.tray.setIcon(tray_icon(True))
        self.sounds.play("start")
        self.max_timer.start()

    def _end_recording(self):
        self.recording = False
        self.max_timer.stop()
        self.tray.setIcon(tray_icon(False))
        return self.recorder.stop()

    def _on_stop(self) -> None:
        if not self.recording:
            return
        audio, sr = self._end_recording()
        if len(audio) < sr * self.cfg.min_duration_ms / 1000:
            self.overlay.set_state("transcribing" if self.pending else "idle")
            return
        self.sounds.play("stop")
        self.pending += 1
        self.overlay.set_state("transcribing")
        self.worker.submit(self._process, audio, sr)

    def _on_cancel(self, silent: bool) -> None:
        if not self.recording:
            return
        self._end_recording()
        self.overlay.set_state("transcribing" if self.pending else "idle")
        if not silent:
            self.sounds.play("cancel")

    def _check_max_duration(self) -> None:
        if self.recording and self.recorder.duration >= self.cfg.max_duration_s:
            self.sm.finish()

    # -- worker thread ---------------------------------------------------
    def _process(self, audio, sr) -> None:
        try:
            if not self.transcriber.ready.wait(timeout=300):
                raise RuntimeError("Speech model is still loading")
            t0 = time.perf_counter()
            raw = self.transcriber.transcribe(audio, sr)
            t_asr = time.perf_counter() - t0
            text = self._clean(raw) if raw else ""
            log.info("%.1fs audio -> %d chars, ASR %.0f ms, total %.0f ms (%s)", len(audio) / sr,
                     len(text), t_asr * 1000, (time.perf_counter() - t0) * 1000,
                     self.transcriber.backend)
            if text:
                self._deliver(text)
                sample = self.training.add(audio, sr, text) if self.training else None
                self._remember(raw, text, len(audio) / sr)
                self._watch_edits(text, sample)
            self.bridge.done.emit(text)
        except Exception as exc:  # noqa: BLE001
            log.exception("Transcription failed")
            self.bridge.failed.emit(str(exc))

    def _clean(self, raw: str) -> str:
        self.hotkeys.wait_for_modifiers_released()
        self._focus = focus.inspect()
        terms, corrections = self.dictionary.snapshot()
        replacements = {**corrections, **self.cfg.replacements}
        text = textproc.process(raw, fillers=self.cfg.remove_fillers,
                                commands=self.cfg.voice_commands,
                                replacements=replacements, terms=terms)
        text = self.polisher.polish(text, app=self._focus.app, terms=terms)
        # Corrections again: the LLM may have reintroduced a misspelling.
        text = textproc.apply_replacements(text, replacements)
        # Never end with a newline: some apps treat it as Enter and send the message.
        return text.strip()

    def _deliver(self, text: str) -> None:
        f = self._focus
        if f.is_text is False and self.cfg.no_text_field == "popup":
            log.info("No text field focused (%s); showing copy card", f.app or "?")
            self.injector.copy(text)
            self.bridge.show_card.emit(text)
            self._before = None  # nothing to watch for edits
            return
        self._before = f.read_text() if self.cfg.learn_from_edits else None
        out = text + (" " if self.cfg.append_space else "")
        self.injector.inject(out)

    def _watch_edits(self, text: str, sample: str | None) -> None:
        if self.cfg.learn_from_edits and self._focus.readable and self._before is not None:
            self.watcher.watch(self._focus, self._before, text, sample)

    def _remember(self, raw: str, text: str, seconds: float) -> None:
        self.history.append(text)
        del self.history[:-10]
        if self.cfg.save_history:
            try:
                with self.history_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"t": time.time(), "seconds": round(seconds, 2),
                                        "raw": raw, "text": text, "app": self._focus.app},
                                       ensure_ascii=False) + "\n")
            except OSError as exc:
                log.warning("Could not write history: %s", exc)
            self.bridge.history_changed.emit()

    def _on_done(self, text: str) -> None:
        self.pending -= 1
        if not self.recording:
            if text:
                self.overlay.set_state("transcribing" if self.pending else "idle")
            else:
                self.overlay.set_state("idle")
                self.overlay.flash("Didn't catch that")

    def _on_failed(self, err: str) -> None:
        self.pending -= 1
        if not self.recording:
            self.overlay.set_state("idle")
        self.sounds.play("error")
        self.overlay.flash(err[:60] or "Transcription failed", error=True, seconds=3)

    # -- tray ------------------------------------------------------------
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(tray_icon(), self.qapp)
        menu = QMenu()
        menu.addAction("Open Whispa", self.show_window)
        self.status_action = QAction("Loading speech model…", menu)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)
        menu.addSeparator()
        menu.addAction("Start hands-free dictation", self.sm.toggle_hands_free)
        self.recent_menu = menu.addMenu("Copy recent transcript")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        menu.addSeparator()
        menu.addAction("Settings…", lambda: self.show_window(page=2))
        menu.addAction("Restart Whispa", self.restart)
        menu.addAction("Quit Whispa", self.quit)
        self._menu = menu
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Whispa")
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick) and sys.platform != "darwin":
            self.show_window()

    def _status(self, text: str) -> None:
        self._status_text = text
        self.status_action.setText(text)
        self.tray.setToolTip(f"Whispa — {text}")
        if self.window is not None and self.window.isVisible():
            self.window.refresh()

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        if not self.history:
            a = self.recent_menu.addAction("Nothing yet")
            a.setEnabled(False)
        for text in reversed(self.history):
            label = text.replace("\n", " ")
            label = label if len(label) <= 60 else label[:57] + "…"
            self.recent_menu.addAction(label, lambda t=text: QGuiApplication.clipboard().setText(t))

    # -- window ----------------------------------------------------------
    def show_window(self, page: int | None = None) -> None:
        if self.window is None:
            from .window import MainWindow

            self.window = MainWindow(self)
        if page is not None:
            self.window.show_page(page)
        self.window.bring_to_front()

    # Controller interface used by the window --------------------------------
    def status_text(self) -> str:
        return self._status_text

    def llm_status(self) -> tuple[bool, str]:
        c = self.cfg.llm_cleanup
        if c.enabled in (False, "off"):
            return False, "Off."
        self.polisher.available = None  # re-check now
        if self.polisher.active(wait=True):
            return True, f"Active · {c.model} via {c.provider}"
        return False, ("Not available. Install Ollama (winget install Ollama.Ollama) and run "
                       f"“ollama pull {c.model}”; it switches on by itself.")

    def capture_hotkey(self, kind: str) -> None:
        single = kind == "hands_free_key"
        self.hotkeys.capture(lambda spec, err: self.bridge.captured.emit(kind, spec, err),
                             single=single)

    def cancel_capture(self) -> None:
        self.hotkeys.cancel_capture()

    def _on_captured(self, kind: str, spec, err) -> None:
        if self.window is not None:
            self.window.settings.capture_finished(kind, spec, err)

    def update_setting(self, key: str, value) -> str | None:
        """Apply a setting from the window immediately and save it. Returns an error message."""
        cfg = self.cfg
        if key == "hotkey":
            try:
                hk = Hotkey.parse(value)
            except ValueError as exc:
                return str(exc)
            self.hotkey = hk
            self.hotkeys.set_hotkey(hk, cfg.hands_free_key or None)
            self.overlay.hint = f"Click or hold {hk.label()} to dictate"
        elif key == "hands_free_key":
            if value and value in self.hotkey.keys:
                return "The hands-free key can't be part of the push-to-talk shortcut."
            self.hotkeys.set_hotkey(self.hotkey, value or None)
        elif key == "launch_at_login":
            try:
                autostart.set_enabled(bool(value))
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't change launch at login: {exc}"
            return None
        elif key == "llm_enabled":
            cfg.llm_cleanup.enabled = value
            self.polisher.available = None
            config_mod.save(cfg)
            return None
        elif key == "sounds":
            self.sounds.enabled = bool(value)
        elif key == "show_idle_pill":
            self.overlay.show_idle = bool(value)
            self.overlay.set_state(self.overlay.state)
            self.overlay._ensure_running()
        elif key == "input_device":
            self.recorder.device = value
            if cfg.keep_mic_open and not self.recording:
                self.recorder.shutdown()
                self._safe_open_mic()
        elif key == "keep_mic_open":
            self.recorder.keep_open = bool(value)
            if not self.recording:
                self.recorder.shutdown()
                if value:
                    self._safe_open_mic()
        elif key == "save_training_audio":
            self.training = TrainingStore(config_mod.config_dir() / "training", cfg.training_max_mb) \
                if value else None
            self.watcher.on_final = self.training.update_text if self.training else None
        elif key == "double_tap_to_lock":
            self.sm.double_tap = bool(value)
        if not hasattr(cfg, key):
            return f"Unknown setting {key}"
        setattr(cfg, key, value)
        config_mod.save(cfg)
        log.info("Setting %s = %r", key, value)
        return None

    def _safe_open_mic(self) -> None:
        try:
            self.recorder._open()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open mic: %s", exc)

    def delete_history(self, t: float) -> None:
        try:
            lines = self.history_path.read_text(encoding="utf-8").splitlines()
            keep = [ln for ln in lines if not ln.strip() or json.loads(ln).get("t") != t]
            self.history_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Could not delete history entry: %s", exc)

    def open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_mod.config_dir())))

    # -- lifecycle -------------------------------------------------------
    def restart(self) -> None:
        import subprocess

        from .shortcuts import launcher

        background = self.window is None or not self.window.isVisible()
        self.shutdown()
        exe, extra = launcher()  # same windowless launch as the Start Menu shortcut
        args = [exe, *extra]
        if background:
            args.append("--background")
        self.instance.release()
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) if sys.platform == "win32" else 0
        subprocess.Popen(args, creationflags=flags, start_new_session=sys.platform != "win32")
        self.qapp.quit()

    def shutdown(self) -> None:
        self.hotkeys.stop()
        self.recorder.shutdown()
        self.tray.hide()

    def quit(self) -> None:
        self.shutdown()
        self.qapp.quit()


class SingleInstance:
    """One Whispa per user. Launching it again just opens the running one's window."""

    NAME = "whispa-" + (os.environ.get("USERNAME") or os.environ.get("USER") or "user")

    def __init__(self):
        self.server: QLocalServer | None = None
        self.lock = QLockFile(str(config_mod.config_dir() / "whispa.lock"))
        self.lock.setStaleLockTime(0)

    def notify_running(self) -> bool:
        """True if another instance is running (and was asked to show itself)."""
        sock = QLocalSocket()
        sock.connectToServer(self.NAME)
        if sock.waitForConnected(400):
            sock.write(b"show\n")
            sock.flush()
            sock.waitForBytesWritten(400)
            sock.disconnectFromServer()
            return True
        # An older Whispa without the socket may still hold the lock.
        return not self.lock.tryLock(200)

    def listen(self, on_show) -> None:
        QLocalServer.removeServer(self.NAME)
        self.server = QLocalServer()
        self.server.newConnection.connect(lambda: self._accept(on_show))
        if not self.server.listen(self.NAME):
            log.warning("Single-instance server failed: %s", self.server.errorString())

    def _accept(self, on_show) -> None:
        while self.server.hasPendingConnections():
            conn = self.server.nextPendingConnection()
            conn.readyRead.connect(lambda c=conn: (c.readAll(), on_show()))

    def release(self) -> None:
        if self.server:
            self.server.close()
        self.lock.unlock()


def ensure_std_streams() -> None:
    """Give a console-less process (pythonw, Start Menu shortcut) real stdout/stderr.

    Without them, native libraries (ONNX Runtime, CUDA) that print warnings write to an
    invalid handle, and the C runtime aborts the whole process. Everything goes to
    stdio.log instead, plus Python-level crash dumps via faulthandler.
    """
    import faulthandler

    def valid(fd: int) -> bool:
        try:
            os.fstat(fd)
            return True
        except OSError:
            return False

    if all(valid(fd) for fd in (1, 2)) and sys.stdout is not None and sys.stderr is not None:
        return
    path = config_mod.config_dir() / "stdio.log"
    try:
        if path.exists() and path.stat().st_size > 2_000_000:
            path.unlink()
    except OSError:
        pass
    f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")  # noqa: SIM115
    for fd, name in ((1, "stdout"), (2, "stderr")):
        os.dup2(f.fileno(), fd)
        setattr(sys, name, f)
    if sys.platform == "win32":
        import ctypes
        import msvcrt

        k32 = ctypes.windll.kernel32
        for std, fd in ((-11, 1), (-12, 2)):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            k32.SetStdHandle(std, msvcrt.get_osfhandle(fd))
    faulthandler.enable(f)


def setup_logging() -> None:
    from logging.handlers import RotatingFileHandler

    handlers: list[logging.Handler] = [RotatingFileHandler(
        config_mod.config_dir() / "whispa.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")]
    if sys.stderr is not None and sys.stderr.isatty():  # console runs only
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("comtypes").setLevel(logging.WARNING)


def run(background: bool = False) -> int:
    ensure_std_streams()
    setup_logging()
    if sys.platform == "win32":
        try:  # own taskbar identity/icon instead of python.exe's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Whispa.Dictation")
        except Exception:  # noqa: BLE001
            pass
    cfg = config_mod.load()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Whispa")
    qapp.setWindowIcon(app_icon())
    qapp.setQuitOnLastWindowClosed(False)

    instance = SingleInstance()
    if instance.notify_running():
        log.info("Whispa is already running; asked it to show its window")
        if not instance.server and instance.lock.error() != QLockFile.LockError.NoError:
            QMessageBox.information(None, "Whispa", "Whispa is already running. Quit the old "
                                    "copy from its tray icon, then start Whispa again.")
        return 0

    if sys.platform == "darwin":
        try:  # menu-bar app: no Dock icon
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

            NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:  # noqa: BLE001
            pass

    app = WhispaApp(qapp, cfg)
    app.instance = instance
    instance.listen(app.show_window)
    if not background:
        app.show_window()
    log.info("Whispa started; hotkey %s, config %s", app.hotkey.label(), config_mod.config_path())
    code = qapp.exec()
    app.worker.shutdown(wait=False, cancel_futures=True)
    return code
