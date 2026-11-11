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

from . import autostart, focus, styles, textproc
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
from .streaming import LiveSession

log = logging.getLogger("whispa")


class Bridge(QObject):
    """Signals let the keyboard-hook and worker threads talk to the Qt main thread."""

    start = Signal(str)  # mode: "dictate" | "command"
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
    update_notice = Signal(str)
    update_restart = Signal()
    history_changed = Signal()
    update_changed = Signal()
    update_auto = Signal()  # an update is downloading for an automatic restart
    paste_last = Signal()
    read_aloud = Signal()
    speak_state = Signal(str)
    meeting_changed = Signal(str)  # meeting id (saved, summary progress or done)
    ai_setup = Signal(str, float)  # message, fraction (-1: unknown); "done:" / "error:" prefixes
    tuning_changed = Signal(str)  # job kind
    notice = Signal(str, bool)  # message, is_error
    sync_event = Signal(str, object)  # kind, info (sync.py)


class MurmrApp:
    def __init__(self, qapp: QApplication, cfg: config_mod.Config):
        self.qapp = qapp
        self.cfg = cfg
        self.bridge = Bridge()
        self.recording = False
        self.pending = 0
        self.live: LiveSession | None = None  # transcribes long dictations while you talk
        self._context_names: list[str] = []  # names seen on screen when the dictation started
        self._speaker = None
        self._focus = focus.Focus()
        self._before: str | None = None
        self.window = None
        self._status_text = "Loading speech model…"
        self.history_path = config_mod.config_dir() / "history.jsonl"
        self.history: list[str] = self._recent_history()

        try:
            hotkey = Hotkey.parse(cfg.hotkey)
        except ValueError as exc:
            log.error("%s Falling back to %s", exc, config_mod.default_hotkey())
            hotkey = Hotkey.parse(config_mod.default_hotkey())
        self.hotkey = hotkey

        self.sounds = Sounds(cfg.sounds, cfg.sound_volume)
        self.recorder = Recorder(cfg.input_device, keep_open=cfg.keep_mic_open)
        self.recorder.limit_s = min(float(cfg.max_duration_s) + 5.0, 3600.0)
        self.transcriber = Transcriber(cfg.model, cfg.device, cfg.quantization)
        self.sm = PTTStateMachine(
            hotkey, hands_free_key=cfg.hands_free_key or None, double_tap=cfg.double_tap_to_lock,
            command_hotkey=self._command_hotkey(cfg, hotkey),
            on_start=lambda: self.bridge.start.emit(self.sm.mode), on_stop=self.bridge.stop.emit,
            on_cancel=self.bridge.cancel.emit, on_lock=self.bridge.lock.emit)
        self.hotkeys = HotkeyService(self.sm)
        self.injector = Injector(cfg.paste_method, cfg.paste_shortcut, cfg.restore_clipboard,
                                 wait_for_release=self.hotkeys.wait_for_modifiers_released)
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.polisher = Polisher(cfg.llm_cleanup)
        cdir = config_mod.config_dir()
        from .privacy import finish_pending_delete

        finish_pending_delete()
        threading.Thread(target=self.managed_reason, name="managed-check", daemon=True).start()
        self.dictionary = Dictionary(cdir / "dictionary.json")
        from . import capabilities

        # No training on this computer (macOS): don't keep dictation audio nothing can use.
        self.training = (TrainingStore(cdir / "training", cfg.training_max_mb)
                         if cfg.save_training_audio and capabilities.can_train() else None)
        self.watcher = CorrectionWatcher(
            self.dictionary,
            on_learned=lambda cs: self.bridge.learned.emit(", ".join(c.meant for c in cs)),
            on_final=self._review_sample)
        self.mode = "dictate"
        self.card = CopyCard()

        self.overlay = Overlay(lambda: self.recorder.level, self._hint(hotkey), cfg.show_idle_pill,
                               anchor=cfg.pill_anchor)
        self.overlay.anchor_changed.connect(lambda a: self.update_setting("pill_anchor", a))
        self.overlay.idle_clicked.connect(self.sm.toggle_hands_free)
        self.overlay.stop_clicked.connect(self.sm.finish)
        self.overlay.cancel_clicked.connect(self.sm.cancel)
        self.overlay.speak_level_fn = lambda: self._speaker.level if self._speaker else 0.0
        self.overlay.speak_pause_clicked.connect(lambda: self._speaker and self._speaker.toggle_pause())
        self.overlay.speak_skip_clicked.connect(lambda: self._speaker and self._speaker.skip())
        self.overlay.speak_stop_clicked.connect(lambda: self._speaker and self._speaker.stop())

        b = self.bridge
        b.start.connect(self._on_start)
        b.stop.connect(self._on_stop)
        b.cancel.connect(self._on_cancel)
        b.lock.connect(lambda: self.overlay.set_locked(True))
        b.done.connect(self._on_done)
        b.failed.connect(self._on_failed)
        b.model_ready.connect(self._on_model_ready)
        b.model_failed.connect(self._on_model_failed)
        b.show_card.connect(lambda text: self.card.show_text(text, self.overlay.anchor))
        b.captured.connect(self._on_captured)
        b.history_changed.connect(lambda: self.window and self.window.history_changed())
        b.learned.connect(lambda words: self.overlay.flash(f"Learned: {words}", seconds=2.5))
        b.paste_last.connect(self.paste_last)
        b.read_aloud.connect(self.read_aloud)
        b.speak_state.connect(self._on_speak_state)
        b.meeting_changed.connect(self._on_meeting_changed)
        b.ai_setup.connect(self._on_ai_setup)
        b.tuning_changed.connect(self._on_tuning_changed)
        b.notice.connect(lambda msg, err: self.overlay.flash(msg, error=err, seconds=3))
        b.sync_event.connect(self._on_sync_event)
        self.sync = None
        if cfg.sync_enabled:
            self._start_sync()

        from .updater import Updater

        self.updater = Updater()
        self._update_busy = ""
        self._update_checked = False
        b.update_changed.connect(self._on_update_changed)
        b.update_notice.connect(self._notify_update)
        b.update_restart.connect(self.restart)
        b.update_auto.connect(lambda: self.auto_timer.isActive() or self.auto_timer.start())
        self.update_timer = QTimer()
        self.update_timer.setInterval(6 * 3600 * 1000)
        self.update_timer.timeout.connect(lambda: self.check_updates(quiet=True))
        self.auto_timer = QTimer()  # a staged update waits for a quiet moment to restart
        self.auto_timer.setInterval(60_000)
        self.auto_timer.timeout.connect(self._maybe_auto_restart)
        self._tray_click: tuple[int, str] | None = None  # where clicking a notification goes
        if self.updater.installed:
            QTimer.singleShot(8000, self.updater.mark_healthy)  # we started: keep this version
            QTimer.singleShot(6000, self._announce_update)
            if cfg.update_check:
                QTimer.singleShot(90_000, lambda: self.check_updates(quiet=True))
                self.update_timer.start()

        self.max_timer = QTimer()
        self.max_timer.setInterval(250)
        self.max_timer.timeout.connect(self._check_max_duration)
        self._tuning: dict = {}  # kind -> tuning.Job (training in WSL)
        self.tune_timer = QTimer()  # overnight training check
        self.tune_timer.setInterval(10 * 60 * 1000)
        self.tune_timer.timeout.connect(self._maybe_auto_tune)
        self.tune_timer.start()
        self._meeting = None  # the running MeetingSession (notes or captions-only)
        self._summary_state: dict[str, str] = {}
        self._captions = None
        self._captions_on = False
        self.captions_timer = QTimer()
        self.captions_timer.setInterval(400)
        self.captions_timer.timeout.connect(self._update_captions)
        self.live_timer = QTimer()
        self.live_timer.setInterval(2000)
        self.live_timer.timeout.connect(self._poll_live)
        self._apply_action_hotkeys()

        self._start_watchdog()
        self._build_tray()
        self.hotkeys.start()
        threading.Thread(target=self._load_model, name="model-load", daemon=True).start()
        self._welcome = None
        self._prune_history()
        self.history_timer = QTimer()
        self.history_timer.setInterval(24 * 3600 * 1000)
        self.history_timer.timeout.connect(self._prune_history)
        self.history_timer.start()
        if not cfg.welcome_done:
            QTimer.singleShot(900, self.show_welcome)  # it asks for the macOS permissions too
        elif sys.platform == "darwin":
            QTimer.singleShot(1500, self._check_mac_permissions)

    def ai_available(self) -> bool:
        """The local AI is installed: Command Mode, AI clean-up and meeting summaries work."""
        from . import capabilities

        return capabilities.local_ai_installed(self.cfg.llm_cleanup) or \
            bool(getattr(self, "polisher", None) and self.polisher.available)

    def _command_hotkey(self, cfg, dictate: Hotkey) -> Hotkey | None:
        """Command Mode's shortcut, only if the local AI is there to carry out commands."""
        from . import capabilities

        if not capabilities.local_ai_installed(cfg.llm_cleanup):
            return None
        return self._parse_command_hotkey(cfg.command_hotkey, dictate)

    # -- tune to my voice (tuning.py) --------------------------------------
    def tuning_active(self) -> bool:
        return any(j.running for j in self._tuning.values())

    def tuning_job(self, kind: str):
        return self._tuning.get(kind)

    def recorded_minutes(self) -> dict[str, float]:
        from .tuning import recorded_minutes

        return recorded_minutes(config_mod.config_dir() / "training")

    def tuning_blocker(self) -> str:
        """Why tuning can't start here, or ""."""
        from . import capabilities
        from .tuning import wsl_status

        if not capabilities.can_train():
            return "Tuning needs a Windows PC with an NVIDIA graphics card."
        if not hasattr(self, "_wsl_status"):
            self._wsl_status = wsl_status()
        return self._wsl_status

    def start_tuning(self, kind: str, hours: float = 4.0) -> str | None:
        from .tuning import Job

        if self.tuning_active():
            return "Training is already running."
        blocker = self.tuning_blocker()
        if blocker:
            return blocker
        if kind == "voice" and not self.cfg.voice_clone:
            return "Switch on Settings > Your voice > Create a voice that sounds like me first."
        if kind == "recognition" and not self.cfg.save_training_audio:
            return "Switch on Settings > Your voice > Improve recognition with my voice first."
        self._free_gpu_for_training()
        job = Job(kind, on_change=lambda j: self.bridge.tuning_changed.emit(j.kind), hours=hours)
        self._tuning[kind] = job
        self._models_before = {p.name for p in self._models_dir().iterdir()} \
            if self._models_dir().exists() else set()
        if kind == "recognition":
            self.cfg.last_tune_minutes = self.recorded_minutes()["recognition"]
            config_mod.save(self.cfg)
        job.start()
        log.info("Tuning started: %s", kind)
        return None

    def cancel_tuning(self, kind: str) -> None:
        job = self._tuning.get(kind)
        if job and job.running:
            job.cancel()

    @staticmethod
    def _models_dir():
        return config_mod.config_dir() / "models"

    def _free_gpu_for_training(self) -> None:
        """Training needs most of the GPU: the local AI and the reading voice unload, and
        dictation moves to the CPU until it's done (slower, but it keeps working)."""
        if self._speaker is not None:
            self._speaker.unload()
        try:
            import urllib.request

            body = json.dumps({"model": self.cfg.llm_cleanup.model, "keep_alive": 0}).encode()
            urllib.request.urlopen(urllib.request.Request(
                self.cfg.llm_cleanup.base_url.rstrip("/") + "/api/generate", data=body,
                headers={"Content-Type": "application/json"}), timeout=5).close()
        except Exception:  # noqa: BLE001 - not running: nothing to unload
            pass
        if self.transcriber.backend != "CPU":
            self._status("Training on your voice · dictation on the CPU meanwhile")
            threading.Thread(target=self.transcriber.switch_device, args=("cpu",),
                             name="asr-to-cpu", daemon=True).start()

    def _on_tuning_changed(self, kind: str) -> None:
        job = self._tuning.get(kind)
        if job is not None and not job.running:
            if self.transcriber.device != self.cfg.device and not self.tuning_active():
                def back():
                    self.transcriber.switch_device(self.cfg.device)
                    self.bridge.model_ready.emit(self.transcriber.backend)

                threading.Thread(target=back, name="asr-back", daemon=True).start()
            if job.ok:
                new = ({p.name for p in self._models_dir().iterdir()} if self._models_dir().exists()
                       else set()) - getattr(self, "_models_before", set())
                if kind == "recognition":
                    msg = ("Your tuned speech model is ready: open Voice training to use it."
                           if new else "Tuning finished, but the result wasn't better than the "
                           "standard model, so it wasn't installed.")
                else:
                    msg = "A new round of your voice is ready: try it in Settings > Read aloud."
                self.tray.showMessage("Murmr training finished", msg, tray_icon(), 8000)
        vt = getattr(self.window, "voice_training", None) if self.window is not None else None
        if vt is not None and vt.tuning is not None:
            vt.tuning.refresh()

    def use_tuned_model(self, path: str) -> None:
        self.cfg.model = path
        config_mod.save(self.cfg)
        log.info("Switching to tuned model %s", path)
        self.restart()

    def _maybe_auto_tune(self) -> None:
        """Overnight training: between 1 and 6 am, when you've been away for 30 minutes."""
        if not self.cfg.auto_tune or self.tuning_active() or self.tuning_blocker():
            return
        hour = time.localtime().tm_hour
        away = time.monotonic() - self.hotkeys.last_activity > 30 * 60
        if not (1 <= hour < 6 and away and not self.recording and not self.meeting_running()):
            return
        from .tuning import RECOGNITION_MIN, VOICE_MIN

        mins = self.recorded_minutes()
        if self.cfg.save_training_audio and mins["recognition"] >= max(RECOGNITION_MIN, 30) and \
                mins["recognition"] - self.cfg.last_tune_minutes >= 15:
            self.start_tuning("recognition")
        elif self.cfg.voice_clone and mins["voice"] >= VOICE_MIN and \
                getattr(self, "_voice_round_day", None) != time.strftime("%Y-%m-%d"):
            self._voice_round_day = time.strftime("%Y-%m-%d")  # one round a night
            self.start_tuning("voice", hours=4)

    # -- one-click local AI (ai_setup.py) -----------------------------------
    def setup_ai(self) -> None:
        if getattr(self, "_ai_cancel", None) is not None:
            return  # already running
        from . import ai_setup

        self._ai_cancel = threading.Event()

        def progress(msg: str, frac: float | None) -> None:
            self.bridge.ai_setup.emit(msg, -1.0 if frac is None else frac)

        def work():
            try:
                model = ai_setup.setup(self.cfg.llm_cleanup, progress, self._ai_cancel)
                self.cfg.llm_cleanup.model = model
                if self.cfg.llm_cleanup.enabled in (False, "off"):
                    self.cfg.llm_cleanup.enabled = "auto"
                config_mod.save(self.cfg)
                self.polisher.available = None
                log.info("Local AI set up: %s", model)
                self.bridge.ai_setup.emit(f"done:Local AI ready ({model})", 1.0)
            except ai_setup.Cancelled:
                self.bridge.ai_setup.emit("error:Setup cancelled", -1.0)
            except Exception as exc:  # noqa: BLE001
                log.exception("Local AI setup failed")
                self.bridge.ai_setup.emit(f"error:Setup failed: {exc}", -1.0)
            finally:
                self._ai_cancel = None

        threading.Thread(target=work, name="ai-setup", daemon=True).start()

    def cancel_ai_setup(self) -> None:
        if getattr(self, "_ai_cancel", None) is not None:
            self._ai_cancel.set()

    def _on_ai_setup(self, msg: str, frac: float) -> None:
        if msg.startswith("done:"):
            from . import capabilities

            capabilities._ai_cache = (0.0, False)  # look again now
            self.refresh_capabilities()
            self.tray.showMessage("Local AI ready", "Command Mode, AI clean-up and meeting "
                                  "summaries are on.", tray_icon(), 6000)
        if self.window is not None:
            self.window.settings.ai_setup_progress(msg, frac)
        if self._welcome is not None and self._welcome.isVisible():
            self._welcome.ai_setup_progress(msg, frac)

    def refresh_capabilities(self) -> None:
        """Ollama installed (or removed) since start: switch Command Mode's shortcut to match."""
        want = self._command_hotkey(self.cfg, self.hotkey) if self.ai_available() else None
        if want != self.sm.command_hotkey:
            self.hotkeys.set_hotkey(self.hotkey, self.cfg.hands_free_key or None, want)
            log.info("Command Mode %s", "on" if want else "off (no local AI)")

    @staticmethod
    def _parse_command_hotkey(spec: str, dictate: Hotkey) -> Hotkey | None:
        if not spec:
            return None
        try:
            hk = Hotkey.parse(spec)
        except ValueError as exc:
            log.error("Command Mode hotkey %r: %s", spec, exc)
            return None
        return None if hk == dictate else hk

    def _review_sample(self, sample_id: str, corrections) -> None:
        if self.training is not None:
            self.training.record_review(sample_id, corrections)

    @staticmethod
    def _hint(hotkey: Hotkey) -> str:
        return f"Click or hold {hotkey.label()} to dictate · drag to move"

    # -- freeze watchdog (watchdog.py) --------------------------------------
    def _start_watchdog(self) -> None:
        """If the main thread (tray, window, pill) freezes: record why, then restart."""
        import faulthandler

        from .watchdog import Watchdog

        try:
            self._crash_file = open(config_mod.config_dir() / "crash.log", "a",  # noqa: SIM115
                                    encoding="utf-8")
        except OSError:
            self._crash_file = None

        def freeze(lag: float) -> None:
            log.error("Murmr has been frozen for %.0f s; thread stacks written to crash.log", lag)
            if self._crash_file:
                self._crash_file.write(f"\n=== frozen for {lag:.0f} s at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._crash_file.flush()
                faulthandler.dump_traceback(self._crash_file, all_threads=True)

        def restart(lag: float) -> None:
            log.error("Still frozen after %.0f s: restarting Murmr", lag)
            for h in logging.getLogger().handlers:
                h.flush()
            relaunch_detached()
            os._exit(3)

        self.watchdog = Watchdog(freeze, restart, dump_after=20, restart_after=85)
        self._heartbeat = QTimer()
        self._heartbeat.setInterval(1000)

        def beat() -> None:
            self.watchdog.beat()
            if self._crash_file and sys.platform != "darwin":
                # Last resort below Python's lock: if even the watchdog thread can't run,
                # the fault handler records every stack and ends the process. (Not on macOS:
                # its timer keeps counting while the Mac sleeps, so waking up would trip it.)
                faulthandler.dump_traceback_later(100, exit=True, file=self._crash_file)

        self._heartbeat.timeout.connect(beat)
        self._heartbeat.start()
        beat()
        self.watchdog.start()

    # -- macOS permissions ------------------------------------------------
    def _check_mac_permissions(self) -> None:
        from . import macperms

        action = macperms.fn_key_action()
        log.info("macOS permissions: %s; 🌐 key: %s", macperms.status(), action or "?")
        missing = macperms.missing()
        if not missing:
            if "fn" in self.hotkey.keys and action and action != "Do Nothing":
                self.tray.showMessage(
                    "Set the 🌐 key to Do Nothing",
                    f"Holding fn currently does “{action}”. System Settings > Keyboard > "
                    "Press 🌐 key to > Do Nothing.", tray_icon(), 10000)
                self.show_window(page=2)
            return
        log.info("macOS permissions missing: %s", ", ".join(missing))
        macperms.request(missing)  # macOS's prompts; they add Murmr to each list
        names = ", ".join(macperms.LABELS[k] for k in missing)
        self.tray.showMessage("Murmr needs your permission",
                              f"Allow Murmr under {names} in System Settings > Privacy & Security.",
                              tray_icon(), 10000)
        self.show_window(page=2)

    def permissions(self) -> dict[str, bool | None]:
        from . import macperms

        st = macperms.status()
        if self.hotkeys.keys_arriving:  # proof, whatever macOS's cached answer says
            st["input"] = True
        return st

    def allow_permission(self, kind: str) -> None:
        from . import macperms

        macperms.request([kind])
        macperms.open_settings(kind)

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
        self.tray.showMessage("Murmr is ready",
                              f"Hold {self.hotkey.label()} and speak. Running locally on {backend}.",
                              tray_icon(), 4000)

    def _on_model_failed(self, err: str) -> None:
        self._status("Model failed to load")
        self.overlay.flash("Speech model failed to load", error=True, seconds=4)
        self.tray.showMessage("Murmr: model failed to load", err[:300],
                              QSystemTrayIcon.MessageIcon.Critical, 8000)

    # -- recording lifecycle ----------------------------------------------
    def _on_start(self, mode: str = "dictate") -> None:
        if self.recording:
            return
        self.watcher.finish()  # a new dictation ends the edit-watch of the previous one
        if self._speaker is not None and self._speaker.speaking:
            self._speaker.stop()  # don't dictate over Murmr reading aloud (the mic would hear it)
        if mode == "command" and self.tuning_active():
            self.sm.reset()
            self.overlay.flash("Command Mode is paused while Murmr trains on your voice", seconds=3)
            return
        if mode == "command" and not self.polisher.reachable():
            self.sm.reset()
            self.sounds.play("error")
            self.overlay.flash("Command Mode needs the local AI (Ollama)", error=True, seconds=3)
            return
        self.mode = mode
        try:
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001
            log.error("Mic start failed: %s", exc)
            self.sm.reset()
            self.sounds.play("error")
            self.overlay.flash("Microphone unavailable", error=True)
            return
        self.recording = True
        self.polisher.prewarm()  # load the AI model while you talk (no-op if it's loaded)
        if mode == "dictate":
            self.live = LiveSession(self.transcriber.transcribe) if self.transcriber.ready.is_set() else None
            self.live_timer.start()
            self._context_names = []
            if self.cfg.screen_context:
                threading.Thread(target=self._grab_screen_names, name="screen-names", daemon=True).start()
        self.overlay.set_mode(mode)
        self.overlay.set_state("listening")
        self.tray.setIcon(tray_icon(True))
        self.sounds.play("start")
        self.max_timer.start()

    def _end_recording(self):
        self.recording = False
        self.max_timer.stop()
        self.live_timer.stop()
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
        live, self.live = self.live, None
        if self.mode == "command":
            self.worker.submit(self._process_command, audio, sr)
        else:
            self.worker.submit(self._process, audio, sr, live)

    def _on_cancel(self, silent: bool) -> None:
        if not self.recording:
            return
        self._end_recording()
        self.live = None
        self.overlay.set_state("transcribing" if self.pending else "idle")
        if not silent:
            self.sounds.play("cancel")

    def _check_max_duration(self) -> None:
        if self.recording and self.recorder.duration >= self.cfg.max_duration_s:
            self.sm.finish()

    def _poll_live(self) -> None:
        if self.recording and self.live is not None:
            try:
                self.live.poll(self.recorder.samples_since, self.recorder.samplerate)
            except Exception:  # noqa: BLE001 - never break the recording over this
                log.exception("Live transcription failed; transcribing at the end instead")
                self.live = None

    def _grab_screen_names(self) -> None:
        """Names in the window you're dictating into, so they come out spelled right."""
        try:
            t0 = time.perf_counter()
            text = focus.screen_text(focus.inspect())
            self._context_names = textproc.candidate_names(text)
            log.debug("Screen names (%.0f ms): %s", (time.perf_counter() - t0) * 1000,
                      ", ".join(self._context_names[:20]))
        except Exception as exc:  # noqa: BLE001
            log.debug("Screen names failed: %s", exc)

    # -- worker thread ---------------------------------------------------
    def _process(self, audio, sr, live: LiveSession | None = None) -> None:
        try:
            if not self.transcriber.ready.wait(timeout=300):
                raise RuntimeError("Speech model is still loading")
            t0 = time.perf_counter()
            if live is not None and live.parts:
                raw = live.finish(audio, sr)  # most of it was transcribed while you talked
            else:
                raw = self.transcriber.transcribe(audio, sr)
            t_asr = time.perf_counter() - t0
            text = self._clean(raw) if raw else ""
            log.info("%.1fs audio -> %d chars, ASR %.0f ms, total %.0f ms (%s)", len(audio) / sr,
                     len(text), t_asr * 1000, (time.perf_counter() - t0) * 1000,
                     self.transcriber.backend)
            if text:
                self._deliver(text)
                # The label is what was heard (the audio doesn't contain AI edits or snippets).
                sample = self.training.add(audio, sr, raw) if self._keep_sample() else None
                self._remember(raw, text, len(audio) / sr)
                self._watch_edits(text, sample)
            self.bridge.done.emit(text)
        except Exception as exc:  # noqa: BLE001
            log.exception("Transcription failed")
            self.bridge.failed.emit(str(exc))

    def _process_command(self, audio, sr) -> None:
        """Command Mode: the recording is an instruction for the selected text."""
        try:
            if not self.transcriber.ready.wait(timeout=300):
                raise RuntimeError("Speech model is still loading")
            t0 = time.perf_counter()
            instruction = textproc.process(self.transcriber.transcribe(audio, sr), commands=False)
            if not instruction:
                self.bridge.done.emit("")
                return
            self.hotkeys.wait_for_modifiers_released()
            self._focus = f = focus.inspect()
            selection = f.read_selection()
            if selection is None and not f.is_terminal:  # Ctrl+C in a terminal would interrupt
                selection = self.injector.copy_selection()
            log.info("Command %r on %s (%d chars selected)", instruction, f.app or "?",
                     len(selection or ""))
            result = self.polisher.command(instruction, selection, app=f.app)
            if not result.strip():
                raise RuntimeError("The AI returned nothing")
            log.info("Command done in %.0f ms", (time.perf_counter() - t0) * 1000)
            if f.is_text is False:
                # Selected text you can't edit (a web page...): show the result to copy.
                self.injector.copy(result)
                self.bridge.show_card.emit(result)
            else:
                self.injector.inject(result)  # replaces the selection, or inserts at the cursor
            self._remember(instruction, result, len(audio) / sr, command=instruction)
            self.bridge.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            log.exception("Command failed")
            self.bridge.failed.emit(f"Command failed: {exc}")

    def _clean(self, raw: str) -> str:
        self.hotkeys.wait_for_modifiers_released()
        self._focus = f = focus.inspect()
        style = styles.style_for(self.cfg.styles, f.app, f.title, f.is_terminal)
        terms, corrections = self.dictionary.snapshot()
        known = {t.lower() for t in terms}
        names = [n for n in self._context_names if n.lower() not in known] \
            if self.cfg.screen_context else []
        replacements = {**corrections, **self.cfg.replacements}
        text = textproc.process(raw, fillers=self.cfg.remove_fillers,
                                commands=self.cfg.voice_commands,
                                replacements=replacements, terms=terms,
                                smart=self.cfg.smart_formatting)
        if names:  # careful matching: only words heard as names, only real misspellings
            text = textproc.apply_terms(text, names, threshold=0.86, min_len=5, case_only=False)
        snippets = self.dictionary.snippets_snapshot()
        triggers = textproc.find_snippets(text, snippets)
        if style == "exact" or self.tuning_active():
            polished = text  # code & terminals: your words; while training: GPU is busy
        else:
            polished = self.polisher.polish(text, app=f.app, terms=terms + names[:40])
        if triggers and textproc.find_snippets(polished, snippets) != triggers:
            polished = text  # the AI reworded a snippet trigger; keep the original wording
        # Corrections again: the LLM may have reintroduced a misspelling.
        text = textproc.apply_replacements(polished, replacements)
        text = styles.apply(text, style, keep=set(terms) | set(names))
        log.info("Style: %s (%s)", style, styles.category(f.app, f.title, f.is_terminal))
        if triggers:  # last, so neither the AI nor the rules touch the saved text
            text = textproc.expand_snippets(text, snippets)
            log.info("Snippet: %s", ", ".join(triggers))
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
        log.info("Pasting into %s (text field: %s)", f.app or "?",
                 {True: "yes", False: "no", None: "can't tell"}[f.is_text])
        self._before = f.read_text() if self.cfg.learn_from_edits else None
        out = text + (" " if self.cfg.append_space else "")
        self.injector.inject(out)

    def _watch_edits(self, text: str, sample: str | None) -> None:
        if self.cfg.learn_from_edits and self._focus.readable and self._before is not None:
            self.watcher.watch(self._focus, self._before, text, sample)

    def _remember(self, raw: str, text: str, seconds: float, command: str | None = None) -> None:
        self.history.append(text)
        del self.history[:-10]
        if self.cfg.save_history:
            row = {"t": time.time(), "seconds": round(seconds, 2), "raw": raw, "text": text,
                   "app": self._focus.app}
            if command:
                row["command"] = command
            try:
                from .history import append

                append(self.history_path, row)
            except OSError as exc:
                log.warning("Could not write history: %s", exc)
            self.bridge.history_changed.emit()

    def _on_done(self, text: str) -> None:
        if self._welcome is not None:
            self._welcome.dictated(text)
        self.pending -= 1
        if not self.pending:
            self.overlay.set_mode("dictate")
        if not self.recording:
            if text:
                self.overlay.set_state("transcribing" if self.pending else "idle")
            else:
                self.overlay.set_state("idle")
                self.overlay.flash("Didn't catch that")

    def _on_failed(self, err: str) -> None:
        self.pending -= 1
        if not self.pending:
            self.overlay.set_mode("dictate")
        if not self.recording:
            self.overlay.set_state("idle")
        self.sounds.play("error")
        self.overlay.flash(err[:60] or "Transcription failed", error=True, seconds=3)

    # -- tray ------------------------------------------------------------
    def _build_tray(self) -> None:
        self.tray = QSystemTrayIcon(tray_icon(), self.qapp)
        menu = QMenu()
        menu.addAction("Open Murmr", self.show_window)
        self.status_action = QAction("Loading speech model…", menu)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)
        menu.addSeparator()
        menu.addAction("Start hands-free dictation", self.sm.toggle_hands_free)
        menu.addAction("Paste last dictation", self.paste_last)
        self.read_action = QAction("Read selection aloud", menu)
        self.read_action.triggered.connect(self.read_aloud)
        menu.addAction(self.read_action)
        self.meeting_action = QAction("Start meeting notes", menu)
        self.meeting_action.triggered.connect(self._toggle_meeting_from_tray)
        menu.addAction(self.meeting_action)
        self.captions_action = QAction("Live captions", menu)
        self.captions_action.setCheckable(True)
        self.captions_action.triggered.connect(self.set_captions)
        menu.addAction(self.captions_action)
        self.recent_menu = menu.addMenu("Copy recent transcript")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        menu.addSeparator()
        menu.addAction("Settings…", lambda: self.show_window(page=2))
        self.update_action = QAction("Update Murmr…", menu)
        self.update_action.triggered.connect(lambda: self.show_window(page=2, section="Updates"))
        self.update_action.setVisible(False)
        menu.addAction(self.update_action)
        menu.addAction("Report a problem…", self.report_problem)
        menu.addAction("Restart Murmr", self.restart)
        menu.addAction("Quit Murmr", self.quit)
        self._menu = menu
        self.tray.setContextMenu(menu)
        self.tray.setToolTip("Murmr")
        self.tray.activated.connect(self._tray_activated)
        self.tray.messageClicked.connect(
            lambda: self._tray_click is not None and self.show_window(*self._tray_click))
        self.tray.show()

    def _tray_activated(self, reason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger,
                      QSystemTrayIcon.ActivationReason.DoubleClick) and sys.platform != "darwin":
            self.show_window()

    def _status(self, text: str) -> None:
        self._status_text = text
        self.status_action.setText(text)
        self.tray.setToolTip(f"Murmr — {text}")
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

    # -- paste again / read aloud -----------------------------------------
    def _recent_history(self) -> list[str]:
        """The last dictations, so "paste last" works right after a restart too."""
        from .history import recent_texts

        try:
            return recent_texts(self.history_path, 10)
        except OSError:
            return []

    def _prune_history(self) -> None:
        from .history import prune

        def work():
            try:
                if prune(self.history_path, int(self.cfg.history_days or 0)):
                    self.bridge.history_changed.emit()
            except (OSError, ValueError) as exc:
                log.warning("History clean-up failed: %s", exc)

        threading.Thread(target=work, name="history-prune", daemon=True).start()

    def delete_all_history(self) -> None:
        from .history import delete_all

        delete_all(self.history_path)
        self.history.clear()
        log.info("History deleted")
        self.bridge.history_changed.emit()

    def _apply_action_hotkeys(self) -> None:
        for name, spec, signal in (("paste_last", self.cfg.paste_last_hotkey, self.bridge.paste_last),
                                   ("read", self.cfg.read_hotkey, self.bridge.read_aloud)):
            hk = None
            if spec:
                try:
                    hk = Hotkey.parse(spec)
                except ValueError as exc:
                    log.error("%s shortcut %r: %s", name, spec, exc)
            self.hotkeys.set_action(name, hk, signal.emit)

    def paste_last(self) -> None:
        if not self.history:
            self.overlay.flash("Nothing to paste yet")
            return
        self.worker.submit(self.injector.inject, self.history[-1])

    def paste_text(self, text: str) -> None:
        """Paste a past dictation into the app you were using before opening Murmr."""
        if self.window is not None:
            self.window.hide()
        if sys.platform == "darwin":
            try:  # give focus back to the previous app
                from AppKit import NSApplication

                NSApplication.sharedApplication().hide_(None)
            except Exception:  # noqa: BLE001
                pass
        QTimer.singleShot(400, lambda: self.worker.submit(self.injector.inject, text))

    def speaker(self):
        if self._speaker is None:
            from .speak import Speaker

            self._speaker = Speaker(self.cfg.read_voice, self.cfg.read_speed, self.cfg.device,
                                    on_state=self.bridge.speak_state.emit)
        return self._speaker

    def read_aloud(self) -> None:
        """Read the selected text aloud; while reading, stop."""
        if self._speaker is not None and self._speaker.speaking:
            log.info("Read aloud: stopped")
            self._speaker.stop()
            return
        log.info("Read aloud: shortcut pressed")
        threading.Thread(target=self._read_selection, name="read-selection", daemon=True).start()

    def _read_selection(self) -> None:
        self.hotkeys.wait_for_modifiers_released()
        try:
            f = focus.inspect()
            text, how = f.read_selection(), "accessibility"
            if text is None and not f.is_terminal:  # Ctrl+C in a terminal would interrupt
                text, how = self.injector.copy_selection(), "copy"
            if not text or not text.strip():
                log.info("Read aloud: no selected text found in %s", f.app or "?")
                self.bridge.notice.emit("Select some text to read aloud", False)
                return
            log.info("Read aloud: %d characters from %s (via %s)", len(text), f.app or "?", how)
            self.speaker().speak(text)
        except Exception as exc:  # noqa: BLE001
            log.exception("Read aloud failed")
            self.bridge.notice.emit(f"Read aloud failed: {exc}"[:60], True)

    def speak_sample(self) -> None:
        self.speaker().speak("Hi! This is how Murmr reads your text aloud.")

    def _on_speak_state(self, state: str) -> None:
        speaking = self._speaker is not None and self._speaker.speaking
        self.read_action.setText("Stop reading" if speaking else "Read selection aloud")
        sp = self._speaker
        self.overlay.set_speaking(speaking, paused=bool(sp and sp.paused), loading=state == "loading",
                                  can_skip=bool(sp and sp.can_skip))
        if state == "loading":
            self.overlay.flash("Loading the reading voice…", seconds=2)
        elif state == "downloading":
            self.overlay.flash("Reading with the system voice · the natural voice is downloading",
                               seconds=4)
        elif state == "ready":
            self.overlay.flash("Natural reading voice ready", seconds=2.5)
        elif state.startswith("error:"):
            self.overlay.flash(state[6:][:60], error=True, seconds=4)

    # -- meetings & live captions (meetings.py) ------------------------------
    def meeting_store(self):
        from .meetings import MeetingStore

        return MeetingStore(config_mod.config_dir() / "meetings")

    def meeting_session(self):
        return self._meeting

    def meeting_running(self) -> bool:
        return bool(self._meeting and self._meeting.running and self._meeting.keep)

    def meeting_status(self) -> str:
        busy = [s for s in self._summary_state.values() if s]
        return busy[0] if busy else ""

    def _captures(self):
        from .meetings import MicCapture, SystemCapture, system_audio_supported

        caps = []
        if self.cfg.meeting_mic:
            caps.append(MicCapture(self.cfg.input_device))
        if self.cfg.meeting_system and system_audio_supported():
            caps.append(SystemCapture())
        return caps

    def _start_session(self, keep: bool) -> str | None:
        from .meetings import MeetingSession

        if not self.transcriber.ready.is_set():
            return "The speech model is still loading. Try again in a moment."
        caps = self._captures()
        if not caps:
            return "Turn on the microphone or computer audio first."
        if getattr(self, "_embedder", None) is None:
            from .speakers import Embedder

            self._embedder = Embedder()
        session = MeetingSession(caps, self.transcriber.transcribe, keep=keep, embedder=self._embedder,
                                 profile_dir=config_mod.config_dir() / "training")
        try:
            session.start()
        except Exception as exc:  # noqa: BLE001
            log.exception("Meeting start failed")
            return f"Couldn't start: {exc}"
        self._meeting = session
        log.info("%s started (%s)", "Meeting notes" if keep else "Live captions",
                 ", ".join(c.capture.who for c in session.channels))
        self._report_capture_problems(session)
        return None

    def _report_capture_problems(self, session) -> None:
        """A source that didn't start (macOS: computer-audio permission): say so, once."""
        from .meetings import SystemAudioDenied, request_mac_system_audio

        for who, exc in getattr(session, "problems", []):
            if isinstance(exc, SystemAudioDenied):
                if not getattr(self, "_asked_system_audio", False):
                    self._asked_system_audio = True
                    threading.Thread(target=request_mac_system_audio, daemon=True).start()
                self.tray.showMessage("Meeting notes: only your microphone", str(exc), tray_icon(), 10000)
            else:
                what = "computer audio" if who == "Others" else "microphone"
                self.tray.showMessage("Meeting notes", f"Couldn't use the {what}: {exc}", tray_icon(), 8000)

    def start_meeting(self) -> str | None:
        if self.meeting_running():
            return None
        if self._meeting is not None:  # a captions-only session: replace it with real notes
            self._meeting.stop()
            self._meeting = None
        err = self._start_session(keep=True)
        if err is None:
            self.meeting_action.setText("Stop meeting notes")
            self.overlay.flash("Meeting notes on", seconds=2)
        return err

    def stop_meeting(self) -> None:
        session = self._meeting
        if session is None:
            return
        self._meeting = None
        self.meeting_action.setText("Start meeting notes")
        if self._captions_on and session.keep:
            self.set_captions(False)

        def work():
            lines = session.stop()
            if not session.keep:
                return
            if not lines:
                log.info("Meeting notes stopped: nothing was said")
                self.bridge.meeting_changed.emit("")
                return
            meeting = {"started": session.started, "duration": round(session.elapsed, 1),
                       "sources": [c.capture.who for c in session.channels],
                       "lines": [line.__dict__ for line in lines], "title": "", "summary": ""}
            mid = self.meeting_store().save(meeting)
            log.info("Meeting saved: %s (%d lines, %.0f min)", mid, len(lines), session.elapsed / 60)
            self.bridge.meeting_changed.emit(mid)
            if self.ai_available():
                self._summarize(mid)

        threading.Thread(target=work, name="meeting-stop", daemon=True).start()

    def summarize_meeting(self, meeting_id: str) -> None:
        threading.Thread(target=self._summarize, args=(meeting_id,), name="meeting-summary",
                         daemon=True).start()

    def summary_state(self, meeting_id: str) -> str:
        return self._summary_state.get(meeting_id, "")

    def _summarize(self, mid: str) -> None:
        from .llm import summarize_meeting
        from .meetings import transcript_text

        store = self.meeting_store()
        meeting = store.load(mid)
        if meeting is None:
            return
        if not self.polisher.reachable(wait=True):
            meeting["summary_error"] = ("Summaries need the local AI. Install Ollama and run "
                                        f"“ollama pull {self.cfg.llm_cleanup.model}”, then "
                                        "click Summarize again. The transcript is saved.")
            store.save(meeting)
            self.bridge.meeting_changed.emit(mid)
            return

        def progress(msg: str) -> None:
            self._summary_state[mid] = msg
            self.bridge.meeting_changed.emit(mid)

        progress("Summarizing…")
        try:
            title, summary = summarize_meeting(self.polisher, transcript_text(meeting["lines"]), progress)
            meeting.update(title=title, summary=summary, summary_error="")
            log.info("Meeting summarized: %s", title)
        except Exception as exc:  # noqa: BLE001
            log.exception("Meeting summary failed")
            meeting["summary_error"] = f"The summary failed ({exc}). Click Summarize again."
        store.save(meeting)
        self._summary_state.pop(mid, None)
        self.bridge.meeting_changed.emit(mid)

    def meetings(self) -> list[dict]:
        return self.meeting_store().list()

    def meeting(self, meeting_id: str) -> dict | None:
        return self.meeting_store().load(meeting_id)

    def delete_meeting(self, meeting_id: str) -> None:
        self.meeting_store().delete(meeting_id)

    def rename_speaker(self, meeting_id: str, old: str, new: str) -> None:
        """Name a speaker: every line of theirs, and the summary, use the new name. Giving
        two labels the same name merges them (one person heard as two)."""
        import re

        new = " ".join(new.split())
        store = self.meeting_store()
        m = store.load(meeting_id)
        if m is None or not new or new == old:
            return
        for line in m.get("lines", []):
            if line.get("who") == old:
                line["who"] = new
        pattern = re.compile(rf"(?<![\w]){re.escape(old)}(?![\w])")
        for key in ("summary", "title"):
            if m.get(key):
                m[key] = pattern.sub(new, m[key])
        names = m.setdefault("speakers", {})
        names[old] = new
        store.save(m)
        log.info("Meeting %s: renamed a speaker", meeting_id)

    def _on_meeting_changed(self, mid: str) -> None:
        if self.window is not None and hasattr(self.window, "meetings"):
            self.window.meetings.show_summary_progress(mid)
            if self.window.pages.currentWidget() is self.window.meetings:
                self.window.meetings.refresh()
        state = self._summary_state.get(mid)
        if mid and not state:
            m = self.meeting_store().load(mid)
            if m and m.get("summary"):
                self.tray.showMessage("Meeting notes ready", m.get("title") or "Your summary is ready",
                                      tray_icon(), 6000)
                self._tray_click = None

    def _toggle_meeting_from_tray(self) -> None:
        if self.meeting_running():
            self.stop_meeting()
        else:
            err = self.start_meeting()
            if err:
                self.overlay.flash(err[:60], error=True, seconds=3)
        if self.window is not None and self.window.isVisible():
            self.window.refresh()

    def captions_on(self) -> bool:
        return self._captions_on

    def set_captions(self, on: bool) -> None:
        from .captions import CaptionsWindow

        on = bool(on)
        if on and self._meeting is None:
            err = self._start_session(keep=False)  # captions only: nothing is saved
            if err:
                self.overlay.flash(err[:60], error=True, seconds=3)
                on = False
        self._captions_on = on
        self.captions_action.setChecked(on)
        if on:
            if self._captions is None:
                self._captions = CaptionsWindow()
                self._captions.closed.connect(lambda: self.set_captions(False))
            self._captions.set_lines([])
            self.captions_timer.start()
        else:
            self.captions_timer.stop()
            if self._captions is not None:
                self._captions.hide()
            if self._meeting is not None and not self._meeting.keep:
                self.stop_meeting()  # captions-only session ends with the captions
        if self.window is not None and self.window.isVisible() and hasattr(self.window, "meetings"):
            self.window.meetings.captions_btn.setChecked(on)

    def _update_captions(self) -> None:
        s = self._meeting
        if s is None or self._captions is None:
            return
        shown = [(line.who, line.text) for line in s.lines[-2:]] + s.live()
        self._captions.set_lines(shown)

    # -- window ----------------------------------------------------------
    def show_window(self, page: int | None = None, section: str | None = None) -> None:
        """Open the window, optionally at a page and (Settings) a section such as "Updates"."""
        if self.window is None:
            from .window import MainWindow

            self.window = MainWindow(self)
        if page is not None:
            self.window.show_page(page)
        self.window.bring_to_front()
        if section and page == 2:
            self.window.settings.show_section(section)

    def _notify_update(self, msg: str) -> None:
        self._tray_click = (2, "Updates")
        self.tray.showMessage("Murmr update available", f"{msg}. Click to see what's new and install.",
                              tray_icon(), 6000)

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
        if self._welcome is not None and self._welcome.isVisible():
            self._welcome.capture_finished(kind, spec, err)
            return
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
            self.overlay.hint = self._hint(hk)
        elif key == "hands_free_key":
            if value and value in self.hotkey.keys:
                return "The hands-free key can't be part of the push-to-talk shortcut."
            self.hotkeys.set_hotkey(self.hotkey, value or None)
        elif key == "command_hotkey":
            if value:
                try:
                    hk = Hotkey.parse(value)
                except ValueError as exc:
                    return str(exc)
                if hk == self.hotkey:
                    return "Command Mode needs a different shortcut from push to talk."
            self.hotkeys.set_hotkey(self.hotkey, cfg.hands_free_key or None,
                                    Hotkey.parse(value) if value else None)
        elif key == "launch_at_login":
            try:
                autostart.set_enabled(bool(value))
            except Exception as exc:  # noqa: BLE001
                return f"Couldn't change launch at login: {exc}"
            return None
        elif key == "llm_mode":  # off | smart | always
            if value == "off":
                cfg.llm_cleanup.enabled = False
            else:
                cfg.llm_cleanup.enabled = "auto"
                cfg.llm_cleanup.when = value
            self.polisher.available = None
            config_mod.save(cfg)
            log.info("AI clean-up: %s", value)
            return None
        elif key == "pill_anchor":
            if value != self.overlay.anchor:
                self.overlay.set_anchor(value)  # picked in Settings (a drag already moved it)
            if self.window is not None:
                self.window.settings.sync_pill_anchor(value)
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
        elif key == "training_excluded_apps":
            if isinstance(value, str):
                value = [a.strip() for a in value.split(",") if a.strip()]
        elif key == "save_training_audio":
            from . import capabilities

            self.training = (TrainingStore(config_mod.config_dir() / "training", cfg.training_max_mb)
                             if value and capabilities.can_train() else None)
        elif key == "update_check":
            if value and self.updater.installed:
                self.update_timer.start()
            else:
                self.update_timer.stop()
        elif key == "double_tap_to_lock":
            self.sm.double_tap = bool(value)
        elif key in ("paste_last_hotkey", "read_hotkey"):
            if value:
                try:
                    hk = Hotkey.parse(value)
                except ValueError as exc:
                    return str(exc)
                if hk == self.hotkey:
                    return "That's already your push-to-talk shortcut."
            setattr(cfg, key, value)
            self._apply_action_hotkeys()
        elif key == "history_days":
            value = int(value)
            setattr(cfg, key, value)
            self._prune_history()
        elif key == "sync_enabled":
            if value:
                err = self._start_sync()
                if err:
                    return err
            else:
                self._stop_sync()
        elif key in ("meeting_mic", "meeting_system", "auto_tune", "sync_models", "sync_voices",
                     "device_name"):
            pass  # used by the next meeting / sync
        elif key in ("read_voice", "read_speed"):
            if self._speaker is not None:
                setattr(self._speaker, key.removeprefix("read_"), value)
        elif key == "styles":
            value = {**cfg.styles, **value}
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

    def training_stats(self) -> dict | None:
        store = self.training or TrainingStore(config_mod.config_dir() / "training")
        return store.stats()

    def delete_training(self) -> None:
        store = self.training or TrainingStore(config_mod.config_dir() / "training")
        store.delete_all()
        log.info("Deleted all voice samples")

    # Your voice (privacy.py) ---------------------------------------------------
    def _keep_sample(self) -> bool:
        """Keep this dictation's audio for training? Never from password fields or excluded apps."""
        from .privacy import excluded

        f = self._focus
        if self.training is None or f.is_password:
            return False
        if excluded(f.app, self.cfg.training_excluded_apps):
            log.info("Not keeping the recording: %s is on the excluded list", f.app)
            return False
        return True

    def voice_storage(self) -> dict:
        from .privacy import storage

        return storage()

    def managed_reason(self) -> str:
        if not hasattr(self, "_managed"):
            from .privacy import managed_device

            self._managed = managed_device()
        return self._managed

    def delete_all_voice(self) -> bool:
        """Delete every recording, tuned model and voice. True if Murmr should restart."""
        from .privacy import delete_everything

        if self._reading_store_ref() is not None:
            self._reading_store_ref().delete_all()
        restart = delete_everything(self.cfg)
        if self.cfg.save_training_audio:
            self.training = TrainingStore(config_mod.config_dir() / "training", self.cfg.training_max_mb)
        self._reading_store = None
        return restart

    def _reading_store_ref(self):
        return self.training or getattr(self, "_reading_store", None)

    # Voice training page (voice_training.py) ----------------------------------
    def training_store(self) -> TrainingStore:
        """Reading sessions always save (they're an explicit action), even with samples off."""
        if self.training is not None:
            return self.training
        if getattr(self, "_reading_store", None) is None:
            self._reading_store = TrainingStore(config_mod.config_dir() / "training",
                                                self.cfg.training_max_mb)
        return self._reading_store

    def transcribe_sample(self, audio, sr: int) -> str:
        """Raw Parakeet output (no clean-up) for checking a reading take. Blocking."""
        if not self.transcriber.ready.wait(timeout=300):
            raise RuntimeError("Speech model is still loading")
        return self.transcriber.transcribe(audio, sr)

    def delete_history(self, t: float) -> None:
        from .history import delete_entry

        try:
            delete_entry(self.history_path, t)
        except OSError as exc:
            log.warning("Could not delete history entry: %s", exc)

    # -- updates (Controller) -------------------------------------------------------
    def update_state(self) -> dict:
        u = self.updater
        last = u.last
        cur = u.current_version()
        staged = u.staged()
        return {"installed": u.installed, "sha": cur.get("sha", ""), "date": cur.get("date", ""),
                "available": bool(last and last.available), "changes": last.changes if last else [],
                "error": last.error if last else "", "checked": self._update_checked,
                "busy": self._update_busy, "has_previous": u.has_previous(),
                "whats_new": cur.get("changes", []),
                "staged": bool(staged and last and staged.get("sha") == last.target_sha)}

    def _announce_update(self) -> None:
        """After an update: a "What's new" note (not on the first run after installing)."""
        cur = self.updater.current_version()
        sha = cur.get("sha", "")
        if not sha or sha == self.cfg.last_seen_version:
            return
        first_run = not self.cfg.last_seen_version
        self.cfg.last_seen_version = sha
        config_mod.save(self.cfg)
        changes = cur.get("changes") or []
        if first_run or not changes:
            return
        lines = "\n".join(f"• {c}" for c in changes[:3]) + \
            (f"\n…and {len(changes) - 3} more" if len(changes) > 3 else "")
        self._tray_click = (2, "Updates")
        self.tray.showMessage(f"Murmr updated · {len(changes)} change{'s' if len(changes) != 1 else ''}",
                              lines, tray_icon(), 12000)
        log.info("What's new shown for %s", sha[:7])

    def _auto_stage(self, info) -> None:
        """Download an update in the background; restart into it when you're away."""
        def work():
            try:
                staged = self.updater.staged()
                if not (staged and staged.get("sha") == info.target_sha):
                    self.updater.stage(info)
                log.info("Update %s ready; restarting when Murmr is idle", info.target_sha[:7])
                self.bridge.update_changed.emit()
            except Exception as exc:  # noqa: BLE001
                log.warning("Background update failed (will retry later): %s", exc)

        threading.Thread(target=work, name="update-stage", daemon=True).start()

    def _quiet(self) -> bool:
        """Nobody's using Murmr: safe to restart for an update."""
        speaking = self._speaker is not None and self._speaker.speaking
        window_open = self.window is not None and self.window.isVisible()
        away = time.monotonic() - self.hotkeys.last_activity > 10 * 60
        return away and not (self.recording or self.pending or speaking or window_open
                             or self._update_busy or self.tuning_active() or self.meeting_running())

    def _maybe_auto_restart(self) -> None:
        if not self.cfg.auto_update:
            self.auto_timer.stop()
            return
        last = self.updater.last
        staged = self.updater.staged()
        if staged and last and staged.get("sha") == last.target_sha and self._quiet():
            log.info("Restarting into update %s (idle)", last.target_sha[:7])
            self.restart()

    def check_updates(self, quiet: bool = False) -> None:
        if not self.updater.installed or self._update_busy:
            return
        self._update_busy = "Checking…"
        self.bridge.update_changed.emit()

        def work():
            info = self.updater.check()
            self._update_busy = ""
            self._update_checked = True
            self.bridge.update_changed.emit()
            if info.available and quiet and self.cfg.auto_update:
                self._auto_stage(info)
                self.bridge.update_auto.emit()
            elif info.available and quiet:
                n = len(info.changes)
                self.bridge.update_notice.emit(f"{n} update{'s' if n != 1 else ''} ready" if n
                                               else "A new version is ready")

        threading.Thread(target=work, name="update-check", daemon=True).start()

    def apply_update(self) -> None:
        info = self.updater.last
        if not info or not info.available or self._update_busy:
            return

        def progress(msg: str) -> None:
            self._update_busy = msg
            self.bridge.update_changed.emit()

        def work():
            try:
                staged = self.updater.staged()
                if not (staged and staged.get("sha") == info.target_sha):  # not downloaded yet
                    self.updater.stage(info, progress)
                progress("Restarting…")
                self.bridge.update_restart.emit()
            except Exception as exc:  # noqa: BLE001
                log.exception("Update failed")
                info.error = f"Update failed: {exc}"
                self._update_busy = ""
                self.bridge.update_changed.emit()

        threading.Thread(target=work, name="update-apply", daemon=True).start()

    def rollback_update(self) -> None:
        try:
            self.updater.request_rollback()
        except RuntimeError as exc:
            if self.updater.last:
                self.updater.last.error = str(exc)
            self.bridge.update_changed.emit()
            return
        self.restart()

    def _on_update_changed(self) -> None:
        last = self.updater.last
        self.update_action.setVisible(bool(last and last.available))
        if self.window is not None and self.window.isVisible():
            self.window.settings.refresh_updates()

    def show_welcome(self) -> None:
        """The first-run guide (welcome.py); Settings can show it again."""
        from .welcome import WelcomeDialog

        if self._welcome is not None and self._welcome.isVisible():
            self._welcome.raise_()
            self._welcome.activateWindow()
            return
        self._welcome = WelcomeDialog(self)
        self._welcome.finished.connect(self._welcome_closed)
        if sys.platform == "darwin":
            try:
                from AppKit import NSApplication

                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
            except Exception:  # noqa: BLE001
                pass
        self._welcome.show()
        self._welcome.raise_()
        self._welcome.activateWindow()
        log.info("Welcome guide shown")

    def _welcome_closed(self, _result=None) -> None:
        self._welcome = None
        if self.window is not None and self.window.isVisible():
            self.window.refresh()

    # -- sync with your other computers (sync.py) ------------------------------
    def device_name(self) -> str:
        import socket

        name = self.cfg.device_name.strip() or socket.gethostname()
        return name.removesuffix(".local")[:60]

    def _sync_accepts(self) -> list[str]:
        return [k for k, on in (("models", self.cfg.sync_models), ("voices", self.cfg.sync_voices)) if on]

    def _start_sync(self) -> str | None:
        from .sync import SyncService

        if self.sync is not None:
            return None
        svc = SyncService(config_mod.config_dir(), self.dictionary, name=self.device_name,
                          accepts=self._sync_accepts,
                          on_event=lambda kind, info: self.bridge.sync_event.emit(kind, info))
        try:
            svc.start()
        except OSError as exc:
            log.warning("Sync couldn't start: %s", exc)
            return f"Sync couldn't start: {exc}"
        self.sync = svc
        return None

    def _stop_sync(self) -> None:
        svc, self.sync = self.sync, None
        if svc is not None:
            threading.Thread(target=svc.stop, name="sync-stop", daemon=True).start()

    def _on_sync_event(self, kind: str, info) -> None:
        if kind == "received" and info.get("kind") == "models":
            self._tray_click = (2, "Speech model")
            self.tray.showMessage("Speech model from " + info.get("from", "your other computer"),
                                  "A model tuned to your voice arrived. Switch to it in Settings > "
                                  "Speech model.", tray_icon(), 10000)
            if self.window is not None and not self.window.isVisible() and \
                    getattr(self.window.settings, "speech_model", None) is None:
                self.window.deleteLater()  # rebuilt with the Speech model section next time
                self.window = None
        elif kind == "paired":
            self.tray.showMessage("Paired with " + info.get("name", "a computer"),
                                  "Your Dictionary will stay in sync between them.", tray_icon(), 5000)
        elif kind == "pair_failed":
            self.tray.showMessage("Pairing stopped", info.get("why", ""), tray_icon(), 6000)
        if self.window is not None and self.window.isVisible():
            sec = getattr(self.window.settings, "sync_section", None)
            if sec is not None:
                sec.refresh()
            if kind == "status" and self.window.pages.currentWidget() is self.window.dictionary:
                self.window.dictionary.refresh()

    def report_problem(self) -> None:
        from . import report

        if self.window is None or not self.window.isVisible():
            self.show_window()
        report.show_dialog(self, self.window)

    def open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(config_mod.config_dir())))

    # -- lifecycle -------------------------------------------------------
    def restart(self) -> None:
        import subprocess

        from .shortcuts import launcher

        from .shortcuts import mac_app_usable, open_mac_app

        background = self.window is None or not self.window.isVisible()
        self.shutdown()
        if mac_app_usable():  # through Murmr.app, so it keeps Murmr's permissions
            self.instance.release()
            if open_mac_app(background):
                self.qapp.quit()
                return
        exe, extra = launcher()  # same windowless launch as the Start Menu shortcut
        args = [exe, *extra]
        if background:
            args.append("--background")
        self.instance.release()
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) if sys.platform == "win32" else 0
        subprocess.Popen(args, creationflags=flags, start_new_session=sys.platform != "win32")
        self.qapp.quit()

    def shutdown(self) -> None:
        import faulthandler

        for job in self._tuning.values():  # training in WSL would otherwise keep running
            if job.running:
                job.cancel()

        if getattr(self, "sync", None) is not None:  # free the network ports
            svc, self.sync = self.sync, None
            svc._stop.set()
            for sock in (svc._tcp, svc._udp):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass

        if self._meeting is not None:  # save the notes taken so far
            session, self._meeting = self._meeting, None
            lines = session.stop()
            if session.keep and lines:
                self.meeting_store().save({
                    "started": session.started, "duration": round(session.elapsed, 1),
                    "sources": [c.capture.who for c in session.channels], "title": "", "summary": "",
                    "lines": [line.__dict__ for line in lines],
                    "summary_error": "Murmr closed before summarizing. Click Summarize again."})

        faulthandler.cancel_dump_traceback_later()
        if getattr(self, "watchdog", None):
            self.watchdog.stop()
        if self._speaker is not None:
            self._speaker.stop()
        self.hotkeys.stop()
        self.recorder.shutdown()
        self.tray.hide()

    def quit(self) -> None:
        self.shutdown()
        self.qapp.quit()


def relaunch_detached() -> None:
    """Start a new Murmr a moment after this process ends (used when it's frozen)."""
    import shlex
    import subprocess

    from .shortcuts import MAC_APP, launcher, mac_app_usable

    try:
        if mac_app_usable():
            subprocess.Popen(["/bin/sh", "-c", f'sleep 2; /usr/bin/open -n -a {shlex.quote(str(MAC_APP))} '
                              "--args --background"], start_new_session=True)
            return
        exe, args = launcher()
        if sys.platform == "win32":
            line = subprocess.list2cmdline([exe, *args, "--background"])
            subprocess.Popen(["cmd", "/c", f'timeout /t 2 /nobreak >nul & start "" {line}'],
                             creationflags=0x08000000 | 0x00000008)  # no window, detached
        else:
            subprocess.Popen(["/bin/sh", "-c", f"sleep 2; exec {shlex.join([exe, *args, '--background'])}"],
                             start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        log.error("Couldn't restart Murmr: %s", exc)


class SingleInstance:
    """One Murmr per user. Launching it again just opens the running one's window."""

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
        # An older Murmr without the socket may still hold the lock.
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


def raise_priority() -> None:
    """Dictation should stay instant even when the CPU is busy with other work."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            k32 = ctypes.windll.kernel32
            # Declare the types: untyped, the 64-bit pseudo-handle is truncated and the call fails.
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000):  # ABOVE_NORMAL
                log.warning("Couldn't raise process priority")
        except Exception:  # noqa: BLE001
            pass


def enable_crash_log() -> None:
    """Python-level crash dumps (native faults included) go to crash.log."""
    import faulthandler

    if faulthandler.is_enabled():
        return  # ensure_std_streams already sent them to stdio.log
    try:
        path = config_mod.config_dir() / "crash.log"
        f = open(path, "a", encoding="utf-8")  # noqa: SIM115 - must stay open
        faulthandler.enable(f)
    except OSError:
        pass


def run(background: bool = False) -> int:
    ensure_std_streams()
    setup_logging()
    enable_crash_log()
    raise_priority()
    if sys.platform == "win32":
        try:  # own taskbar identity/icon instead of python.exe's
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Murmr.Dictation")
        except Exception:  # noqa: BLE001
            pass
    cfg = config_mod.load()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Murmr")
    qapp.setWindowIcon(app_icon())
    qapp.setQuitOnLastWindowClosed(False)

    from .shortcuts import migrate_from_whispa

    if migrate_from_whispa():  # macOS: Murmr.app replaced Whispa.app; continue in there
        from .shortcuts import open_mac_app

        if open_mac_app(background):
            return 0
    if sys.platform == "darwin":
        from .shortcuts import relaunch_through_mac_app

        if relaunch_through_mac_app(background):
            return 0

    instance = SingleInstance()
    if instance.notify_running():
        log.info("Murmr is already running; asked it to show its window")
        if not instance.server and instance.lock.error() != QLockFile.LockError.NoError:
            QMessageBox.information(None, "Murmr", "Murmr is already running. Quit the old "
                                    "copy from its tray icon, then start Murmr again.")
        return 0

    if sys.platform == "darwin":
        try:  # menu-bar app: no Dock icon
            from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

            NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:  # noqa: BLE001
            pass

    app = MurmrApp(qapp, cfg)
    app.instance = instance
    instance.listen(app.show_window)
    if not background:
        app.show_window()
    log.info("Murmr started; hotkey %s, config %s", app.hotkey.label(), config_mod.config_path())
    code = qapp.exec()
    app.worker.shutdown(wait=False, cancel_futures=True)
    return code
