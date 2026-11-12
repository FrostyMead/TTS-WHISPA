"""The first-run guide: a few short steps to get from installed to dictating.

1. Welcome: what Murmr does, and that it all stays on this computer.
2. Permissions (macOS): Microphone, Input Monitoring, Accessibility, the 🌐 key.
3. Microphone: pick one and see it move.
4. Try it: hold the shortcut and talk into a box on the page (change the shortcut here).
5. Extras: local AI, launch at login, voice training.

Shown once (``Config.welcome_done``); Settings > "Welcome guide" shows it again. Every step
can be skipped.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QPlainTextEdit, QProgressBar,
                               QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from . import theme
from .widgets import HotkeyField, Label, Switch, card, guard_wheel, setting_row

log = logging.getLogger(__name__)


def _step(title: str, text: str) -> tuple[QWidget, QVBoxLayout]:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(10)
    h = Label(title)
    h.setObjectName("h1")
    lay.addWidget(h)
    if text:
        s = Label(text)
        s.setObjectName("sub")
        s.setWordWrap(True)
        lay.addWidget(s)
    lay.addSpacing(6)
    return page, lay


class LevelMeter:
    """A microphone level, read from its own short-lived input stream."""

    def __init__(self):
        self.level = 0.0
        self._stream = None

    def open(self, device) -> str | None:
        self.close()
        try:
            import sounddevice as sd

            from .audio import PA_LOCK, resolve_device

            def cb(data, frames, t, status):
                rms = float(np.sqrt(np.mean(data[:, 0] ** 2))) if frames else 0.0
                self.level = max(rms, self.level * 0.8)

            with PA_LOCK:
                dev = resolve_device(device)
                sr = int(sd.query_devices(dev, "input")["default_samplerate"])
                self._stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=dev,
                                              callback=cb, blocksize=sr // 20)
                self._stream.start()
        except Exception as exc:  # noqa: BLE001
            log.info("Level meter: %s", exc)
            self._stream = None
            return str(exc)
        return None

    def close(self) -> None:
        if self._stream is None:
            return
        from .audio import PA_LOCK

        with PA_LOCK:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._stream = None
        self.level = 0.0


class WelcomeDialog(QDialog):
    def showEvent(self, event) -> None:
        super().showEvent(event)
        guard_wheel(self)  # scrolling past the microphone list never changes it

    def __init__(self, ctl, mic_options: list[tuple[str, str]] | None = None, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setObjectName("root")
        self.setWindowTitle("Welcome to Murmr")
        self.setMinimumSize(640, 520)
        pal = theme.current()
        self.setStyleSheet(theme.stylesheet(pal) + f"""
QProgressBar {{ background: {pal.track_off}; border: none; border-radius: 5px; }}
QProgressBar::chunk {{ background: {pal.success}; border-radius: 5px; }}
""")
        self.meter = LevelMeter()
        self._tried = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(36, 30, 36, 22)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack, 1)
        self.steps: list[str] = []
        self._welcome_step()
        if sys.platform == "darwin" and hasattr(ctl, "permissions"):
            self._permissions_step()
        self._mic_step(mic_options)
        self._try_step()
        self._extras_step()

        nav = QHBoxLayout()
        self.dots = Label("")
        self.dots.setObjectName("faint")
        nav.addWidget(self.dots)
        nav.addStretch(1)
        self.skip_btn = QPushButton("Skip the guide")
        self.skip_btn.clicked.connect(self.finish)
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(lambda: self.go(self.stack.currentIndex() - 1))
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("primary")
        self.next_btn.setDefault(True)
        self.next_btn.clicked.connect(self._next)
        for b in (self.skip_btn, self.back_btn, self.next_btn):
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            nav.addWidget(b)
        outer.addLayout(nav)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._ticks = 0
        self.go(0)

    # -- steps -----------------------------------------------------------
    def _welcome_step(self) -> None:
        page, lay = _step("Welcome to Murmr", "Talk instead of typing, in any app.")
        frame, inner = card()
        key = self.ctl.hotkey.label() if hasattr(self.ctl, "hotkey") else "the shortcut"
        for i, (title, text) in enumerate((
                ("Hold, talk, let go", f"Hold {key}, say what you want to write, and let go. The "
                 "text appears wherever your cursor is."),
                ("Private by design", "Your voice is turned into text on this computer. Audio and "
                 "text are never sent anywhere."),
                ("Gets to know you", "Fix a word Murmr got wrong and it learns it. Add names and "
                 "terms in the Dictionary."))):
            inner.addWidget(setting_row(title, text, None, first=i == 0))
        lay.addWidget(frame)
        lay.addStretch(1)
        self._add(page, "welcome")

    def _permissions_step(self) -> None:
        from .macperms import KINDS, LABELS, WHY

        page, lay = _step("Allow Murmr", "macOS asks you to allow each of these once. Click Allow, "
                          "switch Murmr on in System Settings, then come back here.")
        frame, inner = card()
        self.perm_rows: dict[str, tuple[Label, QPushButton]] = {}
        for i, kind in enumerate(KINDS):
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            state = Label("")
            btn = QPushButton("Change…" if kind == "fn_key" else "Allow…")
            btn.setObjectName("primary")
            btn.clicked.connect(lambda _=False, k=kind: self.ctl.allow_permission(k))
            bl.addWidget(state)
            bl.addWidget(btn)
            inner.addWidget(setting_row(LABELS[kind], WHY[kind], box, first=i == 0))
            self.perm_rows[kind] = (state, btn)
        lay.addWidget(frame)
        note = Label("Allowed but it still says no? macOS sometimes needs Murmr restarted. This "
                     "guide comes back after the restart.")
        note.setObjectName("faint")
        note.setWordWrap(True)
        lay.addWidget(note)
        restart = QPushButton("Restart Murmr")
        restart.clicked.connect(self.ctl.restart)
        row = QHBoxLayout()
        row.addWidget(restart)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        self._add(page, "permissions")

    def _mic_step(self, mic_options) -> None:
        page, lay = _step("Your microphone", "Say something: the bar should move.")
        if mic_options is None:
            try:
                from .audio import input_devices

                mic_options = input_devices()
            except Exception:  # noqa: BLE001
                mic_options = []
        frame, inner = card()
        self.mic = QComboBox()
        self.mic.setMinimumWidth(280)
        self.mic.addItem("System default", None)
        for label, value in mic_options:
            self.mic.addItem(label, value)
        current = self.ctl.cfg.input_device
        if current not in (None, ""):
            idx = self.mic.findData(current)
            if idx >= 0:
                self.mic.setCurrentIndex(idx)
        self.mic.currentIndexChanged.connect(self._mic_changed)
        inner.addWidget(setting_row("Microphone", "", self.mic, first=True))
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedSize(280, 10)
        inner.addWidget(setting_row("Level", "", self.level_bar))
        lay.addWidget(frame)
        self.mic_msg = Label("")
        self.mic_msg.setObjectName("faint")
        self.mic_msg.setWordWrap(True)
        lay.addWidget(self.mic_msg)
        lay.addStretch(1)
        self._add(page, "mic")

    def _try_step(self) -> None:
        page, lay = _step("Try it", "")
        self.try_text = Label("")
        self.try_text.setObjectName("sub")
        self.try_text.setWordWrap(True)
        lay.insertWidget(1, self.try_text)
        frame, inner = card()
        labels = self.ctl.hotkey.label().split("+") if hasattr(self.ctl, "hotkey") else []
        from .window import hotkey_labels

        self.hotkey_field = HotkeyField(hotkey_labels(self.ctl.cfg.hotkey) if labels else ["?"])
        self.hotkey_field.change_requested.connect(self._record)
        self.hotkey_field.cancel_requested.connect(self._cancel_record)
        inner.addWidget(setting_row("Push to talk", "Hold it while you talk. Click to change it.",
                                    self.hotkey_field, first=True))
        lay.addWidget(frame)
        self.box = QPlainTextEdit()
        self.box.setPlaceholderText("Click here, then hold the shortcut and say “Hello, this is "
                                    "my first dictation.”")
        self.box.setMinimumHeight(110)
        lay.addWidget(self.box)
        self.try_msg = Label("")
        self.try_msg.setObjectName("faint")
        self.try_msg.setWordWrap(True)
        lay.addWidget(self.try_msg)
        lay.addStretch(1)
        self._update_try_text()
        self._add(page, "try")

    def _extras_step(self) -> None:
        from . import capabilities

        page, lay = _step("A few extras", "All optional. You can change these any time in Settings.")
        frame, inner = card()
        first = True
        self.ai_btn = None
        if hasattr(self.ctl, "setup_ai") and not self.ctl.ai_available():
            self.ai_btn = QPushButton("Set up")
            self.ai_btn.setObjectName("primary")
            self.ai_btn.clicked.connect(self._setup_ai)
            inner.addWidget(setting_row(
                "Local AI", "Tidies up what you say, writes and rewrites text on command, and "
                "summarizes meetings. A one-time download of about 3 GB, and it runs on this "
                "computer too.", self.ai_btn, first=first))
            first = False
            self.ai_note = Label("")
            self.ai_note.setObjectName("faint")
            self.ai_note.setWordWrap(True)
            inner.addWidget(self.ai_note)
        try:
            from . import autostart

            on = autostart.is_enabled()
        except Exception:  # noqa: BLE001
            on = False
        login = Switch(on)
        login.toggled.connect(lambda v: self.ctl.update_setting("launch_at_login", v))
        inner.addWidget(setting_row("Launch at login", "Murmr is ready whenever you are.", login,
                                    first=first))
        if capabilities.can_train():
            inner.addWidget(setting_row(
                "Learn your voice", "Read a few sentences on the Voice training page, and Murmr can "
                "be tuned to how you speak. Takes 10 minutes or more, whenever you like.", None))
        lay.addWidget(frame)
        tray = "menu bar" if sys.platform == "darwin" else "system tray"
        end = Label(f"Murmr keeps running in the {tray}: click its icon for Settings, meeting "
                    "notes and more.")
        end.setObjectName("muted")
        end.setWordWrap(True)
        lay.addWidget(end)
        lay.addStretch(1)
        self._add(page, "extras")

    def _add(self, page: QWidget, name: str) -> None:
        self.stack.addWidget(page)
        self.steps.append(name)

    # -- navigation --------------------------------------------------------
    def go(self, index: int) -> None:
        index = max(0, min(index, self.stack.count() - 1))
        self.stack.setCurrentIndex(index)
        name = self.steps[index]
        last = index == self.stack.count() - 1
        self.back_btn.setVisible(index > 0)
        self.skip_btn.setVisible(not last)
        self.next_btn.setText("Done" if last else "Next")
        self.dots.setText(f"Step {index + 1} of {self.stack.count()}")
        if name == "mic":
            self._mic_changed()
        else:
            self.meter.close()
        if name == "try":
            QTimer.singleShot(50, self.box.setFocus)

    def _next(self) -> None:
        if self.stack.currentIndex() == self.stack.count() - 1:
            self.finish()
        else:
            self.go(self.stack.currentIndex() + 1)

    def finish(self) -> None:
        self.meter.close()
        self._cancel_record()
        try:
            self.ctl.update_setting("welcome_done", True)
        except Exception:  # noqa: BLE001
            log.debug("welcome_done", exc_info=True)
        self.accept()

    def closeEvent(self, event) -> None:
        self.meter.close()
        self._cancel_record()
        try:
            self.ctl.update_setting("welcome_done", True)
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    # -- live parts --------------------------------------------------------
    def _tick(self) -> None:
        self._ticks += 1
        name = self.steps[self.stack.currentIndex()]
        if name == "mic":
            self.level_bar.setValue(min(100, int(self.meter.level * 600)))
        elif name == "permissions" and self._ticks % 15 == 1:
            self.refresh_permissions()
        elif name == "try" and self._ticks % 10 == 1:
            self._update_try_text()

    def refresh_permissions(self) -> None:
        for kind, ok in (self.ctl.permissions() or {}).items():
            if kind not in getattr(self, "perm_rows", {}):
                continue
            state, btn = self.perm_rows[kind]
            if ok:
                state.setText("✓ Done" if kind == "fn_key" else "✓ Allowed")
                state.setObjectName("ok")
            else:
                state.setText("Not set" if kind == "fn_key" else "Not allowed")
                state.setObjectName("error")
            state.style().unpolish(state)
            state.style().polish(state)
            btn.setVisible(not ok)

    def _mic_changed(self) -> None:
        value = self.mic.currentData()
        if value != self.ctl.cfg.input_device:
            self.ctl.update_setting("input_device", value)
        err = self.meter.open(value)
        self.mic_msg.setText(f"Couldn't open this microphone: {err}" if err else
                             "Too quiet? Pick another microphone, or check it isn't muted.")

    def _update_try_text(self) -> None:
        status = self.ctl.status_text() if hasattr(self.ctl, "status_text") else "Ready"
        if not status.startswith("Ready"):
            self.try_text.setText("The speech model is still loading (the first time it's "
                                  "downloaded, about 650 MB). You can try once it says Ready: "
                                  f"{status}")
        else:
            self.try_text.setText("Click in the box below, hold the shortcut, say a sentence, "
                                  "and let go.")

    def _record(self) -> None:
        self.hotkey_field.set_recording(True)
        self.try_msg.setText("Press the keys you want to use…")
        self.ctl.capture_hotkey("hotkey")

    def _cancel_record(self) -> None:
        if hasattr(self, "hotkey_field") and getattr(self.hotkey_field, "recording", False):
            self.ctl.cancel_capture()
        if hasattr(self, "hotkey_field"):
            self.hotkey_field.set_recording(False)

    def capture_finished(self, kind: str, spec: str | None, error: str | None) -> None:
        from .window import hotkey_labels

        self.hotkey_field.set_recording(False)
        if error:
            self.try_msg.setText(error)
            return
        if spec is None:
            self.try_msg.setText("")
            return
        err = self.ctl.update_setting(kind, spec)
        if err:
            self.try_msg.setText(err)
            return
        self.hotkey_field.set_labels(hotkey_labels(spec))
        self.try_msg.setText("Shortcut changed. Try it in the box.")

    def dictated(self, text: str) -> None:
        """A dictation finished while the guide is open."""
        if not text or not self.isVisible() or self.steps[self.stack.currentIndex()] != "try":
            return

        def check():
            if text.strip() and text.strip()[:20] not in self.box.toPlainText():
                self.box.insertPlainText(text)  # pasted somewhere else: show it here
            self.try_msg.setObjectName("ok")
            self.try_msg.style().unpolish(self.try_msg)
            self.try_msg.style().polish(self.try_msg)
            self.try_msg.setText("✓ That's it. It works the same in any app: email, chat, docs, "
                                 "anywhere you can type.")

        QTimer.singleShot(800, check)

    # -- local AI ------------------------------------------------------------
    def _setup_ai(self) -> None:
        if self.ai_btn.text() == "Cancel":
            self.ctl.cancel_ai_setup()
            return
        self.ai_btn.setText("Cancel")
        self.ai_note.setText("Starting…")
        self.ctl.setup_ai()

    def ai_setup_progress(self, msg: str, frac: float) -> None:
        if self.ai_btn is None:
            return
        done = msg.startswith(("done:", "error:"))
        text = msg.split(":", 1)[1] if done else msg
        if frac >= 0 and not done:
            text += f"  ({frac:.0%})"
        self.ai_note.setText(text.strip())
        if done:
            self.ai_btn.setText("Set up")
            self.ai_btn.setVisible(not msg.startswith("done:"))
