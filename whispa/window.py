"""The Whispa window: Home (stats + dictation history), Dictionary, Settings.

It talks to the running app through a small controller interface (see
``WhispaApp`` in app.py), so it can also be rendered with a fake controller.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QLayout,
                               QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget,
                               QVBoxLayout, QWidget)

from . import __version__, theme
from .config import Config
from .hotkey import Hotkey
from .icons import app_icon, app_pixmap
from .learn import Dictionary
from .widgets import HotkeyField, StatCard, Switch, card, setting_row

TYPING_WPM = 40  # average typing speed, for the "time saved" estimate


class Controller(Protocol):
    cfg: Config
    dictionary: Dictionary
    history_path: Path

    def status_text(self) -> str: ...
    def llm_status(self) -> tuple[bool, str]: ...
    def capture_hotkey(self, kind: str) -> None: ...
    def cancel_capture(self) -> None: ...
    def update_setting(self, key: str, value) -> str | None: ...
    def restart(self) -> None: ...
    def open_folder(self) -> None: ...
    def delete_history(self, t: float) -> None: ...


# --------------------------------------------------------------------------- helpers
def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("text"):
                rows.append(row)
    rows.reverse()
    return rows


def hotkey_labels(spec: str) -> list[str]:
    try:
        label = Hotkey.parse(spec).label()
    except ValueError:
        return [spec]
    if sys.platform == "darwin" and "+" not in label:
        return list(label) if all(len(c) == 1 for c in label) and label != "fn" else [label]
    return label.split("+")


def _day_label(day: dt.date) -> str:
    today = dt.date.today()
    if day == today:
        return "Today"
    if day == today - dt.timedelta(days=1):
        return "Yesterday"
    fmt = "%A, %B %d" if day.year == today.year else "%B %d, %Y"
    return day.strftime(fmt).replace(" 0", " ")


def _time_label(t: float) -> str:
    return dt.datetime.fromtimestamp(t).strftime("%I:%M %p").lstrip("0")


def _fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(minutes)} min"
    return f"{minutes / 60:.1f} h"


_APP_NAMES = {
    "code": "VS Code", "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox",
    "outlook": "Outlook", "olk": "Outlook", "winword": "Word", "excel": "Excel",
    "powerpnt": "PowerPoint", "onenote": "OneNote", "ms-teams": "Teams", "teams": "Teams",
    "windowsterminal": "Terminal", "cmd": "Command Prompt", "powershell": "PowerShell",
    "explorer": "File Explorer", "notepad": "Notepad", "slack": "Slack", "discord": "Discord",
    "whatsapp": "WhatsApp", "telegram": "Telegram", "notion": "Notion", "obsidian": "Obsidian",
    "cursor": "Cursor", "claude": "Claude", "chatgpt": "ChatGPT", "spotify": "Spotify",
    "figma": "Figma", "zoom": "Zoom", "signal": "Signal", "thunderbird": "Thunderbird",
}


def pretty_app(name: str) -> str:
    if not name:
        return ""
    known = _APP_NAMES.get(name.lower())
    if known:
        return known
    return name.title() if name.islower() or name.isupper() else name


def clear_layout(layout) -> None:
    """Remove and destroy all widgets in a layout (hidden first so they never flash)."""
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        elif item.layout() is not None:
            clear_layout(item.layout())


class FlowLayout(QLayout):
    """Wraps children onto new lines, for the word chips."""

    def __init__(self, parent=None, spacing: int = 6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), dry=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, dry=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _layout(self, rect, dry):
        x, y, line_h = rect.x(), rect.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            if x + hint.width() > rect.right() and line_h:
                x, y = rect.x(), y + line_h + self._spacing
                line_h = 0
            if not dry:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y()


def _page(title: str, subtitle: str) -> tuple[QScrollArea, QVBoxLayout]:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(40, 32, 40, 32)
    lay.setSpacing(12)
    h = QLabel(title)
    h.setObjectName("h1")
    lay.addWidget(h)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("sub")
        s.setWordWrap(True)
        lay.addWidget(s)
    lay.addSpacing(8)
    area.setWidget(inner)
    return area, lay


def _section(lay: QVBoxLayout, title: str) -> QVBoxLayout:
    h = QLabel(title)
    h.setObjectName("h2")
    lay.addSpacing(10)
    lay.addWidget(h)
    frame, inner = card()
    lay.addWidget(frame)
    return inner


# --------------------------------------------------------------------------- Home
class EntryRow(QFrame):
    def __init__(self, entry: dict, on_delete):
        super().__init__()
        self.setObjectName("entryRow")
        self.entry = entry
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 6, 8)
        lay.setSpacing(14)
        t = QLabel(_time_label(entry.get("t", 0)))
        t.setObjectName("time")
        t.setFixedWidth(62)
        t.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(t, 0, Qt.AlignmentFlag.AlignTop)
        body = QVBoxLayout()
        body.setSpacing(4)
        if entry.get("command"):
            cmd = QLabel(f"✦ {entry['command']}")
            cmd.setObjectName("command")
            cmd.setWordWrap(True)
            body.addWidget(cmd)
        text = QLabel(entry["text"])
        text.setObjectName("entry")
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body.addWidget(text)
        if entry.get("app"):
            app = QLabel(pretty_app(entry["app"]))
            app.setObjectName("app")
            row = QHBoxLayout()
            row.addWidget(app)
            row.addStretch(1)
            body.addLayout(row)
        lay.addLayout(body, 1)
        self.copy = QPushButton("Copy")
        self.copy.setObjectName("icon")
        self.copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy.clicked.connect(self._copy)
        self.delete = QPushButton("✕")
        self.delete.setObjectName("icon")
        self.delete.setToolTip("Delete from history")
        self.delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete.clicked.connect(lambda: on_delete(self))
        for b in (self.copy, self.delete):
            b.setVisible(False)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.entry["text"])
        self.copy.setText("Copied")
        QTimer.singleShot(1200, lambda: self.copy.setText("Copy"))

    def enterEvent(self, e):
        self.copy.setVisible(True)
        self.delete.setVisible(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.copy.setVisible(False)
        self.delete.setVisible(False)
        super().leaveEvent(e)


class HomePage(QScrollArea):
    MAX_ROWS = 300

    def __init__(self, ctl: Controller):
        area, lay = _page(self._greeting(), "")
        super().__init__()
        self.ctl = ctl
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        inner = area.takeWidget()
        self.setWidget(inner)
        self._title = inner.findChild(QLabel, "h1")
        self.hint = QLabel()
        self.hint.setObjectName("sub")
        lay.insertWidget(1, self.hint)

        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.s_words = StatCard("words dictated")
        self.s_wpm = StatCard("average speaking speed (wpm)")
        self.s_saved = StatCard(f"time saved vs typing at {TYPING_WPM} wpm")
        for s in (self.s_words, self.s_wpm, self.s_saved):
            stats.addWidget(s)
        lay.addLayout(stats)

        head = QHBoxLayout()
        h = QLabel("Dictations")
        h.setObjectName("h2")
        head.addWidget(h)
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(240)
        head.addWidget(self.search)
        lay.addSpacing(12)
        lay.addLayout(head)

        self.list = QVBoxLayout()
        self.list.setSpacing(0)
        lay.addLayout(self.list)
        lay.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self.refresh)
        self.search.textChanged.connect(lambda: self._debounce.start())

    @staticmethod
    def _greeting() -> str:
        h = dt.datetime.now().hour
        return "Good morning" if h < 12 else "Good afternoon" if h < 18 else "Good evening"

    def refresh(self) -> None:
        self._title.setText(self._greeting())
        labels = " + ".join(hotkey_labels(self.ctl.cfg.hotkey))
        self.hint.setText(f"Hold {labels} anywhere and start talking. Everything stays on this computer.")
        rows = load_history(self.ctl.history_path)
        dictated = [r for r in rows if not r.get("command")]
        words = sum(len(r["text"].split()) for r in dictated)
        seconds = sum(r.get("seconds", 0) for r in dictated)
        self.s_words.set(_fmt_int(words))
        self.s_wpm.set(_fmt_int(words / seconds * 60) if seconds > 5 else "–")
        saved = max(0.0, words / TYPING_WPM * 60 - seconds)
        self.s_saved.set(_fmt_duration(saved) if rows else "–")

        clear_layout(self.list)
        q = self.search.text().strip().lower()
        if q:
            rows = [r for r in rows if q in r["text"].lower() or q in r.get("app", "").lower()]
        if not rows:
            empty = QLabel("No matches." if q else
                           "No dictations yet. Your transcripts will show up here.")
            empty.setObjectName("muted")
            empty.setContentsMargins(4, 16, 4, 16)
            self.list.addWidget(empty)
            return
        last_day = None
        for r in rows[: self.MAX_ROWS]:
            day = dt.date.fromtimestamp(r.get("t", 0))
            if day != last_day:
                d = QLabel(_day_label(day))
                d.setObjectName("day")
                self.list.addWidget(d)
                last_day = day
            self.list.addWidget(EntryRow(r, self._delete))

    def _delete(self, row: EntryRow) -> None:
        self.ctl.delete_history(row.entry.get("t", 0))
        self.refresh()


# --------------------------------------------------------------------------- Dictionary
class DictionaryPage(QScrollArea):
    def __init__(self, ctl: Controller):
        super().__init__()
        self.ctl = ctl
        area, lay = _page("Dictionary",
                          "Whispa learns names and words it got wrong when you fix them after a "
                          "dictation. Add your own words, corrections and snippets here too.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())

        words = _section(lay, "Your words")
        desc = QLabel("Names, products and jargon. Near-misses snap to your spelling.")
        desc.setObjectName("muted")
        desc.setContentsMargins(0, 12, 0, 8)
        words.addWidget(desc)
        add = QHBoxLayout()
        self.term_in = QLineEdit()
        self.term_in.setPlaceholderText("e.g. Kubernetes, Siobhan, PyTorch")
        self.term_in.returnPressed.connect(self._add_term)
        b = QPushButton("Add")
        b.setObjectName("primary")
        b.clicked.connect(self._add_term)
        add.addWidget(self.term_in, 1)
        add.addWidget(b)
        words.addLayout(add)
        self.chips_host = QWidget()
        self.chips = FlowLayout(self.chips_host)
        self.chips_host.setContentsMargins(0, 10, 0, 12)
        words.addWidget(self.chips_host)

        corr = _section(lay, "Corrections")
        desc2 = QLabel("When Whispa hears the left side, it writes the right side.")
        desc2.setObjectName("muted")
        desc2.setContentsMargins(0, 12, 0, 8)
        corr.addWidget(desc2)
        add2 = QHBoxLayout()
        self.heard_in = QLineEdit()
        self.heard_in.setPlaceholderText("When I say…")
        self.meant_in = QLineEdit()
        self.meant_in.setPlaceholderText("Write…")
        self.meant_in.returnPressed.connect(self._add_correction)
        b2 = QPushButton("Add")
        b2.setObjectName("primary")
        b2.clicked.connect(self._add_correction)
        add2.addWidget(self.heard_in, 1)
        arrow = QLabel("→")
        arrow.setObjectName("muted")
        add2.addWidget(arrow)
        add2.addWidget(self.meant_in, 1)
        add2.addWidget(b2)
        corr.addLayout(add2)
        self.corr_list = QVBoxLayout()
        self.corr_list.setContentsMargins(0, 8, 0, 8)
        self.corr_list.setSpacing(0)
        corr.addLayout(self.corr_list)

        snip = _section(lay, "Snippets")
        desc3 = QLabel("Say the trigger phrase and Whispa types the saved text instead: an email "
                       "address, a link, an address, a sign-off. Pick phrases you wouldn't say by "
                       "accident, like “my calendly link”.")
        desc3.setObjectName("muted")
        desc3.setWordWrap(True)
        desc3.setContentsMargins(0, 12, 0, 8)
        snip.addWidget(desc3)
        add3 = QHBoxLayout()
        add3.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.trigger_in = QLineEdit()
        self.trigger_in.setPlaceholderText("When I say…  e.g. my email")
        self.expansion_in = QPlainTextEdit()
        self.expansion_in.setPlaceholderText("Type this…  (can be several lines)")
        self.expansion_in.setFixedHeight(64)
        self.expansion_in.setTabChangesFocus(True)
        b3 = QPushButton("Add")
        b3.setObjectName("primary")
        b3.clicked.connect(self._add_snippet)
        add3.addWidget(self.trigger_in, 2, Qt.AlignmentFlag.AlignTop)
        arrow3 = QLabel("→")
        arrow3.setObjectName("muted")
        add3.addWidget(arrow3, 0, Qt.AlignmentFlag.AlignTop)
        add3.addWidget(self.expansion_in, 3)
        add3.addWidget(b3, 0, Qt.AlignmentFlag.AlignTop)
        snip.addLayout(add3)
        self.snip_list = QVBoxLayout()
        self.snip_list.setContentsMargins(0, 8, 0, 8)
        self.snip_list.setSpacing(0)
        snip.addLayout(self.snip_list)

        self.pending = QLabel()
        self.pending.setObjectName("faint")
        lay.addWidget(self.pending)
        lay.addStretch(1)

    def refresh(self) -> None:
        terms, corrections = self.ctl.dictionary.snapshot()
        clear_layout(self.chips)
        if not terms:
            none = QLabel("No words yet.")
            none.setObjectName("faint")
            self.chips.addWidget(none)
        for term in terms:
            chip = QFrame()
            chip.setObjectName("chipFrame")
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            lab = QLabel(term)
            lab.setObjectName("chip")
            x = QPushButton("✕")
            x.setObjectName("icon")
            x.setCursor(Qt.CursorShape.PointingHandCursor)
            x.setToolTip(f"Remove {term}")
            x.clicked.connect(lambda _=False, t=term: self._remove_term(t))
            cl.addWidget(lab)
            cl.addWidget(x)
            self.chips.addWidget(chip)
        self.chips_host.updateGeometry()

        clear_layout(self.corr_list)
        if not corrections:
            none = QLabel("Nothing learned yet. Fix a misheard word after dictating and it appears here.")
            none.setObjectName("faint")
            none.setWordWrap(True)
            none.setContentsMargins(0, 8, 0, 8)
            self.corr_list.addWidget(none)
        for i, (heard, meant) in enumerate(sorted(corrections.items())):
            x = QPushButton("✕")
            x.setObjectName("icon")
            x.setCursor(Qt.CursorShape.PointingHandCursor)
            x.clicked.connect(lambda _=False, h=heard: self._remove_correction(h))
            row = setting_row(f"{heard}  →  {meant}", "", x, first=i == 0)
            self.corr_list.addWidget(row)
        clear_layout(self.snip_list)
        snippets = self.ctl.dictionary.snippets_snapshot()
        if not snippets:
            none = QLabel("No snippets yet.")
            none.setObjectName("faint")
            none.setContentsMargins(0, 8, 0, 8)
            self.snip_list.addWidget(none)
        for i, (trigger, text) in enumerate(sorted(snippets.items())):
            x = QPushButton("✕")
            x.setObjectName("icon")
            x.setCursor(Qt.CursorShape.PointingHandCursor)
            x.clicked.connect(lambda _=False, t=trigger: self._remove_snippet(t))
            preview = text.replace("\n", " ⏎ ")
            preview = preview if len(preview) <= 90 else preview[:87] + "…"
            self.snip_list.addWidget(setting_row(f"“{trigger}”", preview, x, first=i == 0))

        n = len(self.ctl.dictionary.pending)
        self.pending.setText(f"{n} more fix{'es' if n != 1 else ''} waiting to be seen a second time."
                             if n else "")

    def _add_term(self) -> None:
        self.ctl.dictionary.add_term(self.term_in.text())
        self.term_in.clear()
        self.refresh()

    def _remove_term(self, term: str) -> None:
        self.ctl.dictionary.remove_term(term)
        self.refresh()

    def _add_correction(self) -> None:
        self.ctl.dictionary.set_correction(self.heard_in.text(), self.meant_in.text())
        self.heard_in.clear()
        self.meant_in.clear()
        self.refresh()

    def _remove_correction(self, heard: str) -> None:
        self.ctl.dictionary.remove_correction(heard)
        self.refresh()

    def _add_snippet(self) -> None:
        self.ctl.dictionary.set_snippet(self.trigger_in.text(), self.expansion_in.toPlainText().strip())
        self.trigger_in.clear()
        self.expansion_in.clear()
        self.refresh()

    def _remove_snippet(self, trigger: str) -> None:
        self.ctl.dictionary.remove_snippet(trigger)
        self.refresh()


# --------------------------------------------------------------------------- Settings
class SettingsPage(QScrollArea):
    def __init__(self, ctl: Controller, mic_options: list[tuple[str, str]]):
        super().__init__()
        self.ctl = ctl
        cfg = ctl.cfg
        area, lay = _page("Settings", "Changes apply immediately.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())

        # Shortcuts
        sec = _section(lay, "Shortcuts")
        self.hotkey = HotkeyField(hotkey_labels(cfg.hotkey))
        self.hotkey.change_requested.connect(lambda: self._record("hotkey"))
        self.hotkey.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Push to talk", "Hold to dictate, release to insert the text.",
                                  self.hotkey, first=True))
        self.hands = HotkeyField([hotkey_labels(cfg.hands_free_key)[0]] if cfg.hands_free_key else ["Off"])
        self.hands.change_requested.connect(lambda: self._record("hands_free_key"))
        self.hands.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Hands-free lock",
                                  "Tap this while holding the shortcut to keep listening without "
                                  "holding. Press the shortcut again to finish.", self.hands))
        self.command = HotkeyField(hotkey_labels(cfg.command_hotkey) if cfg.command_hotkey else ["Off"])
        self.command.change_requested.connect(lambda: self._record("command_hotkey"))
        self.command.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Command Mode",
                                  "Select text, hold this and say what to do: “make this more "
                                  "formal”, “turn this into bullet points”, “translate to "
                                  "Spanish”. With nothing selected, it writes what you ask for. "
                                  "Uses the local AI.", self.command))
        sec.addWidget(self._switch_row("double_tap_to_lock", "Double-tap for hands-free",
                                       "Double-tap the shortcut to start hands-free mode."))
        self.hotkey_msg = QLabel("")
        self.hotkey_msg.setObjectName("error")
        self.hotkey_msg.setWordWrap(True)
        self.hotkey_msg.setVisible(False)
        sec.addWidget(self.hotkey_msg)

        # Microphone
        sec = _section(lay, "Microphone")
        self.mic = QComboBox()
        self.mic.setMinimumWidth(280)
        self.mic.addItem("System default", None)
        current = cfg.input_device
        for label, value in mic_options:
            self.mic.addItem(label, value)
        if current not in (None, ""):
            idx = self.mic.findData(current)
            if idx < 0:
                self.mic.addItem(f"{current} (not connected)", current)
                idx = self.mic.count() - 1
            self.mic.setCurrentIndex(idx)
        self.mic.currentIndexChanged.connect(
            lambda: self._set("input_device", self.mic.currentData()))
        sec.addWidget(setting_row("Input device", "", self.mic, first=True))
        sec.addWidget(self._switch_row("keep_mic_open", "Instant start",
                                       "Keep the microphone open so the first word is never cut "
                                       "off. Your OS will show the mic as in use."))

        # Writing
        sec = _section(lay, "Writing")
        sec.addWidget(self._switch_row("remove_fillers", "Remove filler words",
                                       "Drops um, uh, hmm.", first=True))
        sec.addWidget(self._switch_row("voice_commands", "Voice commands",
                                       "Say “new line”, “new paragraph” or “scratch that”."))
        sec.addWidget(self._switch_row("append_space", "Add a space after each dictation", ""))
        self.no_field = QComboBox()
        self.no_field.addItem("Show a copy card", "popup")
        self.no_field.addItem("Paste anyway", "paste")
        self.no_field.setCurrentIndex(max(0, self.no_field.findData(cfg.no_text_field)))
        self.no_field.currentIndexChanged.connect(
            lambda: self._set("no_text_field", self.no_field.currentData()))
        sec.addWidget(setting_row("When no text field is selected", "", self.no_field))

        # Smart features
        sec = _section(lay, "Smart features")
        self.llm_mode = QComboBox()
        self.llm_mode.addItem("Off", "off")
        self.llm_mode.addItem("Only when needed", "smart")
        self.llm_mode.addItem("Always", "always")
        llm = cfg.llm_cleanup
        current = "off" if llm.enabled in (False, "off") else ("always" if llm.when == "always" else "smart")
        self.llm_mode.setCurrentIndex(self.llm_mode.findData(current))
        self.llm_mode.currentIndexChanged.connect(
            lambda: (self._set("llm_mode", self.llm_mode.currentData()), self.refresh()))
        self.llm_row_desc = QLabel()
        self.llm_row_desc.setWordWrap(True)
        llm_row = setting_row(
            "AI clean-up",
            "Fixes self-corrections (“Tuesday, no wait, Wednesday”), stumbles, lists and grammar "
            "with a local AI model. “Only when needed” runs it just on dictations that look like "
            "they need it, so the rest stay instant.", self.llm_mode, first=True)
        llm_row.layout().itemAt(0).layout().addWidget(self.llm_row_desc)
        sec.addWidget(llm_row)
        sec.addWidget(self._switch_row("learn_from_edits", "Learn from my corrections",
                                       "After pasting, notice when you fix a word and remember it."))
        self.samples_info = QLabel()
        self.samples_info.setObjectName("faint")
        self.samples_info.setWordWrap(True)
        self.samples_delete = QPushButton("Delete samples")
        self.samples_delete.clicked.connect(self._delete_samples)
        sec.addWidget(self._switch_row("save_training_audio", "Keep voice samples",
                                       "Save recordings with their corrected text, to tune the "
                                       "speech model to your voice later. Stored only on this "
                                       "computer."))

        samples = QHBoxLayout()
        samples.addWidget(self.samples_info, 1)
        samples.addWidget(self.samples_delete)
        sec.addLayout(samples)
        sec.addSpacing(10)

        # App
        sec = _section(lay, "App")
        sec.addWidget(self._switch_row("launch_at_login", "Launch at login", "", first=True,
                                       value=self._autostart_state()))
        sec.addWidget(self._switch_row("sounds", "Sounds", "Chimes when dictation starts and stops."))
        sec.addWidget(self._switch_row("show_idle_pill", "Show the pill when idle",
                                       "A small bar at the bottom of the screen. Click it to "
                                       "dictate hands-free."))
        from .overlay import ANCHOR_LABELS

        self.pill_pos = QComboBox()
        for anchor, label in ANCHOR_LABELS.items():
            self.pill_pos.addItem(label, anchor)
        self.pill_pos.setCurrentIndex(max(0, self.pill_pos.findData(cfg.pill_anchor)))
        self.pill_pos.currentIndexChanged.connect(
            lambda: self._set("pill_anchor", self.pill_pos.currentData()))
        sec.addWidget(setting_row("Pill position", "Or just drag the pill. It snaps to the nearest "
                                  "corner or edge.", self.pill_pos))
        self.device = QComboBox()
        for label, value in (("Automatic", "auto"), ("NVIDIA GPU (CUDA)", "cuda"),
                             ("GPU (DirectML)", "directml"), ("CPU", "cpu")):
            self.device.addItem(label, value)
        self.device.setCurrentIndex(max(0, self.device.findData(cfg.device)))
        self.device.currentIndexChanged.connect(self._device_changed)
        self.device_note = QLabel("")
        self.device_note.setObjectName("faint")
        sec.addWidget(setting_row("Speech model runs on", "", self.device))
        sec.addWidget(self.device_note)

        # Updates
        sec = _section(lay, "Updates")
        self.update_title = QLabel()
        self.update_title.setStyleSheet("font-weight: 600;")
        self.update_detail = QLabel()
        self.update_detail.setObjectName("muted")
        self.update_detail.setWordWrap(True)
        self.update_changes = QLabel()
        self.update_changes.setObjectName("muted")
        self.update_changes.setWordWrap(True)
        self.update_check_btn = QPushButton("Check for updates")
        self.update_check_btn.clicked.connect(ctl.check_updates)
        self.update_now_btn = QPushButton("Update now")
        self.update_now_btn.setObjectName("primary")
        self.update_now_btn.clicked.connect(ctl.apply_update)
        head = QHBoxLayout()
        head.setContentsMargins(0, 12, 0, 4)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        texts.addWidget(self.update_title)
        texts.addWidget(self.update_detail)
        head.addLayout(texts, 1)
        head.addWidget(self.update_check_btn, 0, Qt.AlignmentFlag.AlignTop)
        head.addWidget(self.update_now_btn, 0, Qt.AlignmentFlag.AlignTop)
        sec.addLayout(head)
        self.update_changes.setContentsMargins(0, 0, 0, 10)
        sec.addWidget(self.update_changes)
        self.update_auto = self._switch_row("update_check", "Check automatically",
                                            "Once a day. Only public code is downloaded from "
                                            "GitHub; nothing about you is sent.")
        sec.addWidget(self.update_auto)
        self.rollback_btn = QPushButton("Go back to the previous version")
        self.rollback_btn.clicked.connect(ctl.rollback_update)
        self.rollback_row = setting_row("Something wrong after an update?", "", self.rollback_btn)
        sec.addWidget(self.rollback_row)

        about = QHBoxLayout()
        self.status = QLabel()
        self.status.setObjectName("faint")
        about.addWidget(self.status, 1)
        folder = QPushButton("Open data folder")
        folder.clicked.connect(ctl.open_folder)
        restart = QPushButton("Restart Whispa")
        restart.clicked.connect(ctl.restart)
        about.addWidget(folder)
        about.addWidget(restart)
        lay.addSpacing(14)
        lay.addLayout(about)
        lay.addStretch(1)

    # -- helpers ---------------------------------------------------------
    def _switch_row(self, key: str, title: str, desc: str, first: bool = False, value=None) -> QFrame:
        sw = Switch(bool(getattr(self.ctl.cfg, key)) if value is None else value)
        sw.toggled.connect(lambda on, k=key: self._set(k, on))
        return setting_row(title, desc, sw, first=first)

    def _autostart_state(self) -> bool:
        try:
            from . import autostart

            return autostart.is_enabled()
        except Exception:  # noqa: BLE001
            return False

    def _set(self, key: str, value) -> None:
        err = self.ctl.update_setting(key, value)
        if err:
            self.hotkey_msg.setText(err)
            self.hotkey_msg.setVisible(True)

    def _device_changed(self) -> None:
        self._set("device", self.device.currentData())
        self.device_note.setText("Takes effect after a restart.")

    def _record(self, kind: str) -> None:
        self._cancel_record()
        self.hotkey_msg.setVisible(False)
        self._field(kind).set_recording(True)
        self.ctl.capture_hotkey(kind)

    def _cancel_record(self) -> None:
        self.ctl.cancel_capture()
        for field in (self.hotkey, self.hands, self.command):
            field.set_recording(False)

    def _field(self, kind: str) -> HotkeyField:
        return {"hotkey": self.hotkey, "hands_free_key": self.hands, "command_hotkey": self.command}[kind]

    def sync_pill_anchor(self, anchor: str) -> None:
        """The pill was dragged: show its new spot without re-triggering a move."""
        i = self.pill_pos.findData(anchor)
        if i >= 0 and i != self.pill_pos.currentIndex():
            self.pill_pos.blockSignals(True)
            self.pill_pos.setCurrentIndex(i)
            self.pill_pos.blockSignals(False)

    def capture_finished(self, kind: str, spec: str | None, error: str | None) -> None:
        field = self._field(kind)
        field.set_recording(False)
        if error:
            self.hotkey_msg.setText(error)
            self.hotkey_msg.setVisible(True)
            return
        if spec is None:
            return  # cancelled
        err = self.ctl.update_setting(kind, spec)
        if err:
            self.hotkey_msg.setText(err)
            self.hotkey_msg.setVisible(True)
            return
        field.set_labels(hotkey_labels(spec))

    def _delete_samples(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        if QMessageBox.question(self, "Delete voice samples",
                                "Delete all saved voice samples? This can't be undone.") \
                == QMessageBox.StandardButton.Yes:
            self.ctl.delete_training()
            self.refresh()

    def refresh(self) -> None:
        st = self.ctl.training_stats()
        if st:
            minutes = st["seconds"] / 60
            self.samples_info.setText(
                f"{st['clips']} clips · {minutes:.0f} min of speech · {st['bytes'] / 1e6:.0f} MB · "
                f"{st['corrected']} corrected by you. When the limit "
                f"({self.ctl.cfg.training_max_mb / 1000:g} GB) is reached, the oldest uncorrected "
                "clips are removed first.")
            self.samples_delete.setEnabled(st["clips"] > 0)
        ok, text = self.ctl.llm_status()
        self.llm_row_desc.setText(text)
        self.llm_row_desc.setObjectName("ok" if ok else "faint")
        self.llm_row_desc.style().unpolish(self.llm_row_desc)
        self.llm_row_desc.style().polish(self.llm_row_desc)
        self.status.setText(f"Whispa {__version__} · {self.ctl.status_text()}")
        self.refresh_updates()

    def refresh_updates(self) -> None:
        u = self.ctl.update_state()
        if not u["installed"]:
            self.update_title.setText(f"Whispa {__version__} · developer copy")
            self.update_detail.setText("Running from a git checkout: update it with git pull. "
                                       "Copies installed with the installer update here.")
            for w in (self.update_check_btn, self.update_now_btn, self.update_auto, self.rollback_row,
                      self.update_changes):
                w.setVisible(False)
            return
        version = f"Whispa {__version__}"
        if u["sha"]:
            version += f" · {u['sha'][:7]}"
        if u["date"]:
            version += f" · {u['date'][:10]}"
        self.update_title.setText(version)
        busy = bool(u["busy"])
        if u["busy"]:
            detail = u["busy"]
        elif u["error"]:
            detail = u["error"]
        elif u["available"]:
            n = len(u["changes"])
            detail = (f"{n} update{'s' if n != 1 else ''} available." if n else "An update is available.") + \
                " Updating restarts Whispa (a few seconds)."
        elif u["checked"]:
            detail = "You're up to date."
        else:
            detail = "Updates come from GitHub, only versions that passed the tests."
        self.update_detail.setText(detail)
        self.update_detail.setObjectName("error" if u["error"] and not busy else "muted")
        self.update_detail.style().unpolish(self.update_detail)
        self.update_detail.style().polish(self.update_detail)
        changes = u["changes"][:8]
        self.update_changes.setText("\n".join(f"•  {c}" for c in changes)
                                    + (f"\n…and {len(u['changes']) - 8} more" if len(u["changes"]) > 8 else ""))
        self.update_changes.setVisible(bool(u["available"] and changes))
        self.update_now_btn.setVisible(bool(u["available"]))
        self.update_now_btn.setEnabled(not busy)
        self.update_check_btn.setEnabled(not busy)
        self.update_check_btn.setVisible(True)
        self.update_auto.setVisible(True)
        self.rollback_row.setVisible(bool(u["has_previous"]))
        self.rollback_btn.setEnabled(not busy)


# --------------------------------------------------------------------------- window
class MainWindow(QWidget):
    def __init__(self, ctl: Controller, mic_options: list[tuple[str, str]] | None = None):
        super().__init__()
        self.ctl = ctl
        self.setObjectName("root")
        self.setWindowTitle("Whispa")
        self.setWindowIcon(app_icon())
        self.resize(980, 680)
        self.setMinimumSize(820, 540)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 18, 14, 16)
        sl.setSpacing(4)
        brand = QHBoxLayout()
        logo = QLabel()
        logo.setPixmap(app_pixmap(56).scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
        brand.addWidget(logo)
        name = QLabel("Whispa")
        name.setObjectName("brand")
        brand.addWidget(name)
        brand.addStretch(1)
        sl.addLayout(brand)
        sl.addSpacing(18)

        self.pages = QStackedWidget()
        self.home = HomePage(ctl)
        self.dictionary = DictionaryPage(ctl)
        if mic_options is None:
            try:
                from .audio import input_devices

                mic_options = input_devices()
            except Exception:  # noqa: BLE001
                mic_options = []
        self.settings = SettingsPage(ctl, mic_options)
        self.nav = QButtonGroup(self)
        for i, (label, page) in enumerate((("Home", self.home), ("Dictionary", self.dictionary),
                                           ("Settings", self.settings))):
            self.pages.addWidget(page)
            b = QPushButton(label)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(b, i)
            sl.addWidget(b)
        self.nav.idClicked.connect(self.show_page)
        sl.addStretch(1)
        self.side_status = QLabel()
        self.side_status.setObjectName("faint")
        self.side_status.setWordWrap(True)
        sl.addWidget(self.side_status)

        root.addWidget(side)
        root.addWidget(self.pages, 1)
        self.apply_theme()
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _s: self.apply_theme())
        except AttributeError:
            pass
        self.show_page(0)

    def apply_theme(self) -> None:
        self.setStyleSheet(theme.stylesheet(theme.current()))

    def show_page(self, index: int) -> None:
        self.nav.button(index).setChecked(True)
        self.pages.setCurrentIndex(index)
        self.pages.currentWidget().refresh()

    def refresh(self) -> None:
        self.side_status.setText("● " + self.ctl.status_text())
        self.pages.currentWidget().refresh()

    def history_changed(self) -> None:
        if self.isVisible() and self.pages.currentWidget() is self.home:
            self.home.refresh()

    def bring_to_front(self) -> None:
        self.refresh()
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        # Closing the window keeps Whispa running in the tray (like Wispr Flow).
        self.settings._cancel_record()
        event.ignore()
        self.hide()
