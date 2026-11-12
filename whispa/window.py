"""The Murmr window: Home (stats + dictation history), Dictionary, Settings.

It talks to the running app through a small controller interface (see
``MurmrApp`` in app.py), so it can also be rendered with a fake controller.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import sys
from pathlib import Path
from typing import Protocol

from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QGridLayout, QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel, QLayout,
                               QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget,
                               QVBoxLayout, QWidget)

from . import __version__, theme
from .config import Config
from .config import config_dir as config_mod_dir
from .hotkey import Hotkey
from .icons import app_icon, app_pixmap
from .learn import Dictionary
from .widgets import HotkeyField, Label, StatCard, Switch, card, nav_icon, setting_row

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
    def paste_text(self, text: str) -> None: ...
    def speak_sample(self) -> None: ...


# --------------------------------------------------------------------------- helpers
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


def day_streak(days: set[dt.date], today: dt.date | None = None) -> int:
    """Days in a row with a dictation, up to today (or yesterday, if not yet today)."""
    today = today or dt.date.today()
    day = today if today in days else today - dt.timedelta(days=1)
    n = 0
    while day in days:
        n += 1
        day -= dt.timedelta(days=1)
    return n


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


OUTPUT_LANGUAGES = ["", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch",
                    "Japanese", "Chinese", "Korean"]


def _page(title: str, subtitle: str) -> tuple[QScrollArea, QVBoxLayout]:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    inner = QWidget()
    lay = QVBoxLayout(inner)
    lay.setContentsMargins(40, 32, 40, 32)
    lay.setSpacing(12)
    h = Label(title)
    h.setObjectName("h1")
    lay.addWidget(h)
    if subtitle:
        s = Label(subtitle)
        s.setObjectName("sub")
        s.setWordWrap(True)
        lay.addWidget(s)
    lay.addSpacing(8)
    area.setWidget(inner)
    return area, lay


def _section(lay: QVBoxLayout, title: str) -> QVBoxLayout:
    h = Label(title)
    h.setObjectName("h2")
    lay.addSpacing(10)
    lay.addWidget(h)
    frame, inner = card()
    lay.addWidget(frame)
    return inner


# --------------------------------------------------------------------------- Home
class _Relay(QObject):
    done = Signal(str, str)  # result, error


class EntryRow(QFrame):
    def __init__(self, entry: dict, on_delete, on_paste=None, ctl=None):
        super().__init__()
        self.setObjectName("entryRow")
        self.entry = entry
        self.ctl = ctl
        self._relay = _Relay()
        self._relay.done.connect(self._redone)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 6, 8)
        lay.setSpacing(14)
        t = Label(_time_label(entry.get("t", 0)))
        t.setObjectName("time")
        t.setFixedWidth(62)
        t.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(t, 0, Qt.AlignmentFlag.AlignTop)
        body = QVBoxLayout()
        body.setSpacing(4)
        if entry.get("command"):
            cmd = Label(f"✦ {entry['command']}")
            cmd.setObjectName("command")
            cmd.setWordWrap(True)
            body.addWidget(cmd)
        text = Label(entry["text"])
        text.setObjectName("entry")
        text.setWordWrap(True)
        text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        body.addWidget(text)
        raw = (entry.get("raw") or "").strip()
        differs = bool(raw) and " ".join(raw.lower().split()) != " ".join(entry["text"].lower().split())
        self.heard = Label(f"Heard: {raw}")
        self.heard.setObjectName("faint")
        self.heard.setWordWrap(True)
        self.heard.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.heard.setVisible(False)
        body.addWidget(self.heard)
        self.redo_out = Label("")
        self.redo_out.setObjectName("muted")
        self.redo_out.setWordWrap(True)
        self.redo_out.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.redo_out.setVisible(False)
        body.addWidget(self.redo_out)
        if entry.get("app"):
            app = Label(pretty_app(entry["app"]))
            app.setObjectName("app")
            row = QHBoxLayout()
            row.addWidget(app)
            row.addStretch(1)
            body.addLayout(row)
        lay.addLayout(body, 1)
        self.paste = QPushButton("Paste")
        self.paste.setObjectName("icon")
        self.paste.setToolTip("Paste into the app you were just using")
        self.paste.setCursor(Qt.CursorShape.PointingHandCursor)
        self.paste.clicked.connect(lambda: on_paste and on_paste(self.entry["text"]))
        self.copy = QPushButton("Copy")
        self.copy.setObjectName("icon")
        self.copy.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy.clicked.connect(self._copy)
        self.delete = QPushButton("✕")
        self.delete.setObjectName("icon")
        self.delete.setToolTip("Delete from history")
        self.delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.delete.clicked.connect(lambda: on_delete(self))
        self.original = None
        if differs:  # what was heard, before tidying (the history keeps both)
            self.original = QPushButton("Original")
            self.original.setObjectName("icon")
            self.original.setToolTip("Show what Murmr heard, before tidying it")
            self.original.setCursor(Qt.CursorShape.PointingHandCursor)
            self.original.clicked.connect(lambda: self.heard.setVisible(not self.heard.isVisible()))
        self.redo = None
        if ctl is not None and hasattr(ctl, "reprocess") and raw and not entry.get("command"):
            from PySide6.QtWidgets import QMenu

            from .styles import MODE_CYCLE, mode_name

            self.redo = QPushButton("Redo as")
            self.redo.setObjectName("icon")
            self.redo.setToolTip("Redo this dictation in another mode (your current mode stays)")
            self.redo.setCursor(Qt.CursorShape.PointingHandCursor)
            menu = QMenu(self.redo)
            for mode in MODE_CYCLE:
                if mode in ("auto",) or (mode == "custom" and not ctl.cfg.custom_mode_prompt.strip()):
                    continue
                menu.addAction(mode_name(mode), lambda m=mode: self._redo(m))
            self.redo.setMenu(menu)
        self.buttons = [b for b in (self.original, self.redo, self.paste if on_paste else None,
                                    self.copy, self.delete) if b]
        for b in self.buttons:
            b.setVisible(False)
            lay.addWidget(b, 0, Qt.AlignmentFlag.AlignTop)

    def _redo(self, mode: str) -> None:
        from .styles import mode_name

        self._redo_mode = mode_name(mode)
        self.redo_out.setText(f"{self._redo_mode}: working…")
        self.redo_out.setVisible(True)
        raw = self.entry.get("raw", "")

        def work():
            try:
                self._relay.done.emit(self.ctl.reprocess(raw, mode), "")
            except Exception as exc:  # noqa: BLE001
                self._relay.done.emit("", str(exc))

        threading.Thread(target=work, name="redo", daemon=True).start()

    def _redone(self, result: str, error: str) -> None:
        if error:
            self.redo_out.setText(f"{self._redo_mode}: couldn't redo it ({error})")
            return
        QGuiApplication.clipboard().setText(result)
        self.redo_out.setText(f"{self._redo_mode} (copied):\n{result}")

    def _copy(self) -> None:
        QGuiApplication.clipboard().setText(self.entry["text"])
        self.copy.setText("Copied")
        QTimer.singleShot(1200, lambda: self.copy.setText("Copy"))

    def enterEvent(self, e):
        for b in self.buttons:
            b.setVisible(True)
        super().enterEvent(e)

    def leaveEvent(self, e):
        for b in self.buttons:
            b.setVisible(False)
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
        self.hint = Label()
        self.hint.setObjectName("sub")
        self.hint.setWordWrap(True)
        lay.insertWidget(1, self.hint)

        self.stats = QGridLayout()
        self.stats.setSpacing(12)
        self.s_words = StatCard("words dictated")
        self.s_wpm = StatCard("words per minute")
        self.s_saved = StatCard("saved vs typing")
        self.s_saved.setToolTip(f"Compared with typing the same words at {TYPING_WPM} wpm")
        self.s_streak = StatCard("day streak")
        self._stat_cols = 0
        self._place_stats(4)
        lay.addLayout(self.stats)

        head = QHBoxLayout()
        h = Label("Dictations")
        h.setObjectName("h2")
        head.addWidget(h)
        head.addStretch(1)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        self.search.setMinimumWidth(120)
        self.search.setMaximumWidth(240)
        self.search.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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

    def _place_stats(self, cols: int) -> None:
        """Four cards in a row, or two rows of two when the window is narrow."""
        if cols == self._stat_cols:
            return
        self._stat_cols = cols
        cards = (self.s_words, self.s_wpm, self.s_saved, self.s_streak)
        for c in cards:
            self.stats.removeWidget(c)
        for i, c in enumerate(cards):
            self.stats.addWidget(c, i // cols, i % cols)
        for col in range(4):
            self.stats.setColumnStretch(col, 1 if col < cols else 0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cards = (self.s_words, self.s_wpm, self.s_saved, self.s_streak)
        card = max(c.minimumSizeHint().width() for c in cards)
        m = self.widget().layout().contentsMargins()
        room = self.viewport().width() - m.left() - m.right()
        gap = self.stats.spacing()
        self._place_stats(next((n for n in (4, 2) if n * card + (n - 1) * gap <= room), 1))

    @staticmethod
    def _greeting() -> str:
        h = dt.datetime.now().hour
        return "Good morning" if h < 12 else "Good afternoon" if h < 18 else "Good evening"

    def refresh(self) -> None:
        self._title.setText(self._greeting())
        labels = " + ".join(hotkey_labels(self.ctl.cfg.hotkey))
        self.hint.setText(f"Hold {labels} anywhere and start talking. Everything stays on this computer.")
        from . import history

        path = self.ctl.history_path
        st = history.stats(path)  # cached until the file changes; includes removed dictations
        words, seconds = st["words"], st["seconds"]
        self.s_words.set(_fmt_int(words))
        self.s_wpm.set(_fmt_int(words / seconds * 60) if seconds > 5 else "–")
        saved = max(0.0, words / TYPING_WPM * 60 - seconds)
        self.s_saved.set(_fmt_duration(saved) if words else "–")
        streak = day_streak(st["days"])
        self.s_streak.set(str(streak) if streak else "–")
        self.s_words.setToolTip(f"{_fmt_int(st['week'])} in the last 7 days")

        clear_layout(self.list)
        q = self.search.text().strip().lower()
        if q:
            rows = history.scan(path, lambda r: q in r["text"].lower() or q in r.get("app", "").lower())
        else:
            rows = history.recent_rows(path, self.MAX_ROWS)[::-1]  # newest first, only the end
        if not rows:
            empty = Label("No matches." if q else
                           "No dictations yet. Your transcripts will show up here.")
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            empty.setContentsMargins(4, 16, 4, 16)
            self.list.addWidget(empty)
            return
        last_day = None
        for r in rows[: self.MAX_ROWS]:
            day = dt.date.fromtimestamp(r.get("t", 0))
            if day != last_day:
                d = Label(_day_label(day))
                d.setObjectName("day")
                self.list.addWidget(d)
                last_day = day
            self.list.addWidget(EntryRow(r, self._delete, getattr(self.ctl, "paste_text", None), self.ctl))

    def _delete(self, row: EntryRow) -> None:
        self.ctl.delete_history(row.entry.get("t", 0))
        self.refresh()


# --------------------------------------------------------------------------- Dictionary
class DictionaryPage(QScrollArea):
    def __init__(self, ctl: Controller):
        super().__init__()
        self.ctl = ctl
        area, lay = _page("Dictionary",
                          "Murmr learns names and words it got wrong when you fix them after a "
                          "dictation. Add your own words, corrections and snippets here too.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())

        io_row = QHBoxLayout()
        io_row.setContentsMargins(0, 0, 0, 0)
        imp = QPushButton("Import…")
        imp.setToolTip("Add words and corrections from a file: a Murmr export, or a text file with "
                       "one word per line, or \"heard -> meant\" for corrections.")
        imp.clicked.connect(self._import)
        exp = QPushButton("Export…")
        exp.setToolTip("Save your words, corrections and snippets to a file (to back up or move to "
                       "another computer).")
        exp.clicked.connect(self._export)
        self.io_note = Label("")
        self.io_note.setObjectName("faint")
        self.io_note.setWordWrap(True)
        io_row.addWidget(imp)
        io_row.addWidget(exp)
        io_row.addWidget(self.io_note, 1)
        lay.addLayout(io_row)

        words = _section(lay, "Your words")
        desc = Label("Names, products and jargon. Near-misses snap to your spelling.")
        desc.setObjectName("muted")
        desc.setContentsMargins(0, 12, 0, 8)
        desc.setWordWrap(True)
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
        desc2 = Label("When Murmr hears the left side, it writes the right side.")
        desc2.setObjectName("muted")
        desc2.setContentsMargins(0, 12, 0, 8)
        desc2.setWordWrap(True)
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
        arrow = Label("→")
        arrow.setObjectName("arrow")
        add2.addWidget(arrow)
        add2.addWidget(self.meant_in, 1)
        add2.addWidget(b2)
        corr.addLayout(add2)
        self.corr_list = QVBoxLayout()
        self.corr_list.setContentsMargins(0, 8, 0, 8)
        self.corr_list.setSpacing(0)
        corr.addLayout(self.corr_list)

        snip = _section(lay, "Snippets")
        desc3 = Label("Say the trigger phrase and Murmr types the saved text instead: an email "
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
        arrow3 = Label("→")
        arrow3.setObjectName("arrow")
        arrow3.setContentsMargins(0, 6, 0, 0)
        add3.addWidget(arrow3, 0, Qt.AlignmentFlag.AlignTop)
        add3.addWidget(self.expansion_in, 3)
        add3.addWidget(b3, 0, Qt.AlignmentFlag.AlignTop)
        snip.addLayout(add3)
        self.snip_list = QVBoxLayout()
        self.snip_list.setContentsMargins(0, 8, 0, 8)
        self.snip_list.setSpacing(0)
        snip.addLayout(self.snip_list)

        self.pending = Label()
        self.pending.setObjectName("faint")
        lay.addWidget(self.pending)
        lay.addStretch(1)

    def refresh(self) -> None:
        terms, corrections = self.ctl.dictionary.snapshot()
        clear_layout(self.chips)
        if not terms:
            none = Label("No words yet.")
            none.setObjectName("faint")
            self.chips.addWidget(none)
        for term in terms:
            chip = QFrame()
            chip.setObjectName("chipFrame")
            chip.setFixedHeight(26)  # the corner radius needs at least 2 x 12 px
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            lab = Label(term)
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
            none = Label("Nothing learned yet. Fix a misheard word after dictating and it appears here.")
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
            none = Label("No snippets yet.")
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

    def _import(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from .learn import parse_vocabulary

        path, _ = QFileDialog.getOpenFileName(self, "Import vocabulary", "",
                                              "Vocabulary (*.json *.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            data = parse_vocabulary(Path(path).read_text(encoding="utf-8"))
            added = self.ctl.dictionary.import_data(data)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self.io_note.setText(f"Couldn't import that file: {exc}")
            return
        self.io_note.setText(f"Imported {added['terms']} words, {added['corrections']} corrections "
                             f"and {added['snippets']} snippets.")
        self.refresh()

    def _export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Export vocabulary", "Murmr dictionary.json",
                                              "Murmr dictionary (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.ctl.dictionary.export_data(), indent=2,
                                             ensure_ascii=False) + "\n", encoding="utf-8")
        except OSError as exc:
            self.io_note.setText(f"Couldn't save: {exc}")
            return
        self.io_note.setText(f"Saved to {Path(path).name}.")

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
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if getattr(self, "_perm_timer", None) is not None:
            self._perm_timer.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        if getattr(self, "_perm_timer", None) is not None:
            self._perm_timer.stop()

    def __init__(self, ctl: Controller, mic_options: list[tuple[str, str]]):
        super().__init__()
        self.ctl = ctl
        cfg = ctl.cfg
        area, lay = _page("Settings", "Changes apply immediately.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())

        self.perm_rows: dict[str, tuple[QLabel, QPushButton]] = {}
        if sys.platform == "darwin" and hasattr(ctl, "permissions"):
            self._permissions_section(lay)

        # Shortcuts
        sec = _section(lay, "Shortcuts")
        self.hotkey = HotkeyField(hotkey_labels(cfg.hotkey))
        self.hotkey.change_requested.connect(lambda: self._record("hotkey"))
        self.hotkey.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Push to talk", "Hold to dictate, release to insert the text.",
                                  self.hotkey, first=True))
        self.backup = HotkeyField(hotkey_labels(cfg.backup_hotkey) if cfg.backup_hotkey else ["Off"],
                                  optional=True)
        self.backup.change_requested.connect(lambda: self._record("backup_hotkey"))
        self.backup.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Backup push to talk", "A second shortcut that works exactly the "
                                  "same: handy when an app takes over the first one, or for "
                                  "another keyboard.", self.backup))
        self.hands = HotkeyField([hotkey_labels(cfg.hands_free_key)[0]] if cfg.hands_free_key else ["Off"],
                                 optional=True)
        self.hands.change_requested.connect(lambda: self._record("hands_free_key"))
        self.hands.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Hands-free lock",
                                  "Tap while holding the shortcut to keep listening. Press the "
                                  "shortcut again to finish.", self.hands))
        self.command = HotkeyField(hotkey_labels(cfg.command_hotkey) if cfg.command_hotkey else ["Off"],
                                   optional=True)
        self.command.change_requested.connect(lambda: self._record("command_hotkey"))
        self.command.cancel_requested.connect(self._cancel_record)
        self.command_row = setting_row("Command Mode",
                                       "Select text, hold this and say what to do (“make this more "
                                       "formal”, “translate to Spanish”). With nothing selected, it "
                                       "writes what you ask for.", self.command)
        sec.addWidget(self.command_row)
        self.paste_last = HotkeyField(hotkey_labels(cfg.paste_last_hotkey) if cfg.paste_last_hotkey
                                      else ["Off"], optional=True)
        self.paste_last.change_requested.connect(lambda: self._record("paste_last_hotkey"))
        self.paste_last.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Paste last dictation", "Pastes your last dictation again, "
                                  "wherever the cursor is.", self.paste_last))
        self.read_key = HotkeyField(hotkey_labels(cfg.read_hotkey) if cfg.read_hotkey else ["Off"],
                                    optional=True)
        self.read_key.change_requested.connect(lambda: self._record("read_hotkey"))
        self.read_key.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Read aloud", "Select text anywhere and press this to hear it. "
                                  "Press it again to stop.", self.read_key))
        self.mode_key = HotkeyField(hotkey_labels(cfg.mode_hotkey) if cfg.mode_hotkey else ["Off"],
                                    optional=True)
        self.mode_key.change_requested.connect(lambda: self._record("mode_hotkey"))
        self.mode_key.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Switch mode", "Press to go to the next dictation mode (Message, "
                                  "Email, Notes…). The pill shows which one.", self.mode_key))
        self.word_key = HotkeyField(hotkey_labels(cfg.add_word_hotkey) if cfg.add_word_hotkey else ["Off"],
                                    optional=True)
        self.word_key.change_requested.connect(lambda: self._record("add_word_hotkey"))
        self.word_key.cancel_requested.connect(self._cancel_record)
        sec.addWidget(setting_row("Add word", "Select a name or word anywhere and press this: it's "
                                  "added to your Dictionary, spelled exactly like that.", self.word_key))
        for kind in ("backup_hotkey", "hands_free_key", "command_hotkey", "paste_last_hotkey",
                     "read_hotkey", "mode_hotkey", "add_word_hotkey"):
            self._field(kind).clear_requested.connect(lambda k=kind: self._turn_off(k))
        sec.addWidget(self._switch_row("double_tap_to_lock", "Double-tap for hands-free",
                                       "Double-tap the shortcut to start hands-free mode."))
        self.hotkey_msg = Label("")
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
        sec.addWidget(self._switch_row("smart_formatting", "Smart formatting",
                                       "“john at gmail dot com” → john@gmail.com, “20 dollars” → $20, "
                                       "“50 percent” → 50%, “330 pm” → 3:30 pm. Say “bullet point” "
                                       "to start a list item."))
        sec.addWidget(self._switch_row("append_space", "Add a space after each dictation", ""))
        self.no_field = QComboBox()
        self.no_field.addItem("Show a copy card", "popup")
        self.no_field.addItem("Paste anyway", "paste")
        self.no_field.setCurrentIndex(max(0, self.no_field.findData(cfg.no_text_field)))
        self.no_field.currentIndexChanged.connect(
            lambda: self._set("no_text_field", self.no_field.currentData()))
        sec.addWidget(setting_row("When no text field is selected", "", self.no_field))
        sec.addWidget(self._switch_row(
            "insert_original", "Insert into the field I started in",
            "Switched to another window while Murmr was still writing? It goes back to the field "
            "where you started. If that field is gone, the text is shown to copy instead."))

        # Styles
        from .styles import CATEGORIES, CATEGORY_LABELS, STYLE_LABELS

        sec = _section(lay, "Dictation modes")
        from .styles import MODE_CYCLE, mode_name

        self.mode_box = QComboBox()
        for mode in MODE_CYCLE:
            self.mode_box.addItem("Automatic · per app, below" if mode == "auto" else
                                  STYLE_LABELS.get(mode, mode_name(mode)), mode)
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(cfg.mode)))
        self.mode_box.currentIndexChanged.connect(
            lambda: self.ctl.set_mode(self.mode_box.currentData(), announce=False)
            if hasattr(self.ctl, "set_mode") else self._set("mode", self.mode_box.currentData()))
        sec.addWidget(setting_row("Dictation mode", "Automatic picks the style for the kind of app "
                                  "you're in. Pick one to use it everywhere (also from the tray menu "
                                  "or the Switch mode shortcut).", self.mode_box, first=True))
        intro = Label("How your words come out depends on where you're typing. Exact skips the "
                       "AI clean-up, so prompts and commands are never reworded.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        intro.setContentsMargins(0, 12, 0, 4)
        sec.addWidget(intro)
        self.style_boxes: dict[str, QComboBox] = {}
        for cat in CATEGORIES:
            box = QComboBox()
            for style, label in STYLE_LABELS.items():
                box.addItem(label, style)
            box.setCurrentIndex(max(0, box.findData(cfg.styles.get(cat))))
            box.currentIndexChanged.connect(
                lambda _i, c=cat, b=box: self._set("styles", {c: b.currentData()}))
            title, desc = CATEGORY_LABELS[cat]
            sec.addWidget(setting_row(title, desc, box))
            self.style_boxes[cat] = box
        self.custom_prompt = QLineEdit(cfg.custom_mode_prompt)
        self.custom_prompt.setPlaceholderText("e.g. Write in British English and keep it short")
        self.custom_prompt.setMinimumWidth(240)
        self.custom_prompt.editingFinished.connect(
            lambda: self._set("custom_mode_prompt", self.custom_prompt.text().strip()))
        sec.addWidget(setting_row("Custom instructions", "What the Custom mode tells the local AI "
                                  "to do with your dictation.", self.custom_prompt))
        self.language = QComboBox()
        for lang in OUTPUT_LANGUAGES:
            self.language.addItem(lang or "As spoken", lang)
        if cfg.output_language and self.language.findData(cfg.output_language) < 0:
            self.language.addItem(cfg.output_language, cfg.output_language)
        self.language.setCurrentIndex(max(0, self.language.findData(cfg.output_language)))
        self.language.currentIndexChanged.connect(
            lambda: self._set("output_language", self.language.currentData()))
        sec.addWidget(setting_row("Write in", "Speak in one language and Murmr writes in another, "
                                  "with the local AI (on this computer). Exact mode is never "
                                  "translated.", self.language))

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
        self.llm_row_desc = Label()
        self.llm_row_desc.setWordWrap(True)
        llm_row = setting_row(
            "AI clean-up",
            "Fixes self-corrections (“Tuesday, no wait, Wednesday”), stumbles, lists and grammar "
            "with a local AI model. “Only when needed” runs it just on dictations that look like "
            "they need it, so the rest stay instant.", self.llm_mode, first=True)
        llm_row.layout().itemAt(0).widget().layout().addWidget(self.llm_row_desc)
        self.llm_row = llm_row
        sec.addWidget(llm_row)
        # The model: which one, how fast here, and the best one for this computer (ai_choice.py)
        self.model_btn = QPushButton("Switch")
        self.model_btn.clicked.connect(self._ai_setup_clicked)
        self.model_desc = Label()
        self.model_desc.setObjectName("faint")
        self.model_desc.setWordWrap(True)
        self.model_row = setting_row("AI model", "Murmr picks the best model for this computer "
                                     "and checks it's quick enough here.", self.model_btn)
        self.model_row.layout().itemAt(0).widget().layout().addWidget(self.model_desc)
        self.model_row.setVisible(False)
        sec.addWidget(self.model_row)
        # Without the local AI: one button to set it up (then the AI features appear)
        from . import ai_setup

        self.ai_btn = QPushButton("Set up")
        self.ai_btn.setObjectName("primary")
        self.ai_btn.clicked.connect(self._ai_setup_clicked)
        self.ai_note = Label("")
        self.ai_note.setObjectName("faint")
        self.ai_note.setWordWrap(True)
        self.ai_row = setting_row("Local AI", "Turns on Command Mode, AI clean-up and meeting "
                                  "summaries. " + (ai_setup.plan() if ai_setup.supported() else
                                  "Install Ollama from ollama.com."), self.ai_btn, first=True)
        self.ai_row.layout().itemAt(0).widget().layout().addWidget(self.ai_note)
        self.ai_btn.setVisible(ai_setup.supported())
        sec.addWidget(self.ai_row)
        sec.addWidget(self._switch_row(
            "computer_commands", "Computer commands",
            "In Command Mode, say “open Spotify”, “switch to Chrome”, “open Downloads”, "
            "“open youtube.com” or “volume down”. Only these: nothing is ever deleted or sent. "
            "Works without the local AI."))
        sec.addWidget(self._switch_row("learn_from_edits", "Learn from my corrections",
                                       "After pasting, notice when you fix a word and remember it."))
        sec.addWidget(self._switch_row("screen_context", "Spell names from the screen",
                                       "When you start talking, Murmr looks at the names in the "
                                       "window you're typing into (an email thread, a chat), so "
                                       "they come out spelled the same way. Read on this computer "
                                       "only, never saved."))

        # Your voice: opt-ins, exclusions, what's stored, delete everything (privacy.py)
        # History (history.py)
        from .history import RETENTION_CHOICES

        sec = _section(lay, "History")
        sec.addWidget(self._switch_row("save_history", "Keep a history of my dictations",
                                       "Shown on Home to search, copy and paste again. Kept on "
                                       "this computer only.", first=True))
        self.history_keep = QComboBox()
        for days, label in RETENTION_CHOICES:
            self.history_keep.addItem(label, days)
        idx = self.history_keep.findData(int(cfg.history_days or 0))
        if idx < 0:
            self.history_keep.addItem(f"{cfg.history_days} days", int(cfg.history_days))
            idx = self.history_keep.count() - 1
        self.history_keep.setCurrentIndex(idx)
        self.history_keep.currentIndexChanged.connect(
            lambda: self._set("history_days", self.history_keep.currentData()))
        sec.addWidget(setting_row("Keep it for", "Older dictations are deleted. Your totals on "
                                  "Home are kept.", self.history_keep))
        clear_history = QPushButton("Delete all history")
        clear_history.clicked.connect(self._delete_history)
        sec.addWidget(setting_row("Delete it now", "Every dictation in the history and the totals "
                                  "on Home.", clear_history))

        sec = _section(lay, "Your voice")
        promise = Label("🔒  Your voice stays on this computer. Murmr never uploads your "
                        "recordings, transcripts or voice models, and training runs on your own "
                        "computer. Nothing is sent to the developer.")
        promise.setObjectName("muted")
        promise.setWordWrap(True)
        promise.setContentsMargins(0, 12, 0, 6)
        sec.addWidget(promise)
        self.managed_note = Label()
        self.managed_note.setObjectName("error")
        self.managed_note.setWordWrap(True)
        self.managed_note.setVisible(False)
        sec.addWidget(self.managed_note)
        from . import capabilities

        if capabilities.can_train():  # macOS can't train, so it keeps no voice data at all
            sec.addWidget(self._switch_row("save_training_audio", "Improve recognition with my voice",
                                           "Keep your dictations' audio (and your Voice training "
                                           "reading) to tune the speech model to your voice and "
                                           "accent."))
            sec.addWidget(self._switch_row("voice_clone", "Create a voice that sounds like me",
                                           "Use your Voice training reading to build a read-aloud "
                                           "voice like yours. Only from sentences you read live, "
                                           "never from other recordings."))
            self.excluded_apps = QLineEdit(", ".join(cfg.training_excluded_apps))
            self.excluded_apps.setMinimumWidth(280)
            self.excluded_apps.setCursorPosition(0)  # show the start of the list
            self.excluded_apps.editingFinished.connect(
                lambda: self._set("training_excluded_apps", self.excluded_apps.text()))
            sec.addWidget(setting_row("Never keep recordings from",
                                      "Apps where nothing you say is kept, comma separated. "
                                      "Password fields are always skipped.", self.excluded_apps))
        self.samples_info = Label()
        self.samples_info.setObjectName("faint")
        self.samples_info.setWordWrap(True)
        self.samples_delete = QPushButton("Delete all my voice data")
        self.samples_delete.clicked.connect(self._delete_samples)
        folder_btn = QPushButton("Open folder")
        folder_btn.clicked.connect(ctl.open_folder)
        self.samples_info.setContentsMargins(0, 10, 0, 4)
        sec.addWidget(self.samples_info)
        samples_host = QWidget()
        samples = FlowLayout(samples_host, spacing=8)  # wraps in a narrow window
        samples.addWidget(folder_btn)
        samples.addWidget(self.samples_delete)
        samples_host.setContentsMargins(0, 0, 0, 12)
        sec.addWidget(samples_host)

        # Speech model: base or tuned to your voice (speech_model.py)
        from .speech_model import SpeechModelSection

        self.speech_model = None
        models_dir = config_mod_dir() / "models"
        if capabilities.can_train() or (models_dir.exists() and any(models_dir.iterdir())):
            self.speech_model = SpeechModelSection(ctl)  # (a tuned model copied here still shows)
            _section(lay, "Speech model").addWidget(self.speech_model)

        # Sync with your other computers (sync_ui.py)
        self.sync_section = None
        if hasattr(ctl, "device_name"):
            from .sync_ui import SyncSection

            self.sync_section = SyncSection(ctl)
            _section(lay, "Sync").addWidget(self.sync_section)

        # Read aloud
        from .speak import all_voices

        sec = _section(lay, "Read aloud")
        self.voice = QComboBox()
        for vid, label in all_voices():
            self.voice.addItem(label, vid)
        self.voice.setCurrentIndex(max(0, self.voice.findData(cfg.read_voice)))
        self.voice.currentIndexChanged.connect(
            lambda: (self._set("read_voice", self.voice.currentData()), ctl.speak_sample()))
        sec.addWidget(setting_row("Voice", "A natural voice that runs on this computer. The "
                                  "first time, it downloads (90–310 MB).", self.voice, first=True))
        self.speed = QComboBox()
        for v in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6):
            self.speed.addItem(f"{v:g}×", v)
        idx = self.speed.findData(cfg.read_speed)
        self.speed.setCurrentIndex(idx if idx >= 0 else 2)
        self.speed.currentIndexChanged.connect(
            lambda: (self._set("read_speed", self.speed.currentData()), ctl.speak_sample()))
        sec.addWidget(setting_row("Speed", "", self.speed))
        sample = QPushButton("Play a sample")
        sample.clicked.connect(ctl.speak_sample)
        sec.addWidget(setting_row("Try it", "", sample))

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
        self.device_note = Label("")
        self.device_note.setObjectName("faint")
        self.speech_desc = Label("")
        self.speech_desc.setObjectName("faint")
        self.speech_desc.setWordWrap(True)
        self.quality = QComboBox()
        for value, label in (("auto", "Automatic · best for this computer"),
                             ("int8", "Faster · compact model"),
                             ("none", "Most accurate · full precision")):
            self.quality.addItem(label, value)
        q = (cfg.quantization or "auto").lower()
        self.quality.setCurrentIndex(max(0, self.quality.findData("none" if q == "fp32" else q)))
        self.quality.currentIndexChanged.connect(self._quality_changed)
        speech_row = setting_row("Speech recognition", "Speed or accuracy. Runs on this computer; "
                                 "nothing you say is sent anywhere.", self.quality)
        speech_row.layout().itemAt(0).widget().layout().addWidget(self.speech_desc)
        if hasattr(ctl, "speech_info"):
            sec.addWidget(speech_row)
        if capabilities.gpu_choice():  # macOS: always the built-in CPU path
            sec.addWidget(setting_row("Speech model runs on", "", self.device))
            sec.addWidget(self.device_note)

        # Updates
        sec = _section(lay, "Updates")
        self.update_title = Label()
        self.update_title.setStyleSheet("font-weight: 600;")
        self.update_detail = Label()
        self.update_detail.setObjectName("muted")
        self.update_detail.setWordWrap(True)
        self.update_changes = Label()
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
                                            "Every few hours. Only public code is downloaded from "
                                            "GitHub; nothing about you is sent.")
        sec.addWidget(self.update_auto)
        self.update_install = self._switch_row(
            "auto_update", "Install updates automatically",
            "Downloads in the background and restarts Murmr when you've been away for 10 minutes "
            "(takes a few seconds). Off: you choose when, here.")
        sec.addWidget(self.update_install)
        self.whats_new = Label()
        self.whats_new.setObjectName("muted")
        self.whats_new.setWordWrap(True)
        self.whats_new.setContentsMargins(0, 4, 0, 10)
        sec.addWidget(self.whats_new)
        self.rollback_btn = QPushButton("Go back to the previous version")
        self.rollback_btn.clicked.connect(ctl.rollback_update)
        self.rollback_row = setting_row("Something wrong after an update?", "", self.rollback_btn)
        sec.addWidget(self.rollback_row)

        self.status = Label()
        self.status.setObjectName("faint")
        self.status.setWordWrap(True)
        about_host = QWidget()
        about = FlowLayout(about_host, spacing=8)  # wraps onto more lines in a narrow window
        folder = QPushButton("Open data folder")
        folder.clicked.connect(ctl.open_folder)
        restart = QPushButton("Restart Murmr")
        restart.clicked.connect(ctl.restart)
        about.addWidget(folder)
        if hasattr(ctl, "show_welcome"):
            guide = QPushButton("Welcome guide")
            guide.setToolTip("The first-run steps again: microphone, shortcut, permissions, extras.")
            guide.clicked.connect(ctl.show_welcome)
            about.addWidget(guide)
        if hasattr(ctl, "report_problem"):
            report = QPushButton("Report a problem…")
            report.setToolTip("Saves a report file (logs and settings, nothing you said) to send "
                              "to the developer.")
            report.clicked.connect(ctl.report_problem)
            about.addWidget(report)
        about.addWidget(restart)
        lay.addSpacing(14)
        lay.addWidget(self.status)
        lay.addWidget(about_host)
        lay.addStretch(1)
        self._add_jump_links(lay)

    def _refresh_voices(self) -> None:
        """A voice trained on your recordings may have been installed since the window opened."""
        from .speak import all_voices

        voices = all_voices()
        if [self.voice.itemData(i) for i in range(self.voice.count())] == [v for v, _ in voices]:
            return
        self.voice.blockSignals(True)
        self.voice.clear()
        for vid, label in voices:
            self.voice.addItem(label, vid)
        self.voice.setCurrentIndex(max(0, self.voice.findData(self.ctl.cfg.read_voice)))
        self.voice.blockSignals(False)

    # -- local AI setup ----------------------------------------------------
    def _ai_setup_clicked(self) -> None:
        btn = self.sender() if self.sender() in (self.ai_btn, self.model_btn) else self.ai_btn
        if btn.text() == "Cancel":
            self.ctl.cancel_ai_setup()
            return
        self._ai_target = btn
        self._ai_idle_text = btn.text()
        btn.setText("Cancel")
        btn.setObjectName("")
        btn.style().unpolish(btn)
        btn.style().polish(btn)
        (self.model_desc if btn is self.model_btn else self.ai_note).setText("Starting…")
        self.ctl.setup_ai()

    def ai_setup_progress(self, msg: str, frac: float) -> None:
        btn = getattr(self, "_ai_target", self.ai_btn)
        note = self.model_desc if btn is self.model_btn else self.ai_note
        done = msg.startswith(("done:", "error:"))
        text = msg.split(":", 1)[1] if done else msg
        if frac >= 0 and not done:
            text += f"  ({frac:.0%})"
        note.setText(text)
        note.setObjectName("error" if msg.startswith("error:") else
                           "ok" if msg.startswith("done:") else "faint")
        note.style().unpolish(note)
        note.style().polish(note)
        if done:
            btn.setText(getattr(self, "_ai_idle_text", "Set up"))
            btn.setObjectName("primary" if btn is self.ai_btn else "")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            self._ai_target = self.ai_btn
            if msg.startswith("done:"):
                self._model_note_hold = text  # keep "ready: ..., 0.3 s" until the next refresh
                self.refresh()

    # -- jump links ------------------------------------------------------
    def _add_jump_links(self, lay: QVBoxLayout) -> None:
        """A row of section links under the title: the page is long."""
        inner = self.widget()
        heads = [h for h in inner.findChildren(QLabel, "h2")]
        heads.sort(key=lambda h: lay.indexOf(h) if lay.indexOf(h) >= 0 else 999)
        host = QWidget()
        flow = FlowLayout(host, spacing=6)
        for h in heads:
            b = QPushButton(h.text())
            b.setObjectName("jump")
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _=False, w=h: self._scroll_to(w))
            flow.addWidget(b)
        lay.insertWidget(2, host)

    def show_section(self, title: str) -> None:
        """Scroll to a section ("Updates", "Speech model"...), once the page is laid out."""
        heads = [h for h in self.widget().findChildren(QLabel, "h2") if h.text() == title]
        if heads:
            QTimer.singleShot(120, lambda: self._scroll_to(heads[0]))

    def _scroll_to(self, w: QWidget) -> None:
        from PySide6.QtCore import QEasingCurve, QPropertyAnimation

        bar = self.verticalScrollBar()
        anim = QPropertyAnimation(bar, b"value", self)
        anim.setDuration(260)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(bar.value())
        anim.setEndValue(min(bar.maximum(), max(0, w.y() - 16)))
        anim.start()
        self._anim = anim

    # -- macOS permissions -----------------------------------------------
    def _permissions_section(self, lay: QVBoxLayout) -> None:
        from .macperms import KINDS, LABELS, WHY

        sec = _section(lay, "Permissions")
        for i, kind in enumerate(KINDS):
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            state = Label()
            btn = QPushButton("Change…" if kind == "fn_key" else "Allow…")
            btn.setObjectName("primary")
            btn.clicked.connect(lambda _=False, k=kind: self.ctl.allow_permission(k))
            bl.addWidget(state)
            bl.addWidget(btn)
            sec.addWidget(setting_row(LABELS[kind], WHY[kind], box, first=i == 0))
            self.perm_rows[kind] = (state, btn)
        note = Label("Allow opens System Settings: switch Murmr on there, then restart Murmr "
                      "(macOS often only applies it after a restart). Still not allowed? Select "
                      "Murmr in that list, remove it with −, and restart Murmr so it asks again.")
        note.setObjectName("faint")
        note.setWordWrap(True)
        note.setContentsMargins(0, 0, 0, 10)
        sec.addWidget(note)
        self._perm_timer = QTimer(self)
        self._perm_timer.setInterval(2000)  # pick up changes made in System Settings
        self._perm_timer.timeout.connect(self.refresh_permissions)
        # Runs only while Settings is on screen (showEvent / hideEvent).

    def refresh_permissions(self) -> None:
        if not self.perm_rows or not self.isVisible():
            return
        for kind, ok in (self.ctl.permissions() or {}).items():
            if kind not in self.perm_rows:
                continue
            state, btn = self.perm_rows[kind]
            if kind == "fn_key":
                state.setText("Do Nothing" if ok else "Not set" if ok is False else "Unknown")
            else:
                state.setText("Allowed" if ok else "Not allowed" if ok is False else "Unknown")
            state.setObjectName("ok" if ok else "error")
            state.style().unpolish(state)
            state.style().polish(state)
            btn.setVisible(not ok)

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

    def _quality_changed(self) -> None:
        if hasattr(self.ctl, "set_speech_quality"):
            self.ctl.set_speech_quality(self.quality.currentData())
        self.speech_desc.setText("Reloading the speech model…")

    def _refresh_speech(self) -> None:
        if not hasattr(self.ctl, "speech_info"):
            return
        info = self.ctl.speech_info()
        if not info:
            return
        if not info["ready"]:
            self.speech_desc.setText("Loading the speech model…")
            return
        text = f"Now: {info['model']}, {info['precision']} ({info['size']}), on {info['where']}."
        if info["speed"]:
            text += f" About {info['speed']:.0f}× faster than you speak, measured on your dictations."
        if info["slow_choice"]:
            text += " Full precision on the processor is slow and uses more memory: Faster suits this computer."
        self.speech_desc.setText(text)

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
        for field in (self.hotkey, self.backup, self.hands, self.command, self.paste_last,
                      self.read_key, self.mode_key, self.word_key):
            field.set_recording(False)

    def _turn_off(self, kind: str) -> None:
        self._cancel_record()
        self.hotkey_msg.setVisible(False)
        err = self.ctl.update_setting(kind, "")
        if err:
            self.hotkey_msg.setText(err)
            self.hotkey_msg.setVisible(True)
            return
        self._field(kind).set_labels(["Off"])

    def _field(self, kind: str) -> HotkeyField:
        return {"hotkey": self.hotkey, "backup_hotkey": self.backup,
                "hands_free_key": self.hands, "command_hotkey": self.command,
                "paste_last_hotkey": self.paste_last, "read_hotkey": self.read_key,
                "mode_hotkey": self.mode_key, "add_word_hotkey": self.word_key}[kind]

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

    def sync_mode(self, mode: str) -> None:
        """The mode changed elsewhere (tray, shortcut): show it here without re-applying it."""
        self.mode_box.blockSignals(True)
        self.mode_box.setCurrentIndex(max(0, self.mode_box.findData(mode)))
        self.mode_box.blockSignals(False)

    def _delete_history(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        if QMessageBox.question(self, "Delete all history",
                                "Delete every dictation in your history? This can't be undone.",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        if hasattr(self.ctl, "delete_all_history"):
            self.ctl.delete_all_history()

    def _delete_samples(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        if QMessageBox.question(
                self, "Delete everything about my voice",
                "Delete all your recordings, tuned speech models and voices from this computer? "
                "This can't be undone.") != QMessageBox.StandardButton.Yes:
            return
        restart = self.ctl.delete_all_voice()
        from .privacy import remaining_voice_data

        left = remaining_voice_data()
        if left:  # in use (e.g. a loaded model): removed when Murmr starts again
            QMessageBox.information(self, "Murmr", f"Deleted, except {len(left)} file(s) in use. "
                                    "They're removed when Murmr restarts.")
            restart = True
        if restart:
            if not left:
                QMessageBox.information(self, "Murmr", "Deleted. Murmr restarts to use the "
                                        "standard speech model again.")
            self.ctl.restart()
            return
        self.refresh()

    def refresh(self) -> None:
        if hasattr(self.ctl, "voice_storage"):
            v = self.ctl.voice_storage()
            parts = [f"{v['clips']} recordings ({v['seconds'] / 60:.0f} min)"]
            parts.append(f"{v['models']} tuned speech model{'s' if v['models'] != 1 else ''}")
            parts.append(f"{v['voices']} voice{'s' if v['voices'] != 1 else ''}")
            self.samples_info.setText("On this computer: " + " · ".join(parts)
                                      + f" · {v['bytes'] / 1e6:.0f} MB in total.")
            self.samples_delete.setEnabled(bool(v["clips"] or v["models"] or v["voices"]))
            reason = self.ctl.managed_reason()
            self.managed_note.setText(
                f"This looks like a work computer ({reason}). Recordings could contain "
                "confidential work information; consider keeping your voice data on a personal "
                "computer instead." if reason else "")
            self.managed_note.setVisible(bool(reason))
        if self.speech_model is not None:
            self.speech_model.refresh()
        self._refresh_voices()
        ai = self.ctl.ai_available() if hasattr(self.ctl, "ai_available") else True
        # Command Mode needs the local AI, except for computer commands ("open Spotify")
        self.command_row.setVisible(ai or self.ctl.cfg.computer_commands)
        self.llm_row.setVisible(ai)
        self.ai_row.setVisible(not ai or bool(self.ai_note.text()))
        if hasattr(self.ctl, "refresh_capabilities"):
            self.ctl.refresh_capabilities()
        self._refresh_model(ai)
        ok, text = self.ctl.llm_status() if ai else (False, "")
        self.llm_row_desc.setText(text)
        self.llm_row_desc.setObjectName("ok" if ok else "faint")
        self.llm_row_desc.style().unpolish(self.llm_row_desc)
        self.llm_row_desc.style().polish(self.llm_row_desc)
        self.status.setText(f"Murmr {__version__} · {self.ctl.status_text()}")
        self._refresh_speech()
        self.refresh_updates()
        self.refresh_permissions()

    def _refresh_model(self, ai: bool) -> None:
        info = self.ctl.ai_model_info() if ai and hasattr(self.ctl, "ai_model_info") else None
        self.model_row.setVisible(bool(info))
        if not info or self.model_btn.text() == "Cancel":
            return
        hold = getattr(self, "_model_note_hold", "")
        self._model_note_hold = ""
        speed = f", {info['measured_s']:.1f} s per clean-up here" if info["measured_s"] else ""
        if info["differs"]:
            text = (f"Using {info['model']}. Better for this computer: {info['recommended']} "
                    f"({info['reason']}). Switching downloads {info['size']} and removes the old "
                    "model.")
        elif info["auto"]:
            text = f"{info['model']}{speed}. Best for this computer ({info['reason']})."
            if info.get("apple_skipped"):
                text = (f"{info['model']}{speed}. Apple's on-device model wasn't used: "
                        f"{info['apple_skipped']}. Check again after fixing that.")
        else:
            text = f"{info['model']}{speed}. Chosen in config.json."
        self.model_desc.setText(hold or text)
        self.model_desc.setObjectName("ok" if hold else "faint")
        self.model_desc.style().unpolish(self.model_desc)
        self.model_desc.style().polish(self.model_desc)
        self.model_btn.setText("Switch" if info["differs"] else "Check again")
        self.model_btn.setVisible(bool(info["auto"]))

    def refresh_updates(self) -> None:
        u = self.ctl.update_state()
        if not u["installed"]:
            self.update_title.setText(f"Murmr {__version__} · developer copy")
            self.update_detail.setText("Running from a git checkout: update it with git pull. "
                                       "Copies installed with the installer update here.")
            for w in (self.update_check_btn, self.update_now_btn, self.update_auto, self.rollback_row,
                      self.update_changes, self.update_install, self.whats_new):
                w.setVisible(False)
            return
        version = f"Murmr {__version__}"
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
            detail = (f"{n} update{'s' if n != 1 else ''} available." if n else "An update is available.")
            detail += (" Downloaded: it installs when you're away, or now." if u.get("staged")
                       else " Updating restarts Murmr (a few seconds).")
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
        self.update_install.setVisible(True)
        new = u.get("whats_new") or []
        self.whats_new.setText("What's new in this version:\n" + "\n".join(f"•  {c}" for c in new[:10])
                               if new else "")
        self.whats_new.setVisible(bool(new))
        self.rollback_row.setVisible(bool(u["has_previous"]))
        self.rollback_btn.setEnabled(not busy)


# --------------------------------------------------------------------------- window
class MainWindow(QWidget):
    def __init__(self, ctl: Controller, mic_options: list[tuple[str, str]] | None = None):
        super().__init__()
        self.ctl = ctl
        self.setObjectName("root")
        self.setWindowTitle("Murmr")
        self.setWindowIcon(app_icon())
        self.resize(980, 680)
        self.setMinimumSize(520, 420)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        self.side = side
        self._compact = False
        sl = QVBoxLayout(side)
        sl.setContentsMargins(14, 18, 14, 16)
        sl.setSpacing(4)
        brand = QHBoxLayout()
        logo = Label()
        logo.setPixmap(app_pixmap(56).scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
        brand.addWidget(logo)
        name = Label("Murmr")
        name.setObjectName("brand")
        brand.addWidget(name)
        self.brand_name = name
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
            b = QPushButton(f"  {label}")
            b.setObjectName("nav")
            b.setProperty("kind", label.lower())
            b.setIconSize(QSize(18, 18))
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.nav.addButton(b, i)
            sl.addWidget(b)
        self._add_meetings_page(ctl, sl)
        from . import capabilities

        if capabilities.can_train():  # recordings for tuning: only where tuning can run
            self._add_voice_training_page(ctl, sl)
        self.nav.idClicked.connect(self.show_page)
        sl.addStretch(1)
        self.side_status = Label()
        self.side_status.setObjectName("sideStatus")
        self.side_status.setWordWrap(True)
        self.side_status.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(self.side_status)
        version = Label(f"Murmr {__version__}")
        version.setObjectName("version")
        sl.addWidget(version)
        self.version_label = version
        self.side_layout = sl

        root.addWidget(side)
        root.addWidget(self.pages, 1)
        self.apply_theme()
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _s: self.apply_theme())
        except AttributeError:
            pass
        self.show_page(0)

    def _add_meetings_page(self, ctl, sidebar: QVBoxLayout) -> None:
        """Meetings (meetings_page.py), listed after Dictionary."""
        from .meetings_page import MeetingsPage

        self.meetings = MeetingsPage(ctl)
        index = self.pages.addWidget(self.meetings)
        b = QPushButton("  Meetings")
        b.setObjectName("nav")
        b.setProperty("kind", "meetings")
        b.setIconSize(QSize(18, 18))
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.nav.addButton(b, index)
        sidebar.insertWidget(sidebar.indexOf(self.nav.button(self.pages.indexOf(self.settings))), b)

    def _add_voice_training_page(self, ctl, sidebar: QVBoxLayout) -> None:
        """Voice training (voice_training.py): the last page index, listed above Settings."""
        from .voice_training import VoiceTrainingPage

        self.voice_training = VoiceTrainingPage(ctl)
        index = self.pages.addWidget(self.voice_training)
        b = QPushButton("  Voice training")
        b.setObjectName("nav")
        b.setProperty("kind", "training")
        b.setIconSize(QSize(18, 18))
        b.setCheckable(True)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        self.nav.addButton(b, index)
        sidebar.insertWidget(sidebar.indexOf(self.nav.button(self.pages.indexOf(self.settings))), b)

    COMPACT_BELOW = 760  # window width: the sidebar shows only icons

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._set_compact(event.size().width() < self.COMPACT_BELOW)

    def _set_compact(self, compact: bool) -> None:
        if compact == self._compact:
            return
        self._compact = compact
        self.side.setFixedWidth(64 if compact else 210)
        self.side_layout.setContentsMargins(10 if compact else 14, 18, 10 if compact else 14, 16)
        for w in (self.brand_name, self.side_status, self.version_label):
            w.setVisible(not compact)
        side = 20 if compact else 40  # page margins: more room for the content
        for i in range(self.pages.count()):
            page = self.pages.widget(i)
            inner = page.widget() if isinstance(page, QScrollArea) else None
            lay = inner.layout() if inner is not None else None
            if lay is not None and lay.contentsMargins().left() in (20, 40):
                m = lay.contentsMargins()
                lay.setContentsMargins(side, m.top(), side, m.bottom())
        for b in self.nav.buttons():
            if b.property("label") is None:
                b.setProperty("label", b.text())
            b.setText("" if compact else b.property("label"))
            b.setToolTip(b.property("label").strip() if compact else "")

    def apply_theme(self) -> None:
        pal = theme.current()
        self.setStyleSheet(theme.stylesheet(pal))
        for b in self.nav.buttons():
            b.setIcon(nav_icon(b.property("kind"), pal.muted, pal.text))
        if hasattr(self, "side_status"):
            self.refresh_status()

    def show_page(self, index: int) -> None:
        self.nav.button(index).setChecked(True)
        self.pages.setCurrentIndex(index)
        page = self.pages.currentWidget()
        page.refresh()
        QTimer.singleShot(0, lambda: self._relayout(page))

    @staticmethod
    def _relayout(page) -> None:
        """A page laid out while hidden guessed its text widths; redo it now it's visible."""
        inner = page.widget() if isinstance(page, QScrollArea) else page
        if inner is None or inner.layout() is None:
            return
        for lab in inner.findChildren(Label):
            if lab.wordWrap():
                lab.updateGeometry()
        inner.layout().invalidate()
        inner.layout().activate()

    def refresh_status(self) -> None:
        pal = theme.current()
        status = self.ctl.status_text()
        ok = status.startswith("Ready")
        color = pal.success if ok else (pal.danger if "fail" in status.lower() else pal.faint)
        self.side_status.setText(f'<span style="color:{color}">●</span>&nbsp; {status}')

    def refresh(self) -> None:
        self.refresh_status()
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
        # Closing the window keeps Murmr running in the tray (like Wispr Flow).
        self.settings._cancel_record()
        event.ignore()
        self.hide()
