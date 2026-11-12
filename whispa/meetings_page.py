"""The Meetings page: start notes, watch the live transcript, read summaries of past meetings."""

from __future__ import annotations

import datetime as dt
import html
import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QMessageBox, QPushButton,
                               QScrollArea, QTextBrowser, QVBoxLayout, QWidget)

from .meetings import fmt_time, system_audio_supported, to_markdown, transcript_text
from .widgets import Label, Switch, setting_row

WHO_COLOR = {"You": "#10b981", "Others": "#0ea5e9"}
PALETTE = ["#0ea5e9", "#f59e0b", "#a855f7", "#ef4444", "#14b8a6", "#ec4899", "#84cc16", "#6366f1"]


def who_color(who: str, order: list[str]) -> str:
    if who in WHO_COLOR:
        return WHO_COLOR[who]
    if who not in order:
        order.append(who)
    return PALETTE[order.index(who) % len(PALETTE)]


def _browser(min_h: int = 120, fit: bool = False, max_h: int = 0) -> QTextBrowser:
    """Read-only rich text. ``fit``: as tall as its text (no inner scrollbar)."""
    b = QTextBrowser()
    b.setOpenExternalLinks(False)
    b.setMinimumHeight(min_h)
    if max_h:
        b.setMaximumHeight(max_h)
    b.setFrameShape(QFrame.Shape.NoFrame)
    b.setStyleSheet("QTextBrowser { background: transparent; }")
    b.document().setDefaultStyleSheet(
        "h1, h2, h3 { font-size: 14px; font-weight: 600; margin-top: 12px; margin-bottom: 4px; }"
        "ul { margin-top: 0; margin-bottom: 4px; } li { margin-bottom: 2px; }")
    if fit:
        b.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        def resize() -> None:
            b.document().setTextWidth(b.viewport().width())
            b.setFixedHeight(max(min_h, int(b.document().size().height()) + 8))

        b.document().contentsChanged.connect(resize)
        b.resizeEvent = lambda e, f=b.resizeEvent: (f(e), resize())  # noqa: E731
    return b


def lines_html(lines, live=(), links: bool = False, edit_from: int | None = None) -> str:
    """The transcript. ``links``: speaker names are clickable (to rename them).
    ``edit_from``: lines are clickable to correct them; the first one shown is that index."""
    from urllib.parse import quote

    from . import theme

    text_color = theme.current().text
    rows, order = [], []
    for i, d in enumerate(lines):
        d = d if isinstance(d, dict) else d.__dict__
        color = who_color(d["who"], order)
        name = html.escape(d["who"])
        if links:
            name = (f'<a href="speaker:{quote(d["who"])}" style="color:{color};text-decoration:none">'
                    f'{name}</a>')
        text = html.escape(d["text"])
        if edit_from is not None:
            # Qt's rich text has no "inherit": links need the theme's text colour spelled out.
            text = (f'<a href="line:{edit_from + i}" style="color:{text_color};text-decoration:none">'
                    f'{text}</a>')
        mark = ' <span style="color:#999">· edited</span>' if d.get("edited") else ""
        rows.append(f'<p style="margin:0 0 6px 0"><span style="color:{color};font-weight:600">{name}'
                    f'</span> <span style="color:#999">{fmt_time(d["t"])}</span>{mark}'
                    f'<br>{text}</p>')
    for who, text in live:
        rows.append(f'<p style="margin:0 0 6px 0"><span style="color:{who_color(who, order)};'
                    f'font-weight:600">{html.escape(who)}</span> <span style="color:#999">now</span><br>'
                    f'<i style="color:#888">{html.escape(text)}</i></p>')
    return "".join(rows)


class MeetingsPage(QScrollArea):
    def __init__(self, ctl):
        from .window import _page, _section

        super().__init__()
        self.ctl = ctl
        cfg = ctl.cfg
        area, lay = _page("Meetings",
                          "Take notes in calls and meetings: Murmr transcribes you and the other "
                          "people as they talk, then writes a summary with decisions and action "
                          "items. Transcribed on this computer, and no audio is kept, only the "
                          "text.")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidget(area.takeWidget())
        self._viewing: str | None = None
        self.setAcceptDrops(True)  # drop an audio / video file here to transcribe it

        # -- controls
        sec = _section(lay, "Notes")
        self.mic_sw = Switch(cfg.meeting_mic)
        self.mic_sw.toggled.connect(lambda on: ctl.update_setting("meeting_mic", on))
        sec.addWidget(setting_row("My microphone", "What you say, labelled “You”.", self.mic_sw,
                                  first=True))
        self.sys_sw = Switch(cfg.meeting_system and system_audio_supported())
        self.sys_sw.toggled.connect(lambda on: ctl.update_setting("meeting_system", on))
        if system_audio_supported():  # Windows, macOS 13+; elsewhere the mic hears everyone
            desc = ("The other people in your Teams, Zoom or Meet call (what your speakers or "
                    "headphones play), labelled “Others”.")
            if sys.platform == "darwin":
                desc += (" macOS asks once for Screen & System Audio Recording: Murmr only uses "
                         "the sound, never the screen.")
            sec.addWidget(setting_row("Computer audio", desc, self.sys_sw))
        self.room_sw = Switch(cfg.meeting_room)
        self.room_sw.toggled.connect(lambda on: ctl.update_setting("meeting_room", on))
        sec.addWidget(setting_row(
            "Several people share my microphone",
            "In a meeting room: voices on your microphone are told apart too (Person 1, Person 2…; "
            "you're “You” if Murmr knows your voice from Voice training). People on the call stay "
            "separate. Click a name in the live transcript to say who it is.", self.room_sw))
        self.remember_sw = Switch(cfg.remember_voices)
        self.remember_sw.toggled.connect(self._remember_toggled)
        sec.addWidget(setting_row(
            "Remember voices of people I name",
            "Name someone once and Murmr recognises them in later meetings. A voiceprint (no "
            "audio) is kept on this computer only. It's their personal information: only "
            "remember people who are OK with it.", self.remember_sw))
        self.people_host = QWidget()
        self.people_box = QVBoxLayout(self.people_host)
        self.people_box.setContentsMargins(0, 0, 0, 0)
        self.people_box.setSpacing(0)
        sec.addWidget(self.people_host)
        self.start_btn = QPushButton("Start meeting notes")
        self.start_btn.setObjectName("primary")
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.clicked.connect(self._start_stop)
        self.captions_btn = QPushButton("Live captions")
        self.captions_btn.setCheckable(True)
        self.captions_btn.setToolTip("Captions of what's being said, in a small floating bar")
        self.captions_btn.clicked.connect(lambda on: ctl.set_captions(on))
        self.status = Label("")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        self.file_btn = QPushButton("Transcribe a file…")
        self.file_btn.setToolTip("An audio or video file (mp3, m4a, wav, mp4, mov…): transcribed on "
                                 "this computer, with speakers told apart. Or drop it on this page.")
        self.file_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.file_btn.clicked.connect(self._file_clicked)
        from .window import FlowLayout

        host = QWidget()
        flow = FlowLayout(host, spacing=8)  # wraps onto more lines in a narrow window
        for b in (self.start_btn, self.captions_btn, self.file_btn):
            flow.addWidget(b)
        host.setContentsMargins(0, 12, 0, 4)
        sec.addWidget(host)
        self.status.setContentsMargins(0, 0, 0, 6)
        sec.addWidget(self.status)
        consent = Label("Let everyone know you're taking notes: in some places, recording or "
                        "transcribing a conversation needs everyone's consent.")
        consent.setObjectName("faint")
        consent.setWordWrap(True)
        consent.setContentsMargins(0, 0, 0, 10)
        sec.addWidget(consent)

        # -- live transcript
        self.live_head = Label("Live transcript")
        self.live_head.setObjectName("h2")
        lay.addSpacing(10)
        lay.addWidget(self.live_head)
        self.live_card, live = self._card()
        self.live_view = _browser(220)
        self.live_view.setOpenLinks(False)
        self.live_view.anchorClicked.connect(self._live_clicked)
        live.addWidget(self.live_view)
        tip = Label("Click a name to say who it is, or a line to correct it.")
        tip.setObjectName("faint")
        live.addWidget(tip)
        lay.addWidget(self.live_card)

        # -- a meeting's notes
        self.detail_card, d = self._card()
        top = QHBoxLayout()
        self.detail_title = Label("")
        self.detail_title.setObjectName("h2")
        self.detail_title.setWordWrap(True)
        top.addWidget(self.detail_title, 1)
        close = QPushButton("✕")
        close.setObjectName("icon")
        close.setToolTip("Close")
        close.clicked.connect(lambda: self.open_meeting(None))
        top.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        d.addLayout(top)
        self.detail_meta = Label("")
        self.detail_meta.setObjectName("faint")
        d.addWidget(self.detail_meta)
        self.summary_view = _browser(60, fit=True)
        d.addWidget(self.summary_view)
        acts = QHBoxLayout()
        acts.setContentsMargins(0, 6, 0, 6)
        for text, fn in (("Copy summary", self._copy_summary), ("Copy transcript", self._copy_transcript),
                         ("Export…", self._export), ("Summarize again", self._resummarize),
                         ("Delete", self._delete)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            acts.addWidget(b)
            if text == "Summarize again":
                self.resum_btn = b
        acts.addStretch(1)
        d.addLayout(acts)
        self.transcript_view = _browser(160, max_h=420)
        self.transcript_view.setOpenLinks(False)
        self.transcript_view.anchorClicked.connect(self._speaker_clicked)
        self.rename_hint = Label("Click a speaker's name to name them; it changes everywhere in "
                                 "these notes.")
        self.rename_hint.setObjectName("faint")
        self.rename_hint.setContentsMargins(4, 6, 0, 2)
        d.addWidget(self.rename_hint)
        d.addWidget(self.transcript_view)
        lay.addSpacing(10)
        lay.addWidget(self.detail_card)

        # -- past meetings
        head = Label("Past meetings")
        head.setObjectName("h2")
        lay.addSpacing(10)
        lay.addWidget(head)
        self.list_card, self.list = self._card()
        lay.addWidget(self.list_card)
        lay.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    @staticmethod
    def _card() -> tuple[QFrame, QVBoxLayout]:
        from .widgets import card

        frame, inner = card()
        inner.setContentsMargins(18, 12, 18, 12)
        return frame, inner

    # -- refresh ---------------------------------------------------------
    def refresh(self) -> None:
        self._fill_people()
        running = self.ctl.meeting_running()
        self.ai = self.ctl.ai_available() if hasattr(self.ctl, "ai_available") else True
        stop = "Stop and summarize" if self.ai else "Stop"
        self.start_btn.setText(stop if running else "Start meeting notes")
        self.start_btn.setObjectName("danger" if running else "primary")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.captions_btn.setChecked(self.ctl.captions_on())
        self.live_head.setVisible(running)
        self.live_card.setVisible(running)
        if running:
            self._timer.start()
            self._tick()
        else:
            self._timer.stop()
            self.status.setText(self.ctl.meeting_status())
        self._fill_list()
        self.open_meeting(self._viewing)

    def _tick(self) -> None:
        s = self.ctl.meeting_session()
        if s is None or not s.running:
            self._timer.stop()
            self.refresh()
            return
        words = sum(len(line.text.split()) for line in s.lines)
        self.status.setText(f'<span style="color:#ef4444">●</span>&nbsp; Recording '
                            f"{fmt_time(s.elapsed)} · {words} words")
        bar = self.live_view.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 4
        if getattr(self, "_editing", False):
            return  # a correction dialog is open: don't redraw underneath it
        first = max(0, len(s.lines) - 200)
        self.live_view.setHtml(lines_html(s.lines[first:], s.live(), links=True, edit_from=first))
        if at_end:
            bar.setValue(bar.maximum())

    def _fill_list(self) -> None:
        from .window import clear_layout

        clear_layout(self.list)
        meetings = self.ctl.meetings()
        if not meetings:
            none = Label("No meetings yet. Start notes when a call begins.")
            none.setObjectName("faint")
            none.setWordWrap(True)
            none.setContentsMargins(0, 8, 0, 8)
            self.list.addWidget(none)
            return
        for i, m in enumerate(meetings[:100]):
            when = dt.datetime.fromtimestamp(m["started"]).strftime("%a %d %b, %H:%M")
            title = m.get("title") or "Meeting notes"
            b = QPushButton("Open")
            b.clicked.connect(lambda _=False, mid=m["id"]: self.open_meeting(mid))
            self.list.addWidget(setting_row(title, f"{when} · {fmt_time(m.get('duration') or 0)} · "
                                            f"{m.get('words', 0)} words", b, first=i == 0))

    # -- a meeting -------------------------------------------------------
    def open_meeting(self, meeting_id: str | None) -> None:
        self._viewing = meeting_id
        m = self.ctl.meeting(meeting_id) if meeting_id else None
        self.detail_card.setVisible(m is not None)
        if m is None:
            return
        self._meeting = m
        self.detail_title.setText(m.get("title") or "Meeting notes")
        when = dt.datetime.fromtimestamp(m["started"]).strftime("%A %d %B %Y, %H:%M")
        self.detail_meta.setText(f"{when} · {fmt_time(m.get('duration') or 0)}")
        state = self.ctl.summary_state(meeting_id)
        ai = getattr(self, "ai", True)
        self.resum_btn.setVisible(ai)  # summaries need the local AI
        self.summary_view.setVisible(bool(ai or m.get("summary")))
        if m.get("summary"):
            import re

            # Headings as bold labels: Qt draws Markdown headings very large.
            # (blank lines around them, or Markdown glues them onto the bullet before)
            md = re.sub(r"^#{1,6}\s*(.+)$", lambda h: f"\n**{h.group(1)}**\n", m["summary"],
                        flags=re.MULTILINE)
            self.summary_view.setMarkdown(md)
        else:
            self.summary_view.setPlainText(state or m.get("summary_error") or "No summary yet.")
        self.resum_btn.setEnabled(not state)
        self.transcript_view.setHtml(lines_html(m.get("lines", []), links=True))
        QTimer.singleShot(0, lambda: self.ensureWidgetVisible(self.detail_card, 0, 0))

    def show_summary_progress(self, meeting_id: str) -> None:
        if self._viewing == meeting_id:
            self.open_meeting(meeting_id)
        self._fill_list()

    def _remember_toggled(self, on: bool) -> None:
        self.ctl.update_setting("remember_voices", on)
        self._fill_people()

    def _fill_people(self) -> None:
        from .window import clear_layout

        clear_layout(self.people_box)
        if not hasattr(self.ctl, "people_store"):
            self.people_host.setVisible(False)
            return
        store = self.ctl.people_store()
        names = store.names() if store is not None else []
        self.people_host.setVisible(bool(names))
        if not names:
            return
        shown = ", ".join(names[:12]) + (f" and {len(names) - 12} more" if len(names) > 12 else "")
        forget_all = QPushButton("Forget everyone")
        forget_all.clicked.connect(self._forget_all)
        self.people_box.addWidget(setting_row(f"Voices Murmr knows ({len(names)})", shown, forget_all))
        for name in names[:12]:
            b = QPushButton("Forget")
            b.clicked.connect(lambda _=False, n=name: self._forget(n))
            self.people_box.addWidget(setting_row(name, "", b))

    def _forget(self, name: str) -> None:
        self.ctl.people_store().forget(name)
        self._fill_people()

    def _forget_all(self) -> None:
        if QMessageBox.question(self, "Forget everyone", "Forget every voice Murmr has learned? "
                                "Meetings and their names stay as they are.") == \
                QMessageBox.StandardButton.Yes:
            self.ctl.people_store().forget_all()
            self._fill_people()

    def _consent(self, name: str) -> bool:
        """Before remembering someone's voice for the first time: did they agree?"""
        if not hasattr(self.ctl, "needs_voice_consent") or not self.ctl.needs_voice_consent(name):
            return True
        box = QMessageBox(self)
        box.setWindowTitle("Remember this voice?")
        box.setText(f"Remember {name}'s voice?")
        box.setInformativeText(
            f"Murmr would keep a voiceprint of {name} (a description of how they sound: no audio, "
            "nothing they said) on this computer, to recognise them in later meetings. It's their "
            f"personal information: only choose Remember if {name} has agreed. You can forget it "
            "any time on the Meetings page.")
        remember = box.addButton("Remember", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Just this meeting", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is remember

    def _live_clicked(self, url) -> None:
        """During the meeting: name a speaker, or correct a line."""
        from urllib.parse import unquote

        from PySide6.QtWidgets import QInputDialog

        s = self.ctl.meeting_session()
        if s is None:
            return
        self._editing = True
        try:
            if url.scheme() == "speaker":
                old = unquote(url.path() or url.toString()[len("speaker:"):])
                new, ok = QInputDialog.getText(
                    self, "Who is this?", f"Name for “{old}” (the lines so far and from now on). "
                    "The same name as someone else merges them.",
                    text="" if old.startswith(("Person ", "Others", "You")) else old)
                if ok and new.strip():
                    self.ctl.rename_live_speaker(old, new.strip(), self._consent(new.strip()))
            elif url.scheme() == "line":
                index = int(url.path() or url.toString()[len("line:"):])
                if 0 <= index < len(s.lines):
                    text, ok = QInputDialog.getMultiLineText(self, "Correct this line",
                                                             f"{s.lines[index].who}:", s.lines[index].text)
                    if ok and text.strip():
                        self.ctl.edit_live_line(index, text.strip())
        finally:
            self._editing = False
        self._tick()

    def _speaker_clicked(self, url) -> None:
        from urllib.parse import unquote

        from PySide6.QtWidgets import QInputDialog

        if url.scheme() != "speaker" or not getattr(self, "_meeting", None):
            return
        old = unquote(url.path() or url.toString()[len("speaker:"):])
        new, ok = QInputDialog.getText(self, "Name this speaker",
                                       f"Who is “{old}”? (Same name as someone else merges them.)",
                                       text="" if old.startswith(("Person ", "Others")) else old)
        if ok and new.strip():
            self.ctl.rename_speaker(self._meeting["id"], old, new.strip(), self._consent(new.strip()))
            self.open_meeting(self._meeting["id"])

    def _copy_summary(self) -> None:
        m = self._meeting
        QGuiApplication.clipboard().setText(f"{m.get('title') or 'Meeting notes'}\n\n{m.get('summary') or ''}")

    def _copy_transcript(self) -> None:
        QGuiApplication.clipboard().setText(transcript_text(self._meeting.get("lines", [])))

    def _export(self) -> None:
        m = self._meeting
        name = (m.get("title") or "Meeting notes").replace("/", "-")[:60] + ".md"
        path, _ = QFileDialog.getSaveFileName(self, "Export meeting notes", name, "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_markdown(m))

    def _resummarize(self) -> None:
        self.ctl.summarize_meeting(self._meeting["id"])
        self.open_meeting(self._meeting["id"])

    def _delete(self) -> None:
        if QMessageBox.question(self, "Delete meeting", "Delete these notes and transcript?") \
                == QMessageBox.StandardButton.Yes:
            self.ctl.delete_meeting(self._meeting["id"])
            self.open_meeting(None)
            self._fill_list()

    # -- transcribe a file -------------------------------------------------
    def _file_clicked(self) -> None:
        if self.file_btn.text() == "Cancel":
            self.ctl.cancel_file()
            return
        from .filetrans import EXTENSIONS

        path, _ = QFileDialog.getOpenFileName(
            self, "Transcribe a file", "",
            "Audio and video (" + " ".join(f"*{e}" for e in EXTENSIONS) + ");;All files (*)")
        if path:
            self._start_file(path)

    def _start_file(self, path: str) -> None:
        err = self.ctl.transcribe_file(path)
        if err:
            self.status.setText(err)
            return
        self.file_btn.setText("Cancel")
        self.status.setText("Reading the file…")

    def file_progress(self, msg: str, frac: float) -> None:
        if msg.startswith(("done:", "error:")):
            self.file_btn.setText("Transcribe a file…")
            if msg.startswith("done:"):
                self.status.setText("File transcribed.")
                self._viewing = msg[5:]
                self.refresh()
            else:
                self.status.setText(msg[6:])
            return
        self.status.setText(f"Transcribing {msg}" + (f" ({frac:.0%})" if frac >= 0 else ""))

    def dragEnterEvent(self, event) -> None:
        from .filetrans import EXTENSIONS

        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if urls and urls[0].toLocalFile().lower().endswith(EXTENSIONS):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self._start_file(urls[0].toLocalFile())

    def _start_stop(self) -> None:
        if self.ctl.meeting_running():
            self.ctl.stop_meeting()
        else:
            if not self.ctl.cfg.meeting_notice_ok:  # once: everyone should know
                answer = QMessageBox.question(
                    self, "Before you start",
                    "Let everyone in the meeting know you're taking notes. In some places, "
                    "recording or transcribing a conversation needs everyone's consent.\n\n"
                    "Murmr transcribes on this computer and keeps only the text.",
                    QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
                if answer != QMessageBox.StandardButton.Ok:
                    return
                self.ctl.update_setting("meeting_notice_ok", True)
            err = self.ctl.start_meeting()
            if err:
                QMessageBox.warning(self, "Meeting notes", err)
        self.refresh()
