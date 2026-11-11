"""Settings > Sync: switch it on, pair computers, see when each last synced (sync.py)."""

from __future__ import annotations

import sys
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout,
                               QWidget)

from .widgets import Label, Switch, setting_row


def ago(t: float, now: float | None = None) -> str:
    if not t:
        return "never"
    s = (now or time.time()) - t
    if s < 60:
        return "just now"
    if s < 3600:
        return f"{int(s // 60)} min ago"
    if s < 86400:
        return f"{int(s // 3600)} h ago"
    return f"{int(s // 86400)} days ago"


class _Relay(QObject):
    done = Signal(str, str)  # error ("" = ok), detail


class PairDialog(QDialog):
    """Show a code here, or type the other computer's code."""

    def __init__(self, ctl, parent=None):
        super().__init__(parent)
        self.ctl = ctl
        from . import theme

        self.setObjectName("root")
        self.setStyleSheet(theme.stylesheet(theme.current()))
        self.setWindowTitle("Pair a computer")
        self.setMinimumWidth(500)
        self.relay = _Relay()
        self.relay.done.connect(self._paired)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 20, 22, 18)
        lay.setSpacing(10)
        h = Label("Pair a computer")
        h.setObjectName("h2")
        lay.addWidget(h)
        intro = Label("Open Murmr on your other computer, turn on Sync in Settings, and click "
                      "Pair a computer there too. Both need to be on the same network.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # show a code
        show_row = QHBoxLayout()
        self.code = Label("")
        self.code.setStyleSheet("font-size: 30px; font-weight: 700; letter-spacing: 3px;")
        self.code.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        show_btn = QPushButton("Show a code on this computer")
        show_btn.setObjectName("primary")
        show_btn.clicked.connect(self._show_code)
        show_row.addWidget(show_btn)
        show_row.addStretch(1)
        lay.addLayout(show_row)
        lay.addWidget(self.code)
        self.wait = Label("")
        self.wait.setObjectName("faint")
        self.wait.setWordWrap(True)
        lay.addWidget(self.wait)

        # or type one
        lay.addSpacing(6)
        or_label = Label("Or type the code shown on the other computer:")
        or_label.setObjectName("muted")
        lay.addWidget(or_label)
        enter = QHBoxLayout()
        self.entry = QLineEdit()
        self.entry.setPlaceholderText("XXXX-XXXX")
        self.entry.setMaxLength(12)
        self.entry.returnPressed.connect(self._pair)
        self.addr = QLineEdit()
        self.addr.setPlaceholderText("Address (optional)")
        self.addr.setToolTip("Only if it isn't found by itself: the other computer's IP address, "
                             "shown under its code.")
        self.addr.setMaximumWidth(170)
        self.pair_btn = QPushButton("Pair")
        self.pair_btn.clicked.connect(self._pair)
        enter.addWidget(self.entry, 1)
        enter.addWidget(self.addr)
        enter.addWidget(self.pair_btn)
        lay.addLayout(enter)
        self.result = Label("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        close_row.addWidget(close)
        lay.addLayout(close_row)
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._peers_before = set(ctl.sync.peers) if ctl.sync else set()

    def _show_code(self) -> None:
        from .sync import local_addresses, pretty_code

        svc = self.ctl.sync
        if svc is None:
            return
        self.code.setText(pretty_code(svc.start_pairing()))
        addrs = ", ".join(local_addresses()) or "unknown"
        note = ("Windows may ask whether to allow Murmr on private networks: allow it. "
                if sys.platform == "win32" else "")
        self.wait.setText(f"Type this code on the other computer. Waiting… (5 minutes)\n"
                          f"{note}This computer's address: {addrs}, port {svc.port}")
        self._timer.start()

    def _tick(self) -> None:
        svc = self.ctl.sync
        if svc is None:
            return
        new = set(svc.peers) - self._peers_before
        if new:
            self._timer.stop()
            name = svc.peers[next(iter(new))].name
            self._set_result("", f"Paired with {name}. They'll stay in sync from now on.")
            self.code.setText("")
            self.wait.setText("")
        elif svc.pairing is None or svc.pairing["until"] < time.time():
            self._timer.stop()
            self.code.setText("")
            self.wait.setText("The code expired. Show a new one to try again.")

    def _pair(self) -> None:
        svc = self.ctl.sync
        code = self.entry.text()
        if svc is None or not code.strip():
            return
        host, port = self.addr.text().strip(), None
        if ":" in host:
            host, _, p = host.rpartition(":")
            port = int(p) if p.isdigit() else None
        self.pair_btn.setEnabled(False)
        self._set_result("", "Looking for the other computer…", busy=True)

        def work():
            from .sync import SyncError

            try:
                name = svc.pair_with(code, host or None, port)
                self.relay.done.emit("", f"Paired with {name}. They'll stay in sync from now on.")
            except SyncError as exc:
                self.relay.done.emit("error", str(exc))
            except Exception as exc:  # noqa: BLE001
                self.relay.done.emit("error", f"Pairing failed: {exc}")

        threading.Thread(target=work, name="sync-pair", daemon=True).start()

    def _paired(self, error: str, detail: str) -> None:
        self.pair_btn.setEnabled(True)
        self._set_result(error, detail)

    def _set_result(self, error: str, text: str, busy: bool = False) -> None:
        self.result.setObjectName("faint" if busy else "error" if error else "ok")
        self.result.style().unpolish(self.result)
        self.result.style().polish(self.result)
        self.result.setText(text)

    def done(self, r: int) -> None:  # noqa: D401 - closing: stop showing the code
        svc = self.ctl.sync
        if svc is not None and svc.pairing is not None:
            svc.stop_pairing()
        super().done(r)


class SyncSection(QWidget):
    """The rows inside Settings > Sync."""

    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        cfg = ctl.cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.enabled = Switch(cfg.sync_enabled)
        self.enabled.toggled.connect(self._toggle)
        lay.addWidget(setting_row(
            "Sync with my other computers",
            "Keeps your Dictionary (words, corrections, snippets) the same on your computers. "
            "They talk directly over your network, encrypted. Nothing goes through the internet "
            "or any server.", self.enabled, first=True))
        self.name = QLineEdit(cfg.device_name)
        self.name.setPlaceholderText(str(ctl.device_name() or "") if hasattr(ctl, "device_name") else "")
        self.name.setMaximumWidth(240)
        self.name.editingFinished.connect(lambda: ctl.update_setting("device_name",
                                                                     self.name.text().strip()))
        self.name_row = setting_row("This computer's name", "How it appears on your other "
                                    "computers.", self.name)
        lay.addWidget(self.name_row)
        self.models = Switch(cfg.sync_models)
        self.models.toggled.connect(lambda v: ctl.update_setting("sync_models", v))
        self.models_row = setting_row(
            "Speech models tuned to your voice", "Copy a model tuned on one computer to the "
            "others, e.g. trained on your PC, used on your Mac. Large (hundreds of MB).",
            self.models)
        lay.addWidget(self.models_row)
        self.voices = Switch(cfg.sync_voices)
        self.voices.toggled.connect(lambda v: ctl.update_setting("sync_voices", v))
        self.voices_row = setting_row(
            "Read-aloud voices made from your recordings", "A voice that sounds like you is "
            "personal: only switch this on between computers that are yours.", self.voices)
        lay.addWidget(self.voices_row)

        self.peers_box = QVBoxLayout()
        self.peers_box.setContentsMargins(0, 0, 0, 0)
        self.peers_box.setSpacing(0)
        lay.addLayout(self.peers_box)
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 12, 0, 10)
        self.pair_btn = QPushButton("Pair a computer…")
        self.pair_btn.setObjectName("primary")
        self.pair_btn.clicked.connect(self._pair)
        self.sync_btn = QPushButton("Sync now")
        self.sync_btn.clicked.connect(self._sync_now)
        self.note = Label("")
        self.note.setObjectName("faint")
        self.note.setWordWrap(True)
        buttons.addWidget(self.pair_btn)
        buttons.addWidget(self.sync_btn)
        buttons.addWidget(self.note, 1)
        lay.addLayout(buttons)
        self._peer_rows: list[QWidget] = []
        self.refresh()

    def _toggle(self, on: bool) -> None:
        err = self.ctl.update_setting("sync_enabled", on)
        if err:
            self.note.setText(err)
            self.enabled.blockSignals(True)
            self.enabled.setChecked(False)
            self.enabled.blockSignals(False)
        self.refresh()

    def _pair(self) -> None:
        if self.ctl.sync is None:
            return
        PairDialog(self.ctl, self.window()).exec()
        self.refresh()

    def _sync_now(self) -> None:
        svc = self.ctl.sync
        if svc is None or not svc.peers:
            return
        self.note.setText("Syncing…")
        threading.Thread(target=svc.sync_all, name="sync-now", daemon=True).start()

    def refresh(self) -> None:
        on = bool(self.ctl.cfg.sync_enabled and self.ctl.sync is not None)
        for row in (self.name_row, self.models_row, self.voices_row):
            row.setVisible(on)
        self.pair_btn.setVisible(on)
        svc = self.ctl.sync
        peers = list(svc.peers.values()) if on else []
        self.sync_btn.setVisible(bool(peers))
        for w in self._peer_rows:
            w.setParent(None)
            w.deleteLater()
        self._peer_rows = []
        for peer in peers:
            st = svc.status.get(peer.id)
            if st is not None and not st.ok:
                desc = f"Last try {ago(st.when)}: {st.text}"
            else:
                desc = f"Synced {ago(peer.last_sync)}" + (f" · {st.text}" if st and st.text else "")
            remove = QPushButton("Unpair")
            remove.clicked.connect(lambda _=False, pid=peer.id: self._forget(pid))
            row = setting_row(peer.name, desc, remove)
            self.peers_box.addWidget(row)
            self._peer_rows.append(row)
        if on and not peers:
            self.note.setText("No computers paired yet.")
        elif not on:
            self.note.setText("")
        elif self.note.text() in ("No computers paired yet.", "Syncing…"):
            self.note.setText("")

    def _forget(self, pid: str) -> None:
        if self.ctl.sync is not None:
            self.ctl.sync.forget(pid)
        self.refresh()
