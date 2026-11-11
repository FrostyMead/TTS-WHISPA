"""Small reusable widgets for the Murmr window."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (QAbstractButton, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from . import theme


class Switch(QAbstractButton):
    """An iOS/Windows 11 style toggle."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)
        self._pos = 1.0 if checked else 0.0
        self._anim = QPropertyAnimation(self, b"pos", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _get_pos(self) -> float:
        return self._pos

    def _set_pos(self, v: float) -> None:
        self._pos = v
        self.update()

    pos = Property(float, _get_pos, _set_pos)

    def sizeHint(self) -> QSize:
        return QSize(40, 22)

    def paintEvent(self, _event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        off, on = QColor(pal.track_off), QColor(pal.accent)
        t = self._pos
        track = QColor(int(off.red() + (on.red() - off.red()) * t),
                       int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        if not self.isEnabled():
            track.setAlphaF(0.4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0, 0, 40, 22), 11, 11)
        knob = QColor(pal.accent_text) if t > 0.5 else QColor("#ffffff")
        p.setBrush(knob)
        x = 3 + t * 18
        p.drawEllipse(QRectF(x, 3, 16, 16))


def card() -> tuple[QFrame, QVBoxLayout]:
    frame = QFrame()
    frame.setObjectName("card")
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 6, 18, 6)
    lay.setSpacing(0)
    return frame, lay


def setting_row(title: str, description: str, control: QWidget | None, first: bool = False) -> QFrame:
    """A settings line: title + help text on the left, the control on the right."""
    row = QFrame()
    row.setObjectName("row")
    row.setProperty("first", "true" if first else "false")
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 12, 0, 12)
    lay.setSpacing(16)
    text = QVBoxLayout()
    text.setSpacing(2)
    t = QLabel(title)
    t.setStyleSheet("font-weight: 600;")
    text.addWidget(t)
    if description:
        d = QLabel(description)
        d.setObjectName("muted")
        d.setWordWrap(True)
        text.addWidget(d)
    lay.addLayout(text, 1)
    if control is not None:
        lay.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    return row


def keycaps(labels: list[str]) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    for i, label in enumerate(labels):
        if i:
            plus = QLabel("+")
            plus.setObjectName("faint")
            lay.addWidget(plus)
        cap = QLabel(label)
        cap.setObjectName("keycap")
        lay.addWidget(cap)
    return w


class HotkeyField(QWidget):
    """Shows the current shortcut as keycaps with a Change button that records a new one."""

    change_requested = Signal()
    cancel_requested = Signal()

    def __init__(self, labels: list[str]):
        super().__init__()
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(10)
        self._caps = keycaps(labels)
        self._button = QPushButton("Change")
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.clicked.connect(self._clicked)
        self._lay.addWidget(self._caps)
        self._lay.addWidget(self._button)
        self.recording = False

    def _clicked(self) -> None:
        (self.cancel_requested if self.recording else self.change_requested).emit()

    def set_recording(self, on: bool) -> None:
        self.recording = on
        self._caps.setVisible(not on)
        self._button.setText("Press keys…  (Esc to cancel)" if on else "Change")
        self._button.setObjectName("recording" if on else "")
        self._button.style().unpolish(self._button)
        self._button.style().polish(self._button)

    def set_labels(self, labels: list[str]) -> None:
        old = self._caps
        self._caps = keycaps(labels)
        self._lay.replaceWidget(old, self._caps)
        old.deleteLater()
        self._caps.setVisible(not self.recording)


class StatCard(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(2)
        self.value = QLabel("–")
        self.value.setObjectName("statValue")
        cap = QLabel(label)
        cap.setObjectName("statLabel")
        lay.addWidget(self.value)
        lay.addWidget(cap)

    def set(self, value: str) -> None:
        self.value.setText(value)
