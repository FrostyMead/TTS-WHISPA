"""Small reusable widgets for the Murmr window."""

from __future__ import annotations

from PySide6.QtCore import (QEvent, QObject, QTimer, Property, QEasingCurve, QPointF, QPropertyAnimation,
                            QRectF, QSize, Qt, Signal)
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication,
                               QBoxLayout, QAbstractButton, QComboBox, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QVBoxLayout, QWidget)

from . import theme


class Label(QLabel):
    """QLabel whose wrapped text always gets its full height.

    Qt sometimes sizes a word-wrapped label for the wrong width inside nested layouts,
    which cuts off its last lines (or makes text overlap the next row). Each resize pins
    the minimum height to what the text needs at the real width.
    """

    def setWordWrap(self, on: bool) -> None:
        super().setWordWrap(on)
        if on:  # extra height (if any) goes below the text, not between it and the title
            self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

    def sizeHint(self) -> QSize:
        s = super().sizeHint()
        if self.wordWrap() and self.width() > 0:
            # Layouts ask for the height at the *preferred* width, which Qt guesses far
            # narrower than the real one (too many lines, rows too tall): use the real width.
            h = self.heightForWidth(self.width())
            if h > 0:
                return QSize(self.width(), h)
        return s

    _resizing = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.wordWrap():
            return
        if self._resizing:
            # Resized again while fitting (a scroll bar appeared, the width changed): fit
            # once more just after, instead of recursing or keeping a stale height.
            QTimer.singleShot(0, self._fit_height)
            return
        self._fit_height()

    def _fit_height(self) -> None:
        old = self.minimumHeight()
        self._resizing = True
        try:
            # Qt's heightForWidth never answers less than the current minimum height, so a
            # label that was once narrow (tall) would stay tall: measure without it.
            self.setMinimumHeight(0)
            h = self.heightForWidth(self.width())
            if h <= 0:
                self.setMinimumHeight(old)
            elif h != old:
                self.setMinimumHeight(h)
                self.updateGeometry()
            else:
                self.setMinimumHeight(old)
        finally:
            self._resizing = False


def wrapped(text: str = "", name: str = "") -> Label:
    lab = Label(text)
    lab.setWordWrap(True)
    if name:
        lab.setObjectName(name)
    return lab


def nav_icon(kind: str, off: str, on: str, size: int = 18) -> QIcon:
    """Line icons for the sidebar (home, dictionary, settings), muted / highlighted."""
    icon = QIcon()
    for color, state in ((off, QIcon.State.Off), (on, QIcon.State.On)):
        for scale in (1, 2):
            pm = QPixmap(size * scale, size * scale)
            pm.fill(Qt.GlobalColor.transparent)
            pm.setDevicePixelRatio(scale)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            s = size
            if kind == "home":
                path = QPainterPath()
                path.moveTo(s * 0.17, s * 0.45)
                path.lineTo(s * 0.5, s * 0.17)
                path.lineTo(s * 0.83, s * 0.45)
                path.moveTo(s * 0.27, s * 0.38)
                path.lineTo(s * 0.27, s * 0.83)
                path.lineTo(s * 0.73, s * 0.83)
                path.lineTo(s * 0.73, s * 0.38)
                p.drawPath(path)
            elif kind == "dictionary":
                p.drawRoundedRect(QRectF(s * 0.22, s * 0.15, s * 0.56, s * 0.7), 2, 2)
                p.drawLine(QPointF(s * 0.36, s * 0.36), QPointF(s * 0.64, s * 0.36))
                p.drawLine(QPointF(s * 0.36, s * 0.52), QPointF(s * 0.58, s * 0.52))
            elif kind == "meetings":  # speech bubble with lines
                bubble = QPainterPath()
                bubble.addRoundedRect(QRectF(s * 0.14, s * 0.18, s * 0.72, s * 0.5), 3, 3)
                p.drawPath(bubble)
                p.drawLine(QPointF(s * 0.34, s * 0.68), QPointF(s * 0.28, s * 0.84))
                p.drawLine(QPointF(s * 0.28, s * 0.84), QPointF(s * 0.48, s * 0.68))
                p.drawLine(QPointF(s * 0.3, s * 0.36), QPointF(s * 0.7, s * 0.36))
                p.drawLine(QPointF(s * 0.3, s * 0.5), QPointF(s * 0.58, s * 0.5))
            elif kind == "training":  # microphone
                p.drawRoundedRect(QRectF(s * 0.38, s * 0.14, s * 0.24, s * 0.42), s * 0.12, s * 0.12)
                arc = QPainterPath()
                arc.moveTo(s * 0.26, s * 0.44)
                arc.cubicTo(s * 0.26, s * 0.78, s * 0.74, s * 0.78, s * 0.74, s * 0.44)
                p.drawPath(arc)
                p.drawLine(QPointF(s * 0.5, s * 0.7), QPointF(s * 0.5, s * 0.86))
            elif kind == "settings":
                for y, x in ((0.3, 0.62), (0.5, 0.38), (0.7, 0.58)):
                    p.drawLine(QPointF(s * 0.17, s * y), QPointF(s * 0.83, s * y))
                    p.setBrush(QColor(color))
                    p.drawEllipse(QPointF(s * x, s * y), s * 0.07, s * 0.07)
                    p.setBrush(Qt.BrushStyle.NoBrush)
            p.end()
            icon.addPixmap(pm, QIcon.Mode.Normal, state)
    return icon


class Switch(QAbstractButton):
    """An iOS/Windows 11 style toggle."""

    def __init__(self, checked: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)
        self._pos = 1.0 if checked else 0.0
        # Not "pos": that name is QWidget's position, so animating it moved the switch to (0, 0).
        self._anim = QPropertyAnimation(self, b"knob", self)
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

    knob = Property(float, _get_pos, _set_pos)

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


TEXT_MIN = 220  # below this much room beside the control, the control goes underneath


class SettingRow(QFrame):
    """A settings line: title + help text on the left, the control on the right. In a
    narrow window the control moves under the text instead of being cut off."""

    def __init__(self, title: str, description: str, control: QWidget | None, first: bool = False):
        super().__init__()
        self.setObjectName("row")
        self.setProperty("first", "true" if first else "false")
        self.control = control
        self.stacked = False
        self.box = QBoxLayout(QBoxLayout.Direction.LeftToRight, self)
        self.box.setContentsMargins(0, 12, 0, 12)
        self.box.setSpacing(16)
        block = QWidget()  # title + help text stay together, centred against the control
        block.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        text = QVBoxLayout(block)
        text.setContentsMargins(0, 0, 0, 0)
        text.setSpacing(2)
        t = wrapped(title, "rowTitle")
        text.addWidget(t)
        if description:
            text.addWidget(wrapped(description, "muted"))
        self.box.addWidget(block, 1, Qt.AlignmentFlag.AlignVCenter)
        if control is not None:
            if isinstance(control, QComboBox) and control.minimumWidth() == 0:
                # Full width when there's room; in a narrow window it shrinks (long choices
                # are shortened with "…") instead of pushing the page wider than the window.
                control.setMinimumWidth(150)
            self.box.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)

    def _control_width(self, preferred: bool = False) -> int:
        c = self.control
        if c is None:
            return 0
        if preferred:
            return max(c.sizeHint().width(), c.minimumWidth())
        # Layouts honour an explicit minimum width over the widget's own hint.
        return c.minimumWidth() or c.minimumSizeHint().width()

    def minimumSizeHint(self) -> QSize:
        s = super().minimumSizeHint()
        if self.control is None:
            return s
        # Never ask for more than the control itself: the text can always go above it.
        return QSize(min(s.width(), self._control_width()), s.height())

    _pending = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.control is not None and not self._pending:
            # Re-arranging inside a resize makes Qt resize again at once (and again...):
            # do it just after, once things have settled.
            self._pending = True
            QTimer.singleShot(0, self._arrange)

    def _arrange(self) -> None:
        self._pending = False
        try:
            need = self._control_width(preferred=True) + TEXT_MIN + 16
        except RuntimeError:  # the row (or its control) was deleted before this ran
            return
        width = self.width()
        # A little slack either way, so a width right at the edge doesn't flip back and forth.
        stack = width < need if not self.stacked else width < need + 24
        if stack == self.stacked:
            return
        self.stacked = stack
        self.box.setDirection(QBoxLayout.Direction.TopToBottom if stack
                              else QBoxLayout.Direction.LeftToRight)
        self.box.setSpacing(8 if stack else 16)
        align = Qt.AlignmentFlag.AlignLeft if stack else Qt.AlignmentFlag.AlignVCenter
        self.box.setAlignment(self.control, align)


def setting_row(title: str, description: str, control: QWidget | None, first: bool = False) -> QFrame:
    return SettingRow(title, description, control, first)


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
    clear_requested = Signal()  # "Off" pressed (optional shortcuts only)

    def __init__(self, labels: list[str], optional: bool = False):
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
        self._off = None
        if optional:  # an optional shortcut can be switched off again
            self._off = QPushButton("Off")
            self._off.setToolTip("Turn this shortcut off")
            self._off.setCursor(Qt.CursorShape.PointingHandCursor)
            self._off.clicked.connect(self.clear_requested.emit)
            self._lay.addWidget(self._off)
        self.recording = False
        self._is_off = labels == ["Off"]
        self._sync_off()

    def _sync_off(self) -> None:
        if self._off is not None:
            self._off.setVisible(not self.recording and not self._is_off)

    def _clicked(self) -> None:
        (self.cancel_requested if self.recording else self.change_requested).emit()

    def set_recording(self, on: bool) -> None:
        self.recording = on
        self._caps.setVisible(not on)
        self._sync_off()
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
        self._is_off = labels == ["Off"]
        self._sync_off()


class StatCard(QFrame):
    def __init__(self, label: str):
        super().__init__()
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(0)
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


class _WheelGuard(QObject):
    """Scrolling a page with the mouse wheel never changes a drop-down, number box or slider
    the pointer happens to pass over: the wheel only changes one you've clicked into.
    Otherwise the page scrolls as if the control wasn't there."""

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.Wheel or obj.hasFocus():
            return False
        w = obj.parentWidget()
        while w is not None and not isinstance(w, QAbstractScrollArea):
            w = w.parentWidget()
        if w is not None:
            bar = w.verticalScrollBar()
            if bar is not None and bar.isVisible():
                QApplication.sendEvent(bar, event)
        return True


_wheel_guard: _WheelGuard | None = None


def guard_wheel(root: QWidget) -> None:
    """Apply the wheel guard to every drop-down, number box and slider inside ``root``."""
    global _wheel_guard
    if _wheel_guard is None:
        _wheel_guard = _WheelGuard()
    for w in root.findChildren(QWidget):
        if isinstance(w, (QComboBox, QAbstractSpinBox, QAbstractSlider)) \
                and not w.property("_wheel_guarded"):
            w.setProperty("_wheel_guarded", True)
            if w.focusPolicy() == Qt.FocusPolicy.WheelFocus:
                w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the wheel alone doesn't focus it
            w.installEventFilter(_wheel_guard)
