"""The floating "pill" at the bottom of the screen.

idle         a slim bar (hover: hint text, click: hands-free dictation, drag: move + snap)
listening    expands, white bars dance with your voice
locked       hands-free: adds cancel (x) and stop buttons
transcribing a shimmer wave while Parakeet runs

On the left/right screen edge the pill stands upright; text (hints, messages)
still reads horizontally and opens inward from the edge.
message      short status text (errors, "Didn't catch that")
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from typing import Callable

from PySide6.QtCore import (QEasingCurve, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, Qt,
                            QTimer, Signal)
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetricsF, QGuiApplication, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

N_BARS = 11
DRAG_TEXT = "Release to snap"
ANCHORS = ("bottom-center", "bottom-left", "bottom-right", "top-center", "top-left", "top-right",
           "middle-left", "middle-right")
ANCHOR_LABELS = {
    "bottom-center": "Bottom centre", "bottom-left": "Bottom left", "bottom-right": "Bottom right",
    "top-center": "Top centre", "top-left": "Top left", "top-right": "Top right",
    "middle-left": "Left edge", "middle-right": "Right edge",
}
BG = QColor(12, 12, 14, 238)
BORDER = QColor(255, 255, 255, 46)
FG = QColor(255, 255, 255)
STOP_RED = QColor(255, 80, 80)
ERROR_TEXT = QColor(255, 150, 150)
COMMAND = QColor(196, 181, 253)  # violet: Command Mode (editing with your voice)
COMMAND_BORDER = QColor(167, 139, 250, 150)


def _level_to_unit(rms: float) -> float:
    db = 20 * math.log10(max(rms, 1e-7))
    return min(1.0, max(0.0, (db + 52) / 38))


class Overlay(QWidget):
    idle_clicked = Signal()
    stop_clicked = Signal()
    cancel_clicked = Signal()
    anchor_changed = Signal(str)

    WIN_W, WIN_H = 360, 200
    MARGIN = 16  # gap between the pill and the screen edge
    PAD = 6  # pill inset inside this (transparent) window
    DRAG_THRESHOLD = 5

    def __init__(self, level_fn: Callable[[], float], hint: str, show_idle: bool = True,
                 anchor: str = "bottom-center"):
        super().__init__(None)
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus
                 | Qt.WindowType.NoDropShadowWindowHint)
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMouseTracking(True)
        self.setWindowTitle("Whispa")
        self.resize(self.WIN_W, self.WIN_H)

        self.level_fn = level_fn
        self.hint = hint
        self.show_idle = show_idle
        self.state = "idle"
        self.locked = False
        self._message: str | None = None
        self._message_error = False
        self._message_until = 0.0
        self._hover = False
        self._w, self._h = 0.0, 0.0
        self._alpha = 0.0
        self._level = 0.0
        self._bars = [0.0] * N_BARS
        self._phases = [random.uniform(0, math.tau) for _ in range(N_BARS)]
        self._speeds = [random.uniform(5.0, 9.0) for _ in range(N_BARS)]
        self._t0 = time.monotonic()
        self._mac_tweaked = False
        self.anchor = anchor if anchor in ANCHORS else "bottom-center"
        self.mode = "dictate"
        self._press_global: QPoint | None = None
        self._press_win = QPoint()
        self._press_local = QPointF()
        self._dragging = False
        self._snap_anim = QPropertyAnimation(self, b"pos", self)
        self._snap_anim.setDuration(240)
        self._snap_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._font = QFont(self.font())
        self._font.setPixelSize(12)
        self._font.setWeight(QFont.Weight.Medium)

        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._place()
        self._ensure_running()

    # -- public API ------------------------------------------------------
    def set_state(self, state: str) -> None:
        if state == self.state:
            return
        if state != "idle" and self.state == "idle":
            self._place()  # follow the screen the cursor is on
        self.state = state
        if state in ("idle", "transcribing"):
            self.locked = False
        self._ensure_running()

    def set_mode(self, mode: str) -> None:
        """'command' tints the pill violet so you can tell you're editing, not dictating."""
        self.mode = mode
        self.update()

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        self._ensure_running()

    def flash(self, text: str, error: bool = False, seconds: float = 1.8) -> None:
        self._message, self._message_error = text, error
        self._message_until = time.monotonic() + seconds
        self._place()
        self._ensure_running()

    def set_anchor(self, anchor: str) -> None:
        """Move the pill to one of the snap positions (from Settings)."""
        if anchor not in ANCHORS:
            return
        self.anchor = anchor
        self._place()
        self._ensure_running()

    # -- geometry --------------------------------------------------------
    @staticmethod
    def _screen_at(point: QPoint):
        return QGuiApplication.screenAt(point) or QGuiApplication.primaryScreen()

    def _window_pos(self, anchor: str, area: QRect) -> QPoint:
        """Where this window goes so the pill sits MARGIN from the screen edges."""
        v, h = anchor.split("-")
        m, p = self.MARGIN, self.PAD
        if h == "left":
            x = area.left() + m - p
        elif h == "right":
            x = area.left() + area.width() - m - self.WIN_W + p
        else:
            x = area.left() + (area.width() - self.WIN_W) // 2
        if v == "top":
            y = area.top() + m - p
        elif v == "bottom":
            y = area.top() + area.height() - m - self.WIN_H + p
        else:
            y = area.top() + (area.height() - self.WIN_H) // 2
        return QPoint(x, y)

    def _place(self) -> None:
        screen = self._screen_at(QCursor.pos())  # follow the screen you're working on
        if screen is None or self._dragging or self._snap_anim.state() == QPropertyAnimation.State.Running:
            return
        self.move(self._window_pos(self.anchor, screen.availableGeometry()))

    def _anchor_points(self, area: QRect) -> dict[str, QPointF]:
        """Screen point where the pill's centre sits for each snap position."""
        m = self.MARGIN
        points = {}
        for a in ANCHORS:
            v, h = a.split("-")
            half_w = 17 if v == "middle" else 58  # side spots hold a narrow vertical pill
            if h == "left":
                x = area.left() + m + half_w
            elif h == "right":
                x = area.left() + area.width() - m - half_w
            else:
                x = area.center().x()
            if v == "top":
                y = area.top() + m + 17
            elif v == "bottom":
                y = area.top() + area.height() - m - 17
            else:
                y = area.center().y()
            points[a] = QPointF(x, y)
        return points

    @property
    def vertical(self) -> bool:
        """On the left/right edge the pill stands upright (text still reads horizontally)."""
        return self.anchor in ("middle-left", "middle-right")

    def _shape(self, length: float, thickness: float) -> tuple[float, float]:
        return (thickness, length) if self.vertical else (length, thickness)

    def _target(self) -> tuple[float, float, float]:
        """(width, height, opacity) the pill is animating towards."""
        if self._dragging:
            fm = QFontMetricsF(self._font)
            return fm.horizontalAdvance(DRAG_TEXT) + 32, 28, 1.0
        if self._message:
            fm = QFontMetricsF(self._font)
            return min(self.WIN_W - 8, fm.horizontalAdvance(self._message) + 36), 32, 1.0
        if self.state == "listening":
            return (*self._shape(172 if self.locked else 116, 34), 1.0)
        if self.state == "transcribing":
            return (*self._shape(116, 34), 1.0)
        if self._hover:
            fm = QFontMetricsF(self._font)
            return min(self.WIN_W - 8, fm.horizontalAdvance(self.hint) + 32), 28, 1.0
        return (*self._shape(42, 9), 0.85 if self.show_idle else 0.0)

    def _pill_rect(self, anchor: str | None = None) -> QRectF:
        """The pill inside this window. It grows away from the screen edge it's snapped to."""
        v, h = (anchor or self.anchor).split("-")
        p = self.PAD
        if h == "left":
            x = p
        elif h == "right":
            x = self.WIN_W - p - self._w
        else:
            x = (self.WIN_W - self._w) / 2
        if v == "top":
            y = p
        elif v == "bottom":
            y = self.WIN_H - p - self._h
        else:
            y = (self.WIN_H - self._h) / 2
        return QRectF(x, y, self._w, self._h)

    # -- animation -------------------------------------------------------
    def _ensure_running(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        if not self.isVisible():
            self.show()
            self._apply_platform_tweaks()

    def _tick(self) -> None:
        now = time.monotonic()
        if self._message and now > self._message_until:
            self._message = None
        tw, th, ta = self._target()
        k = 0.28
        self._w += (tw - self._w) * k
        self._h += (th - self._h) * k
        self._alpha += (ta - self._alpha) * 0.22

        raw = _level_to_unit(self.level_fn()) if self.state == "listening" else 0.0
        self._level = raw if raw > self._level else self._level * 0.86 + raw * 0.14
        t = now - self._t0
        for i in range(N_BARS):
            centre = 1 - abs(i - (N_BARS - 1) / 2) / ((N_BARS + 1) / 2)
            wobble = 0.55 + 0.45 * math.sin(t * self._speeds[i] + self._phases[i])
            target = self._level * (0.35 + 0.65 * centre) * wobble
            self._bars[i] += (target - self._bars[i]) * 0.45

        settled = (abs(tw - self._w) < 0.3 and abs(th - self._h) < 0.3
                   and abs(ta - self._alpha) < 0.01)
        idle_now = self.state == "idle" and not self._message
        if settled and idle_now:
            self._w, self._h, self._alpha = tw, th, ta
            self._timer.stop()
            if ta == 0.0:
                self.hide()
        r = self._pill_rect().adjusted(-2, -2, 2, 2).toAlignedRect()
        self.setMask(QRegion(r))
        self.update()

    # -- painting --------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(self._alpha)
        rect = self._pill_rect()
        radius = min(rect.width(), rect.height()) / 2
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.fillPath(path, BG)
        command = self.mode == "command" and self.state != "idle"
        p.setPen(QPen(COMMAND_BORDER if command else BORDER, 1.2 if command else 1))
        p.drawPath(path)

        tw, th, _ = self._target()
        # Fade content in only once the pill has nearly reached its size (and shape).
        off = abs(tw - self._w) + abs(th - self._h)
        content = max(0.0, min(1.0, 1 - off / 30)) if min(tw, th) > 20 else 0.0
        if content <= 0:
            return
        p.setOpacity(self._alpha * content)
        if self._dragging:
            self._draw_text(p, rect, DRAG_TEXT, FG)
        elif self._message:
            self._draw_text(p, rect, self._message, ERROR_TEXT if self._message_error else FG)
        elif self.state == "listening":
            inner = rect
            if self.locked:
                self._draw_cancel(p, self._cancel_rect(rect))
                self._draw_stop(p, self._stop_rect(rect))
                inner = rect.adjusted(0, 34, 0, -34) if self.vertical else rect.adjusted(34, 0, -34, 0)
            self._draw_bars(p, inner)
        elif self.state == "transcribing":
            self._draw_shimmer(p, rect)
        elif self._hover:
            self._draw_text(p, rect, self.hint, QColor(255, 255, 255, 220))

    def _bar(self, rect: QRectF, i: int, size: float) -> QRectF:
        """Bar ``i`` of the meter: upright bars in a row, or flat bars in a column when vertical."""
        bw, gap = 3.0, 3.2
        total = N_BARS * bw + (N_BARS - 1) * gap
        c = rect.center()
        if self.vertical:
            y = c.y() - total / 2 + i * (bw + gap)
            return QRectF(c.x() - size / 2, y, size, bw)
        x = c.x() - total / 2 + i * (bw + gap)
        return QRectF(x, c.y() - size / 2, bw, size)

    def _draw_bars(self, p: QPainter, rect: QRectF) -> None:
        span = (rect.width() if self.vertical else rect.height()) - 14
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(COMMAND if self.mode == "command" else FG)
        for i, v in enumerate(self._bars):
            p.drawRoundedRect(self._bar(rect, i, 3 + v * (span - 3)), 1.5, 1.5)

    def _draw_shimmer(self, p: QPainter, rect: QRectF) -> None:
        t = time.monotonic() - self._t0
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(N_BARS):
            wave = max(0.0, math.sin(t * 7 - i * 0.55)) ** 2
            c = QColor(COMMAND if self.mode == "command" else FG)
            c.setAlphaF(0.35 + 0.65 * wave)
            p.setBrush(c)
            p.drawRoundedRect(self._bar(rect, i, 3 + 7 * wave), 1.5, 1.5)

    def _draw_text(self, p: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        p.setFont(self._font)
        p.setPen(color)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _cancel_rect(self, rect: QRectF) -> QRectF:
        if self.vertical:  # top
            return QRectF(rect.center().x() - 11, rect.top() + 6, 22, 22)
        return QRectF(rect.left() + 6, rect.center().y() - 11, 22, 22)

    def _stop_rect(self, rect: QRectF) -> QRectF:
        if self.vertical:  # bottom
            return QRectF(rect.center().x() - 11, rect.bottom() - 28, 22, 22)
        return QRectF(rect.right() - 28, rect.center().y() - 11, 22, 22)

    def _draw_cancel(self, p: QPainter, r: QRectF) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 38))
        p.drawEllipse(r)
        pen = QPen(QColor(255, 255, 255, 210), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        c, d = r.center(), 3.6
        p.drawLine(QPointF(c.x() - d, c.y() - d), QPointF(c.x() + d, c.y() + d))
        p.drawLine(QPointF(c.x() - d, c.y() + d), QPointF(c.x() + d, c.y() - d))

    def _draw_stop(self, p: QPainter, r: QRectF) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(STOP_RED)
        p.drawEllipse(r)
        p.setBrush(FG)
        s = 7.0
        p.drawRoundedRect(QRectF(r.center().x() - s / 2, r.center().y() - s / 2, s, s), 1.5, 1.5)

    # -- mouse -----------------------------------------------------------
    def enterEvent(self, _event) -> None:
        if self.state == "idle":
            self._hover = True
            self._ensure_running()

    def leaveEvent(self, _event) -> None:
        self._hover = False
        self._ensure_running()

    # A press becomes a drag once the mouse moves a few pixels; otherwise it's a click.
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._snap_anim.stop()
        self._press_global = event.globalPosition().toPoint()
        self._press_win = self.pos()
        self._press_local = event.position()

    def mouseMoveEvent(self, event) -> None:
        if self._press_global is None:
            return
        delta = event.globalPosition().toPoint() - self._press_global
        if not self._dragging:
            if delta.manhattanLength() < self.DRAG_THRESHOLD:
                return
            self._dragging = True
            self._hover = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._ensure_running()
        self.move(self._press_win + delta)

    def mouseReleaseEvent(self, event) -> None:
        if self._press_global is None:
            return
        self._press_global = None
        if self._dragging:
            self._dragging = False
            self.unsetCursor()
            self._snap()
            return
        self._click(self._press_local)

    def _click(self, pos: QPointF) -> None:
        rect = self._pill_rect()
        if self.state == "idle" and rect.adjusted(-6, -6, 6, 6).contains(pos):
            self._hover = False
            self.idle_clicked.emit()
        elif self.state == "listening" and self.locked:
            if self._cancel_rect(rect).adjusted(-4, -4, 4, 4).contains(pos):
                self.cancel_clicked.emit()
            elif self._stop_rect(rect).adjusted(-4, -4, 4, 4).contains(pos):
                self.stop_clicked.emit()

    def _snap(self) -> None:
        """Slide to the nearest snap position on the screen the pill was dropped on."""
        pill_center = QPointF(self.mapToGlobal(self._pill_rect().center().toPoint()))
        screen = self._screen_at(pill_center.toPoint())
        if screen is None:
            return
        area = screen.availableGeometry()
        points = self._anchor_points(area)
        anchor = min(points, key=lambda a: (points[a] - pill_center).manhattanLength())
        # Switching anchor changes where the pill sits inside the window; shift the window
        # so the pill doesn't jump, then animate it into place.
        old_top_left = self._pill_rect().topLeft()
        self.anchor = anchor
        new_top_left = self._pill_rect().topLeft()
        start = self.pos() + (old_top_left - new_top_left).toPoint()
        self.move(start)
        self._snap_anim.setStartValue(start)
        self._snap_anim.setEndValue(self._window_pos(anchor, area))
        self._snap_anim.start()
        self._ensure_running()
        self.anchor_changed.emit(anchor)

    # -- platform --------------------------------------------------------
    def _apply_platform_tweaks(self) -> None:
        if sys.platform != "darwin" or self._mac_tweaked:
            return
        self._mac_tweaked = True
        try:
            import objc
            from AppKit import (NSStatusWindowLevel, NSWindowCollectionBehaviorCanJoinAllSpaces,
                                NSWindowCollectionBehaviorFullScreenAuxiliary,
                                NSWindowCollectionBehaviorStationary)

            view = objc.objc_object(c_void_p=int(self.winId()))
            win = view.window()
            win.setHidesOnDeactivate_(False)
            win.setLevel_(NSStatusWindowLevel)
            win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                       | NSWindowCollectionBehaviorFullScreenAuxiliary
                                       | NSWindowCollectionBehaviorStationary)
        except Exception as exc:  # noqa: BLE001
            log.warning("macOS overlay tweaks failed: %s", exc)
