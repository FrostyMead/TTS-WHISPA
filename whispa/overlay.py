"""The floating "pill" at the bottom of the screen.

idle         a slim bar (hover: hint text, click: hands-free dictation)
listening    expands, white bars dance with your voice
locked       hands-free: adds cancel (x) and stop buttons
transcribing a shimmer wave while Parakeet runs
message      short status text (errors, "Didn't catch that")
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QFontMetricsF, QGuiApplication, QPainter, QPainterPath, QPen, QRegion
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

N_BARS = 11
BG = QColor(12, 12, 14, 238)
BORDER = QColor(255, 255, 255, 46)
FG = QColor(255, 255, 255)
STOP_RED = QColor(255, 80, 80)
ERROR_TEXT = QColor(255, 150, 150)


def _level_to_unit(rms: float) -> float:
    db = 20 * math.log10(max(rms, 1e-7))
    return min(1.0, max(0.0, (db + 52) / 38))


class Overlay(QWidget):
    idle_clicked = Signal()
    stop_clicked = Signal()
    cancel_clicked = Signal()

    WIN_W, WIN_H = 340, 96
    MARGIN_BOTTOM = 14

    def __init__(self, level_fn: Callable[[], float], hint: str, show_idle: bool = True):
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

    def set_locked(self, locked: bool) -> None:
        self.locked = locked
        self._ensure_running()

    def flash(self, text: str, error: bool = False, seconds: float = 1.8) -> None:
        self._message, self._message_error = text, error
        self._message_until = time.monotonic() + seconds
        self._place()
        self._ensure_running()

    # -- geometry --------------------------------------------------------
    def _place(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.x() + (area.width() - self.WIN_W) // 2
        y = area.y() + area.height() - self.WIN_H - self.MARGIN_BOTTOM
        self.move(x, y)

    def _target(self) -> tuple[float, float, float]:
        """(width, height, opacity) the pill is animating towards."""
        if self._message:
            fm = QFontMetricsF(self._font)
            return min(self.WIN_W - 8, fm.horizontalAdvance(self._message) + 36), 32, 1.0
        if self.state == "listening":
            return (172 if self.locked else 116), 34, 1.0
        if self.state == "transcribing":
            return 116, 34, 1.0
        if self._hover:
            fm = QFontMetricsF(self._font)
            return min(self.WIN_W - 8, fm.horizontalAdvance(self.hint) + 32), 28, 1.0
        return 42, 9, (0.85 if self.show_idle else 0.0)

    def _pill_rect(self) -> QRectF:
        x = (self.WIN_W - self._w) / 2
        y = self.WIN_H - 6 - self._h
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
        radius = rect.height() / 2
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        p.fillPath(path, BG)
        p.setPen(QPen(BORDER, 1))
        p.drawPath(path)

        tw, th, _ = self._target()
        # Fade content in only once the pill has nearly reached its size.
        content = max(0.0, min(1.0, 1 - abs(tw - self._w) / 30)) if th > 20 else 0.0
        if content <= 0:
            return
        p.setOpacity(self._alpha * content)
        if self._message:
            self._draw_text(p, rect, self._message, ERROR_TEXT if self._message_error else FG)
        elif self.state == "listening":
            inner = rect
            if self.locked:
                self._draw_cancel(p, self._cancel_rect(rect))
                self._draw_stop(p, self._stop_rect(rect))
                inner = rect.adjusted(34, 0, -34, 0)
            self._draw_bars(p, inner)
        elif self.state == "transcribing":
            self._draw_shimmer(p, rect)
        elif self._hover:
            self._draw_text(p, rect, self.hint, QColor(255, 255, 255, 220))

    def _bar_geometry(self, rect: QRectF):
        bw, gap = 3.0, 3.2
        total = N_BARS * bw + (N_BARS - 1) * gap
        x0 = rect.center().x() - total / 2
        return bw, gap, x0, rect.center().y()

    def _draw_bars(self, p: QPainter, rect: QRectF) -> None:
        bw, gap, x0, cy = self._bar_geometry(rect)
        max_h = rect.height() - 14
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(FG)
        for i, v in enumerate(self._bars):
            h = 3 + v * (max_h - 3)
            p.drawRoundedRect(QRectF(x0 + i * (bw + gap), cy - h / 2, bw, h), bw / 2, bw / 2)

    def _draw_shimmer(self, p: QPainter, rect: QRectF) -> None:
        bw, gap, x0, cy = self._bar_geometry(rect)
        t = time.monotonic() - self._t0
        p.setPen(Qt.PenStyle.NoPen)
        for i in range(N_BARS):
            wave = max(0.0, math.sin(t * 7 - i * 0.55)) ** 2
            h = 3 + 7 * wave
            c = QColor(FG)
            c.setAlphaF(0.35 + 0.65 * wave)
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x0 + i * (bw + gap), cy - h / 2, bw, h), bw / 2, bw / 2)

    def _draw_text(self, p: QPainter, rect: QRectF, text: str, color: QColor) -> None:
        p.setFont(self._font)
        p.setPen(color)
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def _cancel_rect(self, rect: QRectF) -> QRectF:
        return QRectF(rect.left() + 6, rect.center().y() - 11, 22, 22)

    def _stop_rect(self, rect: QRectF) -> QRectF:
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

    def mousePressEvent(self, event) -> None:
        pos = event.position()
        rect = self._pill_rect()
        if self.state == "idle" and rect.adjusted(-6, -6, 6, 6).contains(pos):
            self._hover = False
            self.idle_clicked.emit()
        elif self.state == "listening" and self.locked:
            if self._cancel_rect(rect).adjusted(-4, -4, 4, 4).contains(pos):
                self.cancel_clicked.emit()
            elif self._stop_rect(rect).adjusted(-4, -4, 4, 4).contains(pos):
                self.stop_clicked.emit()

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
