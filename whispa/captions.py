"""Live captions: a floating bar with the last thing said, for any audio on the computer.

Frameless, always on top, never takes focus, and drags anywhere. ✕ closes it (and stops a
captions-only session).
"""

from __future__ import annotations

import sys

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QPainter, QPainterPath
from PySide6.QtWidgets import QWidget

BG = QColor(12, 12, 14, 225)
WHO = {"You": QColor(167, 243, 208), "Others": QColor(186, 230, 253)}


class CaptionsWindow(QWidget):
    closed = Signal()

    W, H = 760, 104

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowTitle("Murmr captions")
        self.resize(self.W, self.H)
        self._lines: list[tuple[str, str]] = []
        self._drag: QPoint | None = None
        self._font = QFont(self.font())
        self._font.setPixelSize(17)
        self._who_font = QFont(self.font())
        self._who_font.setPixelSize(11)
        self._who_font.setWeight(QFont.Weight.DemiBold)
        self._placed = False

    def set_lines(self, lines: list[tuple[str, str]]) -> None:
        """(who, text) pairs, oldest first; the last two are shown."""
        self._lines = [(w, t) for w, t in lines if t][-2:]
        if not self.isVisible():
            self._place()
            self.show()
        self.update()

    def _place(self) -> None:
        if self._placed:
            return
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.move(screen.center().x() - self.W // 2, screen.bottom() - self.H - 90)
        self._placed = True

    def _close_rect(self) -> QRectF:
        return QRectF(self.W - 30, 8, 20, 20)

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.W, self.H), 16, 16)
        p.fillPath(path, BG)
        # close
        c = self._close_rect()
        p.setPen(QColor(255, 255, 255, 140))
        p.drawLine(c.topLeft() + QPoint(6, 6), c.bottomRight() - QPoint(6, 6))
        p.drawLine(c.bottomLeft() + QPoint(6, -6), c.topRight() + QPoint(-6, 6))
        if not self._lines:
            p.setFont(self._font)
            p.setPen(QColor(255, 255, 255, 140))
            p.drawText(QRectF(24, 0, self.W - 60, self.H), Qt.AlignmentFlag.AlignVCenter,
                       "Listening… captions appear here")
            return
        fm = QFontMetrics(self._font)
        y = 16
        for who, text in self._lines:
            p.setFont(self._who_font)
            p.setPen(WHO.get(who, QColor(255, 255, 255)))
            p.drawText(QRectF(24, y, 120, 14), Qt.AlignmentFlag.AlignLeft, who.upper())
            p.setFont(self._font)
            p.setPen(QColor(255, 255, 255))
            # keep the end of a long line: that's what's being said now
            shown = text
            while fm.horizontalAdvance(shown) > self.W - 72 and len(shown) > 10:
                shown = "…" + shown[len(shown) // 6:].lstrip("…")
            p.drawText(QRectF(24, y + 13, self.W - 60, 24), Qt.AlignmentFlag.AlignLeft, shown)
            y += 42

    def mousePressEvent(self, e) -> None:
        if self._close_rect().contains(e.position()):
            self.hide()
            self.closed.emit()
            return
        self._drag = e.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, e) -> None:
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e) -> None:
        self._drag = None

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if sys.platform == "darwin" and QGuiApplication.platformName() == "cocoa":
            try:  # stay visible over full-screen apps and on every Space, like the pill
                import objc
                from AppKit import (NSStatusWindowLevel, NSWindowCollectionBehaviorCanJoinAllSpaces,
                                    NSWindowCollectionBehaviorFullScreenAuxiliary)

                win = objc.objc_object(c_void_p=int(self.winId())).window()
                win.setLevel_(NSStatusWindowLevel)
                win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces
                                           | NSWindowCollectionBehaviorFullScreenAuxiliary)
            except Exception:  # noqa: BLE001
                pass
