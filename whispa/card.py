"""Shown when there's no text field to paste into: the transcript, ready to copy."""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
                               QWidget)

_STYLE = """
QLabel#title { color: rgba(255,255,255,0.55); font-size: 12px; }
QPlainTextEdit {
    background: transparent; border: none; color: #f4f4f5;
    font-size: 14px; selection-background-color: rgba(255,255,255,0.25);
}
QPushButton {
    border-radius: 13px; padding: 5px 14px; font-size: 12px; font-weight: 600;
}
QPushButton#copy { background: #ffffff; color: #0c0c0e; border: 1px solid #ffffff; }
QPushButton#copy:hover { background: #e4e4e7; border-color: #e4e4e7; }
QPushButton#close { background: rgba(255,255,255,0.08); color: #e4e4e7;
                    border: 1px solid rgba(255,255,255,0.12); }
QPushButton#close:hover { background: rgba(255,255,255,0.16); }
QScrollBar:vertical { width: 6px; background: transparent; }
QScrollBar::handle:vertical { background: rgba(255,255,255,0.25); border-radius: 3px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


class CopyCard(QWidget):
    copied = Signal(str)

    WIDTH = 400
    MARGIN = 16
    AUTO_HIDE_S = 20

    def __init__(self, bottom_offset: int = 110):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(_STYLE)
        self.bottom_offset = bottom_offset

        self.title = QLabel("No text field selected · copied to clipboard")
        self.title.setObjectName("title")
        self.text = QPlainTextEdit()
        self.text.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self.text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.copy_btn = QPushButton("Copy")
        self.copy_btn.setObjectName("copy")
        self.copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.copy_btn.clicked.connect(self._copy)
        close = QPushButton("Dismiss")
        close.setObjectName("close")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self.hide)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self.copy_btn)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.MARGIN + 2, self.MARGIN, self.MARGIN + 2, self.MARGIN)
        layout.setSpacing(8)
        layout.addWidget(self.title)
        layout.addWidget(self.text)
        layout.addLayout(buttons)

        self._hide_at = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._maybe_hide)

    DEFAULT_TITLE = "No text field selected · copied to clipboard"

    def show_text(self, text: str, anchor: str = "bottom-center", title: str = "") -> None:
        self.title.setText(title or self.DEFAULT_TITLE)
        self.text.setPlainText(text)
        self.copy_btn.setText("Copy")
        fm = self.text.fontMetrics()
        inner = self.WIDTH - 2 * self.MARGIN - 8
        rect = fm.boundingRect(0, 0, inner, 10_000, Qt.TextFlag.TextWordWrap, text)
        text_h = min(max(rect.height() + 12, 28), 180)
        self.text.setFixedHeight(text_h)
        self.setFixedSize(self.WIDTH, text_h + 2 * self.MARGIN + 26 + 34)
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self.move(self._position(screen.availableGeometry(), anchor))
        self._hide_at = time.monotonic() + self.AUTO_HIDE_S
        self.show()
        self.raise_()
        self._timer.start()

    def _position(self, area, anchor: str) -> QPoint:
        """Next to the pill: above it at the bottom, below it at the top, beside it on a side."""
        v, h = anchor.split("-") if "-" in anchor else ("bottom", "center")
        edge, pill_h, pill_w, gap = 16, 34, 34, 10  # on a side the pill is upright: 34 px wide
        w, hgt = self.width(), self.height()
        if h == "left":
            x = area.left() + edge
        elif h == "right":
            x = area.left() + area.width() - edge - w
        else:
            x = area.left() + (area.width() - w) // 2
        if v == "top":
            y = area.top() + edge + pill_h + gap
        elif v == "bottom":
            y = area.top() + area.height() - edge - pill_h - gap - hgt
        else:  # left/right edge, vertically centred: sit beside the pill
            y = area.top() + (area.height() - hgt) // 2
            x += (pill_w + gap) if h == "left" else -(pill_w + gap)
        x = max(area.left() + 8, min(x, area.left() + area.width() - w - 8))
        y = max(area.top() + 8, min(y, area.top() + area.height() - hgt - 8))
        return QPoint(x, y)

    def _copy(self) -> None:
        text = self.text.toPlainText()
        QGuiApplication.clipboard().setText(text)
        self.copy_btn.setText("Copied ✓")
        self.copied.emit(text)
        self._hide_at = time.monotonic() + 1.2

    def _maybe_hide(self) -> None:
        if self.underMouse() or self.text.hasFocus():
            self._hide_at = max(self._hide_at, time.monotonic() + 3)
            return
        if time.monotonic() > self._hide_at:
            self.hide()

    def hideEvent(self, event) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)
        p.fillPath(path, QColor(12, 12, 14, 242))
        p.setPen(QPen(QColor(255, 255, 255, 40), 1))
        p.drawPath(path)
