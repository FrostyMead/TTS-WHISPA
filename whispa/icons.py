"""Icons drawn in code: the app icon (window, taskbar, shortcuts) and the tray icon."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap

_BARS = (0.32, 0.62, 1.0, 0.62, 0.32)


def _draw_bars(p: QPainter, rect: QRectF, color: QColor) -> None:
    n = len(_BARS)
    bw = rect.width() / (n * 2 - 1)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    for i, h in enumerate(_BARS):
        bh = rect.height() * h
        x = rect.left() + i * 2 * bw
        p.drawRoundedRect(QRectF(x, rect.center().y() - bh / 2, bw, bh), bw / 2, bw / 2)


def app_pixmap(size: int = 256) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    s = float(size)
    grad = QLinearGradient(0, 0, 0, s)
    grad.setColorAt(0, QColor(44, 44, 50))
    grad.setColorAt(1, QColor(10, 10, 12))
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(grad)
    m = s * 0.04
    p.drawRoundedRect(QRectF(m, m, s - 2 * m, s - 2 * m), s * 0.23, s * 0.23)
    _draw_bars(p, QRectF(s * 0.27, s * 0.28, s * 0.46, s * 0.44), QColor(255, 255, 255))
    p.end()
    return pm


def app_icon() -> QIcon:
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(app_pixmap(size))
    return icon


def tray_icon(active: bool = False) -> QIcon:
    if sys.platform == "darwin" and not active:
        return _mac_template_icon()
    size = 64
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(239, 68, 68) if active else QColor(24, 24, 28))
    p.drawRoundedRect(QRectF(2, 12, 60, 40), 20, 20)
    p.setPen(QColor(255, 255, 255, 60))
    _draw_bars(p, QRectF(16, 19, 32, 26), QColor(255, 255, 255))
    p.end()
    return QIcon(pm)


def _mac_template_icon() -> QIcon:
    """A menu-bar "template" image: macOS draws it in the menu bar's own colour (black on a
    light bar, white on a dark one). A coloured icon can vanish on a dark menu bar."""
    size = 64
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0))
    p.drawRoundedRect(QRectF(2, 12, 60, 40), 20, 20)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
    _draw_bars(p, QRectF(16, 19, 32, 26), QColor(0, 0, 0))  # cut the bars out
    p.end()
    icon = QIcon(pm)
    icon.setIsMask(True)
    return icon


def write_ico(path: Path) -> Path:
    """Save a multi-size .ico for Windows shortcuts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    app_pixmap(256).save(str(path), "ICO")
    return path
