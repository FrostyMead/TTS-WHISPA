"""Colors and the Qt stylesheet for the Whispa window (follows the OS light/dark mode)."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication


@dataclass(frozen=True)
class Palette:
    bg: str
    sidebar: str
    surface: str
    surface_hover: str
    border: str
    text: str
    muted: str
    faint: str
    accent: str
    accent_text: str
    accent_hover: str
    keycap: str
    keycap_border: str
    danger: str
    success: str
    track_off: str


LIGHT = Palette(
    bg="#f7f7f8", sidebar="#efeff1", surface="#ffffff", surface_hover="#f4f4f5",
    border="#e4e4e7", text="#18181b", muted="#71717a", faint="#a1a1aa",
    accent="#18181b", accent_text="#ffffff", accent_hover="#3f3f46",
    keycap="#ffffff", keycap_border="#d4d4d8", danger="#dc2626", success="#16a34a",
    track_off="#d4d4d8",
)

DARK = Palette(
    bg="#111113", sidebar="#0b0b0d", surface="#19191c", surface_hover="#222226",
    border="#2a2a2f", text="#f4f4f5", muted="#a1a1aa", faint="#71717a",
    accent="#f4f4f5", accent_text="#0c0c0e", accent_hover="#d4d4d8",
    keycap="#26262b", keycap_border="#3f3f46", danger="#f87171", success="#4ade80",
    track_off="#3f3f46",
)


def is_dark() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:  # Qt < 6.5
        return QGuiApplication.palette().window().color().lightness() < 128


def current() -> Palette:
    return DARK if is_dark() else LIGHT


def _arrow(color: str) -> str:
    """A chevron SVG for combo boxes (Qt stylesheets need an image file)."""
    import tempfile
    from pathlib import Path

    path = Path(tempfile.gettempdir()) / f"whispa-chevron-{color.strip('#')}.svg"
    if not path.exists():
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
            f'<path d="M2 3.5 5 6.5 8 3.5" fill="none" stroke="{color}" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>', encoding="utf-8")
    return path.as_posix()


def _font_family() -> str:
    if sys.platform == "win32":
        return 'font-family: "Segoe UI Variable Text", "Segoe UI";'
    if sys.platform == "darwin":
        return 'font-family: ".AppleSystemUIFont", "Helvetica Neue";'
    return ""


def stylesheet(p: Palette) -> str:
    return f"""
* {{ font-size: 13px; color: {p.text}; {_font_family()} }}
QWidget#root, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {p.bg}; }}
QWidget#sidebar {{ background: {p.sidebar}; border-right: 1px solid {p.border}; }}
QLabel#brand {{ font-size: 17px; font-weight: 700; padding-left: 4px; }}
QLabel#version {{ color: {p.faint}; font-size: 11px; padding-left: 4px; }}
QPushButton#nav {{
    text-align: left; padding: 8px 12px; border-radius: 8px; border: none;
    background: transparent; color: {p.muted}; font-weight: 500;
}}
QPushButton#nav:hover {{ background: {p.surface_hover}; color: {p.text}; }}
QPushButton#nav:checked {{ background: {p.surface}; color: {p.text}; font-weight: 600;
    border: 1px solid {p.border}; }}

QLabel#h1 {{ font-size: 24px; font-weight: 700; }}
QLabel#h2 {{ font-size: 15px; font-weight: 650; }}
QLabel#sub, QLabel#muted {{ color: {p.muted}; }}
QLabel#faint {{ color: {p.faint}; font-size: 12px; }}
QLabel#day {{ color: {p.muted}; font-size: 12px; font-weight: 600; padding: 14px 4px 4px 4px; }}
QLabel#statValue {{ font-size: 24px; font-weight: 700; }}
QLabel#statLabel {{ color: {p.muted}; font-size: 12px; }}
QLabel#error {{ color: {p.danger}; font-size: 12px; }}
QLabel#ok {{ color: {p.success}; font-size: 12px; }}
QLabel#time {{ color: {p.faint}; font-size: 12px; }}
QLabel#app {{ color: {p.muted}; font-size: 11px; background: {p.surface_hover};
    border: 1px solid {p.border}; border-radius: 6px; padding: 1px 6px; }}
QLabel#entry {{ font-size: 14px; }}
QLabel#keycap {{ background: {p.keycap}; border: 1px solid {p.keycap_border};
    border-bottom-width: 2px; border-radius: 6px; padding: 3px 9px; font-weight: 600; font-size: 12px; }}
QLabel#chip {{ background: {p.surface_hover}; border: 1px solid {p.border}; border-radius: 12px;
    padding: 4px 4px 4px 10px; }}

QFrame#card {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 12px; }}
QFrame#row {{ background: transparent; border: none; border-top: 1px solid {p.border}; }}
QFrame#row[first="true"] {{ border-top: none; }}
QFrame#entryRow {{ background: transparent; border-radius: 10px; }}
QFrame#entryRow:hover {{ background: {p.surface_hover}; }}

QPushButton {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px;
    padding: 6px 14px; font-weight: 600; }}
QPushButton:hover {{ background: {p.surface_hover}; }}
QPushButton:disabled {{ color: {p.faint}; }}
QPushButton#primary {{ background: {p.accent}; color: {p.accent_text}; border: 1px solid {p.accent}; }}
QPushButton#primary:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton#recording {{ background: {p.accent}; color: {p.accent_text}; border: 1px solid {p.accent}; }}
QPushButton#icon {{ background: transparent; border: none; padding: 4px 6px; color: {p.muted};
    font-size: 13px; border-radius: 6px; }}
QPushButton#icon:hover {{ background: {p.border}; color: {p.text}; }}
QPushButton#link {{ background: transparent; border: none; padding: 0; color: {p.muted};
    text-decoration: underline; font-weight: 500; }}

QLineEdit, QComboBox {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: 8px;
    padding: 6px 10px; selection-background-color: {p.accent}; selection-color: {p.accent_text}; }}
QLineEdit:focus, QComboBox:focus {{ border-color: {p.muted}; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox::down-arrow {{ image: url("{_arrow(p.muted)}"); width: 10px; height: 10px; }}
QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border};
    selection-background-color: {p.surface_hover}; selection-color: {p.text}; outline: none; padding: 4px; }}

QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {p.faint}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}
QToolTip {{ background: {p.surface}; color: {p.text}; border: 1px solid {p.border}; padding: 4px 8px; }}
"""
