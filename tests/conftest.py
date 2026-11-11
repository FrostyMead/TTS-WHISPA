"""Shared test setup."""
import os
import sys

# Offscreen Qt otherwise falls back to a much wider font than people see; measure layouts
# with the real system fonts.
if sys.platform == "win32" and os.path.isdir(r"C:\Windows\Fonts"):
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")
