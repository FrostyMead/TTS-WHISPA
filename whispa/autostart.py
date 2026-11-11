"""Launch-at-login for Windows (registry), macOS (LaunchAgent) and Linux (XDG autostart)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

NAME = "Murmr"
_MAC_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.murmr.app.plist"
_OLD_MAC_PLIST = Path.home() / "Library" / "LaunchAgents" / "com.whispa.app.plist"
OLD_NAME = "Whispa"
_LINUX_DESKTOP = Path.home() / ".config" / "autostart" / "whispa.desktop"


def _command() -> list[str]:
    from .shortcuts import MAC_APP, launcher

    if sys.platform == "darwin" and MAC_APP.exists():
        # Through the app bundle, so macOS privacy permissions belong to Murmr.app.
        return ["/usr/bin/open", "-g", "-a", str(MAC_APP), "--args", "--background"]
    exe, args = launcher()  # pythonw.exe on Windows: no console window
    return [exe, *args, "--background"]


def is_enabled() -> bool:
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run") as k:
                winreg.QueryValueEx(k, NAME)
                return True
        except OSError:
            return False
    if sys.platform == "darwin":
        return _MAC_PLIST.exists()
    return _LINUX_DESKTOP.exists()


def set_enabled(enabled: bool) -> None:
    cmd = _command()
    if sys.platform == "win32":
        import subprocess
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run",
                            0, winreg.KEY_SET_VALUE) as k:
            if enabled:
                winreg.SetValueEx(k, NAME, 0, winreg.REG_SZ, subprocess.list2cmdline(cmd))
            else:
                try:
                    winreg.DeleteValue(k, NAME)
                except FileNotFoundError:
                    pass
    elif sys.platform == "darwin":
        if enabled:
            import plistlib

            _MAC_PLIST.parent.mkdir(parents=True, exist_ok=True)
            _MAC_PLIST.write_bytes(plistlib.dumps({
                "Label": "com.murmr.app", "ProgramArguments": cmd,
                "RunAtLoad": True, "ProcessType": "Interactive",
            }))
        else:
            _MAC_PLIST.unlink(missing_ok=True)
    else:
        if enabled:
            _LINUX_DESKTOP.parent.mkdir(parents=True, exist_ok=True)
            _LINUX_DESKTOP.write_text(
                "[Desktop Entry]\nType=Application\nName=Murmr\n"
                f"Exec={shlex.join(cmd)}\nX-GNOME-Autostart-enabled=true\n", encoding="utf-8")
        else:
            _LINUX_DESKTOP.unlink(missing_ok=True)


def migrate_old(before_app: bool = False) -> bool:
    """Launch at login was set up under the old name (Whispa): remove that entry and, unless
    the caller does it once the new app exists (``before_app``), add Murmr's. True if it was."""
    if sys.platform == "win32":
        import winreg

        key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key, 0,
                                winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE) as k:
                winreg.QueryValueEx(k, OLD_NAME)
                winreg.DeleteValue(k, OLD_NAME)
        except OSError:
            return False
        set_enabled(True)
        return True
    if sys.platform == "darwin":
        if not _OLD_MAC_PLIST.exists():
            return False
        _OLD_MAC_PLIST.unlink(missing_ok=True)
        if not before_app:
            set_enabled(True)
        return True
    return False
