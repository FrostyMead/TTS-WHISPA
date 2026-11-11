"""Start Menu / Desktop shortcuts (Windows) and an app launcher entry (Linux).

They start Whispa with pythonw.exe, so no console window appears.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import config


def launcher() -> tuple[str, list[str]]:
    """(executable, args) that start Whispa without a console window."""
    if getattr(sys, "frozen", False):
        return sys.executable, []
    if sys.platform == "win32":
        # Go straight to the base interpreter's pythonw.exe: virtual-env launcher
        # stubs (e.g. uv's) can start console python.exe, which opens a terminal.
        base_w = Path(sys.base_prefix) / "pythonw.exe"
        script = Path(__file__).resolve().parent.parent / "launcher.pyw"
        if base_w.exists() and script.exists():
            import sysconfig

            packages = sysconfig.get_paths()["purelib"]
            return str(base_w), ["-I", "-S", str(script), packages]
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        return str(pythonw if pythonw.exists() else sys.executable), ["-m", "whispa"]
    return sys.executable, ["-m", "whispa"]


def _icon_path() -> Path:
    from PySide6.QtGui import QGuiApplication

    from .icons import write_ico

    app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841 - needed to draw
    return write_ico(config.config_dir() / "whispa.ico")


def install(desktop: bool = True) -> list[Path]:
    exe, args = launcher()
    made: list[Path] = []
    if sys.platform == "win32":
        icon = _icon_path()
        targets = [Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Whispa.lnk"]
        if desktop:
            targets.append(Path(os.environ["USERPROFILE"]) / "Desktop/Whispa.lnk")
        workdir = str(Path(__file__).resolve().parent.parent)
        for lnk in targets:
            lnk.parent.mkdir(parents=True, exist_ok=True)
            ps = (
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:WHISPA_LNK);"
                "$s.TargetPath=$env:WHISPA_EXE;$s.Arguments=$env:WHISPA_ARGS;"
                "$s.WorkingDirectory=$env:WHISPA_DIR;$s.IconLocation=$env:WHISPA_ICO;"
                "$s.Description='Whispa voice dictation';$s.Save()"
            )
            env = dict(os.environ, WHISPA_LNK=str(lnk), WHISPA_EXE=exe,
                       WHISPA_ARGS=subprocess.list2cmdline(args), WHISPA_DIR=workdir,
                       WHISPA_ICO=str(icon))
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           check=True, env=env, capture_output=True)
            made.append(lnk)
    elif sys.platform.startswith("linux"):
        import shlex

        icon = config.config_dir() / "whispa.png"
        from PySide6.QtGui import QGuiApplication

        from .icons import app_pixmap

        app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
        app_pixmap(256).save(str(icon))
        entry = Path.home() / ".local/share/applications/whispa.desktop"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("[Desktop Entry]\nType=Application\nName=Whispa\n"
                         "Comment=Local voice dictation\n"
                         f"Exec={shlex.join([exe, *args])}\nIcon={icon}\nTerminal=false\n"
                         "Categories=Utility;Accessibility;\n", encoding="utf-8")
        made.append(entry)
    return made
