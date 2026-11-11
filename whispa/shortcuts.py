"""How Whispa is started: Start Menu / Desktop shortcuts (Windows), Whispa.app (macOS),
an app-launcher entry (Linux). None of them opens a console window.
"""

from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__, config

MAC_APP = Path.home() / "Applications" / "Whispa.app"
BUNDLE_ID = "io.github.frostymead.whispa"


def launcher() -> tuple[str, list[str]]:
    """(executable, args) that start Whispa without a console window."""
    if getattr(sys, "frozen", False):
        return sys.executable, []
    from .updater import boot_command, runtime_dir

    if runtime_dir() is not None:  # installed copy: always go through the bootstrap
        cmd = boot_command()
        return cmd[0], cmd[1:]
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


def _qt_app():
    from PySide6.QtGui import QGuiApplication

    return QGuiApplication.instance() or QGuiApplication([])


def _icon_path() -> Path:
    from .icons import write_ico

    _qt_app()  # needed to draw
    return write_ico(config.config_dir() / "whispa.ico")


def install(desktop: bool = True) -> list[Path]:
    exe, args = launcher()
    made: list[Path] = []
    if sys.platform == "win32":
        icon = _icon_path()
        targets = [Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Whispa.lnk"]
        if desktop:
            targets.append(Path(os.environ["USERPROFILE"]) / "Desktop/Whispa.lnk")
        from .updater import runtime_dir

        # Installed copies start in the install root, never inside app/current (which
        # updates have to rename).
        workdir = str(runtime_dir() or Path(__file__).resolve().parent.parent)
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
    elif sys.platform == "darwin":
        made.append(build_mac_app(MAC_APP, exe, args))
    elif sys.platform.startswith("linux"):
        icon = config.config_dir() / "whispa.png"
        from .icons import app_pixmap

        _qt_app()
        app_pixmap(256).save(str(icon))
        entry = Path.home() / ".local/share/applications/whispa.desktop"
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_text("[Desktop Entry]\nType=Application\nName=Whispa\n"
                         "Comment=Local voice dictation\n"
                         f"Exec={shlex.join([exe, *args])}\nIcon={icon}\nTerminal=false\n"
                         "Categories=Utility;Accessibility;\n", encoding="utf-8")
        made.append(entry)
    return made


# --------------------------------------------------------------------------- macOS app
def mac_app_files(exe: str, args: list[str]) -> dict[str, bytes]:
    """Files of a small Whispa.app that starts the bootstrap.

    Updates replace the Python code, not this bundle, so the Microphone / Input
    Monitoring / Accessibility permissions you grant to Whispa stay granted.
    """
    info = {
        "CFBundleName": "Whispa",
        "CFBundleDisplayName": "Whispa",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": "Whispa",
        "CFBundleIconFile": "Whispa",
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": __version__,
        "CFBundleVersion": __version__,
        "LSMinimumSystemVersion": "11.0",
        "LSUIElement": True,  # menu-bar app: no Dock icon
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "Whispa listens while you hold the dictation shortcut "
                                        "and transcribes on this Mac.",
        "NSAppleEventsUsageDescription": "Whispa pastes your dictation into the app you're using.",
    }
    script = "#!/bin/bash\n# Starts Whispa (generated by the installer).\nexec " + \
        shlex.join([exe, *args]) + ' "$@"\n'
    return {"Contents/Info.plist": plistlib.dumps(info),
            "Contents/MacOS/Whispa": script.encode()}


def build_mac_app(dest: Path, exe: str, args: list[str]) -> Path:
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for rel, data in mac_app_files(exe, args).items():
        path = tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp / "Contents/MacOS/Whispa").chmod(0o755)
    _mac_icon(tmp / "Contents/Resources/Whispa.icns")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    if shutil.which("codesign"):  # ad-hoc signature: a stable identity for macOS privacy settings
        subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(dest)],
                       capture_output=True)
    return dest


def _mac_icon(icns: Path) -> None:
    if not shutil.which("iconutil"):
        return
    from .icons import app_pixmap

    _qt_app()
    iconset = icns.with_suffix(".iconset")
    iconset.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        app_pixmap(size).save(str(iconset / f"icon_{size}x{size}.png"))
        app_pixmap(size * 2).save(str(iconset / f"icon_{size}x{size}@2x.png"))
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], capture_output=True)
    shutil.rmtree(iconset, ignore_errors=True)
