"""Report a problem: one zip with what's needed to find a bug, and nothing personal.

In the zip:
* ``report.txt``: what you wrote, Murmr's version, the OS, the speech model status,
  permissions (macOS), the local AI status, which features are on.
* ``settings.json``: your settings, minus the API key and the text of your replacements.
* the logs (``whispa.log``, ``crash.log``, ``stdio.log``, the installer's ``boot.log``), with
  learned words, meeting titles, your user name and home folder removed.

Never in it: audio, recordings, dictation history, your dictionary, meeting transcripts.
The zip is saved on your computer; you choose whether to send it (attach it to a GitHub
issue, or send it however you like). Nothing is uploaded by Murmr.
"""

from __future__ import annotations

import datetime as dt
import getpass
import json
import logging
import os
import platform
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

from . import __version__
from . import config as config_mod

log = logging.getLogger(__name__)

REPO = "FrostyMead/murmr"
LOGS = ("whispa.log", "whispa.log.1", "whispa.log.2", "crash.log", "stdio.log")
MAX_LOG_BYTES = 2_000_000  # per file: the end is what matters

# Log lines that carry words you said or wrote.
_PRIVATE = [
    (re.compile(r"(Learned: ).*"), r"\1[words removed]"),
    (re.compile(r"(Meeting summarized: ).*"), r"\1[title removed]"),
    (re.compile(r"(Dropped echoed line: ).*"), r"\1[text removed]"),
]


def redact(text: str, home: str | None = None, user: str | None = None) -> str:
    home = home if home is not None else str(Path.home())
    try:
        user = user if user is not None else getpass.getuser()
    except Exception:  # noqa: BLE001
        user = ""
    for pattern, repl in _PRIVATE:
        text = pattern.sub(repl, text)
    if home and len(home) > 3:
        for variant in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
            text = text.replace(variant, "~")
    if user and len(user) > 2:
        text = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", "<user>", text)
    return text


def safe_settings(cfg) -> dict:
    from dataclasses import asdict

    data = asdict(cfg)
    llm = data.get("llm_cleanup") or {}
    if llm.get("api_key"):
        llm["api_key"] = "[set]"
    if llm.get("prompt"):
        llm["prompt"] = "[custom prompt]"
    data["replacements"] = f"[{len(data.get('replacements') or {})} replacements]"
    for key in ("custom_mode_prompt", "device_name"):  # your own words / names
        if data.get(key):
            data[key] = "[set]"
    if data.get("training_excluded_apps"):
        data["training_excluded_apps"] = f"[{len(data['training_excluded_apps'])} apps]"
    if llm.get("base_url"):
        from .llm import is_local_url

        llm["base_url"] = "[this computer]" if is_local_url(llm["base_url"]) else "[another address]"
    return data


def _ram_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MEM(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MEM()
            m.dwLength = ctypes.sizeof(MEM)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:  # noqa: BLE001
        return None


def _call(ctl, name: str, *args):
    fn = getattr(ctl, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception as exc:  # noqa: BLE001
        return f"(error: {exc})"


def system_info(ctl=None) -> dict:
    """Facts about this install. ``ctl`` is the running app (optional)."""
    info: dict = {
        "Murmr": __version__,
        "OS": f"{platform.system()} {platform.release()} ({platform.version()})",
        "Machine": platform.machine(),
        "Python": platform.python_version(),
        "When": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if sys.platform == "darwin":
        info["macOS"] = platform.mac_ver()[0]
    ram = _ram_gb()
    if ram:
        info["Memory"] = f"{ram:.1f} GB"
    try:
        import onnxruntime as ort

        info["ONNX Runtime"] = f"{ort.__version__} ({', '.join(ort.get_available_providers())})"
    except Exception as exc:  # noqa: BLE001
        info["ONNX Runtime"] = f"not loaded ({exc})"
    try:
        from .updater import Updater

        up = Updater()
        info["Installed copy"] = up.installed
        if up.installed:
            v = up.current_version()
            info["Build"] = f"{v.get('sha', '?')[:7]} {v.get('date', '')}".strip()
    except Exception:  # noqa: BLE001
        pass
    if ctl is not None:
        info["Status"] = _call(ctl, "status_text")
        if sys.platform == "darwin":
            info["Permissions"] = _call(ctl, "permissions")
        ai = _call(ctl, "ai_available")
        if ai is not None:
            info["Local AI"] = ai
        cfg = getattr(ctl, "cfg", None)
        if cfg is not None:
            info["Model"] = cfg.model if not os.path.isabs(cfg.model) else "a tuned model"
            info["Device"] = cfg.device
            info["Shortcut"] = cfg.hotkey
    try:
        from . import capabilities

        info["Can train here"] = capabilities.can_train()
    except Exception:  # noqa: BLE001
        pass
    return info


def _format_info(info: dict) -> str:
    rows = []
    for k, v in info.items():
        if isinstance(v, dict):
            v = ", ".join(f"{a}: {b}" for a, b in v.items())
        rows.append(f"{k}: {v}")
    return "\n".join(rows)


def _read_tail(path: Path) -> str:
    with open(path, "rb") as f:
        size = path.stat().st_size
        if size > MAX_LOG_BYTES:
            f.seek(size - MAX_LOG_BYTES)
        data = f.read()
    text = data.decode("utf-8", "replace")
    if size > MAX_LOG_BYTES:
        text = "[earlier lines cut]\n" + text.split("\n", 1)[-1]
    return text


def log_files(cdir: Path | None = None) -> list[Path]:
    cdir = cdir or config_mod.config_dir()
    found = [cdir / name for name in LOGS if (cdir / name).is_file()]
    try:
        from .updater import runtime_dir

        rt = runtime_dir()
        if rt and (rt / "boot.log").is_file():
            found.append(rt / "boot.log")
    except Exception:  # noqa: BLE001
        pass
    return found


def default_folder() -> Path:
    for name in ("Desktop", "Downloads"):
        p = Path.home() / name
        if p.is_dir():
            return p
    return Path.home()


def build(description: str, ctl=None, dest: Path | None = None, cdir: Path | None = None,
          home: str | None = None, user: str | None = None) -> Path:
    """Write the report zip and return its path."""
    dest = dest or default_folder()
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"Murmr report {dt.datetime.now():%Y-%m-%d %H%M}.zip"
    info = system_info(ctl)
    text = (f"What happened:\n{description.strip() or '(not described)'}\n\n"
            f"{_format_info(info)}\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.txt", redact(text, home, user))
        cfg = getattr(ctl, "cfg", None)
        if cfg is not None:
            z.writestr("settings.json", redact(json.dumps(safe_settings(cfg), indent=2), home, user))
        for f in log_files(cdir):
            try:
                z.writestr(f"logs/{f.name}", redact(_read_tail(f), home, user))
            except OSError as exc:
                log.debug("Report: skipped %s: %s", f, exc)
    log.info("Problem report saved (%d KB)", path.stat().st_size // 1024)
    return path


def contents(path: Path, max_chars: int = 200_000) -> str:
    """Everything in a report zip, as text: what you'd be sending."""
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            text = z.read(name).decode("utf-8", "replace")
            parts.append(f"===== {name} =====\n{text}")
    out = "\n\n".join(parts)
    return out if len(out) <= max_chars else out[:max_chars] + "\n[…]"


def issue_url(description: str, ctl=None) -> str:
    """A new GitHub issue with the description and system summary filled in."""
    info = system_info(ctl)
    keep = ("Murmr", "Build", "OS", "macOS", "Machine", "Status")
    summary = "\n".join(f"- {k}: {info[k]}" for k in keep if info.get(k))
    description = redact(description)  # your user name / home folder, if you pasted a path
    first = description.strip().splitlines()[0][:80] if description.strip() else "Problem report"
    body = (f"**What happened**\n\n{description.strip()}\n\n**System**\n\n{redact(summary)}\n\n"
            "_Drag the report zip from your Desktop here to attach it._\n")
    return (f"https://github.com/{REPO}/issues/new?title={quote(first)}"
            f"&body={quote(body)}")


# --------------------------------------------------------------------------- dialog
def show_dialog(ctl, parent=None) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import (QDialog, QHBoxLayout, QPlainTextEdit, QPushButton,
                                   QVBoxLayout)

    from .widgets import Label

    from . import theme

    dlg = QDialog(parent)
    dlg.setObjectName("root")
    dlg.setStyleSheet(theme.stylesheet(theme.current()))
    dlg.setWindowTitle("Report a problem")
    dlg.setMinimumWidth(520)
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(22, 20, 22, 18)
    lay.setSpacing(10)
    title = Label("Report a problem")
    title.setObjectName("h2")
    lay.addWidget(title)
    intro = Label("Murmr saves a report file on your computer. Nothing is sent until you choose "
                  "to send it.")
    intro.setWordWrap(True)
    intro.setObjectName("muted")
    lay.addWidget(intro)
    box = QPlainTextEdit()
    box.setPlaceholderText("What happened? What did you expect? (e.g. “On my Mac, dictating into "
                           "Outlook pasted nothing.”)")
    box.setMinimumHeight(110)
    lay.addWidget(box)
    what = Label("Included: Murmr's version, your OS, settings (without your replacements or API "
                 "key) and logs (with learned words, meeting titles and your user name taken "
                 "out).\nNever included: audio, recordings, dictation history, your dictionary, "
                 "meetings.")
    what.setWordWrap(True)
    what.setObjectName("faint")
    lay.addWidget(what)
    result = Label("")
    result.setWordWrap(True)
    result.setObjectName("ok")
    result.setVisible(False)
    lay.addWidget(result)
    preview = QPlainTextEdit()  # exactly what's in the zip, before you send it anywhere
    preview.setReadOnly(True)
    preview.setMinimumHeight(180)
    preview.setVisible(False)
    lay.addWidget(preview)
    row = QHBoxLayout()
    issue = QPushButton("Open a GitHub issue")
    issue.setToolTip("Opens github.com with your description filled in. Attach the report there.")
    issue.setEnabled(False)
    row.addWidget(issue)
    row.addStretch(1)
    close = QPushButton("Close")
    close.clicked.connect(dlg.reject)
    save = QPushButton("Save report")
    save.setObjectName("primary")
    row.addWidget(close)
    row.addWidget(save)
    lay.addLayout(row)
    state: dict = {}

    def do_save():
        try:
            path = build(box.toPlainText(), ctl)
        except Exception as exc:  # noqa: BLE001
            log.exception("Problem report failed")
            result.setObjectName("error")
            result.setText(f"Couldn't save the report: {exc}")
            result.setVisible(True)
            return
        state["path"] = path
        result.setText(f"Saved “{path.name}” to your {path.parent.name}. Check what's in it below, "
                       "then send it to the developer or attach it to a GitHub issue.")
        preview.setPlainText(contents(path))
        preview.setVisible(True)
        result.setVisible(True)
        issue.setEnabled(True)
        save.setText("Show the file")
        save.clicked.disconnect()
        save.clicked.connect(lambda: reveal(path))
        reveal(path)

    save.clicked.connect(do_save)
    issue.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(issue_url(box.toPlainText(), ctl))))
    dlg.exec()


def reveal(path: Path) -> None:
    """Show the file in Explorer / Finder."""
    import subprocess

    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError as exc:
        log.debug("Couldn't show %s: %s", path, exc)
