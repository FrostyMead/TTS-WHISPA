"""Computer commands: a few spoken actions, said in Command Mode (never in dictation).

    "open Spotify"            open an app (or bring it to the front if it's running)
    "switch to Chrome"        bring a running app to the front (opens it if it isn't)
    "open Downloads"          Downloads, Documents, Desktop, Pictures, Music, Videos, Home
    "open youtube.com"        a website, in your browser
    "volume up / down", "turn it up / down", "louder", "quieter", "mute", "unmute"

Only these. They're matched by plain rules (``parse``), not the AI, so they work without it
and never do something you didn't say. Nothing here deletes, sends or buys anything; if
actions like that are ever added, they must ask first.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Action:
    kind: str  # open_app | switch_app | open_folder | open_url | volume
    target: str = ""

    def describe(self) -> str:
        return {"open_app": f"Opening {self.target}", "switch_app": f"Switching to {self.target}",
                "open_folder": f"Opening {self.target}", "open_url": f"Opening {self.target}",
                "volume": {"up": "Volume up", "down": "Volume down",
                           # Windows' mute key can only toggle
                           "mute": "Mute on/off" if sys.platform == "win32" else "Muted",
                           "unmute": "Mute on/off" if sys.platform == "win32" else "Unmuted",
                           }.get(self.target, "Volume")}.get(self.kind, self.kind)


FOLDERS = {"downloads": "Downloads", "documents": "Documents", "desktop": "Desktop",
           "pictures": "Pictures", "photos": "Pictures", "music": "Music", "videos": "Videos",
           "movies": "Movies" if sys.platform == "darwin" else "Videos", "home": ""}

_POLITE = re.compile(r"^(?:(?:hey )?computer,?\s+|please\s+|can you\s+|could you\s+)+|\s*,?\s*please$")
_OPEN = re.compile(r"^(?:open|launch|start|run)\s+(?:up\s+)?(?:the\s+|my\s+)?(.+?)(?:\s+app)?$")
_SWITCH = re.compile(r"^(?:switch|go|change)\s+(?:over\s+)?to\s+(?:the\s+|my\s+)?(.+?)(?:\s+app|\s+window)?$")
_FOLDER = re.compile(r"^(?:my\s+)?(downloads|documents|desktop|pictures|photos|music|videos|movies|home)"
                     r"(?:\s+folder)?$")
_URL = re.compile(r"^([a-z0-9-]+(?:\s*(?:\.|dot)\s*[a-z0-9-]+)*\s*(?:\.|dot)\s*"
                  r"(?:com|org|net|io|ai|dev|app|co|uk|au|nz|ca|de|fr|gov|edu)(?:\s*(?:\.|dot)\s*[a-z]{2})?)$")
_VOLUME = [
    (re.compile(r"^(?:turn\s+(?:the\s+)?(?:volume|sound|it)\s+up|volume\s+up|louder|"
                r"increase\s+(?:the\s+)?volume|raise\s+(?:the\s+)?volume)$"), "up"),
    (re.compile(r"^(?:turn\s+(?:the\s+)?(?:volume|sound|it)\s+down|volume\s+down|quieter|softer|"
                r"decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume)$"), "down"),
    (re.compile(r"^(?:mute|mute\s+(?:the\s+)?(?:sound|volume|audio)|turn\s+(?:the\s+)?sound\s+off)$"), "mute"),
    (re.compile(r"^(?:unmute|unmute\s+(?:the\s+)?(?:sound|volume|audio)|turn\s+(?:the\s+)?sound\s+on)$"),
     "unmute"),
]


def parse(instruction: str) -> Action | None:
    """The action a spoken instruction asks for, or None (then it's a normal command)."""
    text = re.sub(r"[.!?]+$", "", instruction.strip().lower())
    text = _POLITE.sub("", text).strip(" ,")
    if not text or len(text) > 60:
        return None
    for pattern, which in _VOLUME:
        if pattern.match(text):
            return Action("volume", which)
    m = _OPEN.match(text)
    if m:
        target = m.group(1).strip()
        folder = _FOLDER.match(target)
        if folder:
            return Action("open_folder", FOLDERS[folder.group(1)] or "Home")
        url = _URL.match(target)
        if url:
            return Action("open_url", re.sub(r"\s*(?:\.|dot)\s*", ".", url.group(1)))
        if _looks_like_a_name(target):
            return Action("open_app", target)
        return None
    m = _SWITCH.match(text)
    if m and _looks_like_a_name(m.group(1)):
        return Action("switch_app", m.group(1).strip())
    return None


def _looks_like_a_name(target: str) -> bool:
    """An app name, not a sentence ("open the report I sent you yesterday" isn't one)."""
    words = target.split()
    return 0 < len(words) <= 4 and not any(w in {"i", "you", "we", "that", "this", "which", "and"}
                                           for w in words)


# --------------------------------------------------------------------------- finding apps
def best_match(name: str, candidates: dict[str, str], cutoff: float = 0.85) -> str | None:
    """The candidate (display name -> launch target) closest to what was said."""
    said = re.sub(r"[^a-z0-9]", "", name.lower())
    if not said:
        return None
    keys = {re.sub(r"[^a-z0-9]", "", k.lower()): k for k in candidates}
    if said in keys:
        return keys[said]
    starts = sorted((k for k in keys if k.startswith(said) or said.startswith(k) and len(k) >= 4),
                    key=len)
    if starts:
        return keys[starts[0]]
    # A word of the name: "chrome" -> "Google Chrome", "teams" -> "Microsoft Teams"
    words = sorted((k for k in candidates
                    if said in {re.sub(r"[^a-z0-9]", "", w) for w in k.lower().split()}), key=len)
    if words:
        return words[0]
    close = difflib.get_close_matches(said, list(keys), n=1, cutoff=cutoff)
    return keys[close[0]] if close else None


_apps_cache: tuple[float, dict[str, str]] = (0.0, {})
_apps_lock = threading.Lock()


def installed_apps() -> dict[str, str]:
    """Display name -> how to open it. Cached for 10 minutes (Windows asks PowerShell)."""
    global _apps_cache
    with _apps_lock:
        if time.monotonic() - _apps_cache[0] < 600 and _apps_cache[1]:
            return _apps_cache[1]
        apps = _windows_apps() if sys.platform == "win32" else _mac_apps() if sys.platform == "darwin" else {}
        _apps_cache = (time.monotonic(), apps)
        return apps


def _windows_apps() -> dict[str, str]:
    """Everything in the Start menu, Store apps included (Get-StartApps)."""
    apps: dict[str, str] = {}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                            "Get-StartApps | ForEach-Object { $_.Name + \"`t\" + $_.AppID }"],
                           capture_output=True, text=True, timeout=15, creationflags=0x08000000)
        for line in r.stdout.splitlines():
            name, _, app_id = line.partition("\t")
            if name.strip() and app_id.strip():
                apps.setdefault(name.strip(), "shell:AppsFolder\\" + app_id.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        log.info("Couldn't list Start menu apps: %s", exc)
    return apps


def _mac_apps() -> dict[str, str]:
    apps: dict[str, str] = {}
    for root in ("/Applications", "/System/Applications", "/System/Applications/Utilities",
                 str(Path.home() / "Applications"), "/Applications/Utilities"):
        try:
            for entry in os.scandir(root):
                if entry.name.endswith(".app"):
                    apps.setdefault(entry.name[:-4], entry.path)
        except OSError:
            continue
    return apps


# --------------------------------------------------------------------------- doing it
def run(action: Action) -> str:
    """Do it. Returns what to show (the pill); raises RuntimeError with a friendly message."""
    if action.kind == "volume":
        _volume(action.target)
        return action.describe()
    if action.kind == "open_folder":
        path = Path.home() / action.target if action.target != "Home" else Path.home()
        if not path.exists() and sys.platform == "win32":
            path = Path(os.environ.get("USERPROFILE", str(Path.home()))) / action.target
        _open_path(str(path))
        return action.describe()
    if action.kind == "open_url":
        import webbrowser

        webbrowser.open("https://" + action.target)
        return action.describe()
    if action.kind == "switch_app" and _bring_to_front(action.target):
        return action.describe()
    apps = installed_apps()
    name = best_match(action.target, apps)
    if name is None:
        raise RuntimeError(f"Couldn't find an app called “{action.target}”")
    if action.kind == "switch_app" or not _bring_to_front(name):
        _open_path(apps[name])
    return Action(action.kind, name).describe()


def _open_path(target: str) -> None:
    if sys.platform == "win32":
        if target.startswith("shell:"):
            subprocess.Popen(["explorer.exe", target], creationflags=0x08000000)
        else:
            os.startfile(target)  # noqa: S606 - a folder or app the user named
    elif sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def _bring_to_front(name: str) -> bool:
    """A running app whose name or window title matches: bring it to the front."""
    if sys.platform == "darwin":
        r = subprocess.run(["/usr/bin/osascript", "-e",
                            f'tell application "System Events" to (name of processes) contains "{_q(name)}"'],
                           capture_output=True, text=True, timeout=5)
        if r.stdout.strip() != "true":
            return False
        subprocess.run(["/usr/bin/osascript", "-e", f'tell application "{_q(name)}" to activate'],
                       capture_output=True, timeout=5)
        return True
    if sys.platform != "win32":
        return False
    hwnd = _find_window(name)
    if not hwnd:
        return False
    from .focus import Focus, _win_restore

    class _NoField:
        def SetFocus(self):  # noqa: N802 - the app decides which field gets the focus
            pass

    try:
        _win_restore(Focus(_window=hwnd, _handle=_NoField()))
    except Exception as exc:  # noqa: BLE001
        log.info("Couldn't bring %s to the front: %s", name, exc)
        return False
    return True


def _q(s: str) -> str:
    return s.replace("\\", "").replace('"', "")


def _find_window(name: str) -> int:
    """A visible top-level window whose title or process name matches ``name``."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found: dict[str, int] = {}
    buf = ctypes.create_unicode_buffer(512)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
            user32.GetWindowTextW(hwnd, buf, 512)
            title = buf.value
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            from .focus import _process_name

            for key in (_process_name(pid.value), title.rsplit(" - ", 1)[-1], title):
                if key:
                    found.setdefault(key, hwnd)
        return True

    user32.EnumWindows(each, 0)
    match = best_match(name, {k: str(v) for k, v in found.items()}, cutoff=0.8)
    return found.get(match, 0) if match else 0


def _volume(which: str) -> None:
    if sys.platform == "win32":
        import ctypes

        key = {"up": 0xAF, "down": 0xAE, "mute": 0xAD, "unmute": 0xAD}[which]
        for _ in range(5 if which in ("up", "down") else 1):  # ~10% a time
            ctypes.windll.user32.keybd_event(key, 0, 0, 0)
            ctypes.windll.user32.keybd_event(key, 0, 2, 0)
    elif sys.platform == "darwin":
        script = {"up": "set volume output volume ((output volume of (get volume settings)) + 10)",
                  "down": "set volume output volume ((output volume of (get volume settings)) - 10)",
                  "mute": "set volume with output muted", "unmute": "set volume without output muted"}[which]
        subprocess.run(["/usr/bin/osascript", "-e", script], capture_output=True, timeout=5)
    else:
        raise RuntimeError("Volume control isn't available on this system yet")
