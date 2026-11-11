"""Put text into whatever app has focus."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from typing import Callable

log = logging.getLogger(__name__)


def _is_wayland() -> bool:
    return sys.platform.startswith("linux") and bool(os.environ.get("WAYLAND_DISPLAY"))


class Injector:
    def __init__(self, method: str = "auto", shortcut: str = "", restore_clipboard: bool = True,
                 wait_for_release: Callable[[], None] | None = None):
        self.method = method
        self.shortcut = shortcut or ("cmd+v" if sys.platform == "darwin" else "ctrl+v")
        self.restore_clipboard = restore_clipboard
        self.wait_for_release = wait_for_release or (lambda: None)
        self._kb = None

    @property
    def kb(self):
        if self._kb is None:
            from pynput.keyboard import Controller

            self._kb = Controller()
        return self._kb

    def inject(self, text: str) -> None:
        if not text:
            return
        self.wait_for_release()
        method = self.method
        if method == "auto":
            method = "wayland" if _is_wayland() else "clipboard"
        if method == "wayland":
            if self._wayland_type(text):
                return
            method = "clipboard"
        if method == "type":
            self._type(text)
            return
        try:
            self._paste(text)
        except Exception as exc:  # noqa: BLE001 - clipboard backends fail in odd ways
            log.warning("Clipboard paste failed (%s); typing instead", exc)
            self._type(text)

    def copy(self, text: str) -> None:
        """Put text on the clipboard (no paste, no restore)."""
        try:
            import pyperclip

            pyperclip.copy(text)
        except Exception as exc:  # noqa: BLE001
            log.warning("Copy to clipboard failed: %s", exc)

    def copy_selection(self, timeout: float = 0.5) -> str | None:
        """Copy the current selection with Ctrl/Cmd+C and return it, restoring the clipboard.

        Returns None when nothing was selected (the clipboard didn't change).
        """
        import uuid

        import pyperclip

        self.wait_for_release()
        try:
            previous = pyperclip.paste()
        except Exception:  # noqa: BLE001
            previous = None
        sentinel = f"whispa-{uuid.uuid4()}"
        try:
            pyperclip.copy(sentinel)
            self._send_shortcut("cmd+c" if sys.platform == "darwin" else "ctrl+c")
            deadline = time.monotonic() + timeout
            got = sentinel
            while time.monotonic() < deadline:
                time.sleep(0.03)
                got = pyperclip.paste()
                if got != sentinel:
                    break
            return None if got == sentinel or not got.strip() else got
        except Exception as exc:  # noqa: BLE001
            log.warning("Copying the selection failed: %s", exc)
            return None
        finally:
            try:
                pyperclip.copy(previous if previous is not None else "")
            except Exception:  # noqa: BLE001
                pass

    def _type(self, text: str) -> None:
        """Type text; line breaks use Shift+Enter so chat apps don't send the message."""
        if sys.platform == "darwin":
            from .mackeys import type_text

            type_text(text)
            return
        from pynput.keyboard import Key

        for i, line in enumerate(text.split("\n")):
            if i:
                with self.kb.pressed(Key.shift):
                    self.kb.tap(Key.enter)
            if line:
                self.kb.type(line)

    def _paste(self, text: str) -> None:
        import pyperclip

        previous = None
        if self.restore_clipboard:
            try:
                previous = pyperclip.paste()
            except Exception:  # noqa: BLE001 - non-text clipboard content
                previous = None
        pyperclip.copy(text)
        time.sleep(0.04)
        self._send_shortcut(self.shortcut)
        if previous is not None:
            time.sleep(0.3)  # let the target app read the clipboard first
            try:
                if pyperclip.paste() == text:
                    pyperclip.copy(previous)
            except Exception:  # noqa: BLE001
                pass

    def _send_shortcut(self, spec: str) -> None:
        if sys.platform == "darwin":  # pynput's layout lookups crash off the main thread
            from .mackeys import send_shortcut

            send_shortcut(spec)
            return
        from pynput.keyboard import Key

        from .hotkey import normalize

        parts = [normalize(p) for p in spec.split("+")]
        *mods, key = parts
        mod_keys = [getattr(Key, m) for m in mods]
        final = getattr(Key, key) if len(key) > 1 else key
        for m in mod_keys:
            self.kb.press(m)
        try:
            self.kb.tap(final)
        finally:
            for m in reversed(mod_keys):
                self.kb.release(m)

    @staticmethod
    def _wayland_type(text: str) -> bool:
        for cmd in (["wtype", "--", text], ["ydotool", "type", "--", text]):
            if shutil.which(cmd[0]):
                try:
                    subprocess.run(cmd, check=True, timeout=10)
                    return True
                except Exception as exc:  # noqa: BLE001
                    log.warning("%s failed: %s", cmd[0], exc)
        return False
