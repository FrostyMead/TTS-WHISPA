"""What does the user have focused? A text field we can paste into, or nothing?

``inspect()`` returns a :class:`Focus`. ``is_text`` is True or False when we can
tell, or None when the platform gives no answer (then we just paste).
``read_text()`` reads the field's contents so we can learn from the user's edits.

* Windows: UI Automation (works for Win32, WPF, UWP, Chrome/Edge/Electron, Office)
* macOS: Accessibility API (AXUIElement)
* Linux: not implemented yet -> is_text None
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

MAX_READ = 50_000

# Every accessibility call runs on this one thread: COM objects (Windows) and AX
# elements (macOS) are then never shared across threads.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="a11y")


def _call(fn, *args, timeout: float = 5.0):
    if threading.current_thread().name.startswith("a11y"):
        return fn(*args)
    return _executor.submit(fn, *args).result(timeout=timeout)


@dataclass
class Focus:
    is_text: bool | None = None
    app: str = ""
    is_password: bool = False
    _read: Callable[[], str | None] | None = field(default=None, repr=False)
    _same: Callable[["Focus"], bool] | None = field(default=None, repr=False)
    _handle: Any = field(default=None, repr=False)

    @property
    def readable(self) -> bool:
        return self._read is not None and not self.is_password

    def read_text(self) -> str | None:
        if not self.readable:
            return None
        try:
            text = _call(self._read)
        except Exception as exc:  # noqa: BLE001 - element went away
            log.debug("read_text failed: %s", exc)
            return None
        return text if text is None or len(text) <= MAX_READ else None

    def same_element(self, other: "Focus") -> bool:
        try:
            return bool(self._same and other._handle is not None and _call(self._same, other))
        except Exception:  # noqa: BLE001
            return False


def inspect() -> Focus:
    try:
        return _call(_inspect)
    except Exception as exc:  # noqa: BLE001 - includes timeouts
        log.debug("focus inspection failed: %s", exc)
        return Focus()


def _inspect() -> Focus:
    try:
        if sys.platform == "win32":
            return _win_inspect()
        if sys.platform == "darwin":
            return _mac_inspect()
    except Exception as exc:  # noqa: BLE001 - accessibility APIs fail in many ways
        log.debug("focus inspection failed: %s", exc)
    return Focus()


# --------------------------------------------------------------------------- Windows
_local = threading.local()

_EDIT, _DOCUMENT, _COMBO = 50004, 50030, 50003
# Controls that are definitely not somewhere to type: buttons, links, images,
# list/tree items (desktop icons, file lists), menus, tabs, labels, tables...
_NOT_TEXT = {50000, 50002, 50005, 50006, 50007, 50008, 50009, 50010, 50011, 50013, 50014,
             50015, 50018, 50019, 50020, 50021, 50023, 50024, 50028, 50029, 50031, 50034,
             50035, 50036, 50037}
_TERMINAL_CLASSES = {"TermControl", "ConsoleWindowClass", "PseudoConsoleWindow",
                     "CASCADIA_HOSTING_WINDOW_CLASS", "mintty", "VirtualConsoleClass"}
_DESKTOP_CLASSES = {"Progman", "WorkerW", "SHELLDLL_DefView", "SysListView32",
                    "Shell_TrayWnd", "Shell_SecondaryTrayWnd", "TaskListThumbnailWnd"}


def _uia():
    """One IUIAutomation per thread (COM objects are apartment bound)."""
    if getattr(_local, "uia", None) is None:
        sys.coinit_flags = 0  # COINIT_MULTITHREADED, read by comtypes at import
        import comtypes
        import comtypes.client

        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            pass  # thread already initialised (e.g. STA main thread)
        comtypes.client.GetModule("UIAutomationCore.dll")
        from comtypes.gen import UIAutomationClient as U

        _local.U = U
        _local.uia = comtypes.client.CreateObject(U.CUIAutomation, interface=U.IUIAutomation)
    return _local.uia, _local.U


def _pattern(el, pid, iface):
    try:
        p = el.GetCurrentPattern(pid)
        return p.QueryInterface(iface) if p else None
    except Exception:  # noqa: BLE001 - NULL pointer when unsupported
        return None


def _process_name(pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.splitext(os.path.basename(buf.value))[0]
    finally:
        k32.CloseHandle(h)
    return ""


def _win_caret_visible() -> bool:
    """Classic Win32 apps own a system caret while a text field has focus."""
    import ctypes
    from ctypes import wintypes

    class GUITHREADINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD),
                    ("hwndActive", wintypes.HWND), ("hwndFocus", wintypes.HWND),
                    ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                    ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND),
                    ("rcCaret", wintypes.RECT)]

    info = GUITHREADINFO(cbSize=ctypes.sizeof(GUITHREADINFO))
    if not ctypes.windll.user32.GetGUIThreadInfo(0, ctypes.byref(info)):
        return False
    return bool(info.hwndCaret)


def _win_classify(el, U) -> bool | None:
    """True = text field, False = surely not, None = can't tell (caller pastes anyway)."""
    ctype = el.CurrentControlType
    cls = el.CurrentClassName or ""
    if cls in _TERMINAL_CLASSES:
        return True
    if cls in _DESKTOP_CLASSES:
        return False
    value = _pattern(el, U.UIA_ValuePatternId, U.IUIAutomationValuePattern)
    if value is not None and ctype in (_EDIT, _DOCUMENT, _COMBO):
        return not value.CurrentIsReadOnly
    if ctype == _EDIT:
        return True
    if _win_caret_visible():
        return True
    if ctype == _DOCUMENT:
        # Rich editors (Word, new Notepad) expose TextEditPattern without ValuePattern.
        if _pattern(el, U.UIA_TextEditPatternId, U.IUIAutomationTextEditPattern) is not None:
            return True
        return None
    if ctype in _NOT_TEXT:
        return False
    # Panes, groups, custom-drawn surfaces (games, canvas apps, some Electron
    # states): unknown. Pasting is the safer default there.
    return None


def _win_inspect() -> Focus:
    uia, U = _uia()
    el = uia.GetFocusedElement()
    if el is None:
        return Focus(is_text=False)
    app = _process_name(el.CurrentProcessId)
    is_text = _win_classify(el, U)
    is_password = bool(el.CurrentIsPassword)

    def read() -> str | None:
        value = _pattern(el, U.UIA_ValuePatternId, U.IUIAutomationValuePattern)
        if value is not None:
            return value.CurrentValue
        text = _pattern(el, U.UIA_TextPatternId, U.IUIAutomationTextPattern)
        if text is not None:
            return text.DocumentRange.GetText(MAX_READ + 1)
        return None

    def same(other: Focus) -> bool:
        u, _ = _uia()
        return bool(u.CompareElements(el, other._handle))

    return Focus(is_text=is_text, app=app, is_password=is_password,
                 _read=read if is_text is not False else None, _same=same, _handle=el)


# --------------------------------------------------------------------------- macOS
_MAC_TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}


def _mac_inspect() -> Focus:
    from ApplicationServices import (AXIsProcessTrusted, AXUIElementCopyAttributeValue,
                                     AXUIElementCreateSystemWide, AXUIElementIsAttributeSettable)

    if not AXIsProcessTrusted():
        return Focus()  # no Accessibility permission: can't tell, just paste
    system = AXUIElementCreateSystemWide()
    err, el = AXUIElementCopyAttributeValue(system, "AXFocusedUIElement", None)
    if err or el is None:
        return Focus(is_text=False)

    def attr(name):
        e, v = AXUIElementCopyAttributeValue(el, name, None)
        return None if e else v

    role = attr("AXRole") or ""
    subrole = attr("AXSubrole") or ""
    err, settable = AXUIElementIsAttributeSettable(el, "AXValue", None)
    is_text = role in _MAC_TEXT_ROLES or bool(not err and settable) or bool(attr("AXEditable"))
    app = ""
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication().localizedName() or ""
    except Exception:  # noqa: BLE001
        pass

    def read() -> str | None:
        v = attr("AXValue")
        return str(v) if v is not None else None

    def same(other: Focus) -> bool:
        return bool(el == other._handle)

    return Focus(is_text=is_text, app=app, is_password=subrole == "AXSecureTextField",
                 _read=read if is_text else None, _same=same, _handle=el)
