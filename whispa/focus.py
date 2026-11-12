"""What does the user have focused? A text field we can paste into, or nothing?

``inspect()`` returns a :class:`Focus`. ``is_text`` is True or False when we can
tell, or None when the platform gives no answer (then we just paste).
``read_text()`` reads the field's contents so we can learn from the user's edits.
``restore(f)`` brings the window and field of an earlier ``Focus`` back (insert into the
field where dictation started), and only returns True once that's confirmed.

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
    title: str = ""  # the window's title (tells Gmail in a browser from other sites)
    is_password: bool = False
    is_terminal: bool = False
    _read: Callable[[], str | None] | None = field(default=None, repr=False)
    _selection: Callable[[], str | None] | None = field(default=None, repr=False)
    _same: Callable[["Focus"], bool] | None = field(default=None, repr=False)
    _handle: Any = field(default=None, repr=False)
    _window: Any = field(default=None, repr=False)  # Windows: top-level window handle
    _pid: int | None = field(default=None, repr=False)  # macOS: the app's process

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

    def read_selection(self) -> str | None:
        """The selected text, read through accessibility (no clipboard). None if unknown."""
        if self._selection is None or self.is_password:
            return None
        try:
            text = _call(self._selection)
        except Exception as exc:  # noqa: BLE001
            log.debug("read_selection failed: %s", exc)
            return None
        return text if text and len(text) <= MAX_READ else None

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


def _win_title() -> str:
    import ctypes

    u32 = ctypes.windll.user32
    hwnd = u32.GetForegroundWindow()
    buf = ctypes.create_unicode_buffer(512)
    return buf.value if hwnd and u32.GetWindowTextW(hwnd, buf, 512) else ""


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
    title = _win_title()
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

    def selection() -> str | None:
        text = _pattern(el, U.UIA_TextPatternId, U.IUIAutomationTextPattern)
        if text is None:
            return None
        ranges = text.GetSelection()
        if ranges is None or ranges.Length == 0:
            return None
        parts = [ranges.GetElement(i).GetText(MAX_READ + 1) for i in range(ranges.Length)]
        return "\n".join(p for p in parts if p) or None

    import ctypes

    return Focus(is_text=is_text, app=app, title=title, is_password=is_password,
                 is_terminal=(el.CurrentClassName or "") in _TERMINAL_CLASSES,
                 _read=read if is_text is not False else None, _selection=selection,
                 _same=same, _handle=el, _window=ctypes.windll.user32.GetForegroundWindow())


# --------------------------------------------------------------------------- macOS
_MAC_TEXT_ROLES = {"AXTextField", "AXTextArea", "AXComboBox", "AXSearchField"}
# Focus on these is definitely not somewhere to type (Finder icons, buttons, lists...).
_MAC_NOT_TEXT = {"AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton",
                 "AXImage", "AXList", "AXOutline", "AXRow", "AXCell", "AXTable", "AXBrowser",
                 "AXMenu", "AXMenuItem", "AXMenuBar", "AXMenuBarItem", "AXSlider", "AXTabGroup",
                 "AXToolbar", "AXDockItem", "AXDisclosureTriangle", "AXLink", "AXStaticText"}


def mac_classify(role: str, subrole: str, value_settable: bool, editable: bool,
                 has_caret: bool, app: str) -> bool | None:
    """True = text field, False = surely not, None = can't tell (then we paste, like Windows)."""
    if role in _MAC_TEXT_ROLES or value_settable or editable or has_caret:
        return True
    if role in _MAC_NOT_TEXT or (app == "Finder" and role in ("AXScrollArea", "AXGroup", "")):
        return False
    return None  # web areas, groups, custom views (Electron, Chrome, canvas apps): paste


def _mac_inspect() -> Focus:
    from ApplicationServices import (AXIsProcessTrusted, AXUIElementCopyAttributeValue,
                                     AXUIElementCreateApplication, AXUIElementCreateSystemWide,
                                     AXUIElementIsAttributeSettable)

    app, pid = "", None
    try:
        from AppKit import NSWorkspace

        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        app, pid = front.localizedName() or "", front.processIdentifier()
    except Exception:  # noqa: BLE001
        pass
    if not AXIsProcessTrusted():
        return Focus(app=app)  # no Accessibility permission: can't tell, just paste
    err, el = AXUIElementCopyAttributeValue(AXUIElementCreateSystemWide(), "AXFocusedUIElement", None)
    if (err or el is None) and pid is not None:
        # Electron and some other apps don't answer the system-wide query; ask the app itself.
        err, el = AXUIElementCopyAttributeValue(AXUIElementCreateApplication(pid),
                                                "AXFocusedUIElement", None)
    if err or el is None:
        log.debug("No focused element (%s, AX error %s): pasting", app, err)
        return Focus(app=app)

    def attr(name):
        e, v = AXUIElementCopyAttributeValue(el, name, None)
        return None if e else v

    role = str(attr("AXRole") or "")
    subrole = str(attr("AXSubrole") or "")
    e, settable = AXUIElementIsAttributeSettable(el, "AXValue", None)
    is_text = mac_classify(role, subrole, bool(not e and settable), bool(attr("AXEditable")),
                           attr("AXSelectedTextRange") is not None, app)
    log.debug("Focus: %s, role %s/%s -> text field %s", app, role, subrole, is_text)

    def read() -> str | None:
        v = attr("AXValue")
        return str(v) if v is not None else None

    def same(other: Focus) -> bool:
        return bool(el == other._handle)

    def selection() -> str | None:
        v = attr("AXSelectedText")
        return str(v) if v else None

    title = ""
    window = attr("AXWindow")
    if window is not None:
        e, t = AXUIElementCopyAttributeValue(window, "AXTitle", None)
        title = str(t) if not e and t else ""
    return Focus(is_text=is_text, app=app, title=title, is_password=subrole == "AXSecureTextField",
                 _read=read if is_text else None, _selection=selection, _same=same, _handle=el,
                 _pid=pid)


# --------------------------------------------------------------------------- restoring a field
def restore(target: Focus, settle_s: float = 0.25) -> bool:
    """Bring ``target``'s window to the front and focus its field again. True only when a
    fresh look confirms that exact field has the focus (never a guess)."""
    import time

    if target._handle is None:
        return False
    try:
        if sys.platform == "win32":
            _call(_win_restore, target)
        elif sys.platform == "darwin":
            _call(_mac_restore, target)
        else:
            return False
    except Exception as exc:  # noqa: BLE001 - the window or field is gone
        log.info("Couldn't bring the original field back: %s", exc)
        return False
    time.sleep(settle_s)
    return inspect().same_element(target)


def _win_restore(target: Focus) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = target._window
    if not hwnd or not user32.IsWindow(hwnd):
        raise RuntimeError("the window is closed")
    fg = user32.GetForegroundWindow()
    if fg != hwnd:
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        # Windows only lets the foreground app change the foreground window: share input
        # with it for a moment (no keys are pressed, so no menus open anywhere).
        msg = ctypes.create_string_buffer(64)
        user32.PeekMessageW(msg, 0, 0, 0, 0)  # give this thread an input queue
        me = ctypes.windll.kernel32.GetCurrentThreadId()
        other = user32.GetWindowThreadProcessId(fg, None) if fg else 0
        attached = bool(other and other != me and user32.AttachThreadInput(me, other, True))
        try:
            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
        finally:
            if attached:
                user32.AttachThreadInput(me, other, False)
    target._handle.SetFocus()


def _mac_restore(target: Focus) -> None:
    from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication
    from ApplicationServices import AXUIElementSetAttributeValue

    if target._pid is None:
        raise RuntimeError("unknown app")
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(target._pid)
    if app is None or app.isTerminated():
        raise RuntimeError("the app has quit")
    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
    err = AXUIElementSetAttributeValue(target._handle, "AXFocused", True)
    if err:
        raise RuntimeError(f"AX error {err}")


# --------------------------------------------------------------------------- screen text
_BROWSERS = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "arc", "safari",
             "google chrome", "microsoft edge", "zen", "chromium"}
_ctx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screen")


def screen_text(focus: Focus, budget_s: float = 0.4, max_chars: int = 30_000) -> str:
    """Text in the focused window, to spot names you might say. Bounded in time and size.

    Browsers only give their title: walking a web page's accessibility tree makes the
    browser build it in full, which costs it CPU and memory for the rest of the session.
    """
    parts = [focus.title]
    if (focus.app or "").lower() in _BROWSERS or focus.is_password:
        return focus.title
    try:
        if sys.platform == "win32":
            parts.append(_ctx_executor.submit(_win_screen_text, budget_s, max_chars).result(budget_s + 1))
        elif sys.platform == "darwin":
            parts.append(_ctx_executor.submit(_mac_screen_text, budget_s, max_chars).result(budget_s + 1))
    except Exception as exc:  # noqa: BLE001
        log.debug("screen text failed: %s", exc)
    return "\n".join(p for p in parts if p)


def _win_screen_text(budget_s: float, max_chars: int) -> str:
    import ctypes
    import time

    uia, U = _uia()
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    if not hwnd:
        return ""
    deadline = time.monotonic() + budget_s
    walker = uia.ContentViewWalker
    out: list[str] = []
    size = 0
    queue = [uia.ElementFromHandle(hwnd)]
    while queue and time.monotonic() < deadline and size < max_chars:
        el = queue.pop(0)
        try:
            name = el.CurrentName
            if name and len(name) < 2000:
                out.append(name)
                size += len(name)
            child = walker.GetFirstChildElement(el)
            while child is not None and len(queue) < 4000:
                queue.append(child)
                child = walker.GetNextSiblingElement(child)
        except Exception:  # noqa: BLE001 - elements vanish while we walk
            continue
    return "\n".join(out)[:max_chars]


def _mac_screen_text(budget_s: float, max_chars: int) -> str:
    import time

    from ApplicationServices import (AXIsProcessTrusted, AXUIElementCopyAttributeValue,
                                     AXUIElementCreateApplication)
    from AppKit import NSWorkspace

    if not AXIsProcessTrusted():
        return ""
    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    app = AXUIElementCreateApplication(front.processIdentifier())
    err, window = AXUIElementCopyAttributeValue(app, "AXFocusedWindow", None)
    if err or window is None:
        return ""
    deadline = time.monotonic() + budget_s
    out: list[str] = []
    size = 0
    queue = [window]
    while queue and time.monotonic() < deadline and size < max_chars:
        el = queue.pop(0)
        for name in ("AXTitle", "AXValue", "AXDescription"):
            e, v = AXUIElementCopyAttributeValue(el, name, None)
            if not e and isinstance(v, str) and v and len(v) < 5000:
                out.append(v)
                size += len(v)
        e, children = AXUIElementCopyAttributeValue(el, "AXChildren", None)
        if not e and children:
            queue.extend(list(children)[: max(0, 4000 - len(queue))])
    return "\n".join(out)[:max_chars]
