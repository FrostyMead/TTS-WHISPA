"""macOS keyboard: watch keys with a Quartz event tap, post shortcuts and text with CGEvents.

Replaces pynput on macOS. pynput looks up the keyboard layout (TIS/TSM calls) on its
listener thread, which recent macOS versions only allow on the main thread: the app
is killed. Key codes here are layout independent; letters use the ANSI positions, as
macOS shortcuts do.

Watching needs Input Monitoring; posting needs Accessibility (see macperms.py).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

# Virtual key codes (Carbon kVK_*), named like pynput's keys.
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9,
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19,
    "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27, "8": 28,
    "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "enter": 36, "l": 37,
    "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46,
    ".": 47, "tab": 48, "space": 49, "`": 50, "backspace": 51, "esc": 53,
    "f17": 64, "f18": 79, "f19": 80, "f20": 90, "f5": 96, "f6": 97, "f7": 98, "f3": 99,
    "f8": 100, "f9": 101, "f11": 103, "f13": 105, "f16": 106, "f14": 107, "f10": 109,
    "f12": 111, "f15": 113, "home": 115, "page_up": 116, "delete": 117, "f4": 118,
    "end": 119, "f2": 120, "page_down": 121, "f1": 122, "left": 123, "right": 124,
    "down": 125, "up": 126,
}
_NAMES = {code: name for name, code in KEYCODES.items()}

# Modifier key code -> (sided name, generic name, device-dependent flag, generic flag)
_MODIFIERS = {
    59: ("ctrl_l", "ctrl", 0x00000001, 0x00040000),
    62: ("ctrl_r", "ctrl", 0x00002000, 0x00040000),
    56: ("shift_l", "shift", 0x00000002, 0x00020000),
    60: ("shift_r", "shift", 0x00000004, 0x00020000),
    55: ("cmd_l", "cmd", 0x00000008, 0x00100000),
    54: ("cmd_r", "cmd", 0x00000010, 0x00100000),
    58: ("alt_l", "alt", 0x00000020, 0x00080000),
    61: ("alt_r", "alt", 0x00000040, 0x00080000),
}
_DEVICE_BITS = 0x0000217F
FN_FLAG = 0x00800000
FLAGS = {"ctrl": 0x00040000, "shift": 0x00020000, "alt": 0x00080000, "cmd": 0x00100000,
         "fn": FN_FLAG}

KEY_DOWN, KEY_UP, FLAGS_CHANGED = 10, 11, 12  # CGEventType
_TAP_DISABLED = (0xFFFFFFFE, 0xFFFFFFFF)  # by timeout, by user input


def key_names(code: int) -> tuple[set[str], str]:
    """(every name this non-modifier key matches, its primary name)."""
    name = _NAMES.get(code, f"vk{code}")
    return {name}, name


def modifier_change(code: int, flags: int, fn_was_down: bool) -> tuple[str, set[str], str] | None:
    """For a flags-changed event: ("press"|"release", names, primary), or None."""
    if code in _MODIFIERS:
        sided, generic, device, generic_flag = _MODIFIERS[code]
        down = bool(flags & device) if flags & _DEVICE_BITS else bool(flags & generic_flag)
        return ("press" if down else "release"), {sided, generic}, sided
    fn_down = bool(flags & FN_FLAG)
    if fn_down != fn_was_down:  # fn / globe (key code 63 or 179)
        return ("press" if fn_down else "release"), {"fn"}, "fn"
    return None


class KeyTap:
    """Listen-only event tap on its own thread. Retries until Input Monitoring is allowed,
    so granting it takes effect without restarting Murmr."""

    def __init__(self, on_press: Callable[[set[str], str], None],
                 on_release: Callable[[set[str], str], None]):
        self.on_press, self.on_release = on_press, on_release
        self.active = False
        self._stopped = False
        self._loop = None
        self._fn_down = False
        self._mods_down: dict[str, tuple[set[str], int]] = {}  # sided name -> (names, generic flag)
        self._seen_fn = False
        self._seen_key = False

    def start(self) -> None:
        threading.Thread(target=self._run, name="key-tap", daemon=True).start()

    def stop(self) -> None:
        self._stopped = True
        if self._loop is not None:
            import Quartz

            Quartz.CFRunLoopStop(self._loop)

    def _run(self) -> None:
        import Quartz

        mask = (Quartz.CGEventMaskBit(KEY_DOWN) | Quartz.CGEventMaskBit(KEY_UP)
                | Quartz.CGEventMaskBit(FLAGS_CHANGED))
        state = {"tap": None}

        def callback(proxy, etype, event, refcon):
            try:
                if etype in _TAP_DISABLED:
                    Quartz.CGEventTapEnable(state["tap"], True)
                    return event
                self._handle(etype, Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode), Quartz.CGEventGetFlags(event))
            except Exception:  # noqa: BLE001
                log.exception("key tap callback failed")
            return event

        warned = announced = False
        while not self._stopped:
            allowed = _listen_allowed()
            tap = Quartz.CGEventTapCreate(Quartz.kCGSessionEventTap, Quartz.kCGHeadInsertEventTap,
                                          Quartz.kCGEventTapOptionListenOnly, mask, callback, None)
            if tap is None:
                if not warned:
                    log.error("Can't see the keyboard yet: allow Murmr in System Settings > "
                              "Privacy & Security > Input Monitoring. Waiting for it...")
                    warned = True
                self._waits = getattr(self, "_waits", 0) + 1
                time.sleep(min(15.0, 2.0 * 1.3 ** min(self._waits, 10)))
                continue
            state["tap"] = tap
            self.active = True
            if not announced:  # once; the tap may be re-made every few seconds until keys arrive
                log.info("Keyboard watching active (Input Monitoring allowed: %s)", allowed)
                announced = True
            source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            self._loop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(self._loop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            if not self._seen_key:
                # macOS can hand out a tap that gets no key events until Input Monitoring is
                # allowed, and a tap made before then stays deaf. Its "allowed?" check can
                # also keep its first answer until restart, so go by evidence: until a key
                # event arrives, make a fresh tap every few seconds.
                threading.Thread(target=self._refresh_until_events, args=(self._loop,),
                                 name="key-tap-watch", daemon=True).start()
            Quartz.CFRunLoopRun()
            self.active = False
            Quartz.CGEventTapEnable(tap, False)
            Quartz.CFMachPortInvalidate(tap)
            self._loop = None

    def _refresh_until_events(self, loop) -> None:
        import Quartz

        # Every 5 s at first, then less often (up to a minute): an idle Mac before its first
        # key press shouldn't keep rebuilding the tap.
        self._refreshes = getattr(self, "_refreshes", 0) + 1
        time.sleep(min(60.0, 5.0 * 1.5 ** min(self._refreshes - 1, 8)))
        if not self._stopped and not self._seen_key and self._loop is loop:
            log.debug("No keyboard events yet; making a new event tap")
            Quartz.CFRunLoopStop(loop)

    @property
    def working(self) -> bool:
        """Key events have arrived: Input Monitoring is definitely allowed."""
        return self._seen_key

    def _heal(self, flags: int) -> None:
        """A modifier we think is held but macOS says isn't (a missed release, e.g. while the
        tap was briefly disabled): release it, so shortcuts don't misbehave."""
        for sided, (names, generic) in list(self._mods_down.items()):
            if not flags & generic:
                del self._mods_down[sided]
                log.info("Released stuck key %s", sided)
                self.on_release(names, sided)
        if self._fn_down and not flags & FN_FLAG:
            self._fn_down = False
            self.on_release({"fn"}, "fn")

    def _handle(self, etype: int, code: int, flags: int) -> None:
        if not self._seen_key:
            self._seen_key = True
            log.info("Keyboard events are arriving")
        if etype in (KEY_DOWN, FLAGS_CHANGED) and code not in _MODIFIERS and code not in (63, 179):
            self._heal(flags)
        if etype == FLAGS_CHANGED:
            change = modifier_change(code, flags, self._fn_down)
            if change is None:
                return
            kind, names, primary = change
            if code in _MODIFIERS:
                if kind == "press":
                    self._mods_down[primary] = (names, _MODIFIERS[code][3])
                else:
                    self._mods_down.pop(primary, None)
                self._heal(flags)  # also catches the other side's missed release
            if primary == "fn":
                self._fn_down = kind == "press"
                if not self._seen_fn:
                    self._seen_fn = True
                    log.info("fn/globe key seen (key code %d)", code)
            (self.on_press if kind == "press" else self.on_release)(names, primary)
        elif etype == KEY_DOWN:
            names, primary = key_names(code)
            self.on_press(names, primary)
        elif etype == KEY_UP:
            names, primary = key_names(code)
            self.on_release(names, primary)


def _listen_allowed() -> bool | None:
    try:
        import Quartz

        return bool(Quartz.CGPreflightListenEventAccess())
    except Exception:  # noqa: BLE001 - older pyobjc/macOS
        return None


# --------------------------------------------------------------------------- output
def _post(code: int, flags: int, text: str | None = None) -> None:
    import Quartz

    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
    for down in (True, False):
        ev = Quartz.CGEventCreateKeyboardEvent(src, code, down)
        Quartz.CGEventSetFlags(ev, flags)
        if text:
            Quartz.CGEventKeyboardSetUnicodeString(ev, len(text.encode("utf-16-le")) // 2, text)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, ev)


def send_shortcut(spec: str) -> None:
    """Press a shortcut like "cmd+v" once."""
    from .hotkey import normalize

    *mods, key = [normalize(p) for p in spec.split("+")]
    flags = 0
    for m in mods:
        flags |= FLAGS[m.removesuffix("_l").removesuffix("_r")]
    if key not in KEYCODES:
        raise ValueError(f"Unknown key {key!r}")
    _post(KEYCODES[key], flags)


def type_text(text: str) -> None:
    """Type text as-is. Line breaks use Shift+Enter so chat apps don't send the message."""
    for i, line in enumerate(text.split("\n")):
        if i:
            _post(KEYCODES["enter"], FLAGS["shift"])
        for j in range(0, len(line), 16):  # apps accept a few characters per event
            _post(0, 0, line[j:j + 16])
            time.sleep(0.004)
