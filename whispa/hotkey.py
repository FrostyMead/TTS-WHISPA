"""Global push-to-talk hotkey.

``PTTStateMachine`` is pure logic (easy to test). ``HotkeyService`` feeds it
key events from pynput (Windows, Linux) or a Quartz event tap (macOS, see mackeys.py).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)

MODIFIERS = {"ctrl", "alt", "shift", "cmd", "fn"}

_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "option": "alt", "opt": "alt",
    "win": "cmd", "windows": "cmd", "super": "cmd", "meta": "cmd", "command": "cmd",
    "escape": "esc", "return": "enter", "globe": "fn",
    "left_ctrl": "ctrl_l", "right_ctrl": "ctrl_r", "lctrl": "ctrl_l", "rctrl": "ctrl_r",
    "left_alt": "alt_l", "right_alt": "alt_r", "lalt": "alt_l", "ralt": "alt_r",
    "altgr": "alt_gr", "right_option": "alt_r", "left_option": "alt_l",
    "left_shift": "shift_l", "right_shift": "shift_r",
    "left_cmd": "cmd_l", "right_cmd": "cmd_r", "right_win": "cmd_r", "left_win": "cmd_l",
    "right_super": "cmd_r", "left_super": "cmd_l",
}


def normalize(name: str) -> str:
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(name, name)


@dataclass(frozen=True)
class Hotkey:
    keys: frozenset[str]

    @classmethod
    def parse(cls, spec: str) -> "Hotkey":
        keys = frozenset(normalize(p) for p in spec.split("+") if p.strip())
        if not keys:
            raise ValueError(f"Empty hotkey: {spec!r}")
        if "fn" in keys and sys.platform != "darwin":
            raise ValueError(
                "The fn key is only visible to apps on macOS. On Windows/Linux the "
                "keyboard firmware swallows it; pick another hotkey (e.g. ctrl+cmd)."
            )
        return cls(keys)

    def satisfied(self, held: set[str]) -> bool:
        return self.keys <= held

    def label(self) -> str:
        names = {
            "darwin": {"cmd": "⌘", "ctrl": "⌃", "alt": "⌥", "shift": "⇧", "fn": "fn"},
            "win32": {"cmd": "Win", "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift"},
        }.get(sys.platform, {"cmd": "Super", "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift"})
        order = ["fn", "ctrl", "alt", "shift", "cmd"]
        ks = sorted(self.keys, key=lambda k: (order.index(k) if k in order else 99, k))
        parts = []
        for k in ks:
            base, side = (k[:-2], k[-1]) if k.endswith(("_l", "_r")) else (k, "")
            label = names.get(base, base.title())
            parts.append(f"{'Right' if side == 'r' else 'Left'} {label}" if side else label)
        compact = sys.platform == "darwin" and all(k in names for k in ks)
        return ("" if compact else "+").join(parts)


class PTTStateMachine:
    """Hold-to-talk with a hands-free lock, plus an optional Command Mode hotkey.

    * Hold the hotkey: record; release: transcribe.
    * While holding, tap the hands-free key (or double-tap the hotkey): lock.
      Press the hotkey again to finish.
    * Hold the command hotkey: record a spoken instruction (``mode == "command"``).
      If one shortcut contains the other (Ctrl+Win vs Ctrl+Alt+Win), the longer
      one wins, even if its last key arrives just after the shorter one fired.
    * Esc cancels. Pressing an unrelated key right after the hotkey cancels
      silently, so OS shortcuts that share the modifiers keep working.
    """

    IDLE, HOLDING, LOCKED = "idle", "holding", "locked"
    DICTATE, COMMAND = "dictate", "command"

    def __init__(self, hotkey: Hotkey, *, hands_free_key: str | None = "space",
                 double_tap: bool = True, command_hotkey: Hotkey | None = None,
                 on_start: Callable[[], None], on_stop: Callable[[], None],
                 on_cancel: Callable[[bool], None], on_lock: Callable[[], None],
                 clock: Callable[[], float] = time.monotonic,
                 double_tap_s: float = 0.35, shortcut_grace_s: float = 0.4):
        self.hotkey = hotkey
        self.command_hotkey = command_hotkey
        self.hands_free_key = normalize(hands_free_key) if hands_free_key else None
        self.double_tap = double_tap
        self.on_start, self.on_stop = on_start, on_stop
        self.on_cancel, self.on_lock = on_cancel, on_lock
        self.clock = clock
        self.double_tap_s = double_tap_s
        self.shortcut_grace_s = shortcut_grace_s
        self.state = self.IDLE
        self.mode = self.DICTATE
        self.held: set[str] = set()
        self._armed = True
        self._t_start = 0.0
        self._last_tap_end = -1e9
        self._lock = threading.RLock()

    def _hotkeys(self) -> list[tuple[str, Hotkey]]:
        pairs = [(self.DICTATE, self.hotkey)]
        if self.command_hotkey is not None and self.command_hotkey != self.hotkey:
            pairs.append((self.COMMAND, self.command_hotkey))
        return pairs

    def _active(self) -> Hotkey:
        return self.command_hotkey if self.mode == self.COMMAND and self.command_hotkey else self.hotkey

    def _any_satisfied(self) -> bool:
        return any(hk.satisfied(self.held) for _, hk in self._hotkeys())

    def _completed(self, new: set[str]) -> str | None:
        """The mode whose shortcut this key press just completed (longest wins)."""
        hits = [(len(hk.keys), mode) for mode, hk in self._hotkeys()
                if hk.satisfied(self.held) and new & hk.keys]
        return max(hits)[1] if hits else None

    # -- keyboard events -------------------------------------------------
    def press(self, names: set[str]) -> None:
        with self._lock:
            new = names - self.held
            self.held |= names
            if not new:
                return  # auto-repeat
            now = self.clock()
            if self.state == self.IDLE:
                mode = self._completed(new) if self._armed else None
                if mode is not None:
                    double = (mode == self.DICTATE and self.double_tap
                              and now - self._last_tap_end < self.double_tap_s)
                    self._begin(now, mode)
                    if double:
                        self._enter_lock()
            elif self.state == self.HOLDING:
                early = now - self._t_start < self.shortcut_grace_s
                if "esc" in new:
                    self._cancel(silent=False)
                elif (self.mode == self.DICTATE and early and self.command_hotkey is not None
                      and self._completed(new) == self.COMMAND):
                    # Ctrl+Win fired first, then Alt arrived: it was Ctrl+Alt+Win all along.
                    self._cancel(silent=True)
                    self._begin(now, self.COMMAND)
                elif self.mode == self.DICTATE and self.hands_free_key and self.hands_free_key in new:
                    self._enter_lock()
                elif not new & self._active().keys and early:
                    self._cancel(silent=True)  # it was an OS shortcut, not dictation
            elif self.state == self.LOCKED:
                if "esc" in new:
                    self._cancel(silent=False)
                elif self.hotkey.satisfied(self.held) and new & self.hotkey.keys:
                    self._finish()

    def release(self, names: set[str]) -> None:
        with self._lock:
            self.held -= names
            if self._active().satisfied(self.held):
                return
            now = self.clock()
            if self.state == self.HOLDING:
                quick = self.mode == self.DICTATE and now - self._t_start < self.double_tap_s
                self._finish()
                self._last_tap_end = now if quick else -1e9
            self._armed = True

    # -- UI events (overlay clicks, tray) --------------------------------
    def toggle_hands_free(self) -> None:
        with self._lock:
            if self.state == self.IDLE:
                self._begin(self.clock(), self.DICTATE)
                self._enter_lock()
            else:
                self._finish()

    def finish(self) -> None:
        with self._lock:
            if self.state != self.IDLE:
                self._finish()

    def cancel(self) -> None:
        with self._lock:
            if self.state != self.IDLE:
                self._cancel(silent=False)

    def reset(self) -> None:
        """Return to idle without firing callbacks (e.g. the mic failed to open)."""
        with self._lock:
            self.state = self.IDLE
            self._armed = not self._any_satisfied()

    def modifiers_held(self) -> bool:
        return bool(self.held & MODIFIERS)

    # -- internals -------------------------------------------------------
    def _begin(self, now: float, mode: str) -> None:
        self.state = self.HOLDING
        self.mode = mode
        self._t_start = now
        self._last_tap_end = -1e9
        self.on_start()

    def _enter_lock(self) -> None:
        self.state = self.LOCKED
        self.on_lock()

    def _finish(self) -> None:
        self.state = self.IDLE
        self._armed = not self._any_satisfied()
        self.on_stop()

    def _cancel(self, silent: bool) -> None:
        self.state = self.IDLE
        self._armed = not self._any_satisfied()
        self.on_cancel(silent)


_GENERIC_MOD = {"ctrl": "ctrl", "ctrl_l": "ctrl", "ctrl_r": "ctrl", "alt": "alt", "alt_l": "alt",
                "alt_r": "alt", "alt_gr": "alt", "shift": "shift", "shift_l": "shift",
                "shift_r": "shift", "cmd": "cmd", "cmd_l": "cmd", "cmd_r": "cmd", "fn": "fn"}
# pynput reports some left-hand keys without a side (Key.shift, Key.cmd on Windows;
# Key.ctrl/alt/shift/cmd on macOS); these are their sided names.
_SIDED = {"ctrl": "ctrl_l", "alt": "alt_l", "shift": "shift_l", "cmd": "cmd_l", "alt_gr": "alt_r"}
_PLAIN = {"space", "enter", "tab", "backspace", "esc", "delete"}


def combo_spec(keys: list[str], single: bool = False) -> str:
    """Turn the keys pressed in the recorder into a hotkey string, or raise ValueError."""
    if not keys:
        raise ValueError("No keys pressed")
    if single:  # hands-free key: exactly one key
        k = keys[-1]
        if k == "esc":
            raise ValueError("Esc is reserved for cancelling")
        return _GENERIC_MOD.get(k, k)
    if len(keys) == 1:
        k = keys[0]
        if k in _GENERIC_MOD:
            # A lone modifier must be side-specific (e.g. Right Alt), otherwise
            # every Ctrl+C would start dictation.
            return _SIDED.get(k, k)
        if len(k) == 1 or k in _PLAIN:
            raise ValueError("A single letter or key would trigger while typing. "
                             "Add a modifier like Ctrl or Alt, or use a function key.")
        return k
    mods: list[str] = []
    others: list[str] = []
    for k in keys:
        if k in _GENERIC_MOD:
            g = _GENERIC_MOD[k]
            if g not in mods:
                mods.append(g)
        elif k not in others:
            others.append(k)
    if not mods and any(len(k) == 1 or k in _PLAIN for k in others):
        raise ValueError("Letter keys need a modifier like Ctrl or Alt.")
    if mods == ["shift"] and others and all(len(k) == 1 for k in others):
        raise ValueError("Shift + letter types a capital letter. Use Ctrl, Alt or Win instead.")
    order = ["fn", "ctrl", "alt", "shift", "cmd"]
    mods.sort(key=order.index)
    return "+".join(mods + others)


class HotkeyService:
    """Runs the OS keyboard hooks and forwards events to the state machine.

    ``capture()`` temporarily diverts key events to a recorder (the settings
    window's "press your shortcut" button) instead of the state machine.
    """

    def __init__(self, sm: PTTStateMachine):
        self.sm = sm
        self._listener = None
        self._tap = None
        self._capture: dict | None = None
        self._lock = threading.Lock()
        self._actions: dict[str, tuple[Hotkey, Callable[[], None]]] = {}
        self.last_activity = time.monotonic()  # last key press anywhere (auto-update waits for a lull)

    def set_action(self, name: str, hotkey: Hotkey | None, callback: Callable[[], None]) -> None:
        """A shortcut that does something once when pressed (paste last, read aloud).
        It runs on the keyboard thread, so the callback should only hand off (emit a signal)."""
        with self._lock:
            if hotkey is None:
                self._actions.pop(name, None)
            else:
                self._actions[name] = (hotkey, callback)

    def _fire_action(self, new: set[str]) -> None:
        if not new or self.sm.state != self.sm.IDLE:
            return
        with self._lock:
            actions = list(self._actions.values())
        hits = [(len(hk.keys), cb) for hk, cb in actions if hk.satisfied(self.sm.held) and new & hk.keys]
        if hits:
            max(hits, key=lambda h: h[0])[1]()

    def start(self) -> None:
        if sys.platform == "darwin":
            from .mackeys import KeyTap

            self._tap = KeyTap(self._guarded(self._press), self._guarded(self._release))
            self._tap.start()
            return
        from pynput import keyboard

        self._kb = keyboard
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    @property
    def active(self) -> bool:
        return self._tap.active if self._tap is not None else self._listener is not None

    @property
    def keys_arriving(self) -> bool:
        """macOS: key events have reached Whispa (so Input Monitoring works)."""
        return self._tap is not None and self._tap.working

    @staticmethod
    def _guarded(fn):
        def run(names, primary):
            try:
                fn(names, primary)
            except Exception:  # noqa: BLE001
                log.exception("hotkey handler failed")
        return run

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        if self._tap:
            self._tap.stop()

    def set_hotkey(self, hotkey: Hotkey, hands_free_key: str | None,
                   command_hotkey: Hotkey | None = ...) -> None:  # type: ignore[assignment]
        """Swap shortcuts live. ``command_hotkey`` is kept unless passed (None disables it)."""
        with self.sm._lock:
            self.sm.hotkey = hotkey
            self.sm.hands_free_key = normalize(hands_free_key) if hands_free_key else None
            if command_hotkey is not ...:
                self.sm.command_hotkey = command_hotkey
            self.sm.reset()

    def capture(self, on_done: Callable[[str | None, str | None], None], single: bool = False) -> None:
        """Record the next key combination. ``on_done(spec, error)`` runs on the hook thread.

        Esc cancels: ``on_done(None, None)``.
        """
        with self._lock:
            self._capture = {"held": set(), "seen": [], "cb": on_done, "single": single}

    def cancel_capture(self) -> None:
        with self._lock:
            self._capture = None

    def wait_for_modifiers_released(self, timeout: float = 1.5) -> None:
        """Pasting while the user still holds Ctrl/Win would send the wrong shortcut."""
        deadline = time.monotonic() + timeout
        while self.sm.modifiers_held() and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.sm.modifiers_held():
            log.warning("Modifiers still held after %.1fs: %s", timeout, self.sm.held)

    # -- event routing ---------------------------------------------------
    def _press(self, names: set[str], primary: str) -> None:
        cancelled = None
        with self._lock:
            cap = self._capture
            if cap is not None:
                if primary == "esc" and not cap["seen"]:
                    self._capture = None
                    cancelled = cap
                else:
                    cap["held"].add(primary)
                    if primary not in cap["seen"]:
                        cap["seen"].append(primary)
                    return
        if cancelled is not None:
            cancelled["cb"](None, None)
            return
        self.last_activity = time.monotonic()
        new = names - self.sm.held
        self.sm.press(names)
        self._fire_action(new)

    def _release(self, names: set[str], primary: str) -> None:
        done = None
        with self._lock:
            cap = self._capture
            if cap is not None:
                cap["held"].discard(primary)
                if not cap["held"] and cap["seen"]:
                    self._capture = None
                    done = cap
        self.sm.release(names)  # keep the state machine's view of held keys accurate
        if done is not None:
            try:
                done["cb"](combo_spec(done["seen"], done["single"]), None)
            except ValueError as exc:
                done["cb"](None, str(exc))

    # -- pynput ----------------------------------------------------------
    def _names(self, key) -> tuple[set[str], str]:
        """(every name this key matches, its most specific name)."""
        kb = self._kb
        if isinstance(key, kb.Key):
            n = key.name
            names = {n}
            if n.endswith(("_l", "_r")):
                names.add(n[:-2])
            elif n in _SIDED:
                names.add(_SIDED[n])
            if n == "alt_gr":
                names |= {"alt", "alt_r"}
                return names, "alt_r"
            return names, _SIDED.get(n, n) if n in ("ctrl", "alt", "shift", "cmd") else n
        vk = getattr(key, "vk", None)
        try:
            key = self._listener.canonical(key)
        except Exception:
            pass
        ch = getattr(key, "char", None)
        if ch and ch.isprintable():
            name = ch.lower()
        elif vk is not None:
            if sys.platform == "win32" and (48 <= vk <= 57 or 65 <= vk <= 90):
                name = chr(vk).lower()
            else:
                name = f"vk{vk}"
        else:
            return set(), ""
        return {name}, name

    def _on_press(self, key) -> None:
        try:
            names, primary = self._names(key)
            if names:
                self._press(names, primary)
        except Exception:
            log.exception("hotkey press handler failed")

    def _on_release(self, key) -> None:
        try:
            names, primary = self._names(key)
            if names:
                self._release(names, primary)
        except Exception:
            log.exception("hotkey release handler failed")
