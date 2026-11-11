"""macOS privacy permissions Murmr needs, and macOS's own prompts for them.

* Microphone: to hear you.
* Input Monitoring: to see the dictation shortcut (fn and friends) in every app.
* Accessibility: to paste with ⌘V and to read the focused text field.

Asking is what adds Murmr to each list in System Settings > Privacy & Security; until
then there's nothing to switch on. They're granted to Murmr.app (see mac/launcher.c).
"""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger(__name__)

KINDS = ("microphone", "input", "accessibility", "fn_key")
LABELS = {"microphone": "Microphone", "input": "Input Monitoring", "accessibility": "Accessibility",
          "fn_key": "🌐 / fn key"}
WHY = {"microphone": "To hear you while you hold the shortcut.",
       "input": "To notice the dictation shortcut in any app.",
       "accessibility": "To paste the text and read the field you're typing in.",
       "fn_key": "Set “Press 🌐 key to” to Do Nothing, or macOS opens emoji, dictation or "
                 "switches language when you hold fn. (fn on most non-Apple keyboards never "
                 "reaches macOS: use the Mac's own keyboard or pick another shortcut.)"}
PERMISSIONS = KINDS[:3]  # the ones macOS asks about; fn_key is a keyboard setting
_FN_USAGE = {"0": "Do Nothing", "1": "Change Input Source", "2": "Show Emoji & Symbols",
             "3": "Start Dictation"}
_PANES = {"microphone": "Privacy_Microphone", "input": "Privacy_ListenEvent",
          "accessibility": "Privacy_Accessibility"}


def _mic_status() -> int | None:
    """AVAuthorizationStatus: 0 not asked yet, 1 restricted, 2 denied, 3 allowed."""
    try:
        from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

        return int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
    except Exception as exc:  # noqa: BLE001
        log.debug("mic status: %s", exc)
        return None


def fn_key_action() -> str:
    """What macOS does when you press the 🌐/fn key (System Settings > Keyboard)."""
    try:
        r = subprocess.run(["/usr/bin/defaults", "read", "com.apple.HIToolbox", "AppleFnUsageType"],
                           capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return ""
    if r.returncode != 0:
        return "Show Emoji & Symbols"  # never changed: the macOS default
    return _FN_USAGE.get(r.stdout.strip(), "")


def status() -> dict[str, bool | None]:
    """True allowed, False not allowed (or not asked yet), None unknown."""
    if sys.platform != "darwin":
        return {}
    out: dict[str, bool | None] = {k: None for k in KINDS}
    action = fn_key_action()
    out["fn_key"] = None if not action else action == "Do Nothing"
    mic = _mic_status()
    out["microphone"] = None if mic is None else mic == 3
    try:
        import Quartz

        out["input"] = bool(Quartz.CGPreflightListenEventAccess())
    except Exception as exc:  # noqa: BLE001
        log.debug("input monitoring status: %s", exc)
    try:
        from ApplicationServices import AXIsProcessTrusted

        out["accessibility"] = bool(AXIsProcessTrusted())
    except Exception as exc:  # noqa: BLE001
        log.debug("accessibility status: %s", exc)
    return out


def missing() -> list[str]:
    """Permissions not granted yet (not the fn key setting)."""
    return [k for k, ok in status().items() if ok is False and k in PERMISSIONS]


def request(kinds: list[str] | None = None) -> None:
    """Show macOS's prompts for the permissions not granted yet (each one adds Murmr to
    its list in System Settings). macOS shows each prompt only once per app."""
    if sys.platform != "darwin":
        return
    for kind in kinds if kinds is not None else missing():
        if kind not in PERMISSIONS:
            continue
        try:
            if kind == "accessibility":
                from ApplicationServices import (AXIsProcessTrustedWithOptions,
                                                 kAXTrustedCheckOptionPrompt)

                AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
            elif kind == "input":
                import Quartz

                Quartz.CGRequestListenEventAccess()
            elif kind == "microphone" and _mic_status() == 0:
                from AVFoundation import AVCaptureDevice, AVMediaTypeAudio

                AVCaptureDevice.requestAccessForMediaType_completionHandler_(
                    AVMediaTypeAudio, lambda granted: log.info("Microphone allowed: %s", granted))
        except Exception as exc:  # noqa: BLE001
            log.warning("Couldn't ask for %s: %s", kind, exc)


def open_settings(kind: str) -> None:
    """Open System Settings at this permission's list (or the keyboard settings)."""
    if kind == "fn_key":
        url = "x-apple.systempreferences:com.apple.Keyboard-Settings.extension"
    else:
        url = f"x-apple.systempreferences:com.apple.preference.security?{_PANES[kind]}"
    subprocess.Popen(["/usr/bin/open", url])
