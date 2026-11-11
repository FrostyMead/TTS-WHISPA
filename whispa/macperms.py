"""macOS privacy permissions Whispa needs, and macOS's own prompts for them.

* Microphone: to hear you.
* Input Monitoring: to see the dictation shortcut (fn and friends) in every app.
* Accessibility: to paste with ⌘V and to read the focused text field.

Asking is what adds Whispa to each list in System Settings > Privacy & Security; until
then there's nothing to switch on. They're granted to Whispa.app (see mac/launcher.c).
"""

from __future__ import annotations

import logging
import subprocess
import sys

log = logging.getLogger(__name__)

KINDS = ("microphone", "input", "accessibility")
LABELS = {"microphone": "Microphone", "input": "Input Monitoring", "accessibility": "Accessibility"}
WHY = {"microphone": "To hear you while you hold the shortcut.",
       "input": "To notice the dictation shortcut in any app.",
       "accessibility": "To paste the text and read the field you're typing in."}
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


def status() -> dict[str, bool | None]:
    """True allowed, False not allowed (or not asked yet), None unknown."""
    if sys.platform != "darwin":
        return {}
    out: dict[str, bool | None] = {k: None for k in KINDS}
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
    return [k for k, ok in status().items() if ok is False]


def request(kinds: list[str] | None = None) -> None:
    """Show macOS's prompts for the permissions not granted yet (each one adds Whispa to
    its list in System Settings). macOS shows each prompt only once per app."""
    if sys.platform != "darwin":
        return
    for kind in kinds if kinds is not None else missing():
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
    """Open System Settings at this permission's list."""
    url = f"x-apple.systempreferences:com.apple.preference.security?{_PANES[kind]}"
    subprocess.Popen(["/usr/bin/open", url])
