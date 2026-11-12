"""Apple's on-device language model (macOS 26+, Apple Intelligence on), through the small
``murmr-applellm`` helper built from mac/applellm.swift.

The helper runs while Murmr does (so the model stays warm) and answers JSON lines: see the
Swift file for the protocol. Requests can overlap (a meeting summary while you dictate).
Nothing leaves the Mac: it's the on-device model, not Apple's cloud.

Used only when it passed Murmr's checks on this Mac (ai_setup.try_apple): fast enough, and
it tidies real dictations, swearing included, without refusing or rewriting them.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

log = logging.getLogger(__name__)

HELPER = Path(__file__).resolve().parent / "resources" / "murmr-applellm"
CONTEXT_CHARS = 7000  # input per request: the model takes ~4,096 tokens in all, answer included
REASONS = {
    "appleIntelligenceNotEnabled": "Apple Intelligence is off (System Settings > Apple Intelligence "
                                   "& Siri)",
    "deviceNotEligible": "this Mac doesn't support Apple Intelligence",
    "modelNotReady": "Apple's model is still downloading",
    "osTooOld": "it needs macOS 26 or later",
    "notBuilt": "this copy of Murmr was built without it",
    "unavailable": "it isn't available on this Mac",
}


class AppleLLMError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # guardrail | context | unavailable | timeout | other


def _mac_version() -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in platform.mac_ver()[0].split(".") if x)
    except ValueError:
        return ()


def os_version() -> str:
    return platform.mac_ver()[0] if sys.platform == "darwin" else ""


def supported() -> bool:
    """Could Apple's model be here at all (before asking whether it's switched on)?"""
    return (sys.platform == "darwin" and platform.machine() == "arm64" and HELPER.exists()
            and _mac_version() >= (26,))


def reason_text(reason: str) -> str:
    return REASONS.get(reason, reason or "it isn't available on this Mac")


class AppleLLM:
    def __init__(self, helper: Path | None = None, command: list[str] | None = None):
        self.helper = helper or HELPER
        self._command = command
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[threading.Event, list]] = {}
        self._ids = itertools.count(1)

    def _ensure(self) -> subprocess.Popen:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc
            cmd = self._command or [str(self.helper)]
            if self._command is None and not os.access(self.helper, os.X_OK):
                try:
                    os.chmod(self.helper, 0o755)  # an update may not keep the executable bit
                except OSError:
                    pass
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL, bufsize=0)
            threading.Thread(target=self._read, args=(self._proc,), name="apple-llm", daemon=True).start()
            return self._proc

    def _read(self, proc: subprocess.Popen) -> None:
        for raw in iter(proc.stdout.readline, b""):
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            waiter = self._pending.pop(msg.get("id"), None)
            if waiter is not None:
                waiter[1].append(msg)
                waiter[0].set()
        for done, box in list(self._pending.values()):  # the helper stopped: fail what's waiting
            box.append({"error": "other", "message": "Apple's model helper stopped"})
            done.set()
        self._pending.clear()

    def request(self, obj: dict, timeout: float) -> dict:
        proc = self._ensure()
        rid = next(self._ids)
        done, box = threading.Event(), []
        self._pending[rid] = (done, box)
        try:
            proc.stdin.write((json.dumps({**obj, "id": rid}) + "\n").encode())
            proc.stdin.flush()
        except OSError as exc:
            self._pending.pop(rid, None)
            raise AppleLLMError("other", f"Apple's model helper isn't running: {exc}") from exc
        if not done.wait(timeout):
            self._pending.pop(rid, None)
            raise AppleLLMError("timeout", f"Apple's model took longer than {timeout:.0f} s")
        return box[0]

    def availability(self, timeout: float = 10) -> tuple[bool, str]:
        try:
            r = self.request({"op": "availability"}, timeout)
        except (AppleLLMError, OSError) as exc:
            return False, str(exc)
        return bool(r.get("available")), str(r.get("reason", ""))

    def prewarm(self) -> None:
        try:
            self.request({"op": "prewarm"}, 5)
        except (AppleLLMError, OSError):
            pass

    def generate(self, instructions: str, prompt: str, temperature: float = 0.2,
                 max_tokens: int = 600, timeout: float = 30) -> str:
        r = self.request({"op": "generate", "instructions": instructions, "prompt": prompt,
                          "temperature": temperature, "max_tokens": max_tokens}, timeout)
        if "text" in r:
            return str(r["text"])
        raise AppleLLMError(str(r.get("error", "other")), str(r.get("message", "Apple's model failed")))

    def close(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()  # the helper exits when its stdin closes
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                proc.kill()


_shared: AppleLLM | None = None
_shared_lock = threading.Lock()


def shared() -> AppleLLM:
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = AppleLLM()
        return _shared


def close_shared() -> None:
    if _shared is not None:
        _shared.close()


def flatten(msgs: list[dict]) -> tuple[str, str]:
    """Chat messages (system, example pairs, the request) as Apple's instructions + prompt."""
    system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
    turns = [m for m in msgs if m["role"] != "system"]
    prompt = turns[-1]["content"] if turns else ""
    examples = []
    for user, reply in zip(turns[:-1:2], turns[1:-1:2]):
        if user["role"] == "user" and reply["role"] == "assistant":
            examples.append(f"Input: {user['content']}\nOutput: {reply['content']}")
    if examples:
        system += "\n\nExamples:\n\n" + "\n\n".join(examples)
    return system, prompt
