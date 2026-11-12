"""Tune to my voice from inside Murmr (Windows): run the training pipelines in WSL2.

Two jobs, both in Linux on this PC (WSL2), with the NVIDIA GPU:

* ``recognition``: scripts/finetune/pipeline.sh fine-tunes the speech model on your
  recordings, compares it with the standard one, and installs it if it's better.
  About 30-60 minutes.
* ``voice``: scripts/voice_finetune/pipeline.sh trains a read-aloud voice like yours, one
  round of at most a few hours (it resumes next time). The first run also sets up its
  tools (about 4 GB).

The pipelines print "== " lines at each step; the job turns those into the status line.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
CREATE_NO_WINDOW = 0x08000000
RECOGNITION_MIN = 10.0  # minutes of reading/corrected dictation before tuning is offered
RECOGNITION_GOOD = 30.0
VOICE_MIN = 10.0
VOICE_GOOD = 30.0


def to_wsl(path: Path) -> str:
    """C:/Users/me/x -> /mnt/c/Users/me/x (WSL's standard mount of Windows drives)."""
    p = Path(path).resolve()
    return f"/mnt/{p.drive[0].lower()}{p.as_posix()[2:]}"


def _wsl(args: list[str], timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl.exe", *args], capture_output=True, timeout=timeout,
                          creationflags=CREATE_NO_WINDOW)


def wsl_status() -> str:
    """"" when WSL2 with a Linux distribution is ready, else why not (in plain words)."""
    if sys.platform != "win32":
        return "Tuning runs on a Windows PC with an NVIDIA graphics card."
    try:
        r = _wsl(["-l", "-q"])
    except (OSError, subprocess.SubprocessError):
        return "Needs WSL (Windows Subsystem for Linux). Install it with: wsl --install"
    names = r.stdout.decode("utf-16-le", "ignore").replace("\x00", "").split()
    if r.returncode != 0 or not names:
        return "Needs WSL with Ubuntu. Install it with: wsl --install"
    return ""


def recognition_env_ready() -> bool:
    try:
        return _wsl(["--", "bash", "-lc", "test -x ~/whispa-finetune/.venv/bin/python"]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@dataclass
class Job:
    kind: str  # "recognition" | "voice"
    on_change: Callable[["Job"], None] = lambda job: None
    hours: float = 4.0
    status: str = "Starting…"
    step: str = ""
    lines: list[str] = field(default_factory=list)
    running: bool = False
    ok: bool | None = None  # None while running
    started: float = field(default_factory=time.time)
    result: str = ""  # e.g. the installed model's name
    include_unedited: bool = False  # recognition: dictations you left as they were count too

    def __post_init__(self):
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    def command(self) -> list[str]:
        """wsl.exe running a small script file (no quoting through the Windows command line)."""
        from . import config as config_mod

        if self.kind == "recognition":
            folder, env = SCRIPTS / "finetune", "source ~/whispa-finetune/.venv/bin/activate"
            if self.include_unedited:
                env += "\nexport MURMR_INCLUDE_UNEDITED=1"
        else:
            folder, env = SCRIPTS / "voice_finetune", f"export HOURS={self.hours:g}"
        script = "\n".join(["#!/usr/bin/env bash", "set -e", "export PYTHONUNBUFFERED=1", env,
                            f"cd '{to_wsl(folder)}'", "bash ./pipeline.sh 2>&1", ""])
        job = config_mod.config_dir() / "tuning" / f"{self.kind}.sh"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_bytes(script.encode())  # LF line endings: bash needs them
        return ["wsl.exe", "--", "bash", to_wsl(job)]

    def start(self) -> None:
        self.running = True
        threading.Thread(target=self._run, name=f"tune-{self.kind}", daemon=True).start()

    def _run(self) -> None:
        try:
            self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          creationflags=CREATE_NO_WINDOW)
            buf = b""
            while True:
                chunk = self._proc.stdout.read1(4096) if hasattr(self._proc.stdout, "read1") else \
                    self._proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                # progress bars rewrite their line with \r: treat it as a line end too
                parts = re.split(rb"[\r\n]", buf)
                buf = parts.pop()
                for raw in parts:
                    self._line(raw.decode("utf-8", "replace").strip())
            code = self._proc.wait()
            self.ok = code == 0 and not self._cancelled
            if self._cancelled:
                self.status = "Cancelled"
            elif not self.ok:
                self.status = "Stopped with an error: " + (self.lines[-1][:120] if self.lines else f"exit {code}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Tuning job failed")
            self.ok, self.status = False, f"Couldn't run: {exc}"
        finally:
            self.running = False
            log.info("Tuning (%s) finished: %s", self.kind, self.status)
            self.on_change(self)

    def _line(self, line: str) -> None:
        if not line:
            return
        self.lines = (self.lines + [line])[-200:]
        if line.startswith("== done"):
            self.result = line[7:].strip()
            self.status = "Finished"
        elif line.startswith("== "):
            self.step = line[3:]
            self.status = self.step
            log.info("Tuning (%s): %s", self.kind, self.step)
        elif "recommend switching:" in line:
            self.status = line.strip()
        elif re.search(r"\d+%\|", line):  # a tqdm progress bar
            m = re.search(r"(\d+)%\|", line)
            self.status = f"{self.step} · {m.group(1)}%" if m else self.status
        else:
            return
        self.on_change(self)

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.poll() is None:
            pattern = "pipeline.sh" if self.kind == "recognition" else "train_voice.py|pipeline.sh"
            try:
                _wsl(["--", "bash", "-lc", f"pkill -f '{pattern}'; pkill -f 'piper.train|train.py'"])
            except (OSError, subprocess.SubprocessError):
                pass
            self._proc.terminate()

    @property
    def elapsed(self) -> float:
        return time.time() - self.started


def recorded_minutes(training_dir: Path, include_unedited: bool = False) -> dict[str, float]:
    """Minutes usable for each job: recognition (reading + dictations you corrected, plus
    the ones you left as they were with ``include_unedited``) and voice (reading with a
    high-quality copy)."""
    import json

    rows: dict[str, dict] = {}
    manifest = training_dir / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.setdefault(r.get("id", ""), {}).update(r)
    rec = voice = 0.0
    for sid, r in rows.items():
        if not (training_dir / f"{sid}.wav").exists():
            continue
        secs = float(r.get("duration", 0) or 0)
        if r.get("status") in ("read", "corrected") or \
                (include_unedited and r.get("status") == "unedited"):
            rec += secs
        if r.get("status") == "read" and r.get("hq") and (training_dir / r["hq"]).exists():
            voice += secs
    return {"recognition": rec / 60, "voice": voice / 60}
