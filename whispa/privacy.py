"""Your voice stays on this computer: what's stored, where, and how to remove it all.

Murmr never uploads recordings, transcripts or voice models. Training runs on your own
computer. This module answers "what does Murmr keep about my voice?", deletes all of it in
one go, and notices work-managed computers (where recordings may hold confidential work).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as config_mod

log = logging.getLogger(__name__)

def voice_dirs() -> dict[str, Path]:
    d = config_mod.config_dir()
    return {"recordings": d / "training", "models": d / "models", "voices": d / "voices"}


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def storage() -> dict:
    """What's kept about your voice on this computer."""
    dirs = voice_dirs()
    rec = dirs["recordings"]
    clips = list(rec.glob("*.wav")) if rec.exists() else []
    models = [p for p in dirs["models"].iterdir() if p.is_dir()] if dirs["models"].exists() else []
    voices = [p for p in dirs["voices"].iterdir() if p.is_dir()] if dirs["voices"].exists() else []
    return {
        "clips": len(clips),
        "seconds": sum(p.stat().st_size for p in clips) / 32000,  # 16 kHz, 16-bit mono
        "models": len(models),
        "voices": len(voices),
        "bytes": sum(_size(p) for p in dirs.values()),
        "folder": str(config_mod.config_dir()),
    }


def excluded(app: str, patterns: list[str]) -> bool:
    """Is this app on the "never keep recordings" list? (case-insensitive, partial names)"""
    a = (app or "").lower().replace(" ", "")
    return bool(a) and any(p and p.lower().replace(" ", "") in a for p in patterns)


def delete_everything(cfg: config_mod.Config) -> bool:
    """Delete every recording, tuned speech model and voice. Returns True when a tuned
    speech model was active (the app should restart onto the base model)."""
    from .asr import BASE_MODEL

    leftovers = False
    for path in voice_dirs().values():
        shutil.rmtree(path, ignore_errors=True)
        leftovers |= path.exists()  # e.g. the tuned model is still loaded (Windows locks it)
    if leftovers:
        (config_mod.config_dir() / "delete-voice-data").write_text("pending", encoding="utf-8")
    tuned = cfg.model != BASE_MODEL
    if tuned:
        cfg.model = BASE_MODEL
    config_mod.save(cfg)
    log.info("Deleted all voice data%s", " and switched back to the base model" if tuned else "")
    return tuned


def finish_pending_delete() -> None:
    """At startup: finish a "delete everything" that couldn't remove files in use."""
    marker = config_mod.config_dir() / "delete-voice-data"
    if marker.exists():
        for path in voice_dirs().values():
            shutil.rmtree(path, ignore_errors=True)
        marker.unlink(missing_ok=True)
        log.info("Finished deleting voice data")


def managed_device() -> str:
    """A reason if this looks like a work-managed computer (else ""). Best effort, no admin."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(["dsregcmd", "/status"], capture_output=True, text=True,
                                 timeout=10, creationflags=0x08000000).stdout
            for key, reason in (("AzureAdJoined", "joined to a work Microsoft Entra ID"),
                                ("DomainJoined", "joined to a work domain"),
                                ("EnterpriseJoined", "enrolled with an organisation")):
                if any(line.split(":")[0].strip() == key and line.split(":")[-1].strip().upper() == "YES"
                       for line in out.splitlines() if ":" in line):
                    return reason
        elif sys.platform == "darwin":
            out = subprocess.run(["/usr/bin/profiles", "status", "-type", "enrollment"],
                                 capture_output=True, text=True, timeout=10).stdout
            if "MDM enrollment: Yes" in out or "Enrolled via DEP: Yes" in out:
                return "managed by your organisation (MDM)"
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("managed-device check failed: %s", exc)
    return ""
