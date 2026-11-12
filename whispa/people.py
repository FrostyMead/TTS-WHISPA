"""People whose voices Murmr remembers across meetings (opt-in: Meetings > Remember voices).

When you name someone in a meeting (live, or by renaming them afterwards while Murmr is still
open), their voiceprint is kept here, so the next meeting labels them by name right away.
Each meeting you name them in refines it.

* A voiceprint is 256 numbers describing how a voice sounds: no audio, nothing they said.
* Stored only in ``<config>/people.json`` (only your user can read it), never synced.
* It's someone else's personal (biometric) information: only remember people who are OK
  with it. Forget anyone, or everyone, on the Meetings page.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAX_WEIGHT = 20  # later meetings keep refining it, without the first ones dominating forever


class PeopleStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get("people", {}) if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save(self, people: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"_help": "Voices Murmr recognises in meetings (voiceprints, no "
                                   "audio). Forget people on the Meetings page.", "people": people}),
                       encoding="utf-8")
        if sys.platform != "win32":
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        os.replace(tmp, self.path)

    def known(self) -> dict[str, np.ndarray]:
        out = {}
        for name, p in self._load().items():
            v = np.asarray(p.get("v", []), np.float32)
            if v.size and np.linalg.norm(v) > 0:
                out[name] = v / np.linalg.norm(v)
        return out

    def names(self) -> list[str]:
        people = self._load()
        return sorted(people, key=lambda n: -people[n].get("t", 0))

    def learn(self, name: str, voiceprint: np.ndarray, weight: int = 1) -> None:
        name = " ".join(name.split())
        v = np.asarray(voiceprint, np.float32)
        if not name or not v.size or not np.linalg.norm(v):
            return
        v = v / np.linalg.norm(v)
        with self._lock:
            people = self._load()
            p = people.get(name)
            if p and len(p.get("v", [])) == len(v):
                n = min(int(p.get("n", 1)), MAX_WEIGHT)
                old = np.asarray(p["v"], np.float32)
                v = (old * n + v * weight) / (n + weight)
                v = v / np.linalg.norm(v)
                weight = n + weight
            people[name] = {"v": [round(float(x), 5) for x in v], "n": int(weight), "t": time.time()}
            self._save(people)
        log.info("Remembered a voice (%d people known)", len(people))

    def forget(self, name: str) -> None:
        with self._lock:
            people = self._load()
            if people.pop(name, None) is not None:
                self._save(people)

    def forget_all(self) -> None:
        with self._lock:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
