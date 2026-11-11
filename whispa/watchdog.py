"""Notice when Murmr's main thread freezes: record why, then restart instead of hanging.

The Qt main thread runs the menu-bar/tray icon, the window and the pill. If it gets stuck,
none of them respond and the app can't even be quit. A heartbeat timer on that thread
updates a timestamp; a background thread checks it:

* stuck for ``dump_after`` seconds: every thread's stack goes to crash.log (the cause)
* stuck for ``restart_after`` seconds: start a fresh Murmr and end this one

Computer sleep doesn't count: if the checker's own sleep took far longer than asked, the
machine was asleep, and the clock restarts.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


class Watchdog:
    def __init__(self, on_freeze: Callable[[float], None], on_restart: Callable[[float], None],
                 dump_after: float = 20.0, restart_after: float = 90.0, check_every: float = 2.0,
                 clock: Callable[[], float] = time.monotonic):
        self.on_freeze, self.on_restart = on_freeze, on_restart
        self.dump_after, self.restart_after, self.check_every = dump_after, restart_after, check_every
        self.clock = clock
        self.last_beat = clock()
        self._dumped = False
        self._stopped = False

    def beat(self) -> None:
        """Called on the main thread (from a timer)."""
        self.last_beat = self.clock()

    def check(self, slept: float | None = None) -> str:
        """One check. Returns "ok", "frozen" (stacks recorded) or "restart"."""
        now = self.clock()
        if slept is not None and slept > self.check_every * 5:
            self.last_beat = now  # the computer was asleep, not frozen
            return "ok"
        lag = now - self.last_beat
        if lag >= self.restart_after:
            self.on_restart(lag)
            return "restart"
        if lag >= self.dump_after:
            if not self._dumped:
                self._dumped = True
                self.on_freeze(lag)
            return "frozen"
        self._dumped = False
        return "ok"

    def start(self) -> None:
        def run():
            while not self._stopped:
                t0 = self.clock()
                time.sleep(self.check_every)
                if self.check(slept=self.clock() - t0) == "restart":
                    return

        threading.Thread(target=run, name="watchdog", daemon=True).start()

    def stop(self) -> None:
        self._stopped = True
