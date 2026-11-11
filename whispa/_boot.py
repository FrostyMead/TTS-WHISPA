"""Whispa bootstrap for installed copies (copied to <runtime>/boot.py by the installer).

Standard library only: it runs before any of Whispa's code or packages are on the path.

<runtime>/
    boot.py         this file
    install.json    repo, branch, runtime flavour (cuda/cpu), uv path
    venv/           Python packages
    app/current/    the running version (a GitHub snapshot, with version.json)
    app/next/       an update downloaded by the app, applied on the next start
    app/previous/   the version before, for rollback
    trial           present until a freshly applied update has started successfully

Start:   pythonw -I -S boot.py [whispa args...]
Undo:    pythonw -I -S boot.py --rollback
"""

import glob
import json
import os
import shutil
import site
import sys
import time

RUNTIME = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(RUNTIME, "app")
CURRENT = os.path.join(APP, "current")
NEXT = os.path.join(APP, "next")
PREVIOUS = os.path.join(APP, "previous")
TRIAL = os.path.join(RUNTIME, "trial")
LOG = os.path.join(RUNTIME, "boot.log")


def log(msg: str) -> None:
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except OSError:
        pass


def _version(path: str) -> str:
    try:
        with open(os.path.join(path, "version.json"), encoding="utf-8") as f:
            return json.load(f).get("sha", "?")[:7]
    except (OSError, ValueError):
        return "?"


def _replace(src: str, dst: str) -> None:
    for attempt in range(20):  # the previous Whispa may still be shutting down
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(src, dst)


def apply_update() -> None:
    if not os.path.isfile(os.path.join(NEXT, "version.json")):
        return
    shutil.rmtree(PREVIOUS, ignore_errors=True)
    if os.path.isdir(CURRENT):
        _replace(CURRENT, PREVIOUS)
    _replace(NEXT, CURRENT)
    with open(TRIAL, "w", encoding="utf-8") as f:
        f.write(_version(CURRENT))
    log(f"updated {_version(PREVIOUS)} -> {_version(CURRENT)}")
    refresh_self()


def rollback(reason: str) -> bool:
    if not os.path.isfile(os.path.join(PREVIOUS, "version.json")):
        log(f"rollback requested ({reason}) but there is no previous version")
        return False
    bad = os.path.join(APP, "failed")
    shutil.rmtree(bad, ignore_errors=True)
    if os.path.isdir(CURRENT):
        _replace(CURRENT, bad)
    _replace(PREVIOUS, CURRENT)
    shutil.rmtree(bad, ignore_errors=True)
    try:
        os.remove(TRIAL)
    except OSError:
        pass
    log(f"rolled back to {_version(CURRENT)} ({reason})")
    refresh_self()
    return True


def refresh_self() -> None:
    """Keep boot.py in step with the version it starts."""
    src = os.path.join(CURRENT, "whispa", "_boot.py")
    if os.path.isfile(src):
        tmp = os.path.abspath(__file__) + ".new"
        shutil.copyfile(src, tmp)
        os.replace(tmp, os.path.abspath(__file__))


def add_packages() -> None:
    venv = os.path.join(RUNTIME, "venv")
    candidates = [os.path.join(venv, "Lib", "site-packages")]
    candidates += glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    for path in candidates:
        if os.path.isdir(path):
            site.addsitedir(path)  # also runs .pth files (e.g. NVIDIA DLL paths)


def main() -> None:
    args = sys.argv[1:]
    if "--rollback" in args:
        args.remove("--rollback")
        rollback("requested")
    elif os.path.exists(TRIAL) and not os.path.isdir(NEXT):
        # The last update was started but never reported a healthy start: undo it.
        rollback("the updated version did not start")
    apply_update()
    add_packages()
    sys.path.insert(0, CURRENT)
    sys.argv = ["whispa", *args]
    from whispa.__main__ import main as whispa_main

    sys.exit(whispa_main())


if __name__ == "__main__":
    main()
