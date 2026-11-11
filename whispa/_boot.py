"""Murmr bootstrap for installed copies (copied to <runtime>/boot.py by the installer).

Standard library only: it runs before any of Murmr's code or packages are on the path.

<runtime>/
    boot.py         this file
    install.json    repo, branch, runtime flavour (cuda/cpu), uv path
    venv/           Python packages (from the installer)
    envs/<sha>/     packages of a later version whose pyproject.toml changed; each
                    version's version.json says which one it uses ("env"), so a rollback
                    also goes back to that version's packages
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
        with open(os.path.join(path, "version.json"), encoding="utf-8-sig") as f:
            return json.load(f).get("sha", "?")[:7]
    except (OSError, ValueError):
        return "?"


def _replace(src: str, dst: str) -> None:
    for attempt in range(20):  # the previous Murmr may still be shutting down
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
    try:
        _replace(NEXT, CURRENT)
    except OSError:
        _replace(PREVIOUS, CURRENT)  # put the working version back before giving up
        raise
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


def _env_of(app_dir: str) -> str:
    """The environment folder (relative to RUNTIME) a version uses."""
    try:
        with open(os.path.join(app_dir, "version.json"), encoding="utf-8-sig") as f:
            env = str(json.load(f).get("env") or "venv")
    except (OSError, ValueError):
        env = "venv"
    parts = env.replace("\\", "/").split("/")
    if env != "venv" and (len(parts) != 2 or parts[0] != "envs" or parts[1] in ("", ".", "..")):
        return "venv"
    return env


PRUNE_AFTER_S = 2 * 86400


def prune_envs(now: float | None = None) -> None:
    """Remove package environments no kept version uses. One changed in the last two days
    is left alone: it may belong to an update that's being downloaded right now."""
    keep = {_env_of(d) for d in (CURRENT, PREVIOUS, NEXT) if os.path.isdir(d)}
    root = os.path.join(RUNTIME, "envs")
    if not os.path.isdir(root):
        return
    now = now or time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if f"envs/{name}" in keep:
            continue
        try:
            if now - os.path.getmtime(path) < PRUNE_AFTER_S:
                continue
        except OSError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        log(f"removed unused packages envs/{name}")


def add_packages() -> None:
    venv = os.path.join(RUNTIME, _env_of(CURRENT))
    if not os.path.isdir(venv):
        # Missing (deleted by hand?): the installer's packages are the best bet. If they
        # don't fit, this version fails to start and the trial rolls it back.
        log(f"WARNING: packages {venv} missing, using venv")
        venv = os.path.join(RUNTIME, "venv")
    candidates = [os.path.join(venv, "Lib", "site-packages")]
    candidates += glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    for path in candidates:
        if os.path.isdir(path):
            site.addsitedir(path)  # also runs .pth files (e.g. NVIDIA DLL paths)


def main() -> None:
    # Never run from inside app/current: Windows can't rename a process's working folder,
    # which would stop the update from being applied (and the next one, from this process).
    os.chdir(RUNTIME)
    args = sys.argv[1:]
    if "--rollback" in args:
        args.remove("--rollback")
        rollback("requested")
    elif os.path.exists(TRIAL) and not os.path.isdir(NEXT):
        # The last update was started but never reported a healthy start: undo it.
        try:
            rollback("the updated version did not start")
        except OSError as exc:
            log(f"rollback failed, starting the current version: {exc}")
    try:
        apply_update()
    except OSError as exc:
        # e.g. a folder is open in Explorer. Start what we have; the update stays staged.
        log(f"couldn't apply the update yet, starting the current version: {exc}")
    try:
        prune_envs()
    except OSError:
        pass
    add_packages()
    sys.path.insert(0, CURRENT)
    sys.argv = ["whispa", *args]
    from whispa.__main__ import main as whispa_main

    sys.exit(whispa_main())


if __name__ == "__main__":
    main()
