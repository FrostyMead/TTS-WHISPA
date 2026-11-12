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


def _full_version(path: str) -> str:
    with open(os.path.join(path, "version.json"), encoding="utf-8-sig") as f:
        return json.load(f).get("sha", "")


def _write_state(name: str, data: dict) -> None:
    path = os.path.join(RUNTIME, name)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def check_probation() -> bool:
    """Consume crash evidence before another update. Errors must never block startup."""
    rolled_back = False
    try:
        path = os.path.join(RUNTIME, "running")
        with open(path, encoding="utf-8") as f:
            running = json.load(f)
        with open(os.path.join(RUNTIME, "probation"), encoding="utf-8") as f:
            probation = json.load(f)
        if time.time() - probation["time"] >= 86400 or probation.get("clean_exits", 0) >= 3:
            os.remove(os.path.join(RUNTIME, "probation"))
            return False
        crashed = os.path.getsize(running["crash_log"]) > running["size"]
        sha = _full_version(CURRENT)
        if (crashed and running["sha"] == probation["sha"] == sha
                and _full_version(PREVIOUS) == probation["previous_sha"]):
            _write_state("skipped", {"sha": sha})
            if rollback("the updated version crashed during probation"):
                rolled_back = True
                _write_state("rolled_back.json", {"from": sha, "to": _full_version(CURRENT),
                                                   "why": "crash during probation"})
                os.remove(os.path.join(RUNTIME, "probation"))
                shutil.rmtree(NEXT, ignore_errors=True)
                return True
    except FileNotFoundError:
        pass  # no update on probation, or the last run exited cleanly
    except Exception as exc:
        log(f"probation check: {exc}")
    return rolled_back


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
    try:
        with open(os.path.join(RUNTIME, "skipped"), encoding="utf-8") as f:
            if json.load(f).get("sha") == _full_version(NEXT):
                log("not applying a version quarantined after a crash")
                return
    except Exception:
        pass
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
    try:
        _write_state("probation", {"sha": _full_version(CURRENT), "time": time.time(),
                                    "previous_sha": _full_version(PREVIOUS), "clean_exits": 0})
    except Exception as exc:
        log(f"couldn't record probation: {exc}")
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
    try:
        _replace(PREVIOUS, CURRENT)
    except OSError:
        if os.path.isdir(bad) and not os.path.isdir(CURRENT):
            _replace(bad, CURRENT)
        raise
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


STARTING_GRACE_S = 120  # a first start after an update can be slow (new packages, cold caches)


def _alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    if sys.platform == "win32":
        import ctypes

        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
        k32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def trial_state() -> str:
    """For an update that hasn't reported a healthy start: "starting" if its process is
    still alive and young, "stuck" if alive for longer than the grace period (it's ended
    here), else "failed"."""
    try:
        with open(os.path.join(RUNTIME, "starting"), encoding="utf-8") as f:
            starting = json.load(f)
        pid = int(starting.get("pid", 0))
        if starting.get("sha") != _full_version(CURRENT) or not _alive(pid):
            return "failed"
        if time.time() - float(starting.get("time", 0)) < STARTING_GRACE_S:
            return "starting"
        import signal

        os.kill(pid, signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL)
        return "stuck"
    except (OSError, ValueError, TypeError):
        return "failed"


def main() -> None:
    # Never run from inside app/current: Windows can't rename a process's working folder,
    # which would stop the update from being applied (and the next one, from this process).
    os.chdir(RUNTIME)
    args = sys.argv[1:]
    probation_rollback = check_probation()
    if "--rollback" in args:
        args.remove("--rollback")
        rollback("requested")
    elif os.path.exists(TRIAL) and not os.path.isdir(NEXT):
        # The last update was started but never reported a healthy start: undo it, unless
        # it's still starting up in another process (you launched Murmr again meanwhile).
        state = trial_state()
        if state == "starting":
            log("the updated version is still starting; leaving it")
        else:
            try:
                rollback("the updated version did not start" if state != "stuck"
                         else "the updated version got stuck starting")
            except OSError as exc:
                log(f"rollback failed, starting the current version: {exc}")
    try:
        if not probation_rollback:
            apply_update()
    except OSError as exc:
        # e.g. a folder is open in Explorer. Start what we have; the update stays staged.
        log(f"couldn't apply the update yet, starting the current version: {exc}")
    try:
        prune_envs()
    except OSError:
        pass
    add_packages()
    if os.path.exists(TRIAL):  # an update on trial: say which process is starting it
        try:
            _write_state("starting", {"sha": _full_version(CURRENT), "pid": os.getpid(),
                                      "time": time.time()})
        except OSError:
            pass
    sys.path.insert(0, CURRENT)
    sys.argv = ["whispa", *args]
    from whispa.__main__ import main as whispa_main

    sys.exit(whispa_main())


if __name__ == "__main__":
    main()
