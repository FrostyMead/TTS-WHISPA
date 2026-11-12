"""One-click setup of the local AI (Ollama + a model), for Command Mode, AI clean-up and
meeting summaries.

* macOS: Ollama's command-line build (~180 MB) goes into Murmr's own folder; Murmr runs it.
  No installer, no admin password, nothing in /Applications.
* Windows: Ollama's own installer, silently, for this user only.
* Then the best model for this computer is downloaded through Ollama (ai_choice.py: by
  chip, memory and graphics card), timed on a few real clean-ups, and swapped for a smaller
  one if it's too slow here.

Downloads are checked against the SHA-256 list published with each Ollama release.
"""

from __future__ import annotations

import hashlib
import json
import re
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

RELEASES = "https://api.github.com/repos/ollama/ollama/releases/latest"
UA = {"User-Agent": "Murmr-Setup"}


class Cancelled(Exception):
    pass


def memory_gb() -> float:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 2**30
        if sys.platform == "darwin":
            out = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) / 2**30
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 2**20
    except Exception:  # noqa: BLE001
        return 8.0


def recommended_model() -> str:
    from .ai_choice import detect, recommend

    return recommend(detect()).model


def model_size_label(model: str) -> str:
    from .ai_choice import size_label

    return size_label(model)


def local_dir() -> Path:
    from . import config as config_mod

    return config_mod.config_dir() / "ai"


def supported() -> bool:
    return sys.platform in ("win32", "darwin")


def plan() -> str:
    """A sentence about what setting up will download."""
    from .llm import find_ollama

    from .ai_choice import APPLE, detect, recommend

    choice = recommend(detect())
    model = choice.models[1] if choice.model == APPLE else choice.model
    app = "" if find_ollama() else ("Ollama (~180 MB) and " if sys.platform == "darwin"
                                    else "Ollama (~1.7 GB) and ")
    text = f"Downloads {app}the {model} model ({model_size_label(model)}). Runs on this computer only."
    if choice.model == APPLE:
        text = ("Uses Apple's on-device model if it passes Murmr's checks (no download). "
                "Otherwise: " + text[0].lower() + text[1:])
    return text


# --------------------------------------------------------------------------- downloading
def _get(url: str, timeout: float = 30):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


def _download(url: str, dest: Path, progress: Callable[[str, float | None], None], label: str,
              cancel: threading.Event) -> str:
    h = hashlib.sha256()
    with _get(url, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel.is_set():
                raise Cancelled()
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            progress(f"Downloading {label}… {done / 2**20:.0f} of {total / 2**20:.0f} MB" if total
                     else f"Downloading {label}… {done / 2**20:.0f} MB", done / total if total else None)
    return h.hexdigest()


def _release_asset(name: str) -> tuple[str, str]:
    """(download url, expected sha256) of a file in Ollama's latest release."""
    with _get(RELEASES) as r:
        rel = json.load(r)
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    if name not in assets:
        raise RuntimeError(f"Ollama's latest release has no {name}")
    sha = ""
    digest = next((a.get("digest") or "" for a in rel["assets"] if a["name"] == name), "")
    if digest.startswith("sha256:"):  # GitHub's own checksum of the uploaded file
        sha = digest.split(":", 1)[1].lower()
    elif "sha256sum.txt" in assets:
        with _get(assets["sha256sum.txt"]) as r:
            for line in r.read().decode().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[1].lstrip("*./") == name:
                    sha = parts[0].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise RuntimeError("Ollama's download has no checksum to check it against; not installing it.")
    return assets[name], sha


def install_ollama(progress, cancel: threading.Event) -> str:
    """Install Ollama for this user; returns the path of the ollama program."""
    from .llm import find_ollama

    tmp = Path(tempfile.mkdtemp(prefix="murmr-ai-"))
    try:
        if sys.platform == "darwin":
            url, sha = _release_asset("ollama-darwin.tgz")
            got = _download(url, tmp / "ollama.tgz", progress, "Ollama", cancel)
            if got != sha:
                raise RuntimeError("The Ollama download didn't match its checksum; not installing it.")
            progress("Unpacking Ollama…", None)
            dest = local_dir()
            shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            with tarfile.open(tmp / "ollama.tgz") as t:
                t.extractall(dest, filter="data")
            exe = next((p for p in dest.rglob("ollama") if p.is_file()), None)
            if exe is None:
                raise RuntimeError("Ollama's download didn't contain the ollama program")
            exe.chmod(0o755)
            return str(exe)
        if sys.platform == "win32":
            url, sha = _release_asset("OllamaSetup.exe")
            got = _download(url, tmp / "OllamaSetup.exe", progress, "Ollama", cancel)
            if got != sha:
                raise RuntimeError("The Ollama download didn't match its checksum; not installing it.")
            progress("Installing Ollama…", None)
            subprocess.run([str(tmp / "OllamaSetup.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
                           check=True, timeout=900)
            exe = find_ollama()
            if not exe:
                raise RuntimeError("Ollama was installed but can't be found")
            return exe
        raise RuntimeError("Automatic setup isn't available on this system")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _api(base: str, path: str, body: dict | None = None, timeout: float = 5):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def ensure_server(exe: str, base: str, progress, cancel: threading.Event) -> None:
    try:
        _api(base, "/api/tags").close()
        return
    except Exception:  # noqa: BLE001
        pass
    progress("Starting the local AI…", None)
    flags = 0x08000000 if sys.platform == "win32" else 0
    subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, creationflags=flags,
                     start_new_session=sys.platform != "win32", env=dict(os.environ))
    for _ in range(60):
        if cancel.is_set():
            raise Cancelled()
        time.sleep(0.5)
        try:
            _api(base, "/api/tags").close()
            return
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("The local AI didn't start")


def pull(base: str, model: str, progress, cancel: threading.Event) -> None:
    with _api(base, "/api/pull", {"model": model, "stream": True}, timeout=3600) as r:
        for raw in r:
            if cancel.is_set():
                raise Cancelled()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if msg.get("error"):
                raise RuntimeError(msg["error"])
            total, done = msg.get("total"), msg.get("completed")
            if total and total > 2**20 and done is not None:  # (tiny parts: already here)
                progress(f"Downloading {model}… {done / 2**30:.1f} of {total / 2**30:.1f} GB", done / total)
            elif msg.get("status"):
                progress(msg["status"].capitalize() + "…", None)


def try_apple(llm_cfg, progress, cancel: threading.Event) -> tuple[float, str | None]:
    """(seconds per clean-up, None) if Apple's on-device model is available and good enough
    here, else (0, why not)."""
    from . import applellm
    from .ai_choice import APPLE, SLOW_S, quality_check, speed_check

    progress("Checking Apple's on-device model…", None)
    ok, reason = applellm.shared().availability()
    if not ok:
        return 0.0, applellm.reason_text(reason)
    if cancel.is_set():
        raise Cancelled()
    problem = quality_check(llm_cfg, APPLE)
    if problem:
        return 0.0, problem
    took = speed_check(llm_cfg, APPLE, cancel)
    if took > SLOW_S:
        return 0.0, f"it took {took:.1f} s per clean-up here"
    return took, None


def remove_model(base: str, model: str) -> None:
    """Delete a model Murmr installed earlier (frees its disk space)."""
    try:
        req = urllib.request.Request(base.rstrip("/") + "/api/delete", method="DELETE",
                                     data=json.dumps({"model": model}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).close()
        log.info("Removed the old AI model %s", model)
    except Exception as exc:  # noqa: BLE001
        log.info("Couldn't remove %s: %s", model, exc)


def setup(llm_cfg, progress: Callable[[str, float | None], None], cancel: threading.Event) -> str:
    """Install Ollama if needed, start it, download the best model for this computer (or the
    one you chose) and check it's quick enough here. Returns the model name; the measured
    time goes to ``llm_cfg.measured_s``."""
    from .ai_choice import APPLE, KNOWN, SLOW_S, detect, recommend, smaller, speed_check
    from .llm import find_ollama

    if not getattr(llm_cfg, "auto_model", True):
        exe = find_ollama() or install_ollama(progress, cancel)
        ensure_server(exe, llm_cfg.base_url, progress, cancel)
        pull(llm_cfg.base_url, llm_cfg.model, progress, cancel)
        progress("Local AI ready", 1.0)
        return llm_cfg.model
    hw = detect()
    choice = recommend(hw)
    log.info("AI model for %s: %s", hw.describe(), " > ".join(choice.models))
    previous = llm_cfg.model
    choice_had_apple = choice.model == APPLE
    if choice.model == APPLE:
        took, why = try_apple(llm_cfg, progress, cancel)
        if why is None:
            from . import applellm

            llm_cfg.provider, llm_cfg.measured_s = "apple", round(took, 2)
            llm_cfg.apple_skipped = ""
            llm_cfg.apple_os = applellm.os_version()
            if previous in KNOWN and previous != APPLE and find_ollama():
                remove_model(llm_cfg.base_url, previous)  # Murmr's Ollama model isn't needed now
            progress(f"Local AI ready: Apple's on-device model, {took:.1f} s per clean-up", 1.0)
            return APPLE
        log.info("Apple's on-device model not used: %s", why)
        llm_cfg.apple_skipped = why
        progress(f"Apple's model: {why}. Setting up Ollama instead…", None)
        choice.models = [m for m in choice.models if m != APPLE]
    exe = find_ollama() or install_ollama(progress, cancel)
    ensure_server(exe, llm_cfg.base_url, progress, cancel)
    llm_cfg.provider = "ollama"
    had = installed_models(llm_cfg.base_url)  # never remove a model that was already here
    model: str | None = choice.model
    last_error: Exception | None = None
    while model:
        if cancel.is_set():
            raise Cancelled()
        try:
            pull(llm_cfg.base_url, model, progress, cancel)
            progress(f"Checking how fast {model} is on this computer…", None)
            took = speed_check(llm_cfg, model, cancel)
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - e.g. the MLX build can't run here
            log.warning("AI model %s didn't work here: %s", model, exc)
            last_error = exc
            i = choice.models.index(model)
            model = choice.models[i + 1] if i + 1 < len(choice.models) else None
            continue
        log.info("AI model %s: %.2f s per clean-up", model, took)
        step = smaller(choice.models, model) if took > SLOW_S else None
        if step is None:
            llm_cfg.measured_s = round(took, 2)
            if previous != model and previous in KNOWN and previous != APPLE:
                remove_model(llm_cfg.base_url, previous)
            for other in choice.models:  # one downloaded now, then dropped (too slow, broken)
                if other not in (model, previous) and other not in had and                         other in installed_models(llm_cfg.base_url):
                    remove_model(llm_cfg.base_url, other)
            note = f" (Apple's model wasn't used: {llm_cfg.apple_skipped})" if                 getattr(llm_cfg, "apple_skipped", "") and choice_had_apple else ""
            progress(f"Local AI ready: {model}, {took:.1f} s per clean-up{note}", 1.0)
            return model
        progress(f"{model} takes {took:.1f} s here; trying the lighter {step}…", None)
        model = step
    raise RuntimeError(f"No AI model worked on this computer: {last_error}")


def installed_models(base: str) -> set[str]:
    try:
        with _api(base, "/api/tags") as r:
            return {m.get("name", "") for m in json.load(r).get("models", [])}
    except Exception:  # noqa: BLE001
        return set()
