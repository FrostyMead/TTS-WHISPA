"""One-click setup of the local AI (Ollama + a model), for Command Mode, AI clean-up and
meeting summaries.

* macOS: Ollama's command-line build (~180 MB) goes into Murmr's own folder; Murmr runs it.
  No installer, no admin password, nothing in /Applications.
* Windows: Ollama's own installer, silently, for this user only.
* Then the model is downloaded through Ollama: gemma3:4b (3.3 GB) with 16 GB of memory or
  more, else the lighter gemma3:1b (0.8 GB).

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
    return "gemma3:4b" if memory_gb() >= 15 else "gemma3:1b"


def model_size_label(model: str) -> str:
    return {"gemma3:4b": "3.3 GB", "gemma3:1b": "0.8 GB"}.get(model, "a few GB")


def local_dir() -> Path:
    from . import config as config_mod

    return config_mod.config_dir() / "ai"


def supported() -> bool:
    return sys.platform in ("win32", "darwin")


def plan() -> str:
    """A sentence about what setting up will download."""
    from .llm import find_ollama

    model = recommended_model()
    app = "" if find_ollama() else ("Ollama (~180 MB) and " if sys.platform == "darwin"
                                    else "Ollama (~1.7 GB) and ")
    return f"Downloads {app}the {model} model ({model_size_label(model)}). Runs on this computer only."


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
            if total and done is not None:
                progress(f"Downloading {model}… {done / 2**30:.1f} of {total / 2**30:.1f} GB", done / total)
            elif msg.get("status"):
                progress(msg["status"].capitalize() + "…", None)


def setup(llm_cfg, progress: Callable[[str, float | None], None], cancel: threading.Event) -> str:
    """Install Ollama if needed, start it, download the model. Returns the model name."""
    from .llm import find_ollama

    exe = find_ollama() or install_ollama(progress, cancel)
    ensure_server(exe, llm_cfg.base_url, progress, cancel)
    model = recommended_model() if llm_cfg.model in ("gemma3:4b", "gemma3:1b") else llm_cfg.model
    pull(llm_cfg.base_url, model, progress, cancel)
    progress("Local AI ready", 1.0)
    return model
