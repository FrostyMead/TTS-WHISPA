"""Updates from GitHub for installed copies (see whispa/_boot.py for the folder layout).

* check():  newest commit on the branch whose CI run passed, compared with ours,
            plus the list of changes in between
* stage():  download that snapshot to app/next, install new packages if
            pyproject.toml changed; the bootstrap applies it on the next start
* rollback(): ask the bootstrap to go back to app/previous on the next start

Only public data is fetched from GitHub. Nothing about you or your dictations is sent.
A development checkout (a git clone) is never touched: update it with git.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

API = "https://api.github.com"
CODELOAD = "https://codeload.github.com"
USER_AGENT = "Whispa-Updater"
CI_WORKFLOW = "ci.yml"

APP_DIR = Path(__file__).resolve().parent.parent  # .../app/current, or the git checkout


def runtime_dir() -> Path | None:
    """The install root if this is an installed copy (.../app/current/whispa/updater.py)."""
    root = APP_DIR.parent.parent
    if APP_DIR.name == "current" and (root / "install.json").is_file():
        return root
    return None


@dataclass
class UpdateInfo:
    available: bool = False
    current_sha: str = ""
    target_sha: str = ""
    target_date: str = ""
    changes: list[str] = field(default_factory=list)
    error: str = ""


class Updater:
    def __init__(self, repo: str = "", branch: str = "", runtime: Path | None = None,
                 api: str = API, codeload: str = CODELOAD):
        self.runtime = runtime if runtime is not None else runtime_dir()
        info = self.install_info()
        self.repo = repo or info.get("repo", "FrostyMead/TTS-WHISPA")
        self.branch = branch or info.get("branch", "main")
        self.api, self.codeload = api.rstrip("/"), codeload.rstrip("/")
        self.last: UpdateInfo | None = None

    # -- state -------------------------------------------------------------------------
    @property
    def installed(self) -> bool:
        return self.runtime is not None

    def install_info(self) -> dict:
        if not self.runtime:
            return {}
        try:
            return json.loads((self.runtime / "install.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def current_version(self) -> dict:
        app = self.runtime / "app" / "current" if self.runtime else APP_DIR
        try:
            return json.loads((app / "version.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def has_previous(self) -> bool:
        return bool(self.runtime) and (self.runtime / "app" / "previous" / "version.json").is_file()

    def staged(self) -> dict | None:
        if not self.runtime:
            return None
        try:
            return json.loads((self.runtime / "app" / "next" / "version.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def mark_healthy(self) -> None:
        """Called once the app is up: the bootstrap keeps this version."""
        if self.runtime:
            try:
                (self.runtime / "trial").unlink()
                log.info("Update %s started fine", self.current_version().get("sha", "?")[:7])
            except FileNotFoundError:
                pass

    # -- GitHub ------------------------------------------------------------------------
    def _get(self, url: str, timeout: float = 15):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _json(self, path: str):
        return json.loads(self._get(f"{self.api}/repos/{self.repo}{path}"))

    def latest_good_commit(self) -> tuple[str, str]:
        """(sha, date) of the newest commit on the branch whose CI passed (or the branch head
        if the repo has no CI runs yet)."""
        try:
            runs = self._json(f"/actions/workflows/{CI_WORKFLOW}/runs?branch={self.branch}"
                              "&status=success&event=push&per_page=1")
            if runs.get("workflow_runs"):
                run = runs["workflow_runs"][0]
                return run["head_sha"], run.get("head_commit", {}).get("timestamp", "")
        except Exception as exc:  # noqa: BLE001 - no CI configured, or the API is unreachable
            log.debug("CI lookup failed: %s", exc)
        head = self._json(f"/commits/{self.branch}")
        return head["sha"], head["commit"]["committer"]["date"]

    def check(self) -> UpdateInfo:
        info = UpdateInfo(current_sha=self.current_version().get("sha", ""))
        try:
            sha, date = self.latest_good_commit()
            info.target_sha, info.target_date = sha, date
            staged = self.staged()
            if staged and staged.get("sha") == sha:
                info.available, info.changes = True, staged.get("changes", [])
            elif sha != info.current_sha:
                info.available = True
                if info.current_sha:
                    cmp = self._json(f"/compare/{info.current_sha}...{sha}")
                    info.changes = [c["commit"]["message"].splitlines()[0]
                                    for c in reversed(cmp.get("commits", []))][:20]
        except Exception as exc:  # noqa: BLE001
            info.error = f"Couldn't check for updates: {exc}"
            log.warning(info.error)
        self.last = info
        return info

    # -- installing --------------------------------------------------------------------
    def stage(self, info: UpdateInfo, progress: Callable[[str], None] = lambda s: None) -> None:
        """Download ``info.target_sha`` into app/next. Applied by the bootstrap on restart."""
        if not self.runtime:
            raise RuntimeError("This is a development copy; update it with git")
        app = self.runtime / "app"
        nxt, tmp = app / "next", app / "next.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(nxt, ignore_errors=True)
        progress("Downloading…")
        data = self._get(f"{self.codeload}/{self.repo}/zip/{info.target_sha}", timeout=120)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp)
        roots = [p for p in tmp.iterdir() if p.is_dir()]
        src = roots[0] if len(roots) == 1 else tmp  # GitHub zips have one top-level folder
        if not (src / "whispa" / "__main__.py").is_file():
            raise RuntimeError("The download doesn't look like Whispa")
        current_pyproject = app / "current" / "pyproject.toml"
        new_pyproject = src / "pyproject.toml"
        if not current_pyproject.is_file() or \
                current_pyproject.read_bytes() != new_pyproject.read_bytes():
            progress("Installing new packages…")
            self.install_packages(new_pyproject)
        (src / "version.json").write_text(json.dumps({
            "sha": info.target_sha, "date": info.target_date, "repo": self.repo,
            "branch": self.branch, "changes": info.changes, "staged_at": time.time(),
        }, indent=2), encoding="utf-8")
        src.replace(nxt) if src != tmp else tmp.replace(nxt)
        shutil.rmtree(tmp, ignore_errors=True)
        progress("Ready to restart")
        log.info("Staged update %s", info.target_sha[:7])

    def install_packages(self, pyproject: Path) -> None:
        meta = self.install_info()
        uv = meta.get("uv") or shutil.which("uv")
        if not uv or not Path(uv).exists() and not shutil.which(uv):
            raise RuntimeError("uv is missing; re-run the installer")
        venv = self.runtime / "venv"
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cmd = [uv, "pip", "install", "--python", str(python), "-r", str(pyproject)]
        if meta.get("runtime"):
            cmd += ["--extra", meta["runtime"]]
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags, timeout=1800)
        if proc.returncode:
            raise RuntimeError(f"Installing packages failed: {proc.stderr.strip()[-400:]}")

    def request_rollback(self) -> None:
        if not self.has_previous():
            raise RuntimeError("There's no previous version to go back to")
        shutil.rmtree(self.runtime / "app" / "next", ignore_errors=True)
        (self.runtime / "trial").write_text("rollback requested", encoding="utf-8")


def boot_command(extra: list[str] | None = None) -> list[str]:
    """The command that starts this installed copy through its bootstrap, windowless."""
    rt = runtime_dir()
    if rt is None:
        raise RuntimeError("not an installed copy")
    exe = Path(sys.base_prefix) / ("pythonw.exe" if sys.platform == "win32" else "bin/python3")
    if not exe.exists():
        exe = Path(sys.executable)
    return [str(exe), "-I", "-S", str(rt / "boot.py"), *(extra or [])]


def env_without_venv() -> dict:
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    return env
