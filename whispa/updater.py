"""Updates from GitHub for installed copies (see whispa/_boot.py for the folder layout).

* check():  newest signed release whose CI run passed (CI-only until keys are configured),
            plus the list of changes in between. If that can't be confirmed (GitHub
            unreachable, no passing run), there is no update: never an untested one.
* stage():  download that snapshot, check every file against the commit's own git
            hashes (and size limits), and put it in app/next. If pyproject.toml changed,
            the new packages go into a separate environment (envs/<sha>) used only by
            that version, so the running version and a rollback keep their own packages.
            The bootstrap applies it on the next start.
* rollback(): ask the bootstrap to go back to app/previous (and its environment)

Only public data is fetched from GitHub. Nothing about you or your dictations is sent.
A development checkout (a git clone) is never touched: update it with git.
"""

from __future__ import annotations

import io
import hashlib
import json
import logging
import os
import re
from urllib.parse import urlencode
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
USER_AGENT = "Murmr-Updater"
# Public keys of the offline release-signing key (Ed25519, hex). The private key lives only
# on the maintainer's PC, never on GitHub: a release must be signed with it to install.
RELEASE_KEYS: list[str] = ["6f3bf71224602450ad4a8c81a9ba4429c4c0297d09e0d4b5a27c51e28b980125"]
CI_WORKFLOW = "ci.yml"
MAX_DOWNLOAD = 100 << 20  # the snapshot zip (about 1 MB today)
MAX_UNPACKED = 300 << 20

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
    seq: int = 0


class Updater:
    def __init__(self, repo: str = "", branch: str = "", runtime: Path | None = None,
                 api: str = API, codeload: str = CODELOAD):
        self.runtime = runtime if runtime is not None else runtime_dir()
        info = self.install_info()
        self.repo = repo or info.get("repo", "FrostyMead/murmr")
        if self.repo.lower() == "frostymead/tts-whispa":  # renamed on GitHub
            self.repo = "FrostyMead/murmr"
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
            return json.loads((self.runtime / "install.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}

    def current_version(self) -> dict:
        app = self.runtime / "app" / "current" if self.runtime else APP_DIR
        try:
            return json.loads((app / "version.json").read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return {}

    def has_previous(self) -> bool:
        return bool(self.runtime) and (self.runtime / "app" / "previous" / "version.json").is_file()

    def staged(self) -> dict | None:
        if not self.runtime:
            return None
        try:
            return json.loads((self.runtime / "app" / "next" / "version.json").read_text(encoding="utf-8-sig"))
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

    def mark_running(self, crash_log: Path) -> None:
        if not self.runtime:
            return
        try:
            crash_log = crash_log.resolve()
            size = crash_log.stat().st_size if crash_log.exists() else 0
            self._write_state("running", {"sha": self.current_version().get("sha", ""),
                                          "crash_log": str(crash_log), "size": size})
        except Exception:
            log.exception("Couldn't record running marker")

    def _write_state(self, name: str, data: dict) -> None:
        path = self.runtime / name
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)

    def mark_clean_exit(self) -> None:
        if not self.runtime:
            return
        try:
            marker = self.runtime / "running"
            running = json.loads(marker.read_text(encoding="utf-8"))
            marker.unlink()
            path = self.runtime / "probation"
            if path.exists():
                probation = json.loads(path.read_text(encoding="utf-8"))
                if probation["sha"] == running["sha"] == self.current_version().get("sha"):
                    probation["clean_exits"] = probation.get("clean_exits", 0) + 1
                    if probation["clean_exits"] >= 3 or time.time() - probation["time"] >= 86400:
                        path.unlink()
                    else:
                        self._write_state("probation", probation)
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("Couldn't record clean exit")

    def rollback_notice(self, save_report: Callable[[], Path]) -> str:
        if not self.runtime or not (self.runtime / "rolled_back.json").exists():
            return ""
        message = "The last update crashed, so Murmr went back to the version before."
        try:
            path = save_report()
            if path.is_file() and path.parent == Path.home() / "Desktop":
                message += " A problem report is on your Desktop."
        except Exception:
            log.exception("Couldn't save rollback report")
        try:
            (self.runtime / "rolled_back.json").unlink()
        except OSError:
            pass
        return message

    def _check_skipped(self, sha: str, *, clear: bool = False) -> None:
        if not self.runtime:
            return
        path = self.runtime / "skipped"
        try:
            skipped = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        if skipped.get("sha") == sha:
            raise RuntimeError("This update crashed before and has been skipped; waiting for a newer version")
        if clear:
            path.unlink()

    def signed_release(self) -> dict:
        """Choose the highest authenticated sequence among recent published releases."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature

        candidates = []
        for release in self._json("/releases?per_page=10"):
            if release.get("draft") or release.get("prerelease"):
                continue
            try:
                assets = {a["name"]: a["browser_download_url"] for a in release["assets"]}
                raw = self._get_limited(assets["release.json"], 16384)
                sig = self._get_limited(assets["release.json.sig"], 64)
                verified = False
                for key in RELEASE_KEYS:
                    try:
                        Ed25519PublicKey.from_public_bytes(bytes.fromhex(key)).verify(sig, raw)
                        verified = True
                        break
                    except (ValueError, InvalidSignature):
                        continue
                if not verified:
                    continue
                manifest = json.loads(raw)
                if (not re.fullmatch(r"[0-9a-f]{40}", manifest["sha"])
                        or type(manifest["seq"]) is not int or manifest["seq"] <= 0
                        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", manifest["version"])
                        or not isinstance(manifest["date"], str)
                        or release.get("tag_name") != "v" + manifest["version"]):
                    continue
                candidates.append(manifest)
            except Exception:
                log.warning("Ignoring an invalid or unavailable signed release", exc_info=True)
        if not candidates:
            raise RuntimeError("No valid signed release was found; waiting for a signed update")
        manifest = max(candidates, key=lambda m: m["seq"])
        runs = self._json(f"/actions/workflows/{CI_WORKFLOW}/runs?" + urlencode({
            "branch": self.branch, "head_sha": manifest["sha"], "status": "success",
            "event": "push", "per_page": 100}))
        if not any(r.get("head_sha") == manifest["sha"] and r.get("conclusion") == "success"
                   for r in runs.get("workflow_runs", [])):
            raise RuntimeError("The signed version has not passed CI tests")
        return manifest

    # -- GitHub ------------------------------------------------------------------------
    def _get(self, url: str, timeout: float = 15):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()

    def _json(self, path: str):
        return json.loads(self._get(f"{self.api}/repos/{self.repo}{path}"))

    def latest_good_commit(self) -> tuple[str, str]:
        """(sha, date) of the newest commit on the branch whose CI passed. Raises when that
        can't be confirmed: an untested version is never installed."""
        runs = self._json(f"/actions/workflows/{CI_WORKFLOW}/runs?branch={self.branch}"
                          "&status=success&event=push&per_page=1")
        for run in runs.get("workflow_runs") or []:
            if run.get("conclusion", "success") == "success" and run.get("head_sha"):
                return run["head_sha"], (run.get("head_commit") or {}).get("timestamp", "")
        raise RuntimeError("no version that passed the tests was found")

    def check(self) -> UpdateInfo:
        info = UpdateInfo(current_sha=self.current_version().get("sha", ""))
        try:
            if RELEASE_KEYS:
                manifest = self.signed_release()
                if manifest["seq"] <= self.current_version().get("seq", 0):
                    # Up to date (or the newest signed release is older: never go back).
                    info.target_sha = info.current_sha
                    self.last = info
                    return info
                sha, date, info.seq = manifest["sha"], manifest["date"], manifest["seq"]
            else:
                log.warning("Release signing is not configured; using CI-only updates")
                sha, date = self.latest_good_commit()
            self._check_skipped(sha, clear=sha != info.current_sha)
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
        self._check_skipped(info.target_sha)
        if RELEASE_KEYS:
            manifest = self.signed_release()
            if manifest["sha"] != info.target_sha or manifest["seq"] != info.seq:
                raise RuntimeError("The signed release changed; check for updates again")
            if manifest["seq"] <= self.current_version().get("seq", 0):
                raise RuntimeError("That isn't newer than this version")
        app = self.runtime / "app"
        nxt, tmp = app / "next", app / "next.partial"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(nxt, ignore_errors=True)
        progress("Downloading…")
        data = self._get_limited(f"{self.codeload}/{self.repo}/zip/{info.target_sha}", MAX_DOWNLOAD,
                                 timeout=120)
        tree = self._json(f"/git/trees/{info.target_sha}?recursive=1")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            verify_snapshot(zf, tree)
            zf.extractall(tmp)
        roots = [p for p in tmp.iterdir() if p.is_dir()]
        src = roots[0] if len(roots) == 1 else tmp  # GitHub zips have one top-level folder
        if not (src / "whispa" / "__main__.py").is_file():
            raise RuntimeError("The download doesn't look like Murmr")
        current_pyproject = app / "current" / "pyproject.toml"
        new_pyproject = src / "pyproject.toml"
        env = self.current_version().get("env") or "venv"
        if not current_pyproject.is_file() or not (self.runtime / env).is_dir() or \
                current_pyproject.read_bytes() != new_pyproject.read_bytes():
            progress("Installing new packages…")
            env = f"envs/{info.target_sha[:12]}"
            self.install_packages(new_pyproject, self.runtime / env)
        (src / "version.json").write_text(json.dumps({
            "sha": info.target_sha, "date": info.target_date, "repo": self.repo,
            "branch": self.branch, "changes": info.changes, "staged_at": time.time(),
            "env": env, "seq": info.seq,
        }, indent=2), encoding="utf-8")
        src.replace(nxt) if src != tmp else tmp.replace(nxt)
        shutil.rmtree(tmp, ignore_errors=True)
        progress("Ready to restart")
        log.info("Staged update %s", info.target_sha[:7])

    def _get_limited(self, url: str, limit: int, timeout: float = 60) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("The download is far bigger than Murmr; not installing it")
        return data

    def install_packages(self, pyproject: Path, env: Path) -> None:
        """A fresh environment at ``env`` with ``pyproject``'s packages. uv links them from
        its cache, so this is quick and takes little extra disk space."""
        meta = self.install_info()
        uv = meta.get("uv") or shutil.which("uv")
        if not uv or not Path(uv).exists() and not shutil.which(uv):
            raise RuntimeError("uv is missing; re-run the installer")
        base = Path(sys.base_prefix) / ("python.exe" if sys.platform == "win32" else "bin/python3")
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        # Built under a temporary name and renamed when complete: a half-built environment
        # is never used, and the bootstrap never removes one that's being built.
        building = env.with_name(f".building-{env.name}")
        shutil.rmtree(building, ignore_errors=True)
        shutil.rmtree(env, ignore_errors=True)
        env.parent.mkdir(parents=True, exist_ok=True)

        def run(cmd: list[str]) -> None:
            proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags,
                                  timeout=1800, env=env_without_venv())
            if proc.returncode:
                shutil.rmtree(building, ignore_errors=True)
                raise RuntimeError(f"Installing packages failed: {proc.stderr.strip()[-400:]}")

        run([uv, "venv", "--python", str(base if base.exists() else sys.executable), str(building)])
        python = building / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        cmd = [uv, "pip", "install", "--python", str(python), "-r", str(pyproject)]
        if meta.get("runtime"):
            cmd += ["--extra", meta["runtime"]]
        run(cmd)
        building.rename(env)

    def request_rollback(self) -> None:
        if not self.has_previous():
            raise RuntimeError("There's no previous version to go back to")
        shutil.rmtree(self.runtime / "app" / "next", ignore_errors=True)
        (self.runtime / "trial").write_text("rollback requested", encoding="utf-8")


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git's id


def verify_snapshot(zf: zipfile.ZipFile, tree: dict) -> None:
    """Every file in the downloaded zip is exactly the commit's file (git blob hash), none
    missing or extra, within the size limits. Raises otherwise."""
    if tree.get("truncated"):
        raise RuntimeError("Couldn't verify the download (file list incomplete)")
    expected = {e["path"]: e["sha"] for e in tree.get("tree", [])
                if e.get("type") == "blob" and e.get("mode") != "120000"}
    files = [i for i in zf.infolist() if not i.is_dir()]
    if sum(i.file_size for i in files) > MAX_UNPACKED:
        raise RuntimeError("The download unpacks far bigger than Murmr; not installing it")
    seen = set()
    for item in files:
        rel = item.filename.split("/", 1)[1] if "/" in item.filename else item.filename
        if rel not in expected or git_blob_sha(zf.read(item)) != expected[rel]:
            raise RuntimeError(f"The download didn't match the published version ({rel}); "
                               "not installing it")
        seen.add(rel)
    if seen != set(expected):
        raise RuntimeError("The download is missing files; not installing it")


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
