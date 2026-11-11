"""The update flow end to end, against a fake GitHub: check -> stage -> bootstrap applies ->
auto-rollback when the new version never reports a healthy start -> manual rollback."""

import codecs
import io
import json
import shutil
import subprocess
import sys
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from whispa.updater import Updater

REAL_BOOT = Path(__file__).resolve().parent.parent / "whispa" / "_boot.py"
BASE_PY = str(Path(sys.base_prefix) / ("python.exe" if sys.platform == "win32" else "bin/python3"))
STUB_MAIN = '''import json, os
def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = json.load(open(os.path.join(here, "version.json")))["sha"]
    with open(os.environ["WHISPA_TEST_OUT"], "a") as f:
        f.write(sha + "\\n")
    return 0
'''
OLD, NEW = "a" * 40, "b" * 40


def make_app(dest: Path, pyproject: str = "[project]\nname='whispa'\n") -> None:
    (dest / "whispa").mkdir(parents=True)
    (dest / "whispa" / "__init__.py").write_text("")
    (dest / "whispa" / "__main__.py").write_text(STUB_MAIN)
    shutil.copy(REAL_BOOT, dest / "whispa" / "_boot.py")
    (dest / "pyproject.toml").write_text(pyproject)


def zip_of(sha: str, pyproject: str) -> bytes:
    tmp = Path(__import__("tempfile").mkdtemp())
    make_app(tmp / f"TTS-WHISPA-{sha}", pyproject)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in tmp.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(tmp).as_posix())
    shutil.rmtree(tmp)
    return buf.getvalue()


@pytest.fixture
def github():
    state = {"pyproject": "[project]\nname='whispa'\n", "hits": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["hits"].append(self.path)
            if "/actions/workflows/ci.yml/runs" in self.path:
                body = json.dumps({"workflow_runs": [{"head_sha": NEW,
                                                      "head_commit": {"timestamp": "2026-11-12T10:00:00Z"}}]})
            elif "/compare/" in self.path:
                body = json.dumps({"commits": [{"commit": {"message": "Fix the pill\n\nDetails"}},
                                               {"commit": {"message": "Add snippets"}}]})
            elif "/zip/" in self.path:
                data = zip_of(NEW, state["pyproject"])
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_port}"
    yield state
    srv.shutdown()


@pytest.fixture
def runtime(tmp_path):
    rt = tmp_path / "Whispa"
    make_app(rt / "app" / "current")
    (rt / "app" / "current" / "version.json").write_text(json.dumps({"sha": OLD, "date": "2026-11-11"}))
    shutil.copy(REAL_BOOT, rt / "boot.py")
    (rt / "install.json").write_text(json.dumps({"repo": "o/r", "branch": "main", "runtime": "cpu"}))
    return rt


def boot(rt: Path, out: Path, *args: str) -> list[str]:
    subprocess.run([BASE_PY, "-I", "-S", str(rt / "boot.py"), *args], check=True,
                   env={"WHISPA_TEST_OUT": str(out), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
    return out.read_text().split()


def updater(rt, gh):
    return Updater(runtime=rt, api=gh["url"], codeload=gh["url"])


def test_check_finds_newer_passing_version(runtime, github):
    info = updater(runtime, github).check()
    assert info.available and info.target_sha == NEW and info.current_sha == OLD
    assert info.changes == ["Add snippets", "Fix the pill"]  # newest first, first line only


def test_full_update_cycle(runtime, github, tmp_path):
    out = tmp_path / "ran.txt"
    u = updater(runtime, github)
    info = u.check()
    u.stage(info)
    assert u.staged()["sha"] == NEW
    # restart -> the bootstrap applies the update and runs the new version
    assert boot(runtime, out) == [NEW]
    assert (runtime / "trial").exists() and u.current_version()["sha"] == NEW
    # the new version never reported a healthy start -> next start rolls back automatically
    assert boot(runtime, out) == [NEW, OLD]
    assert u.current_version()["sha"] == OLD and not (runtime / "trial").exists()


def test_healthy_update_is_kept(runtime, github, tmp_path):
    out = tmp_path / "ran.txt"
    u = updater(runtime, github)
    u.stage(u.check())
    boot(runtime, out)
    u.mark_healthy()
    assert boot(runtime, out) == [NEW, NEW]
    assert u.has_previous()


def test_manual_rollback(runtime, github, tmp_path):
    out = tmp_path / "ran.txt"
    u = updater(runtime, github)
    u.stage(u.check())
    boot(runtime, out)
    u.mark_healthy()
    u.request_rollback()
    assert boot(runtime, out) == [NEW, OLD]
    assert u.current_version()["sha"] == OLD


def test_packages_reinstalled_only_when_pyproject_changes(runtime, github, monkeypatch):
    calls = []
    monkeypatch.setattr(Updater, "install_packages", lambda self, p: calls.append(p))
    u = updater(runtime, github)
    u.stage(u.check())
    assert calls == []
    github["pyproject"] = "[project]\nname='whispa'\ndependencies=['new-thing']\n"
    u.stage(u.check())
    assert len(calls) == 1


def test_up_to_date(runtime, github):
    (runtime / "app" / "current" / "version.json").write_text(json.dumps({"sha": NEW}))
    assert updater(runtime, github).check().available is False


def test_development_copy_is_left_alone(tmp_path):
    u = Updater(runtime=None)
    assert not u.installed
    with pytest.raises(RuntimeError):
        from whispa.updater import UpdateInfo
        u.stage(UpdateInfo(available=True, target_sha=NEW))


def test_offline_is_reported_not_raised(runtime):
    info = Updater(runtime=runtime, api="http://127.0.0.1:9", codeload="http://127.0.0.1:9").check()
    assert not info.available and "Couldn't check" in info.error


def test_windows_installer_files_with_bom_are_read(runtime, github, tmp_path):
    """Windows PowerShell 5.1 writes UTF-8 with a byte-order mark."""
    (runtime / "app" / "current" / "version.json").write_bytes(
        codecs.BOM_UTF8 + json.dumps({"sha": OLD}).encode())
    (runtime / "install.json").write_bytes(codecs.BOM_UTF8 + json.dumps({"repo": "o/r", "runtime": "cpu"}).encode())
    u = updater(runtime, github)
    assert u.current_version()["sha"] == OLD and u.install_info()["runtime"] == "cpu"
    info = u.check()
    assert info.current_sha == OLD and info.changes  # compare worked
    u.stage(info)
    out = tmp_path / "ran.txt"
    assert boot(runtime, out) == [NEW]
    assert f"{OLD[:7]} -> {NEW[:7]}" in (runtime / "boot.log").read_text()


def test_boot_works_when_started_inside_the_app_folder(runtime, github, tmp_path):
    """The old shortcut started Whispa with app/current as its working folder."""
    u = updater(runtime, github)
    u.stage(u.check())
    out = tmp_path / "ran.txt"
    subprocess.run([BASE_PY, "-I", "-S", str(runtime / "boot.py")], check=True,
                   cwd=runtime / "app" / "current",
                   env={"WHISPA_TEST_OUT": str(out), "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})
    assert out.read_text().split() == [NEW]


def test_locked_folder_starts_current_version_and_keeps_update(runtime, github, tmp_path, monkeypatch):
    """If the update can't be applied (folder open elsewhere), Whispa still starts."""
    u = updater(runtime, github)
    u.stage(u.check())
    import os
    out = tmp_path / "ran.txt"
    lock = open(runtime / "app" / "current" / "pyproject.toml")  # open file: Windows can't rename the folder
    try:
        if sys.platform == "win32":
            env = {"WHISPA_TEST_OUT": str(out), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
            subprocess.run([BASE_PY, "-I", "-S", str(runtime / "boot.py")], check=True, env=env)
            assert out.read_text().split() == [OLD]  # started the version it had
            assert (runtime / "app" / "next").is_dir()  # update kept for next time
            assert "couldn't apply" in (runtime / "boot.log").read_text()
    finally:
        lock.close()
