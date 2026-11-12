import importlib.util
import json
from pathlib import Path
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from whispa import _boot, updater as mod
from whispa.updater import Updater, UpdateInfo
from test_updater import runtime, github, NEW, OLD, make_app, zip_of, tree_of


@pytest.fixture
def probation(runtime, monkeypatch, tmp_path):
    for name, value in {"RUNTIME": runtime, "APP": runtime / "app",
                        "CURRENT": runtime / "app/current", "NEXT": runtime / "app/next",
                        "PREVIOUS": runtime / "app/previous", "TRIAL": runtime / "trial",
                        "LOG": runtime / "boot.log"}.items():
        monkeypatch.setattr(_boot, name, str(value))
    monkeypatch.setattr(_boot, "refresh_self", lambda: None)
    make_app(runtime / "app/next")
    (runtime / "app/next/version.json").write_text(json.dumps({"sha": NEW}))
    _boot.apply_update()
    u = Updater(runtime=runtime)
    u.mark_healthy()
    crash = tmp_path / "crash.log"
    crash.write_text("old dump\n")
    u.mark_running(crash)
    return u, crash


def test_probation_crash_rolls_back(probation, runtime):
    u, crash = probation
    crash.write_text(crash.read_text() + "Fatal Python error: stack overflow\n")
    assert _boot.check_probation()
    assert u.current_version()["sha"] == OLD
    assert json.loads((runtime / "skipped").read_text())["sha"] == NEW
    assert json.loads((runtime / "rolled_back.json").read_text()) == {
        "from": NEW, "to": OLD, "why": "crash during probation"}


@pytest.mark.parametrize("kind", ["power_cut", "expired", "other_sha", "three_exits", "broken"])
def test_no_probation_rollback(probation, runtime, kind):
    u, crash = probation
    if kind != "power_cut":
        crash.write_text("a larger crash dump than before\n")
    path = runtime / "probation"
    data = json.loads(path.read_text())
    if kind == "expired":
        data["time"] = time.time() - 86401
    if kind == "other_sha":
        data["sha"] = OLD
    if kind == "three_exits":
        data["clean_exits"] = 3
    path.write_text("invalid" if kind == "broken" else json.dumps(data))
    assert not _boot.check_probation()
    assert u.current_version()["sha"] == NEW


def test_three_clean_exits_end_probation(probation, runtime):
    u, crash = probation
    for i in range(3):
        u.mark_running(crash)
        u.mark_clean_exit()
        u.mark_clean_exit()  # shutdown is idempotent
        assert not (runtime / "running").exists()
        assert (runtime / "probation").exists() == (i < 2)
    assert not _boot.check_probation()


def test_skip_blocks_check_and_stage(runtime, github):
    u = Updater(runtime=runtime, api=github["url"])
    (runtime / "skipped").write_text(json.dumps({"sha": NEW}))
    assert not u.check().available
    with pytest.raises(RuntimeError, match="skipped"):
        u.stage(UpdateInfo(target_sha=NEW))
    github["runs"][0]["head_sha"] = "c" * 40
    assert u.check().available
    assert not (runtime / "skipped").exists()


@pytest.mark.parametrize("saved", [True, False])
def test_notice_once(probation, runtime, tmp_path, monkeypatch, saved):
    u, _ = probation
    (runtime / "rolled_back.json").write_text("{}")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    report = tmp_path / "Desktop/report.zip"
    report.parent.mkdir()
    if saved:
        report.write_bytes(b"zip")
    notice = u.rollback_notice(lambda: report)
    assert ("on your Desktop" in notice) == saved
    assert u.rollback_notice(lambda: report) == ""


@pytest.fixture
def signed(runtime, monkeypatch):
    # Ephemeral test-only key; never serialized to disk.
    key = Ed25519PrivateKey.generate()
    monkeypatch.setattr(mod, "RELEASE_KEYS", [key.public_key().public_bytes_raw().hex()])
    manifest = {"sha": NEW, "version": "0.8.16", "seq": 100, "date": "2026-11-13T00:00:00Z"}
    raw = json.dumps(manifest).encode()
    state = {"raw": raw, "sig": key.sign(raw), "ci": True}
    u = Updater(runtime=runtime)

    def fake_json(path):
        if path.startswith("/releases"):
            return [{"tag_name": "v0.8.16", "assets": [
                {"name": "release.json", "browser_download_url": "manifest"},
                {"name": "release.json.sig", "browser_download_url": "signature"}]}]
        if path.startswith("/actions"):
            return {"workflow_runs": [{"head_sha": NEW if state["ci"] else OLD,
                                       "conclusion": "success"}]}
        return {"commits": []}

    monkeypatch.setattr(u, "_json", fake_json)
    monkeypatch.setattr(u, "_get_limited", lambda url, limit: state["raw" if url == "manifest" else "sig"])
    return u, key, state


def test_signed_release_accepted(signed):
    u, _, _ = signed
    info = u.check()
    assert info.available and info.target_sha == NEW and info.seq == 100


def test_signed_stage_records_sequence(signed, monkeypatch):
    u, _, state = signed
    snapshot = zip_of(NEW, "[project]\nname='whispa'\n")
    original_json = u._json
    original_get = u._get_limited
    monkeypatch.setattr(u, "_json", lambda path: tree_of(snapshot) if path.startswith("/git/trees")
                        else original_json(path))
    monkeypatch.setattr(u, "_get_limited", lambda url, limit, **kw: snapshot if "/zip/" in url
                        else original_get(url, limit))
    info = u.check()
    u.stage(info)
    assert u.staged()["seq"] == 100
    state["sig"] = b"x" * 64
    with pytest.raises(RuntimeError):
        u.stage(info)


def test_boot_wont_apply_quarantined_snapshot(probation, runtime):
    _, _ = probation
    make_app(runtime / "app/next")
    (runtime / "app/next/version.json").write_text(json.dumps({"sha": OLD}))
    (runtime / "skipped").write_text(json.dumps({"sha": OLD}))
    _boot.apply_update()
    assert Updater(runtime=runtime).current_version()["sha"] == NEW


def test_failed_probation_rollback_keeps_current(probation, runtime, monkeypatch):
    u, crash = probation
    crash.write_text("a larger fatal crash dump than the original\n")
    replace = _boot._replace
    def locked_previous(src, dst):
        if src == str(runtime / "app/previous"):
            raise PermissionError("locked")
        replace(src, dst)
    monkeypatch.setattr(_boot, "_replace", locked_previous)
    assert not _boot.check_probation()
    assert u.current_version()["sha"] == NEW
    assert (runtime / "running").exists()  # evidence retained for retry


@pytest.mark.parametrize("bad", ["signature", "key", "tamper", "downgrade", "ci", "no_assets"])
def test_signed_release_rejected(signed, runtime, monkeypatch, bad):
    u, _, state = signed
    if bad == "signature":
        state["sig"] = b"x" * 64
    elif bad == "key":
        monkeypatch.setattr(mod, "RELEASE_KEYS", ["00" * 32])
    elif bad == "tamper":
        state["raw"] += b" "
    elif bad == "downgrade":
        (runtime / "app/current/version.json").write_text(json.dumps({"sha": OLD, "seq": 100}))
    elif bad == "ci":
        state["ci"] = False
    else:
        monkeypatch.setattr(u, "_json", lambda path: [])
    info = u.check()
    assert not info.available and info.error
    with pytest.raises(RuntimeError):
        u.stage(UpdateInfo(target_sha=NEW, seq=100))


def test_empty_keys_ci_fallback(runtime, github, monkeypatch, caplog):
    monkeypatch.setattr(mod, "RELEASE_KEYS", [])
    assert Updater(runtime=runtime, api=github["url"]).check().available
    assert "CI-only" in caplog.text


def test_sign_script_publish_commands(monkeypatch):
    spec = importlib.util.spec_from_file_location("sign_release", Path(__file__).parents[1] / "scripts/sign_release.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []
    class Result:
        returncode = 0
    monkeypatch.setattr(script.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Result())
    script.publish("0.8.16", NEW)
    assert calls[-1][-1] == "--clobber"
    Result.returncode = 1
    script.publish("0.8.16", NEW)
    assert calls[-1][:3] == ["gh", "release", "create"]


def test_sign_script_manifest_roundtrip(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization

    spec = importlib.util.spec_from_file_location("sign_release", Path(__file__).parents[1] / "scripts/sign_release.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    key = Ed25519PrivateKey.generate()  # memory only; no private key file is written
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    monkeypatch.chdir(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", lambda self: pem)
        patch.setattr(script.time, "time", lambda: 1234567890)
        script.sign("0.8.16", NEW, Path("dummy-test-key"))
    raw = (tmp_path / "release.json").read_bytes()
    key.public_key().verify((tmp_path / "release.json.sig").read_bytes(), raw)
    assert json.loads(raw)["seq"] == 1234567890
    assert json.loads(raw)["sha"] == NEW


def test_notice_report_failure_still_notifies(probation, runtime):
    u, _ = probation
    (runtime / "rolled_back.json").write_text("{}")
    def fail():
        raise OSError("Desktop unavailable")
    assert "Desktop" not in u.rollback_notice(fail)
    assert not (runtime / "rolled_back.json").exists()
