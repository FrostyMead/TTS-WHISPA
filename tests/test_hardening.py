"""Fixes from the security / reliability review: history retention, AI endpoint guard, the
API key in the credential store, sync connection limits, live transcription backlog, and
the Ollama checksum requirement."""
import json
import socket
import tempfile
import threading
import time
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from whispa import history


# --------------------------------------------------------------------------- history
def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_prune_keeps_recent_and_totals(tmp_path):
    h = tmp_path / "history.jsonl"
    now = time.time()
    _write(h, [{"t": now - 400 * 86400, "seconds": 10, "text": "old words here"},
               {"t": now - 400 * 86400, "seconds": 5, "text": "a command", "command": "x"},
               {"t": now - 3600, "seconds": 2, "text": "new one"}])
    assert history.prune(h, 365, now) == 2
    assert [json.loads(x)["text"] for x in h.read_text().splitlines()] == ["new one"]
    assert history.totals(h) == {"words": 3, "seconds": 10.0}
    assert history.prune(h, 0, now) == 0  # forever


def test_recent_texts_reads_only_the_end(tmp_path):
    h = tmp_path / "history.jsonl"
    _write(h, [{"t": i, "text": f"dictation {i} " + "x" * 200} for i in range(5000)])
    got = history.recent_texts(h, 10, block=4096)
    assert len(got) == 10 and got[-1].startswith("dictation 4999 ") and got[0].startswith("dictation 4990 ")


def test_delete_all(tmp_path):
    h = tmp_path / "history.jsonl"
    _write(h, [{"t": 1, "text": "a"}])
    history.totals_path(h).write_text('{"words": 3}')
    history.delete_all(h)
    assert not h.exists() and not history.totals_path(h).exists()


# --------------------------------------------------------------------------- AI endpoint
def test_remote_ai_server_needs_explicit_permission(monkeypatch):
    from whispa.config import LLMCleanup
    from whispa.llm import Polisher, is_local_url

    assert is_local_url("http://127.0.0.1:11434") and is_local_url("http://localhost:8080/v1")
    assert not is_local_url("https://api.example.com/v1") and not is_local_url("http://192.168.1.5:11434")
    sent = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: sent.append(a) or (_ for _ in ()).throw(OSError()))
    p = Polisher(LLMCleanup(provider="openai", base_url="https://api.example.com/v1"))
    assert p._probe() is False
    with pytest.raises(RuntimeError, match="allow_remote"):
        p.chat_once("system", "my private dictation")
    assert sent == []  # nothing left the computer
    p.cfg.allow_remote = True
    assert p.endpoint_allowed()


# --------------------------------------------------------------------------- API key
def test_api_key_goes_to_the_credential_store(tmp_path, monkeypatch):
    from whispa import config as config_mod
    from whispa import credentials

    store = {}

    class FakeKeyring:
        @staticmethod
        def set_password(service, account, key):
            store[(service, account)] = key

        @staticmethod
        def get_password(service, account):
            return store.get((service, account))

    monkeypatch.setitem(__import__("sys").modules, "keyring", FakeKeyring)
    monkeypatch.setattr(credentials, "_stored", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 3, "llm_cleanup": {"api_key": "sk-secret"}}))
    cfg = config_mod.load(path)
    assert cfg.llm_cleanup.api_key == "sk-secret"
    assert "sk-secret" not in path.read_text()
    assert store[("Murmr", "llm_api_key")] == "sk-secret"
    monkeypatch.setattr(credentials, "_stored", None)
    assert config_mod.load(path).llm_cleanup.api_key == "sk-secret"  # read back from the store


def test_api_key_stays_in_file_without_a_credential_store(tmp_path, monkeypatch):
    from whispa import config as config_mod
    from whispa import credentials

    class Broken:
        @staticmethod
        def set_password(*a):
            raise RuntimeError("no backend")

    monkeypatch.setitem(__import__("sys").modules, "keyring", Broken)
    monkeypatch.setattr(credentials, "_stored", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 3, "llm_cleanup": {"api_key": "sk-x"}}))
    assert config_mod.load(path).llm_cleanup.api_key == "sk-x"
    assert "sk-x" in path.read_text()  # not lost


# --------------------------------------------------------------------------- sync
def test_sync_limits_connections_and_drops_silent_ones(monkeypatch):
    from whispa import sync
    from whispa.learn import Dictionary

    monkeypatch.setattr(sync, "HANDSHAKE_S", 0.5)
    d = Path(tempfile.mkdtemp())
    svc = sync.SyncService(d, Dictionary(d / "d.json"), name=lambda: "PC", udp_port=None,
                           tcp_port=0, auto=False, host="127.0.0.1")
    svc.start()
    try:
        idle = [socket.create_connection(("127.0.0.1", svc.port)) for _ in range(sync.MAX_CONNECTIONS)]
        time.sleep(0.2)
        extra = socket.create_connection(("127.0.0.1", svc.port))
        extra.settimeout(2)
        assert extra.recv(10) == b""  # closed straight away: all slots taken
        time.sleep(1.2)  # the silent ones time out and free their slots
        for s in idle:
            s.settimeout(2)
            assert s.recv(1024) is not None
        ok = socket.create_connection(("127.0.0.1", svc.port))
        ch = sync.Channel(ok)
        ch.send({"t": "hello", "id": "stranger", "nonce": "AAAA"})
        with pytest.raises(sync.SyncError, match="not paired"):
            ch.recv()
        for s in idle + [extra, ok]:
            s.close()
    finally:
        svc.stop()


# --------------------------------------------------------------------------- live transcription
def test_live_session_never_queues_a_second_piece():
    from whispa.streaming import LiveSession

    sr = 16000
    session = LiveSession(lambda a, s: "x")
    pending = Future()
    session.parts.append(pending)
    calls = []
    session.poll(lambda start: calls.append(start) or np.zeros(sr * 60, np.float32), sr)
    assert calls == [] and len(session.parts) == 1  # waits for the busy piece
    pending.set_result("done")
    audio = np.concatenate([np.ones(sr * 25, np.float32) * 0.3, np.zeros(sr, np.float32),
                            np.ones(sr * 2, np.float32) * 0.3])
    session.poll(lambda start: audio[start:], sr)
    assert len(session.parts) == 2


# --------------------------------------------------------------------------- Ollama checksum
def test_ollama_without_checksum_is_refused(monkeypatch):
    import io

    from whispa import ai_setup

    release = {"assets": [{"name": "OllamaSetup.exe", "browser_download_url": "https://x/OllamaSetup.exe"}]}
    monkeypatch.setattr(ai_setup, "_get", lambda url, *a, **k: io.BytesIO(json.dumps(release).encode()))
    with pytest.raises(RuntimeError, match="no checksum"):
        ai_setup._release_asset("OllamaSetup.exe")
    release["assets"][0]["digest"] = "sha256:" + "a" * 64
    assert ai_setup._release_asset("OllamaSetup.exe")[1] == "a" * 64


# --------------------------------------------------------------------------- round 2
def test_names_that_only_look_local_are_remote():
    from whispa.llm import is_local_url

    for remote in ("http://127.attacker.example:11434", "http://localhost.evil.com",
                   "http://user@evil.com:11434", "file:///etc/passwd", "http://[::ffff:8.8.8.8]/"):
        assert not is_local_url(remote), remote
    for local in ("http://127.0.0.1:11434", "http://127.5.6.7", "http://[::1]:8080", "http://localhost"):
        assert is_local_url(local), local


def test_clearing_the_key_removes_it_from_the_store(tmp_path, monkeypatch):
    from whispa import config as config_mod
    from whispa import credentials

    store = {}

    class FakeKeyring:
        set_password = staticmethod(lambda s, a, k: store.__setitem__((s, a), k))
        get_password = staticmethod(lambda s, a: store.get((s, a)))
        delete_password = staticmethod(lambda s, a: store.pop((s, a)))

    monkeypatch.setitem(__import__("sys").modules, "keyring", FakeKeyring)
    monkeypatch.setattr(credentials, "_stored", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 3, "llm_cleanup": {"api_key": "sk-1"}}))
    cfg = config_mod.load(path)
    cfg.llm_cleanup.api_key = ""
    config_mod.save(cfg, path)
    assert store == {}
    monkeypatch.setattr(credentials, "_stored", None)
    assert config_mod.load(path).llm_cleanup.api_key == ""


def test_unreadable_keychain_never_deletes_the_key(tmp_path, monkeypatch):
    from whispa import config as config_mod
    from whispa import credentials

    deleted = []

    class Locked:
        @staticmethod
        def get_password(s, a):
            raise RuntimeError("locked")

        delete_password = staticmethod(lambda s, a: deleted.append(1))

    monkeypatch.setitem(__import__("sys").modules, "keyring", Locked)
    monkeypatch.setattr(credentials, "_stored", None)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config_version": 3, "llm_cleanup": {"api_key_stored": "keychain"}}))
    config_mod.load(path)
    assert deleted == [] and '"api_key_stored": "keychain"' in path.read_text()


def test_slow_client_is_cut_off_at_the_handshake_deadline(monkeypatch):
    from whispa import sync
    from whispa.learn import Dictionary

    monkeypatch.setattr(sync, "HANDSHAKE_S", 1.0)
    d = Path(tempfile.mkdtemp())
    svc = sync.SyncService(d, Dictionary(d / "d.json"), name=lambda: "PC", udp_port=None,
                           tcp_port=0, auto=False, host="127.0.0.1")
    svc.start()
    try:
        s = socket.create_connection(("127.0.0.1", svc.port))
        s.sendall(b"\x00\x00\x01\x00")  # a 256-byte frame, then a trickle
        t0 = time.monotonic()
        closed = False
        while time.monotonic() - t0 < 4:
            try:
                s.sendall(b"x")
            except OSError:
                closed = True
                break
            time.sleep(0.2)
        s.settimeout(1)
        try:
            data = s.recv(4096)
            closed = closed or data is not None
        except OSError:
            closed = True
        assert closed and time.monotonic() - t0 < 3.5
        s.close()
    finally:
        svc.stop()


def test_dictations_added_while_pruning_are_kept(tmp_path, monkeypatch):
    h = tmp_path / "history.jsonl"
    now = time.time()
    _write(h, [{"t": now - 400 * 86400, "text": "old"}] + [{"t": now, "text": f"n{i}"} for i in range(2000)])
    real = history._rewrite

    def slow(path, keep):  # an append arrives in the middle of the rewrite
        threading.Thread(target=history.append, args=(path, {"t": now, "text": "during"})).start()
        time.sleep(0.2)
        return real(path, keep)

    monkeypatch.setattr(history, "_rewrite", slow)
    assert history.prune(h, 365, now) == 1
    time.sleep(0.3)
    texts = [json.loads(x)["text"] for x in h.read_text().splitlines()]
    assert "during" in texts and "old" not in texts and len(texts) == 2001


def test_home_stats_and_list_read_cheaply(tmp_path):
    h = tmp_path / "history.jsonl"
    now = time.time()
    _write(h, [{"t": now - 86400 * 30, "seconds": 1, "text": "one two"},
               {"t": now, "seconds": 2, "text": "three four five"},
               {"t": now, "text": "rewrite it", "command": "x"}])
    history.totals_path(h).write_text('{"words": 10, "seconds": 5}')
    st = history.stats(h)
    assert st["words"] == 15 and st["seconds"] == 8 and st["week"] == 3
    assert [r["text"] for r in history.recent_rows(h, 2)] == ["three four five", "rewrite it"]
    assert [r["text"] for r in history.scan(h, lambda r: "two" in r["text"])] == ["one two"]
    history.delete_entry(h, now - 86400 * 30)
    assert history.stats(h)["words"] == 13


def test_recent_package_environments_are_not_pruned(tmp_path, monkeypatch):
    import importlib.util
    import os

    spec = importlib.util.spec_from_file_location("boot_under_test", Path("whispa/_boot.py").resolve())
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    rt = tmp_path
    for name in ("current", "previous"):
        (rt / "app" / name).mkdir(parents=True)
    (rt / "app" / "current" / "version.json").write_text('{"env": "envs/aaa"}')
    for name in ("aaa", "old", "new", ".building-ccc"):
        (rt / "envs" / name).mkdir(parents=True)
    week_ago = time.time() - 7 * 86400
    os.utime(rt / "envs" / "old", (week_ago, week_ago))
    os.utime(rt / "envs" / "aaa", (week_ago, week_ago))
    monkeypatch.setattr(boot, "RUNTIME", str(rt))
    monkeypatch.setattr(boot, "CURRENT", str(rt / "app" / "current"))
    monkeypatch.setattr(boot, "PREVIOUS", str(rt / "app" / "previous"))
    monkeypatch.setattr(boot, "NEXT", str(rt / "app" / "next"))
    monkeypatch.setattr(boot, "LOG", str(rt / "boot.log"))
    boot.prune_envs()
    assert sorted(p.name for p in (rt / "envs").iterdir()) == [".building-ccc", "aaa", "new"]


def test_recorder_stops_keeping_audio_past_its_hard_limit():
    from whispa.audio import Recorder

    r = Recorder.__new__(Recorder)
    r._lock = threading.Lock()
    r._recording, r.keep_open, r.samplerate, r.limit_s = True, False, 16000, 1.0
    r._chunks, r._samples, r.level = [], 0, 0.0
    block = np.zeros((1600, 1), np.float32)
    for _ in range(50):  # 5 s of audio
        r._callback(block, 1600, None, None)
    assert r._samples == 16000


# --------------------------------------------------------------------------- privacy review
def test_sync_never_lists_or_sends_kinds_switched_off_here(tmp_path):
    from whispa import sync
    from whispa.learn import Dictionary

    def svc(name, accepts):
        d = tmp_path / name
        d.mkdir()
        s = sync.SyncService(d, Dictionary(d / "d.json"), name=lambda: name, accepts=lambda: accepts,
                             udp_port=None, tcp_port=0, auto=False, host="127.0.0.1")
        s.start()
        return s

    pc, mac = svc("PC", []), svc("Mac", ["models", "voices"])  # the PC shares neither
    try:
        voice = pc.folders["voices"] / "me"
        voice.mkdir(parents=True)
        (voice / "voice.onnx").write_bytes(b"x")
        assert pc._local_files() == {"models": {}, "voices": {}}  # not even listed
        code = pc.start_pairing()
        mac.pair_with(code, "127.0.0.1", pc.port)
        assert mac.sync_with(pc.id).ok
        assert not (mac.folders["voices"] / "me").exists()
    finally:
        pc.stop()
        mac.stop()


def test_logs_keep_out_what_you_said(tmp_path, caplog):
    import logging

    from whispa.learn import Correction, CorrectionWatcher, Dictionary

    caplog.set_level(logging.INFO)
    d = Dictionary(tmp_path / "d.json")
    d.learn(Correction("cooper netties", "Kubernetes"))
    d.learn(Correction("cooper netties", "Kubernetes"))
    from whispa.llm import Polisher
    from whispa.config import LLMCleanup

    p = Polisher(LLMCleanup(enabled=True, when="always"))
    p.active = lambda wait=False: True
    p._complete = lambda *a, **k: "Here is a secret poem about the private launch plans"
    p.polish("write me a poem about the launch")
    assert "secret poem" not in caplog.text and "launch plans" not in caplog.text


def test_report_settings_leave_out_your_text():
    from whispa.config import Config
    from whispa.report import safe_settings

    cfg = Config()
    cfg.custom_mode_prompt = "Always mention Project Falcon"
    cfg.device_name = "Angus's MacBook"
    cfg.llm_cleanup.base_url = "https://ai.example.com"
    data = safe_settings(cfg)
    flat = str(data)
    assert "Falcon" not in flat and "Angus" not in flat and "example.com" not in flat


def test_one_click_setup_refuses_a_server_elsewhere():
    import threading

    from whispa import ai_setup
    from whispa.config import LLMCleanup

    with pytest.raises(RuntimeError, match="only installs the AI on this computer"):
        ai_setup.setup(LLMCleanup(base_url="https://ai.example.com"), lambda *a: None, threading.Event())


def test_pending_voice_delete_keeps_trying_until_gone(tmp_path, monkeypatch):
    from whispa import config as config_mod
    from whispa import privacy

    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    stuck = tmp_path / "training"
    stuck.mkdir()
    (stuck / "clip.wav").write_bytes(b"x")
    (tmp_path / "delete-voice-data").write_text("pending")
    monkeypatch.setattr(privacy.shutil, "rmtree", lambda *a, **k: None)  # "in use"
    privacy.finish_pending_delete()
    assert (tmp_path / "delete-voice-data").exists() and privacy.remaining_voice_data()
