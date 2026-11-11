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
