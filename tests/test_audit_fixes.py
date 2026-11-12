"""Fixes from the November 2026 three-way audit: each test is one finding."""
import json
import socket
import tempfile
import types
from pathlib import Path

import pytest

from whispa import sync
from whispa.learn import Dictionary
from whispa.sync import SyncError, SyncService

GOOD_ID = "0123456789abcdef0123456789abcdef"


def _svc(tmp: Path, **kw) -> SyncService:
    tmp.mkdir(parents=True, exist_ok=True)
    return SyncService(tmp, Dictionary(tmp / "d.json"), name=lambda: "Desk", auto=False,
                       host="127.0.0.1", **kw)


# --- sync: a paired computer can't write outside the sync folder ---------------------------
@pytest.mark.parametrize("bad", ["../config", "..\\config", "", "x" * 32, GOOD_ID.upper(), None, []])
def test_peer_ids_must_be_ours(tmp_path, bad):
    svc = _svc(tmp_path)
    assert not sync.valid_id(bad)
    with pytest.raises(SyncError):
        svc._base_path(bad)
    with pytest.raises(SyncError):
        svc._add_peer(bad, "Evil", b"k" * 32, "1.2.3.4:5")
    assert not (tmp_path / "config.json").exists()


def test_saved_peers_with_bad_ids_are_ignored(tmp_path):
    svc = _svc(tmp_path)
    data = json.loads(svc._path.read_text(encoding="utf-8"))
    data["peers"] = {"../config": {"name": "Evil", "key": "a2V5", "addr": ""}}
    svc._path.write_text(json.dumps(data), encoding="utf-8")
    assert _svc(tmp_path).peers == {}


# --- sync: sizes can't be negative (that dodged the 8 GB limit) -----------------------------
class _Head:
    def __init__(self, files):
        self.files = files

    def expect(self, kind):
        return {"files": self.files}


@pytest.mark.parametrize("files", [
    [{"path": "big", "size": 9 << 30}, {"path": "neg", "size": -(9 << 30)}],
    [{"path": "a", "size": "12"}],
    [{"path": "a", "size": 1}, {"path": "a", "size": 1}],
    [{"path": "a", "size": True}],
    "not a list",
])
def test_bad_file_lists_are_refused(tmp_path, files):
    with pytest.raises(SyncError):
        sync.recv_folder(_Head(files), tmp_path, "model")
    assert not any(tmp_path.iterdir())


# --- sync: a malformed packet doesn't stop discovery ----------------------------------------
def test_discovery_survives_junk_packets(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    svc = _svc(tmp_path / "a", udp_port=port, tcp_port=0)
    svc.start()
    try:
        svc.start_pairing()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(3)
            for junk in ({"murmr": "who", "id": []}, {"murmr": "who", "id": {"a": 1}}, [1, 2], "x"):
                s.sendto(json.dumps(junk).encode(), ("127.0.0.1", port))
            s.sendto(json.dumps({"murmr": "who", "id": GOOD_ID}).encode(), ("127.0.0.1", port))
            reply = json.loads(s.recvfrom(2048)[0])
        assert reply["name"] == "Desk"  # still listening after the junk
    finally:
        svc.stop()


# --- the text never goes to a window you switched to while the AI worked -------------------
def test_switching_windows_while_polishing(monkeypatch):
    from whispa import focus
    from whispa.app import MurmrApp

    f = focus.Focus(is_text=True, app="Notes")
    monkeypatch.setattr(focus, "same_window", lambda t: True)
    assert MurmrApp._moved_away(f) == ""
    monkeypatch.setattr(focus, "same_window", lambda t: None)  # can't tell: paste as before
    assert MurmrApp._moved_away(f) == ""
    monkeypatch.setattr(focus, "same_window", lambda t: False)
    monkeypatch.setattr(focus, "restore", lambda t: True)  # brought back: paste there
    assert MurmrApp._moved_away(f) == ""
    monkeypatch.setattr(focus, "restore", lambda t: False)  # couldn't: clipboard instead
    assert "switched away from Notes" in MurmrApp._moved_away(f)


# --- passwords aren't kept ----------------------------------------------------------------
def test_password_dictation_is_not_remembered(tmp_path):
    from whispa import focus
    from whispa.app import MurmrApp

    ns = types.SimpleNamespace(_focus=focus.Focus(is_text=True, app="Bank", is_password=True),
                               history=[], cfg=types.SimpleNamespace(save_history=True),
                               history_path=tmp_path / "h.jsonl")
    MurmrApp._remember(ns, "hunter2", "hunter2", 1.0)
    assert ns.history == [] and not ns.history_path.exists()


# --- Settings never waits for the AI check ---------------------------------------------------
def test_llm_status_answers_at_once():
    from whispa.app import MurmrApp

    started = []
    ns = types.SimpleNamespace(cfg=types.SimpleNamespace(llm_cleanup=types.SimpleNamespace(enabled="auto")),
                               _check_llm_status=lambda: started.append(1))
    assert MurmrApp.llm_status(ns) == (False, "Checking…")
    for _ in range(50):
        if started:
            break
        import time
        time.sleep(0.01)
    assert started  # the real check runs in the background
    ns._llm_status = (True, "Active")
    assert MurmrApp.llm_status(ns) == (True, "Active")


# --- an old edit watch can't end the next one ----------------------------------------------
def test_stale_watch_does_not_end_the_new_one(tmp_path):
    from whispa.learn import CorrectionWatcher

    class F:
        readable = True

        def read_text(self):
            return "x"

    w = CorrectionWatcher(Dictionary(tmp_path / "d.json"))
    w.watch(F(), "", "first", None)
    old = w._session
    w.watch(F(), "", "second", None)
    w.finish(old)  # what a slow poll of the first dictation would do
    assert w._session is not None and w._session["inserted"] == "second"
    w.finish()


# --- Cancel stops only Murmr's own training ------------------------------------------------
@pytest.mark.skipif(__import__("sys").platform != "win32", reason="training runs in WSL on Windows")
def test_training_runs_in_its_own_process_group(monkeypatch, tmp_path):
    from whispa import config as config_mod
    from whispa import tuning

    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    job = tuning.Job("recognition")
    job.command()
    script = (tmp_path / "tuning" / "recognition.sh").read_text()
    assert "setsid bash ./pipeline.sh" in script and job.pid_file in script
    calls = []
    monkeypatch.setattr(tuning, "_wsl", lambda args, timeout=30: calls.append(args))
    job._proc = types.SimpleNamespace(poll=lambda: None, terminate=lambda: None, kill=lambda: None,
                                      wait=lambda timeout=None: 0)
    job.cancel()
    cmd = " ".join(calls[0])
    assert "pkill" not in cmd and 'kill -TERM -- "-$p"' in cmd and job.pid_file in cmd


# --- the downloaded base voice is pinned ---------------------------------------------------
def test_voice_base_checkpoint_is_pinned():
    src = (Path(__file__).parent.parent / "scripts" / "voice_finetune" / "train_voice.py").read_text()
    assert "BASE_SHA256 = " in src and "revision=BASE_REVISION" in src
    assert "list_repo_files" not in src  # never "whatever is newest"
