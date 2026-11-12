"""Sync between computers: merging, the encrypted channel, pairing and a full sync."""
import socket
import tempfile
from pathlib import Path

import pytest

from whispa import sync
from whispa.learn import Dictionary
from whispa.sync import Channel, SyncError, SyncService, merge3


def test_merge3_takes_each_sides_changes_and_deletions():
    base = {"a": "1", "b": "1", "c": "1"}
    local = {"a": "2", "b": "1"}  # changed a, deleted c
    remote = {"a": "1", "b": "3", "c": "1", "d": "4"}  # changed b, added d
    assert merge3(base, local, remote) == {"a": "2", "b": "3", "d": "4"}


def test_merge3_conflict_keeps_local_and_data_over_deletion():
    assert merge3({"a": "1"}, {"a": "2"}, {"a": "3"}) == {"a": "2"}
    assert merge3({"a": "1"}, {}, {"a": "3"}) == {"a": "3"}
    assert merge3({}, {"x": "1"}, {"y": "2"}) == {"x": "1", "y": "2"}  # first sync: union


def test_codes():
    code = sync.new_code()
    assert len(code) == 8 and sync.normalize_code(sync.pretty_code(code).lower()) == code
    assert sync.normalize_code("ol1i-ABCD") == "0111ABCD"


def test_channel_encrypts_and_detects_tampering():
    a, b = socket.socketpair()
    ca, cb = Channel(a), Channel(b)
    k1, k2 = b"1" * 32, b"2" * 32
    ca.secure(k1, k2)
    cb.secure(k2, k1)
    ca.send({"t": "x", "v": "secret words"})
    ca.send(b"\x00\x01binary")
    assert cb.recv() == {"t": "x", "v": "secret words"}
    assert cb.recv() == b"\x00\x01binary"
    # what's on the wire isn't readable, and a changed byte is rejected
    raw_a, raw_b = socket.socketpair()
    c = Channel(raw_a)
    c.secure(k1, k2)
    c.send({"t": "x", "v": "secret words"})
    wire = raw_b.recv(1000)
    assert b"secret" not in wire
    tampered = wire[:-1] + bytes([wire[-1] ^ 1])
    x, y = socket.socketpair()
    x.sendall(tampered)
    cy = Channel(y)
    cy.secure(k2, k1)
    with pytest.raises(SyncError):
        cy.recv()


def test_unsafe_paths_rejected():
    for bad in ("../x", "/etc/passwd", "a/../../b", "C:/x"):
        with pytest.raises(SyncError):
            sync._safe_rel(bad)
    assert sync._safe_rel("a/b.onnx").as_posix() == "a/b.onnx"


def _service(tmp: Path, name: str, accepts=("models",)) -> SyncService:
    d = tmp / name
    d.mkdir()
    svc = SyncService(d, Dictionary(d / "dictionary.json"), name=lambda: name,
                      accepts=lambda: list(accepts), udp_port=None, tcp_port=0, auto=False, host="127.0.0.1")
    svc.start()
    return svc


@pytest.fixture
def pair():
    tmp = Path(tempfile.mkdtemp())
    pc, mac = _service(tmp, "PC"), _service(tmp, "Mac")
    yield pc, mac
    pc.stop()
    mac.stop()


def _pair(pc, mac):
    code = pc.start_pairing()
    assert mac.pair_with(code.lower(), "127.0.0.1", pc.port) == "PC"
    assert pc.peers[mac.id].key == mac.peers[pc.id].key and pc.pairing is None


def test_wrong_codes_end_the_pairing_window(pair):
    pc, mac = pair
    pc.start_pairing()
    for _ in range(sync.PAIR_MAX_FAILS):
        with pytest.raises(SyncError, match="didn't match"):
            mac.pair_with("ZZZZZZZZ", "127.0.0.1", pc.port)
    assert pc.pairing is None and not pc.peers and not mac.peers


def test_sync_dictionary_and_models(pair):
    pc, mac = pair
    _pair(pc, mac)
    pc.dictionary.add_term("Kubernetes")
    pc.dictionary.set_correction("cooper netties", "Kubernetes")
    mac.dictionary.set_snippet("my address", "1 Example St")
    model = pc.folders["models"] / "tuned-2026-11-12"
    (model / "sub").mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (model / "sub" / "encoder.onnx").write_bytes(bytes(range(256)) * 9000)  # > 1 chunk

    st = mac.sync_with(pc.id)
    assert st.ok, st.text
    for svc in (pc, mac):
        terms, corrections = svc.dictionary.snapshot()
        assert "Kubernetes" in terms and corrections["cooper netties"] == "Kubernetes"
        assert svc.dictionary.snippets_snapshot() == {"my address": "1 Example St"}
    got = mac.folders["models"] / "tuned-2026-11-12" / "sub" / "encoder.onnx"
    assert got.read_bytes() == bytes(range(256)) * 9000

    # a deletion on one side carries over; a model deleted on purpose doesn't come back
    pc.dictionary.remove_term("Kubernetes")
    import shutil
    shutil.rmtree(mac.folders["models"] / "tuned-2026-11-12")
    assert pc.sync_with(mac.id).ok
    assert "Kubernetes" not in mac.dictionary.snapshot()[0]
    assert not (mac.folders["models"] / "tuned-2026-11-12").exists()
    assert (pc.folders["models"] / "tuned-2026-11-12").exists()


def test_voices_only_go_where_accepted(pair):
    pc, mac = pair
    _pair(pc, mac)
    v = pc.folders["voices"] / "me"
    v.mkdir(parents=True)
    (v / "voice.onnx").write_bytes(b"x")
    assert mac.sync_with(pc.id).ok
    assert not (mac.folders["voices"] / "me").exists()  # mac accepts models only


def test_unpaired_computer_gets_nothing(pair):
    pc, mac = pair
    from whispa.sync import Peer

    mac.peers["fake"] = Peer(pc.id, "PC", b"k" * 32, f"127.0.0.1:{pc.port}")
    mac.peers[pc.id] = mac.peers.pop("fake")
    st = mac.sync_with(pc.id, retries=0)
    assert not st.ok and "pair again" in st.text


def test_discovery_reply():
    import json

    tmp = Path(tempfile.mkdtemp())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    d = tmp / "a"
    d.mkdir()
    svc = SyncService(d, Dictionary(d / "d.json"), name=lambda: "Desk", udp_port=port, tcp_port=0,
                      auto=False, host="127.0.0.1")
    svc.start()
    try:
        svc.start_pairing()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(3)
            s.sendto(json.dumps({"murmr": "who", "id": "0123456789abcdef0123456789abcdef"}).encode(), ("127.0.0.1", port))
            reply = json.loads(s.recvfrom(2048)[0])
        assert reply["name"] == "Desk" and reply["port"] == svc.port and reply["pairing"] is True
        assert "key" not in json.dumps(reply)
    finally:
        svc.stop()
