"""Sync with your other computers on the same network: direct, encrypted, no cloud.

What's shared:
* your Dictionary: words, corrections and snippets (merged both ways; a deletion on one
  computer deletes on the others);
* speech models tuned to your voice (``<config>/models``), e.g. trained on the PC and
  used on the Mac (optional, on by default);
* read-aloud voices made from your recordings (``<config>/voices``; optional, off by default).
Never shared: recordings, dictation history, meetings, settings.

How:
* Computers find each other with a UDP broadcast on the local network (port 47811) and
  talk over TCP (port 47812, or any free port). Only while sync is switched on.
* Pairing: one computer shows an 8-character code, you type it on the other. They agree
  a key with X25519, confirmed by the code (scrypt-hardened, so a recorded exchange can't
  be brute-forced cheaply). Five wrong guesses end the pairing window.
* Each sync: mutual proof of the paired key (HMAC), then everything is encrypted with
  ChaCha20-Poly1305 under fresh keys. Unpaired devices get nothing but "not paired".
* Merging is 3-way against what both had after the last sync, so edits on either side win
  over the unchanged side and deletions carry over. If both changed the same entry, the
  computer that started the sync wins.

Files: ``<config>/sync.json`` (this computer's id and the paired computers' keys) and
``<config>/sync/<peer>.json`` (the state after the last sync with each).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

UDP_PORT = 47811
TCP_PORT = 47812
VERSION = 1
CODE_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32: no I, L, O, U
CODE_LEN = 8
PAIR_WINDOW_S = 300
PAIR_MAX_FAILS = 5
SYNC_EVERY_S = 600
CHUNK = 1 << 20
MAX_FRAME = 8 << 20
MAX_FRAME_PLAIN = 64 << 10  # before encryption is set up
MAX_FOLDER_BYTES = 8 << 30
KINDS = ("models", "voices")
_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,120}$")


class SyncError(Exception):
    pass


# --------------------------------------------------------------------------- merging
def merge3(base: dict, local: dict, remote: dict) -> dict:
    """Per key: whichever side changed it since ``base`` wins; both changed: ``local``.
    A missing key is a deletion (or never existed)."""
    out = {}
    for key in set(base) | set(local) | set(remote):
        b, mine, theirs = base.get(key), local.get(key), remote.get(key)
        if mine == theirs:
            value = mine
        elif mine == b:
            value = theirs
        elif theirs == b:
            value = mine
        else:  # both changed it: keep the starting side's edit, and data over a deletion
            value = mine if mine is not None else theirs
        if value is not None:
            out[key] = value
    return out


def changes(before: dict, after: dict) -> int:
    return sum(1 for k in set(before) | set(after) if before.get(k) != after.get(k))


def dictionary_state(d) -> dict[str, str]:
    """A Dictionary as flat keys: term:<word>, corr:<heard>, snip:<trigger>."""
    terms, corrections = d.snapshot()
    snippets = d.snippets_snapshot()
    state = {f"term:{t}": "1" for t in terms}
    state.update({f"corr:{k}": v for k, v in corrections.items()})
    state.update({f"snip:{k}": v for k, v in snippets.items()})
    return state


def apply_dictionary_state(d, state: dict[str, str]) -> None:
    with d._lock:
        d.snapshot()  # pick up hand edits first
        wanted = [k[5:] for k in state if k.startswith("term:")]
        d.terms = [t for t in d.terms if t in wanted] + [t for t in wanted if t not in d.terms]
        d.corrections = {k[5:]: v for k, v in state.items() if k.startswith("corr:")}
        d.snippets = {k[5:]: v for k, v in state.items() if k.startswith("snip:")}
        for heard in d.corrections:
            d.pending.pop(heard, None)
        d.save()


def folder_manifest(root: Path) -> dict[str, int]:
    """name -> total bytes, for each complete folder in ``root``."""
    out = {}
    if not root.is_dir():
        return out
    for d in root.iterdir():
        if d.is_dir() and not d.name.startswith(".") and _NAME_OK.match(d.name):
            try:
                out[d.name] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            except OSError:
                continue
    return out


def new_code() -> str:
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LEN))


def normalize_code(code: str) -> str:
    code = code.upper().replace("O", "0").replace("I", "1").replace("L", "1")
    return "".join(c for c in code if c in CODE_ALPHABET)


def pretty_code(code: str) -> str:
    return f"{code[:4]}-{code[4:]}"


# --------------------------------------------------------------------------- crypto
def _hkdf(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)


def _mac(key: bytes, *parts: bytes) -> bytes:
    h = hmac.new(key, digestmod=hashlib.sha256)
    for p in parts:
        h.update(struct.pack(">I", len(p)) + p)
    return h.digest()


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode())


class Channel:
    """Length-prefixed frames over a socket; JSON ("J") or bytes ("B"); encrypted once
    ``secure()`` is called (a separate key and counter nonce per direction)."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._tx = self._rx = None
        self._ntx = self._nrx = 0

    def secure(self, tx_key: bytes, rx_key: bytes) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

        self._tx, self._rx = ChaCha20Poly1305(tx_key), ChaCha20Poly1305(rx_key)

    def send(self, msg) -> None:
        payload = (b"B" + bytes(msg)) if isinstance(msg, (bytes, bytearray, memoryview)) else \
            b"J" + json.dumps(msg, separators=(",", ":")).encode()
        if self._tx is not None:
            payload = self._tx.encrypt(self._ntx.to_bytes(12, "big"), payload, None)
            self._ntx += 1
        self.sock.sendall(struct.pack(">I", len(payload)) + payload)

    def _read(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(n - len(buf), 1 << 20))
            if not chunk:
                raise SyncError("connection closed")
            buf += chunk
        return bytes(buf)

    def recv(self):
        (n,) = struct.unpack(">I", self._read(4))
        if n > (MAX_FRAME + 64 if self._rx is not None else MAX_FRAME_PLAIN):
            raise SyncError("frame too large")
        payload = self._read(n)
        if self._rx is not None:
            from cryptography.exceptions import InvalidTag

            try:
                payload = self._rx.decrypt(self._nrx.to_bytes(12, "big"), payload, None)
            except InvalidTag:
                raise SyncError("message failed authentication") from None
            self._nrx += 1
        if payload[:1] == b"B":
            return payload[1:]
        try:
            msg = json.loads(payload[1:])
        except ValueError:
            raise SyncError("bad message") from None
        if not isinstance(msg, dict):
            raise SyncError("bad message")
        if msg.get("t") == "error":
            raise SyncError(msg.get("why", "error"))
        return msg

    def expect(self, kind: str) -> dict:
        msg = self.recv()
        if not isinstance(msg, dict) or msg.get("t") != kind:
            raise SyncError(f"expected {kind}")
        return msg


# --------------------------------------------------------------------------- folders
def _safe_rel(path: str) -> Path:
    from pathlib import PurePosixPath

    text = path.replace("\\", "/")
    p = PurePosixPath(text)
    if text.startswith("/") or ".." in p.parts or not p.parts or             any(":" in part or part in ("", ".") for part in p.parts):
        raise SyncError(f"unsafe path {path!r}")
    return Path(*p.parts)


def send_folder(ch: Channel, folder: Path) -> None:
    files = sorted(f for f in folder.rglob("*") if f.is_file())
    ch.send({"t": "folder", "files": [{"path": f.relative_to(folder).as_posix(),
                                        "size": f.stat().st_size} for f in files]})
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
                ch.send(chunk)
        ch.send({"t": "filedone", "sha": h.hexdigest()})


def recv_folder(ch: Channel, root: Path, name: str) -> Path:
    if not _NAME_OK.match(name):
        raise SyncError(f"bad name {name!r}")
    head = ch.expect("folder")
    specs = head.get("files") or []
    total = sum(int(s.get("size", 0)) for s in specs)
    if total > MAX_FOLDER_BYTES:
        raise SyncError("too large")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total + (1 << 30):
        raise SyncError("not enough free disk space")
    tmp = root / f".incoming-{name}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        for spec in specs:
            out = tmp / _safe_rel(str(spec["path"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            left = int(spec["size"])
            h = hashlib.sha256()
            with open(out, "wb") as fh:
                while left > 0:
                    chunk = ch.recv()
                    if not isinstance(chunk, bytes) or len(chunk) > left:
                        raise SyncError("bad file data")
                    fh.write(chunk)
                    h.update(chunk)
                    left -= len(chunk)
            if ch.expect("filedone").get("sha") != h.hexdigest():
                raise SyncError("file damaged in transfer")
        final = root / name
        if final.exists():
            raise SyncError(f"{name} already exists")
        tmp.rename(final)
        return final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- network helpers
def local_addresses() -> list[str]:
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    try:  # the address used for the default route (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            found.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(a for a in found if not a.startswith(("127.", "169.254.")))


def broadcast_targets() -> list[str]:
    out = ["255.255.255.255"]
    for ip in local_addresses():
        out.append(".".join(ip.split(".")[:3] + ["255"]))
    return list(dict.fromkeys(out))


@dataclass
class Peer:
    id: str
    name: str
    key: bytes
    addr: str = ""  # host:port last seen
    last_sync: float = 0.0


@dataclass
class Status:
    when: float = 0.0
    ok: bool = True
    text: str = ""


# --------------------------------------------------------------------------- the service
class SyncService:
    """Runs while sync is on: answers discovery and sync requests, and syncs with each
    paired computer every 10 minutes and soon after the Dictionary changes."""

    def __init__(self, cdir: Path, dictionary, name: Callable[[], str],
                 accepts: Callable[[], list[str]] = lambda: ["models"],
                 on_event: Callable[[str, dict], None] = lambda kind, info: None,
                 udp_port: int | None = UDP_PORT, tcp_port: int = TCP_PORT,
                 folders: dict[str, Path] | None = None, auto: bool = True, host: str = ""):
        self.cdir = cdir
        self.dictionary = dictionary
        self.name = name
        self.accepts = accepts
        self.on_event = on_event
        self.udp_port = udp_port
        self.tcp_port = tcp_port
        self.folders = folders or {"models": cdir / "models", "voices": cdir / "voices"}
        self.auto = auto
        self.host = host  # "" = every network; tests use 127.0.0.1
        self.status: dict[str, Status] = {}
        self.pairing: dict | None = None  # {"code", "until", "fails"} while showing a code
        self._lock = threading.Lock()  # one sync or pairing at a time
        self._file_lock = threading.Lock()
        self._stop = threading.Event()
        self._kick = threading.Event()
        self._threads: list[threading.Thread] = []
        self._tcp: socket.socket | None = None
        self._udp: socket.socket | None = None
        self.port = 0
        self._dict_mtime = 0.0
        self._load()

    # -- identity & peers ------------------------------------------------
    @property
    def _path(self) -> Path:
        return self.cdir / "sync.json"

    def _load(self) -> None:
        data = {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.id = data.get("id") or uuid.uuid4().hex
        self.peers: dict[str, Peer] = {}
        for pid, p in (data.get("peers") or {}).items():
            try:
                self.peers[pid] = Peer(pid, p["name"], _unb64(p["key"]), p.get("addr", ""),
                                       p.get("last_sync", 0.0))
            except (KeyError, ValueError):
                continue
        if not data.get("id"):
            self._save()

    def _save(self) -> None:
        with self._file_lock:
            self.cdir.mkdir(parents=True, exist_ok=True)
            data = {"_help": "Murmr sync: this computer's id and the computers paired with it. "
                             "The keys are secret: don't share this file.",
                    "id": self.id,
                    "peers": {p.id: {"name": p.name, "key": _b64(p.key), "addr": p.addr,
                                     "last_sync": p.last_sync} for p in self.peers.values()}}
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, self._path)

    def _base_path(self, peer_id: str) -> Path:
        return self.cdir / "sync" / f"{peer_id}.json"

    def _base(self, peer_id: str) -> dict:
        try:
            return json.loads(self._base_path(peer_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_base(self, peer_id: str, state: dict, files: dict) -> None:
        p = self._base_path(peer_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"state": state, "files": files}, ensure_ascii=False),
                     encoding="utf-8")

    def forget(self, peer_id: str) -> None:
        self.peers.pop(peer_id, None)
        self.status.pop(peer_id, None)
        self._save()
        try:
            self._base_path(peer_id).unlink()
        except OSError:
            pass

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._stop.clear()
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp.bind((self.host, self.tcp_port))
        except OSError:
            tcp.bind((self.host, 0))  # the usual port is taken
        tcp.listen(4)
        tcp.settimeout(1.0)
        self._tcp = tcp
        self.port = tcp.getsockname()[1]
        self._threads = [threading.Thread(target=self._accept_loop, name="sync-tcp", daemon=True)]
        if self.udp_port is not None:
            try:
                udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                udp.bind((self.host, self.udp_port))
                udp.settimeout(1.0)
                self._udp = udp
                self._threads.append(threading.Thread(target=self._udp_loop, name="sync-udp",
                                                      daemon=True))
            except OSError as exc:
                log.warning("Sync: discovery port %s unavailable: %s", self.udp_port, exc)
        if self.auto:
            self._threads.append(threading.Thread(target=self._schedule_loop, name="sync",
                                                  daemon=True))
        for t in self._threads:
            t.start()
        log.info("Sync on: port %d, %d paired computer(s)", self.port, len(self.peers))

    def stop(self) -> None:
        self._stop.set()
        self._kick.set()
        for s in (self._tcp, self._udp):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
        for t in self._threads:
            t.join(timeout=3)
        self._tcp = self._udp = None
        self.pairing = None

    def sync_soon(self) -> None:
        self._kick.set()

    # -- discovery -------------------------------------------------------
    def _hello(self) -> dict:
        return {"murmr": "here", "v": VERSION, "id": self.id, "name": self.name(), "port": self.port,
                "pairing": bool(self.pairing and self.pairing["until"] > time.time())}

    def _udp_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, addr = self._udp.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            if isinstance(msg, dict) and msg.get("murmr") == "who" and msg.get("id") != self.id:
                try:
                    self._udp.sendto(json.dumps(self._hello()).encode(), addr)
                except OSError:
                    pass

    def discover(self, timeout: float = 1.5) -> list[dict]:
        """Murmr computers on this network with sync on: [{id, name, host, port, pairing}]."""
        found: dict[str, dict] = {}
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(0.25)
            query = json.dumps({"murmr": "who", "v": VERSION, "id": self.id}).encode()
            for target in broadcast_targets():
                try:
                    s.sendto(query, (target, self.udp_port or UDP_PORT))
                except OSError:
                    pass
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                try:
                    data, addr = s.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    break
                try:
                    msg = json.loads(data)
                except ValueError:
                    continue
                if isinstance(msg, dict) and msg.get("murmr") == "here" and msg.get("id") != self.id:
                    found[msg["id"]] = {"id": msg["id"], "name": str(msg.get("name", "?"))[:60],
                                        "host": addr[0], "port": int(msg.get("port", TCP_PORT)),
                                        "pairing": bool(msg.get("pairing"))}
        return list(found.values())

    @staticmethod
    def _connect(host: str, port: int, timeout: float = 4.0) -> socket.socket:
        s = socket.create_connection((host, port), timeout=timeout)
        s.settimeout(60)
        return s

    # -- serving ---------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._tcp.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._serve, args=(conn, addr), name="sync-serve",
                             daemon=True).start()

    def _serve(self, conn: socket.socket, addr) -> None:
        conn.settimeout(60)
        ch = Channel(conn)
        try:
            msg = ch.recv()
            if not isinstance(msg, dict):
                raise SyncError("bad hello")
            if msg.get("t") == "pair":
                self._serve_pair(ch, msg, addr)
            elif msg.get("t") == "hello":
                if not self._lock.acquire(timeout=5):
                    ch.send({"t": "error", "why": "busy"})
                    return
                try:
                    self._serve_sync(ch, msg, addr)
                finally:
                    self._lock.release()
            else:
                ch.send({"t": "error", "why": "unknown request"})
        except (SyncError, OSError, ValueError, KeyError) as exc:
            log.info("Sync request from %s ended: %s", addr[0], exc)
            try:
                ch.send({"t": "error", "why": str(exc)[:200]})
            except OSError:
                pass
        finally:
            conn.close()

    # -- pairing ---------------------------------------------------------
    def start_pairing(self) -> str:
        code = new_code()
        self.pairing = {"code": code, "until": time.time() + PAIR_WINDOW_S, "fails": 0}
        return code

    def stop_pairing(self) -> None:
        self.pairing = None

    @staticmethod
    def _pair_keys(code: str, shared: bytes, n_client: bytes, n_server: bytes,
                   ids: list[str]) -> tuple[bytes, bytes]:
        salt = hashlib.scrypt(normalize_code(code).encode(), salt=b"murmr-pair" + n_client + n_server,
                              n=1 << 14, r=8, p=1, dklen=32)
        okm = _hkdf(shared, salt, b"murmr pair v1|" + "|".join(sorted(ids)).encode(), 64)
        return okm[:32], okm[32:]  # confirm key, link key

    def _serve_pair(self, ch: Channel, msg: dict, addr) -> None:
        from cryptography.hazmat.primitives.asymmetric.x25519 import (X25519PrivateKey,
                                                                      X25519PublicKey)
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        pairing = self.pairing
        if not pairing or pairing["until"] < time.time():
            raise SyncError("this computer isn't showing a pairing code")
        priv = X25519PrivateKey.generate()
        mine = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        nonce = secrets.token_bytes(16)
        ch.send({"t": "pair_ok", "id": self.id, "name": self.name(), "pub": _b64(mine),
                 "nonce": _b64(nonce)})
        theirs = _unb64(msg["pub"])
        shared = priv.exchange(X25519PublicKey.from_public_bytes(theirs))
        transcript = theirs + mine + _unb64(msg["nonce"]) + nonce
        confirm, link = self._pair_keys(pairing["code"], shared, _unb64(msg["nonce"]), nonce,
                                        [self.id, msg["id"]])
        proof = ch.expect("confirm")
        if not hmac.compare_digest(_unb64(proof["mac"]), _mac(confirm, b"client", transcript)):
            pairing["fails"] += 1
            if pairing["fails"] >= PAIR_MAX_FAILS:
                self.pairing = None
                self.on_event("pair_failed", {"why": "Too many wrong codes. Show a new code."})
            raise SyncError("wrong code")
        ch.send({"t": "confirm", "mac": _b64(_mac(confirm, b"server", transcript))})
        self._add_peer(msg["id"], str(msg.get("name", "?"))[:60], link, f"{addr[0]}:{msg.get('port', 0)}")

    def pair_with(self, code: str, host: str | None = None, port: int | None = None) -> str:
        """Pair with the computer showing ``code``. Returns its name. Raises SyncError."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import (X25519PrivateKey,
                                                                      X25519PublicKey)
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        code = normalize_code(code)
        if len(code) != CODE_LEN:
            raise SyncError(f"The code has {CODE_LEN} letters and numbers.")
        if host:
            targets = [(host, port or TCP_PORT)]
        else:
            targets = [(d["host"], d["port"]) for d in self.discover() if d["pairing"]]
            if not targets:
                raise SyncError("No computer showing a code was found on this network. Check "
                                "both are on the same Wi-Fi, and that the firewall allows Murmr.")
        last: Exception | None = None
        for h, p in targets:
            try:
                with self._connect(h, p) as sock:
                    ch = Channel(sock)
                    priv = X25519PrivateKey.generate()
                    mine = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
                    nonce = secrets.token_bytes(16)
                    ch.send({"t": "pair", "v": VERSION, "id": self.id, "name": self.name(),
                             "pub": _b64(mine), "nonce": _b64(nonce), "port": self.port})
                    reply = ch.expect("pair_ok")
                    theirs = _unb64(reply["pub"])
                    shared = priv.exchange(X25519PublicKey.from_public_bytes(theirs))
                    transcript = mine + theirs + nonce + _unb64(reply["nonce"])
                    confirm, link = self._pair_keys(code, shared, nonce, _unb64(reply["nonce"]),
                                                    [self.id, reply["id"]])
                    ch.send({"t": "confirm", "mac": _b64(_mac(confirm, b"client", transcript))})
                    proof = ch.expect("confirm")
                    if not hmac.compare_digest(_unb64(proof["mac"]),
                                               _mac(confirm, b"server", transcript)):
                        raise SyncError("the other computer couldn't prove the code")
                    name = str(reply.get("name", "?"))[:60]
                    self._add_peer(reply["id"], name, link, f"{h}:{p}")
                    return name
            except (OSError, SyncError, KeyError, ValueError) as exc:
                last = exc
        if isinstance(last, SyncError) and str(last) == "wrong code":
            raise SyncError("That code didn't match. Check it and try again.")
        raise SyncError(f"Pairing failed: {last}")

    def _add_peer(self, pid: str, name: str, key: bytes, addr: str) -> None:
        self.peers[pid] = Peer(pid, name, key, addr)
        self.pairing = None
        self._save()
        log.info("Sync: paired with %s", name)
        self.on_event("paired", {"id": pid, "name": name})
        self.sync_soon()

    # -- syncing ---------------------------------------------------------
    def _session_keys(self, key: bytes, n_init: bytes, n_resp: bytes) -> tuple[bytes, bytes]:
        okm = _hkdf(key, n_init + n_resp, b"murmr session v1", 64)
        return okm[:32], okm[32:]  # initiator -> responder, responder -> initiator

    def _local_files(self) -> dict[str, dict[str, int]]:
        return {kind: folder_manifest(self.folders[kind]) for kind in KINDS}

    def _serve_sync(self, ch: Channel, msg: dict, addr) -> None:
        peer = self.peers.get(str(msg.get("id")))
        if peer is None:
            raise SyncError("not paired")
        n_init = _unb64(msg["nonce"])
        n_resp = secrets.token_bytes(16)
        ch.send({"t": "hello", "id": self.id, "nonce": _b64(n_resp),
                 "mac": _b64(_mac(peer.key, b"responder", n_init, n_resp))})
        auth = ch.expect("auth")
        if not hmac.compare_digest(_unb64(auth["mac"]), _mac(peer.key, b"initiator", n_init, n_resp)):
            raise SyncError("not paired")
        to_peer, from_peer = self._session_keys(peer.key, n_init, n_resp)
        ch.secure(tx_key=from_peer, rx_key=to_peer)
        if msg.get("port"):
            peer.addr = f"{addr[0]}:{msg['port']}"

        sent = dictionary_state(self.dictionary)
        ch.expect("state")
        ch.send({"t": "state", "data": sent, "files": self._local_files(), "accepts": self.accepts(),
                 "name": self.name()})
        got_files: list[tuple[str, str]] = []
        while True:
            req = ch.recv()
            if not isinstance(req, dict):
                raise SyncError("bad request")
            t = req.get("t")
            if t == "merged":
                now = dictionary_state(self.dictionary)
                final = merge3(sent, now, req["data"])
                if final != now:
                    apply_dictionary_state(self.dictionary, final)
                self._save_base(peer.id, req["data"], req.get("files") or {})
                ch.send({"t": "ok"})
                n = changes(now, final)
                peer.last_sync = time.time()
                peer.name = str(req.get("name") or peer.name)[:60]
                self._save()
                self._report(peer, n, got_files)
                return
            if t == "get" and req.get("kind") in KINDS:
                folder = self.folders[req["kind"]] / str(req.get("name", ""))
                if not _NAME_OK.match(str(req.get("name", ""))) or not folder.is_dir():
                    raise SyncError("no such folder")
                send_folder(ch, folder)
            elif t == "put" and req.get("kind") in KINDS and req["kind"] in self.accepts():
                recv_folder(ch, self.folders[req["kind"]], str(req.get("name", "")))
                got_files.append((req["kind"], req["name"]))
                ch.send({"t": "ok"})
            else:
                raise SyncError("unexpected request")

    def sync_with(self, peer_id: str, retries: int = 2) -> Status:
        peer = self.peers[peer_id]
        status = Status(time.time(), False, "busy")
        for attempt in range(retries + 1):
            if attempt:  # both computers started at once: one backs off at random
                self._stop.wait(secrets.randbelow(8000) / 1000 + 2)
            if not self._lock.acquire(timeout=30):
                continue
            try:
                status = self._sync_locked(peer)
            except (OSError, SyncError, KeyError, ValueError) as exc:
                status = Status(time.time(), False, _friendly(exc))
                log.info("Sync with %s failed: %s", peer.name, exc)
            finally:
                self._lock.release()
            if status.ok or status.text != "busy":
                break
        self.status[peer_id] = status
        self.on_event("status", {"id": peer_id})
        return status

    def _find(self, peer: Peer) -> list[tuple[str, int]]:
        targets = []
        if peer.addr and ":" in peer.addr:
            host, _, port = peer.addr.rpartition(":")
            if port.isdigit() and int(port):
                targets.append((host, int(port)))
        return targets

    def _sync_locked(self, peer: Peer) -> Status:
        sock = None
        for host, port in self._find(peer):
            try:
                sock = self._connect(host, port, timeout=3)
                break
            except OSError:
                continue
        if sock is None:
            seen = {d["id"]: d for d in self.discover()} if self.udp_port is not None else {}
            if peer.id not in seen:
                return Status(time.time(), False, "Not found on this network")
            d = seen[peer.id]
            peer.addr = f"{d['host']}:{d['port']}"
            sock = self._connect(d["host"], d["port"])
        with sock:
            ch = Channel(sock)
            n_init = secrets.token_bytes(16)
            ch.send({"t": "hello", "v": VERSION, "id": self.id, "nonce": _b64(n_init), "port": self.port})
            hello = ch.expect("hello")
            n_resp = _unb64(hello["nonce"])
            if hello.get("id") != peer.id or not hmac.compare_digest(
                    _unb64(hello["mac"]), _mac(peer.key, b"responder", n_init, n_resp)):
                raise SyncError("the other computer couldn't prove it's paired")
            ch.send({"t": "auth", "mac": _b64(_mac(peer.key, b"initiator", n_init, n_resp))})
            to_peer, from_peer = self._session_keys(peer.key, n_init, n_resp)
            ch.secure(tx_key=to_peer, rx_key=from_peer)

            ch.send({"t": "state"})
            remote = ch.expect("state")
            base = self._base(peer.id)
            local = dictionary_state(self.dictionary)
            merged = merge3(base.get("state") or {}, local, remote["data"])

            # Folders (tuned models, voices): copy what one side has and the other doesn't,
            # unless it was on both last time (then it was deleted on purpose).
            got: list[tuple[str, str]] = []
            sent: list[tuple[str, str]] = []
            mine_files = self._local_files()
            theirs_files = remote.get("files") or {}
            had = base.get("files") or {}
            both: dict[str, list[str]] = {}
            for kind in KINDS:
                mine, theirs = mine_files.get(kind, {}), theirs_files.get(kind, {})
                gone = set(had.get(kind, []))
                if kind in self.accepts():
                    for name in sorted(set(theirs) - set(mine) - gone):
                        ch.send({"t": "get", "kind": kind, "name": name})
                        recv_folder(ch, self.folders[kind], name)
                        got.append((kind, name))
                if kind in (remote.get("accepts") or []):
                    for name in sorted(set(mine) - set(theirs) - gone):
                        if not _NAME_OK.match(name):
                            continue
                        ch.send({"t": "put", "kind": kind, "name": name})
                        send_folder(ch, self.folders[kind] / name)
                        ch.expect("ok")
                        sent.append((kind, name))
                now_mine = set(folder_manifest(self.folders[kind]))
                now_theirs = set(theirs) | {n for k, n in sent if k == kind}
                both[kind] = sorted(now_mine & now_theirs)

            ch.send({"t": "merged", "data": merged, "files": both, "name": self.name()})
            ch.expect("ok")
        now = dictionary_state(self.dictionary)
        final = merge3(local, now, merged)  # keep edits made while syncing
        if final != now:
            apply_dictionary_state(self.dictionary, final)
        self._save_base(peer.id, merged, both)
        peer.name = str(remote.get("name") or peer.name)[:60]
        peer.last_sync = time.time()
        self._save()
        n = changes(local, final)
        return self._report(peer, n, got, sent)

    def _report(self, peer: Peer, n: int, got: list, sent: list | None = None) -> Status:
        parts = []
        if n:
            parts.append(f"{n} dictionary change{'s' if n != 1 else ''}")
        if got:
            parts.append(f"received {len(got)} {'item' if len(got) == 1 else 'items'}")
        if sent:
            parts.append(f"sent {len(sent)}")
        status = Status(time.time(), True, ", ".join(parts) or "Up to date")
        self.status[peer.id] = status
        log.info("Synced with %s: %s", peer.name, status.text)
        for kind, name in got:
            self.on_event("received", {"kind": kind, "name": name, "from": peer.name})
        self.on_event("status", {"id": peer.id})
        return status

    def sync_all(self) -> None:
        for pid in list(self.peers):
            if self._stop.is_set():
                return
            self.sync_with(pid)

    def _dictionary_changed(self) -> bool:
        try:
            mtime = self.dictionary.path.stat().st_mtime
        except OSError:
            return False
        changed = self._dict_mtime and mtime != self._dict_mtime
        self._dict_mtime = mtime
        return bool(changed)

    def _schedule_loop(self) -> None:
        self._dictionary_changed()
        next_full = time.time() + 15  # soon after starting
        while not self._stop.is_set():
            kicked = self._kick.wait(30)
            self._kick.clear()
            if self._stop.is_set():
                return
            if not self.peers:
                continue
            if kicked or self._dictionary_changed() or time.time() >= next_full:
                self._stop.wait(2 + secrets.randbelow(4000) / 1000)  # edits settle; no lockstep
                self.sync_all()
                next_full = time.time() + SYNC_EVERY_S


def _friendly(exc: Exception) -> str:
    text = str(exc)
    if isinstance(exc, (ConnectionRefusedError, socket.timeout, TimeoutError)):
        return "Couldn't connect (is Murmr running there, with sync on?)"
    if text == "not paired":
        return "The other computer doesn't know this one any more: pair again"
    return text[:120] or exc.__class__.__name__
