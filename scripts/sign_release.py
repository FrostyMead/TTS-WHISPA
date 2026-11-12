"""Offline Murmr release signing. Never put the private key in GitHub or this repo."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import time

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_key(path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    key = Ed25519PrivateKey.generate()
    # Exclusive creation: never overwrite an existing signing identity.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                 serialization.PrivateFormat.PKCS8,
                                 serialization.NoEncryption()))
    print(key.public_key().public_bytes_raw().hex())


def sign(version: str, sha: str, key_path: Path) -> None:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Version must be X.Y.Z")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Use the full 40-character commit SHA")
    key = serialization.load_pem_private_key(key_path.expanduser().read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("An Ed25519 private key is required")
    raw = json.dumps({"sha": sha, "version": version, "seq": int(time.time()),
                      "date": dt.datetime.now(dt.timezone.utc).isoformat()},
                     sort_keys=True, separators=(",", ":")).encode()
    Path("release.json").write_bytes(raw)
    Path("release.json.sig").write_bytes(key.sign(raw))


def publish(version: str, sha: str) -> None:
    tag = f"v{version}"
    exists = subprocess.run(["gh", "release", "view", tag], capture_output=True).returncode == 0
    if exists:
        cmd = ["gh", "release", "upload", tag, "release.json", "release.json.sig", "--clobber"]
    else:
        cmd = ["gh", "release", "create", tag, "release.json", "release.json.sig",
               "--target", sha, "--title", f"Murmr {version}", "--notes", "Signed Murmr release"]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-key", type=Path)
    parser.add_argument("--sign", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--version")
    parser.add_argument("--sha")
    parser.add_argument("--key", type=Path, default=Path(os.environ.get(
        "MURMR_SIGNING_KEY", "~/.murmr-signing/release_key.pem")))
    args = parser.parse_args()
    if args.generate_key:
        if args.sign or args.publish:
            parser.error("Generate a key separately from signing/publishing")
        generate_key(args.generate_key)
        return
    if not (args.sign or args.publish):
        parser.error("Choose --generate-key, --sign, or --publish")
    if args.sign:
        if not args.version or not args.sha:
            parser.error("--sign requires --version and --sha")
        sign(args.version, args.sha, args.key)
    if args.publish:
        manifest = json.loads(Path("release.json").read_bytes())
        if args.version and args.version != manifest["version"]:
            parser.error("--version differs from release.json")
        if args.sha and args.sha != manifest["sha"]:
            parser.error("--sha differs from release.json")
        publish(manifest["version"], manifest["sha"])


if __name__ == "__main__":
    main()
