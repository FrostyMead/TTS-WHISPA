"""Step 2: general English speech, to keep the model good at everyone else's voice too.

Downloads (about 700 MB in total, from openslr.org, CC BY 4.0):
  LibriSpeech dev-clean   (337 MB, 5.4 h)   mixed into training ("replay") against forgetting
  LibriSpeech test-clean  (346 MB, 5.4 h)   a fixed 200-utterance general benchmark
  LibriSpeech-PC manifests (25 MB)          punctuated + capitalized transcripts, so the
                                            model doesn't unlearn punctuation and capitals

Writes WORK/data/general/{replay,test}.json. Re-running skips finished downloads.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import WORK, write_manifest  # noqa: E402

URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "pc": "https://www.openslr.org/resources/145/manifests.tar.gz",
}


def fetch(name: str, dest: Path) -> Path:
    archive = dest / f"{name}.tar.gz"
    done = dest / f".{name}.extracted"
    if done.exists():
        return dest
    if not archive.exists():
        print(f"downloading {URLS[name]}")
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(URLS[name], tmp)
        tmp.rename(archive)
    print(f"extracting {archive.name}")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    done.touch()
    archive.unlink()
    return dest


def pc_manifest(root: Path, split: str) -> list[dict]:
    """The LibriSpeech-PC manifest for a split, with audio paths pointing at our copy."""
    manifest = root / f"{split}.json"  # from manifests.tar.gz
    if not manifest.exists():
        raise SystemExit(f"No LibriSpeech-PC manifest {manifest}")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        # e.g. "dev-clean/84/121123/84-121123-0000.flac" relative to the LibriSpeech folder
        rel = r["audio_filepath"].split("LibriSpeech/")[-1].lstrip("./")
        path = root / "LibriSpeech" / rel
        if path.exists():
            rows.append({"audio_filepath": str(path), "duration": r["duration"], "text": r["text"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=WORK / "data" / "general")
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--max-seconds", type=float, default=20.0)
    args = ap.parse_args()
    raw = args.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in URLS:
        fetch(name, raw)

    replay = [r for r in pc_manifest(raw, "dev-clean") if r["duration"] <= args.max_seconds]
    test = [r for r in pc_manifest(raw, "test-clean") if r["duration"] <= args.max_seconds]
    test = random.Random(0).sample(test, min(args.test_size, len(test)))
    write_manifest(args.out / "replay.json", replay)
    write_manifest(args.out / "test.json", test)
    for name, rows in (("replay", replay), ("test", test)):
        print(f"  {name:6s} {len(rows):5d} utterances  {sum(r['duration'] for r in rows) / 3600:.2f} h")


if __name__ == "__main__":
    main()
