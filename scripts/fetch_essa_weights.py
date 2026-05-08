"""Download the ESSA Mask R-CNN checkpoint from Zenodo.

Le Corre et al. (2025), "New candidate cave entrances on the Moon found using
deep learning", Icarus 441, 116675. doi:10.1016/j.icarus.2025.116675
Weights record: doi:10.5281/zenodo.15438463 (CC BY 4.0).

Usage:
    python scripts/fetch_essa_weights.py
    python scripts/fetch_essa_weights.py --dest data/weights/essa.pt --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data" / "weights" / "essa.pt"

ZENODO_RECORD = 15438463
FILENAME = "ESSA_ResNet50FPN_best_version.pt"
URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{FILENAME}/content"
EXPECTED_MD5 = "b49c17eae6d7f217fa3df14f5f4ac8e5"
EXPECTED_BYTES = 551_700_000  # ~552 MB; informational only


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def fetch(dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        digest = _md5(dest)
        if digest == EXPECTED_MD5:
            print(f"[ok] checkpoint already present and md5 matches: {dest}")
            return dest
        print(f"[warn] existing file md5 mismatch ({digest}); re-downloading")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", EXPECTED_BYTES))
        with tmp.open("wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=FILENAME
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))

    digest = _md5(tmp)
    if digest != EXPECTED_MD5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"md5 mismatch: got {digest}, expected {EXPECTED_MD5}. Refusing to keep file."
        )
    tmp.rename(dest)
    print(f"[ok] {dest}  md5={digest}")
    return dest


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    fetch(args.dest, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
