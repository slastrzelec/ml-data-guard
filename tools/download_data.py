"""Download a public benchmark dataset once and verify its SHA-256 checksum.

This is the only place in the repository that uses the network. The library, the tests and the
benchmark read the local cache in ``data/`` (never tracked by git).

Usage: python tools/download_data.py covertype [--accept-unverified]

Exit codes: 0 verified, 1 checksum mismatch or download error, 2 no checksum pinned yet.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The Covertype checksum was pinned on 2026-10-10 from a download whose content passed
# mlguard.datasets.load_covertype (581,012 rows, 55 columns, valid one-hot groups).
SOURCES = {
    "covertype": {
        "url": "https://archive.ics.uci.edu/static/public/31/covertype.zip",
        "path": "data/covertype/covertype.zip",
        "sha256": "89a975c2457cd48e824238ae43c5a3cb762e42c4b4078d9b44a4514055105f6d",
        "licence": "CC BY 4.0, https://doi.org/10.24432/C50K5N",
    },
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"only https downloads are allowed, got {url!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", choices=sorted(SOURCES))
    parser.add_argument(
        "--accept-unverified",
        action="store_true",
        help="exit with 0 even though no checksum is pinned yet (first download only)",
    )
    arguments = parser.parse_args(argv)
    source = SOURCES[arguments.dataset]
    target = REPO / source["path"]

    if target.is_file():
        print(f"using cached {target.relative_to(REPO)}")
    else:
        print(f"downloading {source['url']} ({source['licence']})")
        try:
            download(source["url"], target)
        except OSError as error:
            print(f"ERROR: download failed: {error}")
            return 1

    actual = sha256_of(target)
    print(f"sha256 {actual}  {target.stat().st_size} bytes")
    expected = source["sha256"]
    if expected is None:
        print("WARNING: no checksum is pinned for this dataset yet; record the value above.")
        return 0 if arguments.accept_unverified else 2
    if actual != expected:
        rejected = target.with_suffix(target.suffix + ".rejected")
        target.replace(rejected)
        print(f"ERROR: checksum mismatch, expected {expected}; file moved to {rejected.name}")
        return 1
    print("checksum verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
