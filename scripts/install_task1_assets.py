#!/usr/bin/env python3
"""Download, verify, and install the project-owned task1 release asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/task1-assets-v1.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="use an already downloaded release asset")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archive_path = args.archive or ROOT / "downloads" / manifest["asset"]
    if not archive_path.is_file() or archive_path.stat().st_size != manifest["bytes"]:
        if args.archive:
            raise SystemExit(f"archive is missing or incomplete: {archive_path}")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://github.com/Zxy-MLlab/AutoPolicy/releases/download/{manifest['tag']}/{manifest['asset']}"
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "5", "--continue-at", "-", "--output", str(archive_path), url],
            check=True,
        )
    if archive_path.stat().st_size != manifest["bytes"] or digest(archive_path) != manifest["sha256"]:
        raise SystemExit(f"release asset size or SHA-256 mismatch: {archive_path}")
    expected = set(manifest["files"])
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        actual = {member.name for member in members}
        if actual != expected or any(not member.isfile() for member in members):
            raise SystemExit("release asset contains unexpected paths or non-file entries")
        archive.extractall(ROOT, filter="data")
    old_root = manifest["source_root"]
    for name in expected:
        path = ROOT / name
        if path.suffix in {".json", ".jsonl"}:
            contents = path.read_text(encoding="utf-8")
            relocated = contents.replace(old_root, str(ROOT))
            if relocated != contents:
                path.write_text(relocated, encoding="utf-8")
    print(f"installed {len(expected)} task1 files under {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
