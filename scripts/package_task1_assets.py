#!/usr/bin/env python3
"""Build the minimal project-owned task1 release asset from this workspace."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAG = "task1-assets-v1"
ARCHIVE = ROOT / "downloads" / f"{TAG}.tar.gz"
MANIFEST = ROOT / "assets" / f"{TAG}.json"
FILES = [
    "data/real/task1/VID_20260611_160742_1000810407.mp4",
    "data/real/task1/videos/camera_00.mp4",
    "data/task1_smolvla_v5_homereset_100episodes/collection_report.json",
    "runs/task1-real2sim-modeling/report.json",
    "runs/task1-smolvla-v5-robustness-20seed8080/robustness_report.json",
    "runs/task1-vla-unified-artifact-verification-20260912/iteration-000/improve/sampling_requests.jsonl",
]
DIRECTORIES = [
    "data/task1_smolvla_v5_homereset_success99",
    "checkpoints/task1_smolvla_v5_homereset_stage2_restartlr_pyav_2000steps_b32/checkpoints/002000/pretrained_model",
]
ROBUSTNESS = ROOT / "runs/task1-smolvla-v5-robustness-20seed8080"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    selected = {ROOT / name for name in FILES}
    for name in DIRECTORIES:
        selected.update(path for path in (ROOT / name).rglob("*") if path.is_file())
    selected.update(path for path in ROBUSTNESS.rglob("*.json") if path.is_file())
    missing = [str(path) for path in selected if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required task1 assets: {missing}")
    if any(path.is_symlink() for path in selected):
        raise SystemExit("release assets must not contain symlinks")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "w:gz", compresslevel=1) as archive:
        for path in sorted(selected):
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    manifest = {
        "schema": "autopolicy.task1_assets/v1",
        "tag": TAG,
        "asset": ARCHIVE.name,
        "bytes": ARCHIVE.stat().st_size,
        "sha256": sha256(ARCHIVE),
        "source_root": str(ROOT),
        "files": [str(path.relative_to(ROOT)) for path in sorted(selected)],
        "contents": "task1 source videos, 99-episode expert dataset, v5 trained checkpoint, and validation records",
        "excluded": "public upstream source/model assets, caches, optimizer states, and unrelated experiments",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{ARCHIVE} ({manifest['bytes']} bytes, sha256 {manifest['sha256']})")
    print(MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
