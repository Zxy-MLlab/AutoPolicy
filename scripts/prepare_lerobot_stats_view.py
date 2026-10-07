#!/usr/bin/env python3
"""Create a lightweight LeRobotDataset view with selected statistics replaced.

This is useful for controlled continuation experiments where the trajectory data
comes from a new dataset but state/action normalization must stay in the coordinate
system used by a pretrained policy. The source datasets are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_children(source: Path, destination: Path, excluded: set[str] | None = None) -> None:
    excluded = excluded or set()
    for child in source.iterdir():
        if child.name in excluded:
            continue
        os.symlink(child.resolve(), destination / child.name, target_is_directory=child.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=Path, required=True)
    parser.add_argument("--stats-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["observation.state", "action"],
        help="Statistics keys copied from --stats-source.",
    )
    args = parser.parse_args()

    data_source = args.data_source.resolve()
    stats_source = args.stats_source.resolve()
    output = args.output.resolve()
    if output in {data_source, stats_source}:
        raise ValueError("output must differ from both source datasets")
    for source in (data_source, stats_source):
        if not (source / "meta/info.json").is_file() or not (source / "meta/stats.json").is_file():
            raise FileNotFoundError(f"not a LeRobotDataset root: {source}")

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    (output / "meta").mkdir()
    link_children(data_source, output, excluded={"meta"})
    link_children(data_source / "meta", output / "meta", excluded={"stats.json"})

    data_stats_path = data_source / "meta/stats.json"
    replacement_stats_path = stats_source / "meta/stats.json"
    merged_stats = json.loads(data_stats_path.read_text(encoding="utf-8"))
    replacement_stats = json.loads(replacement_stats_path.read_text(encoding="utf-8"))
    for feature in args.features:
        if feature not in merged_stats or feature not in replacement_stats:
            raise KeyError(f"missing requested statistics key: {feature}")
        merged_stats[feature] = replacement_stats[feature]
    output_stats_path = output / "meta/stats.json"
    output_stats_path.write_text(
        json.dumps(merged_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema": "autopolicy.lerobot_stats_view/v1",
        "data_source": str(data_source),
        "stats_source": str(stats_source),
        "replaced_features": args.features,
        "source_stats_sha256": {
            "data_source": sha256(data_stats_path),
            "stats_source": sha256(replacement_stats_path),
        },
        "output_stats_sha256": sha256(output_stats_path),
        "storage": "source content is referenced by absolute symlinks; only merged stats and this manifest are new files",
    }
    (output / "stats_view_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
