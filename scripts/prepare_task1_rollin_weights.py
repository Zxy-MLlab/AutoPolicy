#!/usr/bin/env python3
"""Build auditable episode weights for a nominal-plus-roll-in merged dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-manifest", type=Path, required=True)
    parser.add_argument("--correction-weight", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.correction_weight <= 0:
        raise ValueError("correction-weight must be positive")

    manifest = json.loads(args.merge_manifest.read_text(encoding="utf-8"))
    first, stop = map(int, manifest["correction_episode_index_range"])
    episodes = int(manifest["episodes"])
    if not (0 <= first < stop <= episodes):
        raise ValueError(f"invalid correction episode range: {[first, stop]} for {episodes} episodes")
    base_frames = int(manifest["sources"][0]["frames"])
    correction_frames = int(manifest["sources"][1]["frames"])
    weighted_correction_frames = correction_frames * args.correction_weight
    payload = {
        "schema": "autopolicy.task1_rollin_episode_weights/v1",
        "merge_manifest": str(args.merge_manifest.resolve()),
        "weights": {"nominal": 1.0, "policy_rollin_correction": args.correction_weight},
        "episode_counts": {"nominal": first, "policy_rollin_correction": stop - first},
        "frame_counts": {"nominal": base_frames, "policy_rollin_correction": correction_frames},
        "effective_weighted_frame_mass": {
            "nominal": base_frames / (base_frames + weighted_correction_frames),
            "policy_rollin_correction": weighted_correction_frames
            / (base_frames + weighted_correction_frames),
        },
        "episode_weights": {
            str(index): args.correction_weight if first <= index < stop else 1.0
            for index in range(episodes)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "episode_weights"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
