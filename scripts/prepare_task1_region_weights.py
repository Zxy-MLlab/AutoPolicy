#!/usr/bin/env python3
"""Build auditable per-episode weights from a task1 collection report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold-x", type=float, required=True)
    parser.add_argument("--left-weight", type=float, default=1.5)
    parser.add_argument("--right-weight", type=float, default=1.0)
    args = parser.parse_args()
    if args.left_weight <= 0 or args.right_weight <= 0:
        raise ValueError("episode weights must be positive")

    source = args.collection_report.resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    metrics = report.get("episode_metrics", [])
    expected_episodes = int(report["episodes"])
    if len(metrics) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} episode metrics, found {len(metrics)}")

    episode_weights: dict[str, float] = {}
    regions: dict[str, str] = {}
    counts = {"left": 0, "right": 0}
    for row in metrics:
        episode = int(row["episode"])
        x = float(row["bottle_xy"][0])
        region = "left" if x < args.threshold_x else "right"
        episode_weights[str(episode)] = args.left_weight if region == "left" else args.right_weight
        regions[str(episode)] = region
        counts[region] += 1
    if sorted(map(int, episode_weights)) != list(range(expected_episodes)):
        raise ValueError("episode metrics must contain each contiguous episode index exactly once")

    left_mass = counts["left"] * args.left_weight
    right_mass = counts["right"] * args.right_weight
    total_mass = left_mass + right_mass
    payload = {
        "schema": "autopolicy.task1_episode_region_weights/v1",
        "collection_report": str(source),
        "threshold_x_m": args.threshold_x,
        "rule": "left if bottle_x < threshold_x_m, otherwise right",
        "weights": {"left": args.left_weight, "right": args.right_weight},
        "episode_counts": counts,
        "effective_weight_mass": {
            "left": left_mass / total_mass,
            "right": right_mass / total_mass,
        },
        "episode_weights": episode_weights,
        "episode_regions": regions,
        "policy_input_note": "bottle_x is used only offline to construct training weights and is not a policy input",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "schema", "threshold_x_m", "weights", "episode_counts", "effective_weight_mass"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
