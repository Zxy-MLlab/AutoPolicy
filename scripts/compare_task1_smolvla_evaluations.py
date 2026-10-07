#!/usr/bin/env python3
"""Compare two task1 SmolVLA evaluations episode by episode."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, episodes: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes))
        / denominator
    )
    return [center - radius, center + radius]


def summarize(rows: list[dict], threshold_x: float) -> dict:
    result: dict[str, object] = {}
    for label, selected in (
        ("all", rows),
        ("left", [r for r in rows if r["bottle_xy_ground_truth_for_scoring_only"][0] < threshold_x]),
        ("right", [r for r in rows if r["bottle_xy_ground_truth_for_scoring_only"][0] >= threshold_x]),
    ):
        successes = sum(bool(row["success"]) for row in selected)
        episodes = len(selected)
        result[label] = {
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "wilson_95_interval": wilson_interval(successes, episodes),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold-x", type=float, default=0.4839584794)
    args = parser.parse_args()

    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_report = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = baseline_report["episode_results"]
    candidate = candidate_report["episode_results"]
    if len(baseline) != len(candidate):
        raise ValueError("evaluation episode counts differ")

    paired_outcomes = {
        "both_success": 0,
        "both_failure": 0,
        "baseline_only_success": 0,
        "candidate_only_success": 0,
    }
    flips = []
    for baseline_row, candidate_row in zip(baseline, candidate, strict=True):
        if baseline_row["episode"] != candidate_row["episode"]:
            raise ValueError("episode indices differ")
        baseline_xy = baseline_row["bottle_xy_ground_truth_for_scoring_only"]
        candidate_xy = candidate_row["bottle_xy_ground_truth_for_scoring_only"]
        if baseline_xy != candidate_xy:
            raise ValueError("paired bottle positions differ")
        if baseline_row.get("perturbations") != candidate_row.get("perturbations"):
            raise ValueError("paired perturbations differ")
        baseline_success = bool(baseline_row["success"])
        candidate_success = bool(candidate_row["success"])
        if baseline_success and candidate_success:
            paired_outcomes["both_success"] += 1
        elif not baseline_success and not candidate_success:
            paired_outcomes["both_failure"] += 1
        elif baseline_success:
            paired_outcomes["baseline_only_success"] += 1
        else:
            paired_outcomes["candidate_only_success"] += 1
        if baseline_success != candidate_success:
            flips.append(
                {
                    "episode": baseline_row["episode"],
                    "bottle_xy_m": baseline_xy,
                    "baseline_success": baseline_success,
                    "candidate_success": candidate_success,
                }
            )

    baseline_only = paired_outcomes["baseline_only_success"]
    candidate_only = paired_outcomes["candidate_only_success"]
    discordant = baseline_only + candidate_only
    smaller = min(baseline_only, candidate_only)
    exact_p = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant),
    )
    delta = candidate_report["success_rate"] - baseline_report["success_rate"]
    if delta == 0:
        interpretation = (
            "The candidate and baseline have the same aggregate success rate on this paired "
            "evaluation set. Region results and episode-level flips should still be inspected."
        )
    elif exact_p < 0.05:
        direction = "better" if delta > 0 else "worse"
        interpretation = (
            f"The candidate is {direction}, and the paired difference is statistically significant "
            "at alpha=0.05 on this evaluation set. Region results should still be inspected."
        )
    else:
        direction = "better" if delta > 0 else "worse"
        interpretation = (
            f"The candidate is {direction} on this holdout, but the paired difference is not "
            "statistically significant at alpha=0.05. Region results should be inspected "
            "because aggregate performance can hide regional regressions."
        )
    result = {
        "schema": "autopolicy.task1_paired_evaluation_comparison/v1",
        "baseline_report": str(args.baseline.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "episodes": len(baseline),
        "position_pairing_verified": True,
        "perturbation_pairing_verified": True,
        "region_threshold_x_m": args.threshold_x,
        "baseline": summarize(baseline, args.threshold_x),
        "candidate": summarize(candidate, args.threshold_x),
        "paired_outcomes": paired_outcomes,
        "discordant_pairs": discordant,
        "exact_mcnemar_two_sided_p": exact_p,
        "success_rate_absolute_delta": delta,
        "flipped_episodes": flips,
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
