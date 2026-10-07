#!/usr/bin/env python3
"""Aggregate multiple task1 paired-evaluation comparison reports."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.comparison]
    thresholds = {report["region_threshold_x_m"] for report in reports}
    if len(thresholds) != 1:
        raise ValueError("comparison reports use different region thresholds")

    totals = {
        "both_success": 0,
        "both_failure": 0,
        "baseline_only_success": 0,
        "candidate_only_success": 0,
    }
    side_totals = {
        model: {region: {"successes": 0, "episodes": 0} for region in ("all", "left", "right")}
        for model in ("baseline", "candidate")
    }
    for report in reports:
        for key in totals:
            totals[key] += report["paired_outcomes"][key]
        for model in side_totals:
            for region in side_totals[model]:
                side_totals[model][region]["successes"] += report[model][region]["successes"]
                side_totals[model][region]["episodes"] += report[model][region]["episodes"]

    for model in side_totals:
        for region in side_totals[model]:
            values = side_totals[model][region]
            values["success_rate"] = values["successes"] / values["episodes"]
            values["wilson_95_interval"] = wilson_interval(values["successes"], values["episodes"])

    baseline_only = totals["baseline_only_success"]
    candidate_only = totals["candidate_only_success"]
    discordant = baseline_only + candidate_only
    smaller = min(baseline_only, candidate_only)
    exact_p = min(
        1.0,
        2 * sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant),
    )
    delta = (
        side_totals["candidate"]["all"]["success_rate"]
        - side_totals["baseline"]["all"]["success_rate"]
    )
    if delta == 0:
        interpretation = "The candidate and baseline have the same aggregate success rate."
    elif exact_p < 0.05:
        direction = "improvement" if delta > 0 else "regression"
        interpretation = (
            f"The candidate has a statistically significant aggregate paired {direction} "
            "at alpha=0.05. The regional totals quantify whether the effect is balanced."
        )
    else:
        direction = "better" if delta > 0 else "worse"
        interpretation = (
            f"The candidate is aggregate {direction}, but the paired difference is not "
            "statistically significant at alpha=0.05."
        )
    result = {
        "schema": "autopolicy.task1_paired_evaluation_aggregate/v1",
        "comparison_reports": [str(path.resolve()) for path in args.comparison],
        "sets": len(reports),
        "episodes_per_set": [report["episodes"] for report in reports],
        "total_paired_episodes": sum(report["episodes"] for report in reports),
        "position_pairing_verified_in_all_sets": all(
            report["position_pairing_verified"] for report in reports
        ),
        "perturbation_pairing_verified_in_all_sets": all(
            report.get("perturbation_pairing_verified", False) for report in reports
        ),
        "region_threshold_x_m": thresholds.pop(),
        "baseline": side_totals["baseline"],
        "candidate": side_totals["candidate"],
        "paired_outcomes": totals,
        "discordant_pairs": discordant,
        "exact_mcnemar_two_sided_p": exact_p,
        "success_rate_absolute_delta": delta,
        "interpretation": interpretation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
