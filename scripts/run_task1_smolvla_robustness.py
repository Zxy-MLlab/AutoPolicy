#!/usr/bin/env python3
"""Run and aggregate paired task1 SmolVLA replica-sim perturbation scenarios."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts/evaluate_task1_smolvla.py"


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials == 0:
        return [0.0, 0.0]
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def exact_mcnemar_p_value(nominal_only: int, scenario_only: int) -> float:
    discordant = nominal_only + scenario_only
    if discordant == 0:
        return 1.0
    tail = min(nominal_only, scenario_only)
    one_sided = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * one_sided)


def read_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "autopolicy.task1_smolvla_evaluation/v2":
        raise ValueError(f"unexpected evaluation schema in {path}: {report.get('schema')}")
    return report


def same_positions(left: dict, right: dict) -> bool:
    left_positions = [row["bottle_xy_ground_truth_for_scoring_only"] for row in left["episode_results"]]
    right_positions = [row["bottle_xy_ground_truth_for_scoring_only"] for row in right["episode_results"]]
    return left_positions == right_positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/task1_smolvla_robustness_v1.json",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--perturbation-seed", type=int, default=9001)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--render-scenarios", nargs="*", default=["combined_ood"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    suite = json.loads(args.config.read_text(encoding="utf-8"))
    scenarios = suite.get("scenarios", [])
    if not scenarios or scenarios[0].get("name") != "nominal":
        raise ValueError("robustness config must start with a nominal scenario")
    names = [scenario.get("name") for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError(f"scenario names must be unique: {names}")

    args.output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict] = {}
    commands: list[dict] = []
    for scenario in scenarios:
        name = scenario["name"]
        scenario_output = args.output / name
        report_path = scenario_output / "evaluation_report.json"
        command = [
            sys.executable,
            str(EVALUATOR),
            "--checkpoint",
            str(args.checkpoint),
            "--dataset-root",
            str(args.dataset_root),
            "--output",
            str(scenario_output),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--perturbation-seed",
            str(args.perturbation_seed),
            "--scenario",
            name,
            *map(str, scenario.get("arguments", [])),
        ]
        if args.n_action_steps is not None:
            command.extend(["--n-action-steps", str(args.n_action_steps)])
        if name in args.render_scenarios:
            command.append("--render")

        resumed = bool(args.resume and report_path.is_file())
        if not resumed:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise SystemExit(f"scenario {name} failed with return code {completed.returncode}")
        reports[name] = read_report(report_path)
        commands.append({"scenario": name, "command": command, "resumed": resumed})

    nominal = reports["nominal"]
    scenario_rows = []
    for name, report in reports.items():
        if not same_positions(nominal, report):
            raise ValueError(f"scenario {name} does not use the nominal bottle positions")
        nominal_only = sum(
            bool(nominal_row["success"]) and not bool(scenario_row["success"])
            for nominal_row, scenario_row in zip(
                nominal["episode_results"], report["episode_results"], strict=True
            )
        )
        scenario_only = sum(
            not bool(nominal_row["success"]) and bool(scenario_row["success"])
            for nominal_row, scenario_row in zip(
                nominal["episode_results"], report["episode_results"], strict=True
            )
        )
        successes = int(report["successes"])
        scenario_rows.append(
            {
                "scenario": name,
                "successes": successes,
                "episodes": args.episodes,
                "success_rate": successes / args.episodes,
                "wilson_95_interval": wilson_interval(successes, args.episodes),
                "delta_vs_nominal_percentage_points": 100.0
                * (successes - int(nominal["successes"]))
                / args.episodes,
                "paired_flips": {
                    "nominal_only_success": nominal_only,
                    "scenario_only_success": scenario_only,
                    "exact_mcnemar_p_value": exact_mcnemar_p_value(nominal_only, scenario_only),
                },
                "evaluation_report": str((args.output / name / "evaluation_report.json").resolve()),
                "video": report.get("video"),
            }
        )

    aggregate = {
        "schema": "autopolicy.task1_smolvla_robustness_report/v1",
        "evaluation_type": "paired MuJoCo replica-sim perturbation evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset_root.resolve()),
        "suite_config": str(args.config.resolve()),
        "position_seed": args.seed,
        "perturbation_seed": args.perturbation_seed,
        "episodes_per_scenario": args.episodes,
        "policy_inputs": nominal["policy_inputs"],
        "oracle_inputs_to_policy": nominal["oracle_inputs_to_policy"],
        "same_bottle_positions_across_scenarios": True,
        "scenarios": scenario_rows,
        "commands": commands,
        "limitations": [
            "These are MuJoCo replica-sim perturbations, not real-world robustness measurements.",
            "Perturbation ranges are engineering stress-test hypotheses, not calibrated sensor/physics uncertainty.",
            "Intervals are per-scenario binomial Wilson intervals; paired McNemar p-values are exploratory and uncorrected for multiple comparisons.",
        ],
    }
    json_path = args.output / "robustness_report.json"
    json_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown_lines = [
        "# Task1 SmolVLA robustness evaluation",
        "",
        f"- Checkpoint: `{aggregate['checkpoint']}`",
        f"- Dataset: `{aggregate['dataset']}`",
        f"- Paired episodes per scenario: {args.episodes}",
        f"- Position seed: {args.seed}; perturbation seed: {args.perturbation_seed}",
        "",
        "| Scenario | Success | Rate | 95% Wilson CI | Delta vs nominal | Nominal-only / OOD-only | Exact McNemar p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scenario_rows:
        low, high = row["wilson_95_interval"]
        flips = row["paired_flips"]
        markdown_lines.append(
            f"| {row['scenario']} | {row['successes']}/{row['episodes']} | "
            f"{100 * row['success_rate']:.1f}% | {100 * low:.1f}–{100 * high:.1f}% | "
            f"{row['delta_vs_nominal_percentage_points']:+.1f} pp | "
            f"{flips['nominal_only_success']} / {flips['scenario_only_success']} | "
            f"{flips['exact_mcnemar_p_value']:.6g} |"
        )
    markdown_lines.extend(
        [
            "",
            "This is an exploratory MuJoCo replica-sim stress test. It is not a real-robot or calibrated Sim2Real result.",
            "",
        ]
    )
    markdown_path = args.output / "robustness_report.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    print(json.dumps({"report": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
