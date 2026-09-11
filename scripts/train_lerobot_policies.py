#!/usr/bin/env python3
"""Launch standard LeRobot VLA and WAM training without writing outside AutoPolicy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lerobot-root", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profiles", nargs="+", choices=("smolvla", "fastwam"), default=("smolvla", "fastwam"))
    return parser.parse_args()


def main() -> int:
    args = arguments()
    root = Path("/data/zxy/autopolicy").resolve()
    for label, path in (("lerobot root", args.lerobot_root), ("dataset root", args.dataset_root), ("output dir", args.output_dir)):
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise SystemExit(f"{label} must be under {root}: {resolved}")
    if not (args.dataset_root / "meta/info.json").is_file():
        raise SystemExit(f"not a single LeRobotDataset root: {args.dataset_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.lerobot_root / "src")
    commands: list[dict[str, object]] = []
    for profile in args.profiles:
        output = args.output_dir / profile
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id=autopolicy/{args.dataset_root.name}",
            f"--dataset.root={args.dataset_root}",
            f"--policy.type={profile}",
            f"--policy.device={args.device}",
            f"--output_dir={output}",
            f"--job_name=autopolicy_{profile}",
            f"--steps={args.steps}",
            f"--batch_size={args.batch_size}",
            "--wandb.enable=false",
        ]
        completed = subprocess.run(command, cwd=args.lerobot_root, env=environment, check=False)
        commands.append({"profile": profile, "command": command, "returncode": completed.returncode, "output": str(output)})
        if completed.returncode != 0:
            return completed.returncode
    result = {
        "authenticity": "trained_with_lerobot",
        "dataset": str(args.dataset_root.resolve()),
        "profiles": list(args.profiles),
        "runs": commands,
    }
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
