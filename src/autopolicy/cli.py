from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import STAGE_NAMES, load_config
from .dataset import audit_lerobot_dataset
from .errors import AutoPolicyError
from .pipeline import PipelineRunner, doctor, summarize_run


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="autopolicy", description="Closed-loop Real2Sim policy pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    doctor_parser = commands.add_parser("doctor", help="validate paths, storage policy, and adapters")
    doctor_parser.add_argument("--config", required=True)

    run_parser = commands.add_parser("run", help="run or resume the pipeline")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--run-dir")
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve stages and commands without creating a run or executing commands",
    )
    run_parser.add_argument("--from-stage", choices=STAGE_NAMES)
    run_parser.add_argument("--to-stage", choices=STAGE_NAMES)

    status_parser = commands.add_parser("status", help="show a compact run summary")
    status_parser.add_argument("run_dir")

    audit_parser = commands.add_parser("audit-dataset", help="validate one or more LeRobot datasets")
    audit_parser.add_argument("dataset_root")
    audit_parser.add_argument("--metadata-only", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "doctor":
            report = doctor(load_config(arguments.config))
            _print(report)
            return 0 if report["ok"] else 2
        if arguments.command == "run":
            config = load_config(arguments.config)
            runner = PipelineRunner(config, Path(arguments.run_dir) if arguments.run_dir else None)
            if arguments.dry_run:
                if arguments.resume:
                    raise ValueError("--dry-run and --resume cannot be used together")
                _print(
                    runner.dry_run(
                        from_stage=arguments.from_stage,
                        to_stage=arguments.to_stage,
                    )
                )
                return 0
            state = runner.run(
                resume=arguments.resume,
                from_stage=arguments.from_stage,
                to_stage=arguments.to_stage,
            )
            _print(summarize_run(Path(state["run_dir"])))
            return 0
        if arguments.command == "status":
            _print(summarize_run(Path(arguments.run_dir)))
            return 0
        if arguments.command == "audit-dataset":
            report = audit_lerobot_dataset(Path(arguments.dataset_root).resolve(), not arguments.metadata_only)
            _print(report)
            return 0 if report["valid"] else 2
    except (AutoPolicyError, OSError, ValueError) as exc:
        print(f"autopolicy: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
