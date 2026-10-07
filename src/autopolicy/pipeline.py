from __future__ import annotations

import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import STAGE_NAMES, PipelineConfig, storage_environment
from .errors import ConfigError
from .io import file_record, read_json, sha256_file, utc_now, write_json
from .stages import STAGE_FUNCTIONS, external_stage_plan


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _git_revision(path: Path) -> str | None:
    if not (path / ".git").exists():
        marker = path / ".upstream-commit"
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty(path: Path) -> bool | None:
    if not (path / ".git").exists():
        return None
    completed = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def doctor(config: PipelineConfig) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    needs_upstreams = any(
        stage.backend not in {"mock", "disabled"} for stage in config.stages.values()
    )
    for name in ("gpt6_real2sim", "robotwin", "lerobot"):
        path = config.paths[name]
        checks[name] = {
            "path": str(path),
            "exists": path.is_dir(),
            "required": needs_upstreams,
            "git_revision": _git_revision(path),
        }
    for name in ("runs", "datasets", "checkpoints", "models", "cache", "tmp"):
        path = config.paths[name]
        path.mkdir(parents=True, exist_ok=True)
        checks[name] = {"path": str(path), "exists": path.is_dir(), "writable": os.access(path, os.W_OK)}
    environment = storage_environment(config)
    path_environment = {
        key: value
        for key, value in environment.items()
        if key not in {"PYTHONNOUSERSITE", "MUJOCO_GL"}
    }
    storage_ok = all(
        all(Path(part).is_relative_to(config.root) for part in value.split(os.pathsep))
        for value in path_environment.values()
    )
    stage_checks = {
        name: {
            "backend": stage.backend,
            "has_command": bool(stage.command),
            "valid": stage.backend != "external" or bool(stage.command),
        }
        for name, stage in config.stages.items()
    }
    ok = (
        all(item["exists"] or not item.get("required", True) for item in checks.values())
        and all(item.get("writable", True) for item in checks.values())
        and all(item["valid"] for item in stage_checks.values())
        and storage_ok
    )
    return {
        "ok": ok,
        "workspace": str(config.root),
        "python": platform.python_version(),
        "host": socket.gethostname(),
        "checks": checks,
        "stages": stage_checks,
        "storage_environment": environment,
        "storage_policy_satisfied": storage_ok,
    }


def _select_stages(from_stage: str | None, to_stage: str | None) -> tuple[str, ...]:
    if from_stage and from_stage not in STAGE_NAMES:
        raise ConfigError(f"unknown --from-stage {from_stage!r}")
    if to_stage and to_stage not in STAGE_NAMES:
        raise ConfigError(f"unknown --to-stage {to_stage!r}")
    start = STAGE_NAMES.index(from_stage) if from_stage else 0
    stop = STAGE_NAMES.index(to_stage) + 1 if to_stage else len(STAGE_NAMES)
    if start >= stop:
        raise ConfigError("--from-stage must precede or equal --to-stage")
    return STAGE_NAMES[start:stop]


class PipelineRunner:
    def __init__(self, config: PipelineConfig, run_dir: Path | None = None):
        self.config = config
        self.run_dir = run_dir.resolve() if run_dir else (config.paths["runs"] / _run_id()).resolve()
        if config.root != self.run_dir and config.root not in self.run_dir.parents:
            raise ConfigError(f"run directory must be under {config.root}: {self.run_dir}")
        self.state_path = self.run_dir / "pipeline_state.json"

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": "autopolicy.run/v1",
            "name": self.config.name,
            "run_dir": str(self.run_dir),
            "status": "running",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "seed": self.config.seed,
            "success_target": self.config.success_target,
            "configuration": file_record(self.config.source),
            "workspace_git_revision": _git_revision(self.config.root),
            "workspace_git_dirty": _git_dirty(self.config.root),
            "iterations": [],
            "upstreams": {
                key: {"path": str(self.config.paths[key]), "git_revision": _git_revision(self.config.paths[key])}
                for key in ("gpt6_real2sim", "robotwin", "lerobot")
            },
        }

    def dry_run(
        self,
        *,
        from_stage: str | None = None,
        to_stage: str | None = None,
    ) -> dict[str, Any]:
        """Resolve an execution plan without creating a run directory or invoking a stage."""
        selected = _select_stages(from_stage, to_stage)
        iterations: list[dict[str, Any]] = []
        for iteration in range(self.config.max_iterations):
            iteration_dir = self.run_dir / f"iteration-{iteration:03d}"
            stage_plans: list[dict[str, Any]] = []
            for stage_name in selected:
                stage = self.config.stage(stage_name)
                stage_dir = iteration_dir / stage_name
                if stage.backend == "external":
                    detail = external_stage_plan(
                        self.config, stage, stage_name, stage_dir, iteration
                    )
                else:
                    detail = {
                        "stage": stage_name,
                        "backend": stage.backend,
                        "options": stage.options,
                    }
                stage_plans.append(detail)
            iterations.append({"index": iteration, "stages": stage_plans})
        return {
            "schema": "autopolicy.dry_run/v1",
            "name": self.config.name,
            "config": str(self.config.source),
            "run_dir": str(self.run_dir),
            "selected_stages": list(selected),
            "iterations": iterations,
            "side_effects": False,
        }

    def _state(self, resume: bool, selected: tuple[str, ...]) -> dict[str, Any]:
        if resume:
            if not self.state_path.is_file():
                raise ConfigError(f"cannot resume: missing {self.state_path}")
            state = read_json(self.state_path)
            recorded_config = state.get("configuration")
            if recorded_config and recorded_config.get("sha256") != sha256_file(self.config.source):
                raise ConfigError(
                    "cannot resume: pipeline configuration changed since the run was created"
                )
            if state.get("status") == "completed":
                selected_are_complete = bool(state.get("iterations")) and all(
                    all(
                        item.get("stages", {}).get(name, {}).get("status") == "completed"
                        and self._stage_artifacts_valid(item["stages"][name])
                        for name in selected
                    )
                    for item in state["iterations"]
                )
                if selected_are_complete:
                    return state
                state.pop("finished_at", None)
                state.pop("stop_reason", None)
            state["status"] = "running"
            state.pop("error", None)
            return state
        if self.state_path.exists():
            raise ConfigError(f"run already exists; pass --resume: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "resolved_config.json", self.config.raw)
        return self._initial_state()

    def _stage_artifacts_valid(self, manifest: dict[str, Any]) -> bool:
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            return False
        records = [(record, self.run_dir / record.get("path", "")) for record in artifacts]
        command_inputs = manifest.get("command_inputs", [])
        if not isinstance(command_inputs, list):
            return False
        records.extend((record, Path(record.get("path", ""))) for record in command_inputs)
        for record, path in records:
            try:
                if not path.is_file():
                    return False
                if path.stat().st_size != int(record["bytes"]):
                    return False
                if sha256_file(path) != record["sha256"]:
                    return False
            except (KeyError, OSError, TypeError, ValueError):
                return False
        return True

    def run(
        self,
        *,
        resume: bool = False,
        from_stage: str | None = None,
        to_stage: str | None = None,
    ) -> dict[str, Any]:
        selected = _select_stages(from_stage, to_stage)
        state = self._state(resume, selected)
        if state.get("status") == "completed":
            return state
        existing = {int(item["index"]): item for item in state.get("iterations", [])}
        try:
            for iteration in range(self.config.max_iterations):
                iteration_dir = self.run_dir / f"iteration-{iteration:03d}"
                iteration_state = existing.get(iteration, {"index": iteration, "stages": {}, "status": "running"})
                if iteration not in existing:
                    state["iterations"].append(iteration_state)
                for stage_name in selected:
                    previous = iteration_state["stages"].get(stage_name)
                    if (
                        resume
                        and previous
                        and previous.get("status") == "completed"
                        and self._stage_artifacts_valid(previous)
                    ):
                        continue
                    stage_dir = iteration_dir / stage_name
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    stage_manifest = {
                        "schema": "autopolicy.stage/v1",
                        "name": stage_name,
                        "backend": self.config.stage(stage_name).backend,
                        "iteration": iteration,
                        "status": "running",
                        "started_at": utc_now(),
                    }
                    if self.config.stage(stage_name).backend == "external":
                        execution_plan = external_stage_plan(
                            self.config,
                            self.config.stage(stage_name),
                            stage_name,
                            stage_dir,
                            iteration,
                        )
                        command_inputs = []
                        for argument in execution_plan["command"]:
                            candidate = Path(argument)
                            if (
                                candidate.is_absolute()
                                and candidate.is_file()
                                and (candidate == self.config.root or self.config.root in candidate.parents)
                            ):
                                command_inputs.append(file_record(candidate))
                        stage_manifest["execution_plan"] = execution_plan
                        stage_manifest["command_inputs"] = command_inputs
                    iteration_state["stages"][stage_name] = stage_manifest
                    write_json(stage_dir / "manifest.json", stage_manifest)
                    write_json(self.state_path, state)
                    try:
                        result = STAGE_FUNCTIONS[stage_name](self.config, stage_dir, iteration)
                        artifact_files = [
                            file_record(path, self.run_dir)
                            for path in sorted(stage_dir.rglob("*"))
                            if path.is_file() and path.name != "manifest.json"
                        ]
                        stage_manifest.update(
                            status="completed",
                            finished_at=utc_now(),
                            result=result,
                            artifacts=artifact_files,
                        )
                    except BaseException as exc:
                        stage_manifest.update(
                            status="failed",
                            finished_at=utc_now(),
                            error={"type": type(exc).__name__, "message": str(exc)},
                        )
                        write_json(stage_dir / "manifest.json", stage_manifest)
                        raise
                    write_json(stage_dir / "manifest.json", stage_manifest)
                    state["updated_at"] = utc_now()
                    write_json(self.state_path, state)
                iteration_state["status"] = "completed"
                evaluate_result = iteration_state["stages"].get("evaluate", {}).get("result", {})
                iteration_state["success_rate"] = evaluate_result.get("success_rate")
                state["updated_at"] = utc_now()
                write_json(self.state_path, state)
                if "evaluate" in selected and evaluate_result.get("target_met") is True:
                    state["stop_reason"] = "simulation_success_target_met"
                    break
            state["status"] = "completed"
            state["finished_at"] = utc_now()
            state.setdefault("stop_reason", "iteration_budget_exhausted")
        except BaseException as exc:
            state["status"] = "failed"
            state["finished_at"] = utc_now()
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(self.state_path, state)
            raise
        write_json(self.state_path, state)
        return state


def summarize_run(run_dir: Path) -> dict[str, Any]:
    state_path = run_dir.resolve() / "pipeline_state.json"
    if not state_path.is_file():
        raise ConfigError(f"not an AutoPolicy run: {run_dir}")
    state = read_json(state_path)
    return {
        "run_dir": state.get("run_dir"),
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "iterations": [
            {
                "index": item["index"],
                "status": item.get("status"),
                "success_rate": item.get("success_rate"),
                "stages": {name: value.get("status") for name, value in item.get("stages", {}).items()},
            }
            for item in state.get("iterations", [])
        ],
        "error": state.get("error"),
    }
