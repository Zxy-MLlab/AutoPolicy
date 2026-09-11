from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .config import PipelineConfig, StageConfig, storage_environment
from .dataset import audit_lerobot_dataset
from .errors import SafetyError, StageError
from .execution import run_command
from .io import read_json, read_jsonl, write_json, write_jsonl


def _float(options: dict[str, Any], name: str, default: float) -> float:
    return float(options.get(name, default))


def _external(
    config: PipelineConfig,
    stage_config: StageConfig,
    stage_name: str,
    stage_dir: Path,
    iteration: int,
) -> dict[str, Any]:
    variables = {
        "root": config.root,
        "run_dir": stage_dir.parent,
        "stage_dir": stage_dir,
        "output_dir": stage_dir,
        "iteration": iteration,
        "gpt6_real2sim": config.paths["gpt6_real2sim"],
        "robotwin": config.paths["robotwin"],
        "lerobot": config.paths["lerobot"],
        "datasets": config.paths["datasets"],
        "checkpoints": config.paths["checkpoints"],
        "models": config.paths["models"],
    }
    cwd_value = stage_config.options.get("cwd", str(config.root))
    cwd = Path(str(cwd_value).format_map({key: str(value) for key, value in variables.items()})).resolve()
    argv = run_command(
        stage_config.command,
        cwd=cwd,
        environment=storage_environment(config),
        log_path=stage_dir / "command.log",
        variables=variables,
    )
    result_name = str(stage_config.options.get("result", "result.json"))
    result_path = stage_dir / result_name
    if not result_path.is_file():
        raise StageError(f"external stage {stage_name!r} did not write required result {result_path}")
    payload = read_json(result_path)
    if not isinstance(payload, dict):
        raise StageError(f"external stage result must be a JSON object: {result_path}")
    return {
        **payload,
        "authenticity": payload.get("authenticity", "external"),
        "execution": {"command": argv, "result_file": str(result_path)},
    }


def reconstruct(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("reconstruct")
    if stage.backend == "external":
        return _external(config, stage, "reconstruct", stage_dir, iteration)
    if stage.backend == "artifact":
        source = Path(str(stage.options.get("source", config.paths["gpt6_real2sim"] / "real2sim_ep0"))).resolve()
        scene = source / str(stage.options.get("scene", "scene.xml"))
        validation = source / str(stage.options.get("validation", "validation.json"))
        if not scene.is_file() or not validation.is_file():
            raise StageError(f"incomplete GPT6-real2sim artifact at {source}")
        payload = {
            "authenticity": "archived_real2sim_artifact",
            "source": str(source),
            "scene": str(scene),
            "validation": str(validation),
            "simulator": "mujoco",
            "executable": True,
        }
        write_json(stage_dir / "digital_twin.json", payload)
        return payload
    if stage.backend != "mock":
        raise StageError(f"reconstruct backend {stage.backend!r} cannot run")
    scene = {
        "schema": "autopolicy.digital_twin/v1",
        "units": "metres-kilograms-seconds-radians",
        "world": {"gravity": [0.0, 0.0, -9.81], "timestep_s": 0.002},
        "robot": {"name": "mock_dual_arm", "dof": 16, "aligned": True},
        "cameras": [{"name": "head", "calibrated": True}],
        "objects": [{"name": "target", "collision": "convex", "dynamic": True}],
    }
    write_json(stage_dir / "scene.json", scene)
    payload = {
        "authenticity": "synthetic_smoke_test",
        "scene": str(stage_dir / "scene.json"),
        "simulator": "contract-only",
        "executable": True,
    }
    write_json(stage_dir / "digital_twin.json", payload)
    return payload


def _normalise_real2sim_validation(validation: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    camera_values: list[float] = []
    for camera in validation.get("camera_alignment", {}).values():
        for rim in camera.get("rim_errors", {}).values():
            camera_values.append(float(rim.get("mean_px", 0.0)))
        if "held_out_flange_error_px" in camera:
            camera_values.append(float(camera["held_out_flange_error_px"]))
    physics = validation.get("physics", {})
    placement = validation.get("placement_alignment", {})
    metrics = {
        "camera_reprojection_mean_px": sum(camera_values) / len(camera_values) if camera_values else None,
        "max_penetration_m": physics.get("max_penetration_m"),
        "trajectory_rmse_rad": physics.get("arm_tracking_rmse_rad"),
        "placement_error_m": placement.get("position_difference_m"),
        "joint_limits_ok": validation.get("joint_validation", {}).get("within_position_limits"),
        "velocity_limits_ok": validation.get("joint_validation", {}).get("within_velocity_limits"),
        "task_executable": bool(physics.get("out_of_bowl_at_end") and physics.get("on_counter_at_placement")),
    }
    thresholds = {
        "max_camera_reprojection_px": _float(options, "max_camera_reprojection_px", 15.0),
        "max_penetration_m": _float(options, "max_penetration_m", 0.005),
        "max_trajectory_rmse_rad": _float(options, "max_trajectory_rmse_rad", 0.02),
        "max_placement_error_m": _float(options, "max_placement_error_m", 0.15),
    }
    checks = {
        "camera": metrics["camera_reprojection_mean_px"] is not None
        and metrics["camera_reprojection_mean_px"] <= thresholds["max_camera_reprojection_px"],
        "collision": metrics["max_penetration_m"] is not None
        and metrics["max_penetration_m"] <= thresholds["max_penetration_m"],
        "trajectory": metrics["trajectory_rmse_rad"] is not None
        and metrics["trajectory_rmse_rad"] <= thresholds["max_trajectory_rmse_rad"],
        "placement": metrics["placement_error_m"] is not None
        and metrics["placement_error_m"] <= thresholds["max_placement_error_m"],
        "joint_limits": metrics["joint_limits_ok"] is True,
        "velocity_limits": metrics["velocity_limits_ok"] is True,
        "task_executable": metrics["task_executable"] is True,
    }
    return {"metrics": metrics, "thresholds": thresholds, "checks": checks, "passed": all(checks.values())}


def validate(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("validate")
    if stage.backend == "external":
        payload = _external(config, stage, "validate", stage_dir, iteration)
        if payload.get("passed") is not True:
            raise StageError("external simulation validation did not pass")
        return payload
    if stage.backend == "artifact":
        source = Path(
            str(stage.options.get("validation", config.paths["gpt6_real2sim"] / "real2sim_ep0/validation.json"))
        ).resolve()
        if not source.is_file():
            raise StageError(f"validation artifact does not exist: {source}")
        report = _normalise_real2sim_validation(read_json(source), stage.options)
        report.update({"authenticity": "archived_real2sim_artifact", "source": str(source)})
    elif stage.backend == "mock":
        metrics = {
            "camera_reprojection_mean_px": 2.0,
            "geometry_chamfer_m": 0.004,
            "max_penetration_m": 0.0004,
            "trajectory_rmse_rad": 0.003,
            "robot_camera_alignment_ok": True,
            "articulation_ok": True,
            "task_executable": True,
        }
        report = {
            "authenticity": "synthetic_smoke_test",
            "metrics": metrics,
            "thresholds": {
                "max_camera_reprojection_px": _float(stage.options, "max_camera_reprojection_px", 5.0),
                "max_penetration_m": _float(stage.options, "max_penetration_m", 0.005),
                "max_trajectory_rmse_rad": _float(stage.options, "max_trajectory_rmse_rad", 0.02),
            },
        }
        report["checks"] = {
            "visual": metrics["camera_reprojection_mean_px"] <= report["thresholds"]["max_camera_reprojection_px"],
            "geometry": metrics["geometry_chamfer_m"] <= 0.01,
            "alignment": metrics["robot_camera_alignment_ok"],
            "collision": metrics["max_penetration_m"] <= report["thresholds"]["max_penetration_m"],
            "articulation": metrics["articulation_ok"],
            "physics": metrics["trajectory_rmse_rad"] <= report["thresholds"]["max_trajectory_rmse_rad"],
            "task_executable": metrics["task_executable"],
        }
        report["passed"] = all(report["checks"].values())
    else:
        raise StageError(f"validate backend {stage.backend!r} cannot run")
    write_json(stage_dir / "validation.json", report)
    if not report["passed"]:
        raise StageError("digital twin validation gate failed")
    return report


def generate(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("generate")
    if stage.backend == "external":
        return _external(config, stage, "generate", stage_dir, iteration)
    if stage.backend != "mock":
        raise StageError(f"generate backend {stage.backend!r} cannot run")
    count = int(stage.options.get("episodes", 8))
    horizon = int(stage.options.get("horizon", 12))
    rng = random.Random(config.seed + iteration * 1009)
    feedback: list[dict[str, Any]] = []
    if iteration > 0:
        feedback_path = stage_dir.parent.parent / f"iteration-{iteration - 1:03d}/improve/sampling_requests.jsonl"
        if feedback_path.is_file():
            feedback = read_jsonl(feedback_path)
            requested = sum(int(item.get("requested_episodes", 0)) for item in feedback)
            count += min(requested, int(stage.options.get("max_targeted_episodes", 20)))
    targeted_scenarios = [
        item["scenario"]
        for item in feedback
        for _ in range(max(1, int(item.get("priority", 1))))
        if item.get("scenario")
    ]
    rows = []
    accepted = 0
    for episode in range(count):
        score = round(0.72 + 0.27 * rng.random(), 5)
        success = score >= _float(stage.options, "quality_threshold", 0.75)
        trajectory = []
        for frame in range(horizon):
            state = [round(rng.uniform(-1, 1), 6) for _ in range(16)]
            action = [round(value + rng.uniform(-0.03, 0.03), 6) for value in state]
            trajectory.append({"frame": frame, "observation.state": state, "action": action})
        scenario = rng.choice(targeted_scenarios) if targeted_scenarios else "nominal"
        row = {
            "episode_id": f"iter{iteration:03d}-ep{episode:05d}",
            "task": "pick the target and place it in the receptacle",
            "target_scenario": scenario,
            "generator": "mock_cap_expert",
            "success": success,
            "quality_score": score,
            "trajectory": trajectory,
        }
        rows.append(row)
        accepted += int(success)
    write_jsonl(stage_dir / "rollouts.jsonl", rows)
    filtered = [row for row in rows if row["success"]]
    write_jsonl(stage_dir / "successful_rollouts.jsonl", filtered)
    return {
        "authenticity": "synthetic_smoke_test",
        "task_generator": "GPT/CaP contract mock",
        "rollout_engine": "RoboTwin contract mock",
        "attempted": count,
        "accepted": accepted,
        "success_rate": accepted / count if count else 0.0,
        "feedback_requests_consumed": len(feedback),
        "targeted_scenarios": sorted(set(targeted_scenarios)),
        "rollouts": str(stage_dir / "successful_rollouts.jsonl"),
    }


def dataset(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("dataset")
    if stage.backend == "external":
        return _external(config, stage, "dataset", stage_dir, iteration)
    if stage.backend == "catalog":
        source = Path(str(stage.options.get("source", ""))).resolve()
        report = audit_lerobot_dataset(source, bool(stage.options.get("verify_files", True)))
        report.update({"authenticity": "existing_dataset", "source": str(source)})
        write_json(stage_dir / "dataset_audit.json", report)
        if not report["valid"]:
            raise StageError(f"LeRobot dataset audit failed with {len(report['errors'])} errors")
        return report
    if stage.backend != "mock":
        raise StageError(f"dataset backend {stage.backend!r} cannot run")
    rollout_path = stage_dir.parent / "generate/successful_rollouts.jsonl"
    rows = read_jsonl(rollout_path)
    metadata = {
        "codebase_version": "autopolicy-smoke-v1",
        "format": "LeRobotDataset-compatible contract fixture",
        "robot_type": "mock_dual_arm",
        "fps": 10,
        "total_episodes": len(rows),
        "total_frames": sum(len(row["trajectory"]) for row in rows),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16]},
            "action": {"dtype": "float32", "shape": [16]},
        },
        "note": "Smoke fixture uses JSONL; external conversion writes canonical Parquet/video LeRobotDataset.",
    }
    write_json(stage_dir / "info.json", metadata)
    write_jsonl(stage_dir / "episodes.jsonl", rows)
    return {"authenticity": "synthetic_smoke_test", **metadata, "root": str(stage_dir)}


def train(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("train")
    if stage.backend == "external":
        return _external(config, stage, "train", stage_dir, iteration)
    if stage.backend == "disabled":
        return {"authenticity": "none", "skipped": True, "reason": "training disabled"}
    if stage.backend != "mock":
        raise StageError(f"train backend {stage.backend!r} cannot run")
    steps = int(stage.options.get("steps", 20))
    curves = {
        family: [round(start * (0.92**step), 6) for step in range(steps)]
        for family, start in (("vla", 1.2), ("wam", 0.9))
    }
    write_json(stage_dir / "learning_curves.json", curves)
    model = {
        "schema": "autopolicy.policy/v1",
        "authenticity": "synthetic_smoke_test",
        "families": ["VLA", "WAM"],
        "dataset_iteration": iteration,
        "steps": steps,
        "deployable": False,
    }
    write_json(stage_dir / "policy.json", model)
    return {**model, "checkpoint": str(stage_dir / "policy.json")}


def evaluate(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("evaluate")
    if stage.backend == "external":
        return _external(config, stage, "evaluate", stage_dir, iteration)
    if stage.backend == "disabled":
        return {"authenticity": "none", "skipped": True, "success_rate": 0.0}
    if stage.backend != "mock":
        raise StageError(f"evaluate backend {stage.backend!r} cannot run")
    scenarios = ["nominal", "camera_shift", "lighting_ood", "friction_ood", "object_pose_ood"]
    rng = random.Random(config.seed + iteration * 8191)
    rows = []
    for scenario in scenarios:
        trials = int(stage.options.get("trials_per_scenario", 10))
        difficulty = 0.08 * scenarios.index(scenario)
        successes = sum(rng.random() < min(0.98, 0.72 + 0.08 * iteration - difficulty) for _ in range(trials))
        rows.append(
            {
                "scenario": scenario,
                "trials": trials,
                "successes": successes,
                "success_rate": successes / trials,
                "failure_count": trials - successes,
            }
        )
    total_trials = sum(row["trials"] for row in rows)
    total_successes = sum(row["successes"] for row in rows)
    failures = [row for row in rows if row["failure_count"]]
    write_jsonl(stage_dir / "scenarios.jsonl", rows)
    write_jsonl(stage_dir / "failures.jsonl", failures)
    return {
        "authenticity": "synthetic_smoke_test",
        "success_rate": total_successes / total_trials,
        "target": config.success_target,
        "target_met": total_successes / total_trials >= config.success_target,
        "scenarios": rows,
        "failures": str(stage_dir / "failures.jsonl"),
    }


def deploy(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("deploy")
    if stage.backend == "disabled":
        return {
            "authenticity": "none",
            "skipped": True,
            "reason": "real robot deployment is disabled by default and requires explicit operator approval",
        }
    if stage.backend == "external":
        if stage.options.get("operator_approved") is not True:
            raise SafetyError("external deployment requires stages.deploy.options.operator_approved=true")
        if not stage.options.get("emergency_stop"):
            raise SafetyError("external deployment requires a documented emergency_stop setting")
        return _external(config, stage, "deploy", stage_dir, iteration)
    if stage.backend == "mock":
        report = {
            "authenticity": "synthetic_smoke_test",
            "mode": "dry_run",
            "commands_sent_to_hardware": 0,
            "safety_interlocks_checked": ["workspace", "joint_limits", "velocity", "emergency_stop"],
            "deployable": False,
        }
        write_json(stage_dir / "deployment.json", report)
        return report
    raise StageError(f"deploy backend {stage.backend!r} cannot run")


def improve(config: PipelineConfig, stage_dir: Path, iteration: int) -> dict[str, Any]:
    stage = config.stage("improve")
    if stage.backend == "external":
        return _external(config, stage, "improve", stage_dir, iteration)
    if stage.backend == "disabled":
        return {"authenticity": "none", "skipped": True}
    failures_path = stage_dir.parent / "evaluate/failures.jsonl"
    failures = read_jsonl(failures_path) if failures_path.is_file() else []
    requests = [
        {
            "priority": row["failure_count"],
            "scenario": row["scenario"],
            "requested_episodes": max(4, row["failure_count"] * 2),
            "strategy": "targeted_domain_randomization_and_expert_rollout",
            "source_iteration": iteration,
        }
        for row in sorted(failures, key=lambda item: item["failure_count"], reverse=True)
    ]
    write_jsonl(stage_dir / "sampling_requests.jsonl", requests)
    return {
        "authenticity": "synthetic_smoke_test" if stage.backend == "mock" else "derived_feedback",
        "failure_regions": len(requests),
        "requested_episodes": sum(row["requested_episodes"] for row in requests),
        "requests": str(stage_dir / "sampling_requests.jsonl"),
    }


STAGE_FUNCTIONS = {
    "reconstruct": reconstruct,
    "validate": validate,
    "generate": generate,
    "dataset": dataset,
    "train": train,
    "evaluate": evaluate,
    "deploy": deploy,
    "improve": improve,
}
