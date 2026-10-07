#!/usr/bin/env python3
"""Auditable adapters for the real task1 SmolVLA pipeline.

These adapters deliberately distinguish re-validation of existing experiment
artifacts from work performed in the current run.  They never label the
single-video, hypothesis-fitted replica as an automatic metric reconstruction
or its MuJoCo results as real-robot evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from autopolicy.dataset import audit_lerobot_dataset
from autopolicy.io import file_record, read_json as _read_json, read_jsonl, sha256_file, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1].resolve()
TASK = "抓取桌面上的水瓶并抬离桌面"


def read_json(path: Path) -> Any:
    """Read archived task1 metadata after mapping the source checkout to this one."""
    def relocate(value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith(str(ROOT)):
                return value
            source_root = "/data/zxy/autopolicy"
            if value == source_root or value.startswith(source_root + "/"):
                return str(ROOT) + value[len(source_root):]
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    return relocate(_read_json(path))


def project_path(value: str | Path, *, file: bool | None = None) -> Path:
    path = Path(value).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"path must stay under {ROOT}: {path}")
    if file is True and not path.is_file():
        raise FileNotFoundError(path)
    if file is False and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def output_dir(value: str | Path) -> Path:
    path = project_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def finish(directory: Path, payload: dict[str, Any]) -> int:
    write_json(directory / "result.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_child(command: list[str], *, cuda_device: str | None = None) -> None:
    environment = os.environ.copy()
    if cuda_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_device
    print(json.dumps({"child_command": command}, ensure_ascii=False), flush=True)
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"child command failed with exit code {completed.returncode}: {command}")


def resolve_cuda_device(requested: str) -> str:
    if requested != "auto":
        return requested
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("cannot select a CUDA device automatically: nvidia-smi failed")
    choices = []
    for line in completed.stdout.splitlines():
        index, free_mib = (item.strip() for item in line.split(",", 1))
        choices.append((int(free_mib), index))
    if not choices:
        raise RuntimeError("cannot select a CUDA device automatically: no GPUs reported")
    return max(choices)[1]


def reconstruct(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    source_video = project_path(args.source_video, file=True)
    normalized_video = project_path(args.normalized_video, file=True)
    metadata_path = project_path(args.metadata, file=True)
    scene = project_path(args.scene, file=True)
    modeling_report_path = project_path(args.modeling_report, file=True)
    metadata = read_json(metadata_path)
    modeling = read_json(modeling_report_path)

    checks = {
        "source_video_sha256": sha256_file(source_video) == metadata["source"]["sha256"],
        "normalized_video_sha256": sha256_file(normalized_video)
        == metadata["normalized"]["sha256"],
        "scene_exists": scene.is_file(),
        "modeling_report_passed": modeling.get("passed") is True,
        "task_fitted_replica_declares_limitations": bool(modeling.get("limitations")),
        "camera_is_explicitly_uncalibrated": modeling.get("camera_model", {}).get(
            "calibrated_from_video"
        )
        is False,
        "robot_identity_is_explicitly_unconfirmed": modeling.get("robot_model", {}).get(
            "exact_identity_confirmed"
        )
        is False,
    }
    if not all(checks.values()):
        raise ValueError(f"task1 reconstruction artifact checks failed: {checks}")
    payload = {
        "schema": "autopolicy.task1_reconstruction_reference/v1",
        "authenticity": "existing_single_video_task_fitted_replica",
        "automated_in_this_run": False,
        "reconstruction_scope": "task-focused executable replica, not a full-scene digital twin",
        "source_video": file_record(source_video),
        "normalized_video": file_record(normalized_video),
        "metadata": file_record(metadata_path),
        "scene": str(scene),
        "modeling_report": str(modeling_report_path),
        "task": modeling.get("task", TASK),
        "checks": checks,
        "assumptions": {
            "robot": modeling["robot_model"],
            "camera": modeling["camera_model"],
            "scale": modeling["scale_hypotheses"],
        },
        "limitations": modeling["limitations"],
        "executable": True,
    }
    return finish(destination, payload)


def validate(args: argparse.Namespace) -> int:
    import mujoco
    import numpy as np

    destination = output_dir(args.output)
    scene = project_path(args.scene, file=True)
    modeling_report_path = project_path(args.modeling_report, file=True)
    source_scene = project_path(args.robot_source_scene, file=True)
    archived = read_json(modeling_report_path)
    model = mujoco.MjModel.from_xml_path(str(scene))
    source_model = mujoco.MjModel.from_xml_path(str(source_scene))
    data = mujoco.MjData(model)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "task1_head_camera")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task1_bottle_free")
    grasp_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    if source_model.nkey < 1:
        raise ValueError("robot source scene has no home-state keyframe")
    data.qpos[:16] = source_model.key_qpos[0, :16]
    if bottle_joint >= 0:
        qpos_index = int(model.jnt_qposadr[bottle_joint])
        data.qpos[qpos_index : qpos_index + 7] = [0.47, 0.16, 0.762, 1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    named_checks = {
        "camera_present": camera_id >= 0,
        "bottle_present": bottle_id >= 0,
        "bottle_free_joint_present": bottle_joint >= 0,
        "left_grasp_site_present": grasp_site >= 0,
        "state_dimensions_at_least_16": model.nq >= 16,
        "action_dimensions_equal_14": model.nu == 14,
        "finite_forward_dynamics_state": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.xpos).all()
        ),
        "archived_scripted_grasp_passed": archived.get("passed") is True,
        "archived_penetration_below_1cm": float(
            archived.get("metrics", {}).get("max_collision_penetration_m", math.inf)
        )
        < 0.01,
        "archived_bottle_lifted": archived.get("metrics", {}).get(
            "bottle_lifted_above_0_83m"
        )
        is True,
    }
    passed = all(named_checks.values())
    payload = {
        "schema": "autopolicy.task1_simulation_validation/v1",
        "authenticity": "current_executable_check_plus_archived_mujoco_task_validation",
        "automated_in_this_run": True,
        "scene": str(scene),
        "archived_task_validation": str(modeling_report_path),
        "simulator": "MuJoCo",
        "model_dimensions": {
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
            "bodies": model.nbody,
            "geometries": model.ngeom,
            "cameras": model.ncam,
        },
        "checks": named_checks,
        "passed": passed,
        "limitations": archived.get("limitations", []),
    }
    finish(destination, payload)
    if not passed:
        raise SystemExit("task1 simulation validation failed")
    return 0


def generate(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    collection_path = project_path(args.collection_report, file=True)
    filter_manifest_path = project_path(args.filter_manifest, file=True)
    dataset_root = project_path(args.dataset, file=False)
    collection = read_json(collection_path)
    filtering = read_json(filter_manifest_path)
    audit = audit_lerobot_dataset(dataset_root, verify_files=True)
    if not audit["valid"]:
        raise ValueError(f"expert dataset audit failed: {audit['errors']}")
    policy_inputs = collection.get("policy_input", filtering.get("policy_inputs", []))
    if policy_inputs != ["observation.images.camera_00", "observation.state", "task"]:
        raise ValueError(f"unexpected policy inputs: {policy_inputs}")
    attempted = int(filtering["source_episodes"])
    accepted = int(filtering["output_episodes"])
    payload = {
        "schema": "autopolicy.task1_expert_generation_reference/v1",
        "authenticity": "existing_mujoco_expert_rollouts",
        "automated_in_this_run": False,
        "generator": "MuJoCo position-IK expert with success filtering",
        "task": collection.get("task", TASK),
        "attempted": attempted,
        "accepted": accepted,
        "success_rate": accepted / attempted,
        "frames": audit["total_frames"],
        "policy_inputs": policy_inputs,
        "oracle_inputs_to_policy_or_dataset": filtering.get("oracle_inputs_to_policy", []),
        "dataset": str(dataset_root),
        "collection_report": str(collection_path),
        "filter_manifest": str(filter_manifest_path),
        "audit": audit,
    }
    return finish(destination, payload)


def dataset(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    dataset_root = project_path(args.dataset, file=False)
    audit = audit_lerobot_dataset(dataset_root, verify_files=True)
    info = read_json(dataset_root / "meta/info.json")
    expected = {
        "observation.images.camera_00": [240, 320, 3],
        "observation.state": [16],
        "action": [14],
    }
    shapes = {name: info.get("features", {}).get(name, {}).get("shape") for name in expected}
    checks = {
        "lerobot_audit_valid": audit["valid"],
        "feature_shapes_match": shapes == expected,
        "single_language_task": int(info.get("total_tasks", -1)) == 1,
        "non_empty": audit["total_episodes"] > 0 and audit["total_frames"] > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"task1 VLA dataset checks failed: {checks}; shapes={shapes}")
    payload = {
        "schema": "autopolicy.task1_lerobot_dataset/v1",
        "authenticity": (
            "new_lerobot_vla_dataset" if args.created_in_this_run else "existing_lerobot_vla_dataset"
        ),
        "automated_in_this_run": args.created_in_this_run,
        "root": str(dataset_root),
        "task": TASK,
        "policy_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "action": {"shape": [14], "chunk_shape": [20, 14]},
        "checks": checks,
        **audit,
    }
    return finish(destination, payload)


def train(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    checkpoint = project_path(args.checkpoint, file=False)
    dataset_root = project_path(args.dataset, file=False)
    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
        checkpoint / "train_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete checkpoint: {missing}")
    policy = read_json(checkpoint / "config.json")
    training = read_json(checkpoint / "train_config.json")
    checks = {
        "policy_type_smolvla": training.get("policy", {}).get("type") == "smolvla",
        "rgb_input": "observation.images.camera_00" in policy.get("input_features", {}),
        "state_shape_16": policy.get("input_features", {})
        .get("observation.state", {})
        .get("shape")
        == [16],
        "action_shape_14": policy.get("output_features", {}).get("action", {}).get("shape")
        == [14],
        "chunk_size_20": policy.get("chunk_size") == 20,
        "dataset_matches": Path(training.get("dataset", {}).get("root", "")).resolve()
        == dataset_root,
        "model_weights_nonempty": (checkpoint / "model.safetensors").stat().st_size > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"checkpoint provenance checks failed: {checks}")
    payload = {
        "schema": "autopolicy.task1_smolvla_training_reference/v1",
        "authenticity": "existing_lerobot_smolvla_checkpoint",
        "automated_in_this_run": False,
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "policy_type": "SmolVLA",
        "policy_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "state_shape": [16],
        "action_chunk_shape": [policy["chunk_size"], 14],
        "n_action_steps": policy["n_action_steps"],
        "training_steps": training["steps"],
        "checks": checks,
        "artifacts": [file_record(path) for path in required if path.name != "model.safetensors"],
        "model_weights": {
            "path": str(checkpoint / "model.safetensors"),
            "bytes": (checkpoint / "model.safetensors").stat().st_size,
            "sha256": sha256_file(checkpoint / "model.safetensors"),
        },
    }
    return finish(destination, payload)


def evaluate(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    report_path = project_path(args.robustness_report, file=True)
    checkpoint = project_path(args.checkpoint, file=False)
    dataset_root = project_path(args.dataset, file=False)
    report = read_json(report_path)
    if Path(report["checkpoint"]).resolve() != checkpoint:
        raise ValueError("robustness report checkpoint does not match the configured checkpoint")
    if Path(report["dataset"]).resolve() != dataset_root:
        raise ValueError("robustness report dataset does not match the configured dataset")
    scenarios = report.get("scenarios", [])
    by_name = {row["scenario"]: row for row in scenarios}
    required = {
        "nominal",
        "joint_reset_ood",
        "camera_calibration_ood",
        "dynamics_ood",
        "visual_ood",
        "combined_ood",
    }
    if set(by_name) != required:
        raise ValueError(f"robustness suite mismatch: {sorted(by_name)}")
    gates = {
        "nominal": {"minimum": args.nominal_min, "actual": by_name["nominal"]["success_rate"]},
        "combined_ood": {
            "minimum": args.combined_min,
            "actual": by_name["combined_ood"]["success_rate"],
        },
    }
    for value in gates.values():
        value["passed"] = value["actual"] >= value["minimum"]
    total_successes = sum(int(row["successes"]) for row in scenarios)
    total_episodes = sum(int(row["episodes"]) for row in scenarios)
    failures = [
        {
            "scenario": row["scenario"],
            "successes": row["successes"],
            "trials": row["episodes"],
            "failure_count": int(row["episodes"]) - int(row["successes"]),
            "success_rate": row["success_rate"],
        }
        for row in scenarios
        if int(row["successes"]) < int(row["episodes"])
    ]
    write_jsonl(destination / "failures.jsonl", failures)
    target_met = all(value["passed"] for value in gates.values())
    payload = {
        "schema": "autopolicy.task1_smolvla_evaluation_reference/v1",
        "authenticity": "existing_paired_mujoco_replica_sim_evaluation",
        "automated_in_this_run": False,
        "evaluation_report": str(report_path),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "success_rate": total_successes / total_episodes,
        "target_met": target_met,
        "gates": gates,
        "scenarios": scenarios,
        "failures": str(destination / "failures.jsonl"),
        "videos": [row["video"] for row in scenarios if row.get("video")],
        "limitations": report.get("limitations", []),
    }
    return finish(destination, payload)


def improve(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    evaluation_path = project_path(args.evaluation, file=True)
    evaluation = read_json(evaluation_path)
    requests = []
    for row in evaluation.get("scenarios", []):
        failures = int(row["episodes"]) - int(row["successes"])
        if failures:
            requests.append(
                {
                    "source_iteration": args.iteration,
                    "scenario": row["scenario"],
                    "priority": failures,
                    "requested_episodes": max(8, failures * 2),
                    "strategy": "paired_failure_replay_plus_nominal_replay",
                    "preserve_baseline_behavior": True,
                }
            )
    requests.sort(key=lambda row: row["priority"], reverse=True)
    write_jsonl(destination / "sampling_requests.jsonl", requests)
    payload = {
        "schema": "autopolicy.task1_failure_driven_request/v1",
        "authenticity": "derived_from_mujoco_replica_sim_failures",
        "automated_in_this_run": True,
        "source_evaluation": str(evaluation_path),
        "failure_regions": len(requests),
        "requested_episodes": sum(row["requested_episodes"] for row in requests),
        "requests": str(destination / "sampling_requests.jsonl"),
        "selection_policy": {
            "candidate_must_not_regress_nominal": True,
            "candidate_must_improve_combined_ood": True,
            "fresh_position_seed_required": True,
        },
    }
    return finish(destination, payload)


def collect_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    dataset_root = destination / "lerobot_dataset"
    collection_path = dataset_root / "collection_report.json"
    resumed = False
    command: list[str] | None = None
    if collection_path.is_file():
        resumed = True
    else:
        if dataset_root.exists():
            raise RuntimeError(
                f"incomplete collection exists at {dataset_root}; use a fresh run directory"
            )
        collector = project_path(args.collector, file=True)
        command = [
            sys.executable,
            str(collector),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--sampling",
            "stratified",
            "--repo-id",
            args.repo_id,
            "--root",
            str(dataset_root),
            "--successful-only",
        ]
        run_child(command)
    collection = read_json(collection_path)
    audit = audit_lerobot_dataset(dataset_root, verify_files=True)
    if not audit["valid"] or audit["total_episodes"] < 1:
        raise ValueError(f"newly collected dataset failed audit: {audit}")
    payload = {
        "schema": "autopolicy.task1_expert_generation/v1",
        "authenticity": "new_mujoco_expert_rollouts",
        "automated_in_this_run": not resumed,
        "resumed_existing_completed_stage_output": resumed,
        "task": collection["task"],
        "attempted": collection["attempted_episodes"],
        "accepted": collection["dataset_episodes"],
        "success_rate": collection["expert_success_rate"],
        "frames": collection["frames"],
        "policy_inputs": collection["policy_input"],
        "oracle_inputs_to_policy_or_dataset": [],
        "dataset": str(dataset_root),
        "collection_report": str(collection_path),
        "command": command,
        "audit": audit,
    }
    return finish(destination, payload)


def latest_checkpoint(root: Path) -> Path:
    candidates = sorted(root.glob("checkpoints/*/pretrained_model"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found under {root}")
    return candidates[-1].resolve()


def train_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    dataset_root = project_path(args.dataset, file=False)
    train_config = project_path(args.train_config, file=True)
    base_checkpoint = project_path(args.base_checkpoint, file=False)
    vlm_assets = project_path(args.vlm_assets, file=False)
    checkpoint_root = destination / "checkpoints"
    completed = sorted(checkpoint_root.glob("checkpoints/*/pretrained_model/train_config.json"))
    resumed = bool(completed)
    command: list[str] | None = None
    archived_incomplete: str | None = None
    selected_cuda_device: str | None = None
    if not resumed:
        if checkpoint_root.exists():
            attempt = 1
            archive = destination / f"incomplete-checkpoints-attempt-{attempt:03d}"
            while archive.exists():
                attempt += 1
                archive = destination / f"incomplete-checkpoints-attempt-{attempt:03d}"
            checkpoint_root.replace(archive)
            archived_incomplete = str(archive)
        selected_cuda_device = resolve_cuda_device(args.cuda_device)
        resolved_training = read_json(train_config)
        resolved_training["dataset"]["root"] = str(dataset_root)
        resolved_training["dataset"]["repo_id"] = args.repo_id
        resolved_training["policy"]["pretrained_path"] = str(base_checkpoint)
        resolved_training["policy"]["vlm_model_name"] = str(vlm_assets)
        resolved_training["policy"]["use_peft"] = False
        resolved_training["peft"] = None
        resolved_training["output_dir"] = str(checkpoint_root)
        resolved_training["job_name"] = args.job_name
        resolved_training["steps"] = args.steps
        resolved_training["batch_size"] = args.batch_size
        resolved_training["save_freq"] = args.steps
        resolved_training["wandb"]["enable"] = False
        resolved_training_path = destination / "resolved_train_config.json"
        write_json(resolved_training_path, resolved_training)
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            "--config_path",
            str(resolved_training_path),
            "--dataset.repo_id",
            args.repo_id,
            "--dataset.root",
            str(dataset_root),
            "--output_dir",
            str(checkpoint_root),
            "--job_name",
            args.job_name,
            "--steps",
            str(args.steps),
            "--batch_size",
            str(args.batch_size),
            "--save_freq",
            str(args.steps),
            "--log_freq",
            "1",
            "--wandb.enable",
            "false",
            "--policy.use_peft",
            "false",
        ]
        run_child(command, cuda_device=selected_cuda_device)
    checkpoint = latest_checkpoint(checkpoint_root)
    config = read_json(checkpoint / "config.json")
    weight_candidates = [checkpoint / "adapter_model.safetensors", checkpoint / "model.safetensors"]
    weights = next((path for path in weight_candidates if path.is_file()), None)
    if weights is None:
        raise FileNotFoundError(f"checkpoint has no policy weights: {checkpoint}")
    checks = {
        "smolvla": config.get("type") == "smolvla" or "input_features" in config,
        "rgb_input": "observation.images.camera_00" in config.get("input_features", {}),
        "state_shape_16": config.get("input_features", {})
        .get("observation.state", {})
        .get("shape")
        == [16],
        "action_shape_14": config.get("output_features", {}).get("action", {}).get("shape")
        == [14],
        "chunk_size_20": config.get("chunk_size") == 20,
        "weights_nonempty": weights.stat().st_size > 0,
    }
    if not all(checks.values()):
        raise ValueError(f"new checkpoint validation failed: {checks}")
    payload = {
        "schema": "autopolicy.task1_smolvla_training/v1",
        "authenticity": "new_lerobot_smolvla_smoke_checkpoint",
        "automated_in_this_run": not resumed,
        "resumed_existing_completed_stage_output": resumed,
        "qualification": "pipeline execution smoke test; not a deployment candidate",
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "training_steps": args.steps,
        "batch_size": args.batch_size,
        "policy_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "action_chunk_shape": [20, 14],
        "checks": checks,
        "weights": file_record(weights),
        "command": command,
        "archived_incomplete_attempt": archived_incomplete,
        "cuda_device": selected_cuda_device,
    }
    return finish(destination, payload)


def evaluate_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    checkpoint_root = project_path(args.checkpoint_root, file=False)
    dataset_root = project_path(args.dataset, file=False)
    suite = project_path(args.suite, file=True)
    checkpoint = latest_checkpoint(checkpoint_root)
    evaluation_root = destination / "robustness"
    report_path = evaluation_root / "robustness_report.json"
    resumed = report_path.is_file()
    command: list[str] | None = None
    selected_cuda_device: str | None = None
    if not resumed:
        runner = project_path(args.runner, file=True)
        selected_cuda_device = resolve_cuda_device(args.cuda_device)
        command = [
            sys.executable,
            str(runner),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(dataset_root),
            "--output",
            str(evaluation_root),
            "--config",
            str(suite),
            "--episodes",
            str(args.episodes),
            "--seed",
            str(args.seed),
            "--perturbation-seed",
            str(args.perturbation_seed),
            "--render-scenarios",
            "combined_ood",
        ]
        run_child(command, cuda_device=selected_cuda_device)
    report = read_json(report_path)
    scenarios = report["scenarios"]
    by_name = {row["scenario"]: row for row in scenarios}
    required = {"nominal", "combined_ood"}
    if set(by_name) != required:
        raise ValueError(f"smoke robustness suite mismatch: {sorted(by_name)}")
    failures = [
        {
            "scenario": row["scenario"],
            "successes": row["successes"],
            "trials": row["episodes"],
            "failure_count": int(row["episodes"]) - int(row["successes"]),
            "success_rate": row["success_rate"],
        }
        for row in scenarios
        if int(row["successes"]) < int(row["episodes"])
    ]
    write_jsonl(destination / "failures.jsonl", failures)
    total_successes = sum(int(row["successes"]) for row in scenarios)
    total_episodes = sum(int(row["episodes"]) for row in scenarios)
    payload = {
        "schema": "autopolicy.task1_smolvla_smoke_evaluation/v1",
        "authenticity": "new_mujoco_replica_sim_smoke_evaluation",
        "automated_in_this_run": not resumed,
        "resumed_existing_completed_stage_output": resumed,
        "qualification": "pipeline execution smoke test; sample size is not a performance estimate",
        "evaluation_report": str(report_path),
        "checkpoint": str(checkpoint),
        "dataset": str(dataset_root),
        "success_rate": total_successes / total_episodes,
        "target_met": False,
        "target_not_evaluated_reason": "smoke suite is intentionally too small for promotion",
        "scenarios": scenarios,
        "failures": str(destination / "failures.jsonl"),
        "videos": [row["video"] for row in scenarios if row.get("video")],
        "command": command,
        "cuda_device": selected_cuda_device,
        "limitations": report.get("limitations", []),
    }
    return finish(destination, payload)


def correction_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    checkpoint = project_path(args.checkpoint, file=False)
    policy_dataset = project_path(args.dataset, file=False)
    failure_report = project_path(args.failure_report, file=True)
    correction_root = destination / "lerobot_dataset"
    report_path = correction_root / "collection_report.json"
    request = None
    if args.sampling_requests:
        requests_path = project_path(args.sampling_requests, file=True)
        report_scenario = read_json(failure_report).get("scenario")
        matching = [row for row in read_jsonl(requests_path) if row.get("scenario") == report_scenario]
        if not matching:
            raise ValueError(
                f"failure scenario {report_scenario!r} is absent from {requests_path}"
            )
        request = matching[0]
    resumed = False
    commands: list[list[str]] = []
    selected_cuda_device: str | None = None
    archived_attempts: list[str] = []
    case_offset = 0

    def archive_current() -> None:
        nonlocal case_offset
        if not correction_root.exists():
            return
        collection = read_json(report_path) if report_path.is_file() else None
        if collection is not None:
            # Retry the same failure after upgrading an older correction planner.
            advance = int(collection.get("source_failures", args.max_cases))
            if collection.get("correction_planner") != "orientation_aware_wrist_regrasp_v5":
                advance = 0
            case_offset = int(collection.get("case_offset", case_offset)) + advance
        attempt = 1
        archive = destination / f"rejected-corrections-attempt-{attempt:03d}"
        while archive.exists():
            attempt += 1
            archive = destination / f"rejected-corrections-attempt-{attempt:03d}"
        correction_root.replace(archive)
        archived_attempts.append(str(archive))

    if report_path.is_file():
        existing = read_json(report_path)
        if int(existing.get("saved_expert_corrections", 0)) > 0:
            resumed = True
        else:
            archive_current()
    elif correction_root.exists():
        archive_current()

    source_failure_count = sum(
        not row["success"] for row in read_json(failure_report)["episode_results"]
    )
    while not resumed:
        if case_offset >= source_failure_count:
            raise ValueError(
                f"all {source_failure_count} policy failures were attempted without a successful correction"
            )
        collector = project_path(args.collector, file=True)
        if selected_cuda_device is None:
            selected_cuda_device = resolve_cuda_device(args.cuda_device)
        command = [
            sys.executable,
            str(collector),
            "--checkpoint",
            str(checkpoint),
            "--dataset-root",
            str(policy_dataset),
            "--failure-report",
            str(failure_report),
            "--root",
            str(correction_root),
            "--repo-id",
            args.repo_id,
            "--rollin-seconds",
            str(args.rollin_seconds),
            "--max-cases",
            str(args.max_cases),
            "--case-offset",
            str(case_offset),
        ]
        run_child(command, cuda_device=selected_cuda_device)
        commands.append(command)
        collection = read_json(report_path)
        if int(collection.get("saved_expert_corrections", 0)) > 0:
            break
        archive_current()
    collection = read_json(report_path)
    if int(collection["saved_expert_corrections"]) < 1:
        raise ValueError("policy roll-in produced no successful expert correction")
    audit = audit_lerobot_dataset(correction_root, verify_files=True)
    if not audit["valid"]:
        raise ValueError(f"correction dataset failed audit: {audit}")
    attempt_reports = [
        read_json(Path(path) / "collection_report.json")
        for path in archived_attempts
        if (Path(path) / "collection_report.json").is_file()
    ] + [collection]
    payload = {
        "schema": "autopolicy.task1_failure_correction_generation/v1",
        "authenticity": "new_policy_rollin_expert_corrections",
        "automated_in_this_run": not resumed,
        "resumed_existing_completed_stage_output": resumed,
        "sampling_request_consumed": request,
        "failure_report": str(failure_report),
        "rollin_checkpoint": str(checkpoint),
        "policy_dataset": str(policy_dataset),
        "attempted": sum(int(report["source_failures"]) for report in attempt_reports),
        "accepted": collection["saved_expert_corrections"],
        "rejected": sum(
            int(report["rejected_expert_corrections"]) for report in attempt_reports
        ),
        "frames": collection["frames"],
        "dataset": str(correction_root),
        "audit": audit,
        "cuda_device": selected_cuda_device,
        "commands": commands,
        "archived_rejected_attempts": archived_attempts,
        "attempted_case_offsets": [report.get("case_offset", 0) for report in attempt_reports],
    }
    return finish(destination, payload)


def merge_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    base = project_path(args.base_dataset, file=False)
    augmentation = project_path(args.augmentation_dataset, file=False)
    merged_root = destination / "lerobot_dataset"
    manifest_path = merged_root / "merge_manifest.json"
    resumed = manifest_path.is_file()
    command: list[str] | None = None
    if not resumed:
        if merged_root.exists():
            raise RuntimeError(f"incomplete merged dataset exists at {merged_root}")
        merger = project_path(args.merger, file=True)
        command = [
            sys.executable,
            str(merger),
            "--base-root",
            str(base),
            "--augmentation-roots",
            str(augmentation),
            "--output-root",
            str(merged_root),
            "--repo-id",
            args.repo_id,
            "--base-role",
            args.base_role,
            "--augmentation-role",
            args.augmentation_role,
        ]
        run_child(command)
    manifest = read_json(manifest_path)
    audit = audit_lerobot_dataset(merged_root, verify_files=True)
    if not audit["valid"]:
        raise ValueError(f"merged dataset failed audit: {audit}")
    payload = {
        "schema": "autopolicy.task1_failure_replay_dataset/v1",
        "authenticity": "new_nominal_replay_plus_policy_rollin_corrections",
        "automated_in_this_run": not resumed,
        "resumed_existing_completed_stage_output": resumed,
        "root": str(merged_root),
        "base_dataset": str(base),
        "augmentation_dataset": str(augmentation),
        "base_role": args.base_role,
        "augmentation_role": args.augmentation_role,
        "episodes": audit["total_episodes"],
        "frames": audit["total_frames"],
        "merge_manifest": str(manifest_path),
        "audit": audit,
        "command": command,
    }
    return finish(destination, payload)


def cycle_correction_run(args: argparse.Namespace) -> int:
    pipeline_run = project_path(args.pipeline_run_dir)
    if args.iteration == 0:
        checkpoint = project_path(args.initial_checkpoint, file=False)
        dataset_root = project_path(args.initial_dataset, file=False)
        failure_report = project_path(args.initial_failure_report, file=True)
        sampling_requests = project_path(args.initial_sampling_requests, file=True)
        source = "configured_initial_failure"
    else:
        previous = pipeline_run / f"iteration-{args.iteration - 1:03d}"
        checkpoint = latest_checkpoint(previous / "train/checkpoints")
        dataset_root = project_path(previous / "dataset/lerobot_dataset", file=False)
        evaluation = read_json(project_path(previous / "evaluate/result.json", file=True))
        failed_scenarios = [
            row
            for row in evaluation.get("scenarios", [])
            if int(row["successes"]) < int(row["episodes"])
        ]
        if not failed_scenarios:
            raise ValueError("previous candidate evaluation contains no failed episodes to correct")
        failed_scenarios.sort(
            key=lambda row: (
                row["scenario"] != "combined_ood",
                -(int(row["episodes"]) - int(row["successes"])),
            )
        )
        scenario = failed_scenarios[0]["scenario"]
        failure_report = project_path(
            previous / f"evaluate/candidate/{scenario}/evaluation_report.json", file=True
        )
        sampling_requests = project_path(
            previous / "improve/sampling_requests.jsonl", file=True
        )
        source = f"previous_iteration_{args.iteration - 1:03d}_{scenario}_failure"
    namespace = argparse.Namespace(
        output=args.output,
        collector=args.collector,
        checkpoint=str(checkpoint),
        dataset=str(dataset_root),
        failure_report=str(failure_report),
        sampling_requests=str(sampling_requests),
        repo_id=f"{args.repo_id_prefix}_iter{args.iteration:03d}",
        rollin_seconds=args.rollin_seconds,
        max_cases=args.max_cases,
        cuda_device=args.cuda_device,
    )
    result = correction_run(namespace)
    result_path = project_path(Path(args.output) / "result.json", file=True)
    payload = read_json(result_path)
    payload["cycle_input_source"] = source
    payload["cycle_iteration"] = args.iteration
    write_json(result_path, payload)
    return result


def cycle_merge_run(args: argparse.Namespace) -> int:
    pipeline_run = project_path(args.pipeline_run_dir)
    if args.iteration == 0:
        base = project_path(args.initial_dataset, file=False)
        base_role = "nominal_success_filtered"
    else:
        base = project_path(
            pipeline_run / f"iteration-{args.iteration - 1:03d}/dataset/lerobot_dataset",
            file=False,
        )
        base_role = "accumulated_nominal_and_failure_replay"
    augmentation = project_path(
        pipeline_run / f"iteration-{args.iteration:03d}/generate/lerobot_dataset",
        file=False,
    )
    return merge_run(
        argparse.Namespace(
            output=args.output,
            merger=args.merger,
            base_dataset=str(base),
            augmentation_dataset=str(augmentation),
            repo_id=f"{args.repo_id_prefix}_iter{args.iteration:03d}",
            base_role=base_role,
            augmentation_role="failure_driven_policy_rollin_correction",
        )
    )


def paired_outcomes(baseline_report: dict[str, Any], candidate_report: dict[str, Any]) -> dict[str, int]:
    baseline_rows = baseline_report["episode_results"]
    candidate_rows = candidate_report["episode_results"]
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("paired evaluation episode counts differ")
    outcomes = {
        "both_success": 0,
        "both_failure": 0,
        "baseline_only_success": 0,
        "candidate_only_success": 0,
    }
    for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
        if baseline["episode"] != candidate["episode"]:
            raise ValueError("paired evaluation episode indices differ")
        if baseline["bottle_xy_ground_truth_for_scoring_only"] != candidate[
            "bottle_xy_ground_truth_for_scoring_only"
        ]:
            raise ValueError("paired evaluation bottle positions differ")
        if baseline.get("perturbations") != candidate.get("perturbations"):
            raise ValueError("paired evaluation perturbations differ")
        baseline_success = bool(baseline["success"])
        candidate_success = bool(candidate["success"])
        key = (
            "both_success"
            if baseline_success and candidate_success
            else "both_failure"
            if not baseline_success and not candidate_success
            else "baseline_only_success"
            if baseline_success
            else "candidate_only_success"
        )
        outcomes[key] += 1
    return outcomes


def paired_evaluate_run(args: argparse.Namespace) -> int:
    destination = output_dir(args.output)
    baseline_checkpoint = project_path(args.baseline_checkpoint, file=False)
    candidate_root = project_path(args.candidate_checkpoint_root, file=False)
    candidate_checkpoint = latest_checkpoint(candidate_root)
    dataset_root = project_path(args.dataset, file=False)
    suite = project_path(args.suite, file=True)
    runner = project_path(args.runner, file=True)
    selected_cuda_device: str | None = None
    commands: list[list[str]] = []
    reports: dict[str, dict[str, Any]] = {}
    newly_executed = False
    position_seed = args.seed + args.iteration * args.seed_stride
    perturbation_seed = args.perturbation_seed + args.iteration * args.seed_stride
    for label, checkpoint in (
        ("baseline", baseline_checkpoint),
        ("candidate", candidate_checkpoint),
    ):
        evaluation_root = destination / label
        report_path = evaluation_root / "robustness_report.json"
        if not report_path.is_file():
            if selected_cuda_device is None:
                selected_cuda_device = resolve_cuda_device(args.cuda_device)
            command = [
                sys.executable,
                str(runner),
                "--checkpoint",
                str(checkpoint),
                "--dataset-root",
                str(dataset_root),
                "--output",
                str(evaluation_root),
                "--config",
                str(suite),
                "--episodes",
                str(args.episodes),
                "--seed",
                str(position_seed),
                "--perturbation-seed",
                str(perturbation_seed),
                "--render-scenarios",
                "combined_ood",
            ]
            run_child(command, cuda_device=selected_cuda_device)
            commands.append(command)
            newly_executed = True
        reports[label] = read_json(report_path)
    baseline_scenarios = {row["scenario"]: row for row in reports["baseline"]["scenarios"]}
    candidate_scenarios = {row["scenario"]: row for row in reports["candidate"]["scenarios"]}
    if baseline_scenarios.keys() != candidate_scenarios.keys():
        raise ValueError("baseline and candidate robustness suites differ")
    comparisons = []
    failures = []
    for scenario in baseline_scenarios:
        baseline_summary = baseline_scenarios[scenario]
        candidate_summary = candidate_scenarios[scenario]
        baseline_detail = read_json(project_path(baseline_summary["evaluation_report"], file=True))
        candidate_detail = read_json(project_path(candidate_summary["evaluation_report"], file=True))
        outcomes = paired_outcomes(baseline_detail, candidate_detail)
        comparisons.append(
            {
                "scenario": scenario,
                "episodes": args.episodes,
                "baseline_successes": baseline_summary["successes"],
                "baseline_success_rate": baseline_summary["success_rate"],
                "candidate_successes": candidate_summary["successes"],
                "candidate_success_rate": candidate_summary["success_rate"],
                "delta_percentage_points": 100
                * (candidate_summary["success_rate"] - baseline_summary["success_rate"]),
                "paired_outcomes": outcomes,
            }
        )
        failure_count = args.episodes - int(candidate_summary["successes"])
        if failure_count:
            failures.append(
                {
                    "scenario": scenario,
                    "successes": candidate_summary["successes"],
                    "trials": args.episodes,
                    "failure_count": failure_count,
                    "success_rate": candidate_summary["success_rate"],
                }
            )
    write_jsonl(destination / "failures.jsonl", failures)
    comparison_by_name = {row["scenario"]: row for row in comparisons}
    enough_evidence = args.episodes >= args.minimum_qualification_episodes
    nominal_no_regression = comparison_by_name["nominal"]["candidate_successes"] >= comparison_by_name[
        "nominal"
    ]["baseline_successes"]
    combined_improved = comparison_by_name["combined_ood"]["candidate_successes"] > comparison_by_name[
        "combined_ood"
    ]["baseline_successes"]
    promotion_gate = {
        "minimum_episodes_per_scenario": args.minimum_qualification_episodes,
        "episodes_per_scenario": args.episodes,
        "enough_evidence": enough_evidence,
        "nominal_no_regression": nominal_no_regression,
        "combined_ood_strict_improvement": combined_improved,
    }
    promotion_gate["passed"] = all(
        [enough_evidence, nominal_no_regression, combined_improved]
    )
    total_candidate_successes = sum(int(row["successes"]) for row in candidate_scenarios.values())
    total_episodes = args.episodes * len(candidate_scenarios)
    candidate_rows = [candidate_scenarios[name] for name in candidate_scenarios]
    payload = {
        "schema": "autopolicy.task1_paired_candidate_gate/v1",
        "authenticity": "new_paired_mujoco_replica_sim_candidate_evaluation",
        "automated_in_this_run": newly_executed,
        "qualification": (
            "candidate qualification" if enough_evidence else "execution smoke; insufficient sample size"
        ),
        "baseline_checkpoint": str(baseline_checkpoint),
        "candidate_checkpoint": str(candidate_checkpoint),
        "dataset": str(dataset_root),
        "cycle_iteration": args.iteration,
        "position_seed": position_seed,
        "perturbation_seed": perturbation_seed,
        "success_rate": total_candidate_successes / total_episodes,
        "target_met": promotion_gate["passed"],
        "promotion_gate": promotion_gate,
        "comparisons": comparisons,
        "scenarios": candidate_rows,
        "failures": str(destination / "failures.jsonl"),
        "videos": [
            row["video"]
            for report in reports.values()
            for row in report["scenarios"]
            if row.get("video")
        ],
        "cuda_device": selected_cuda_device,
        "commands": commands,
        "limitations": [
            "MuJoCo replica-sim results are not real-robot performance.",
            "A smoke run below the minimum episode count can never promote a checkpoint.",
        ],
    }
    return finish(destination, payload)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="stage", required=True)

    command = commands.add_parser("reconstruct")
    command.add_argument("--output", required=True)
    command.add_argument("--source-video", required=True)
    command.add_argument("--normalized-video", required=True)
    command.add_argument("--metadata", required=True)
    command.add_argument("--scene", required=True)
    command.add_argument("--modeling-report", required=True)
    command.set_defaults(function=reconstruct)

    command = commands.add_parser("validate")
    command.add_argument("--output", required=True)
    command.add_argument("--scene", required=True)
    command.add_argument("--modeling-report", required=True)
    command.add_argument("--robot-source-scene", required=True)
    command.set_defaults(function=validate)

    command = commands.add_parser("generate")
    command.add_argument("--output", required=True)
    command.add_argument("--collection-report", required=True)
    command.add_argument("--filter-manifest", required=True)
    command.add_argument("--dataset", required=True)
    command.set_defaults(function=generate)

    command = commands.add_parser("dataset")
    command.add_argument("--output", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--created-in-this-run", action="store_true")
    command.set_defaults(function=dataset)

    command = commands.add_parser("train")
    command.add_argument("--output", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--dataset", required=True)
    command.set_defaults(function=train)

    command = commands.add_parser("evaluate")
    command.add_argument("--output", required=True)
    command.add_argument("--robustness-report", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--nominal-min", type=float, default=0.85)
    command.add_argument("--combined-min", type=float, default=0.50)
    command.set_defaults(function=evaluate)

    command = commands.add_parser("improve")
    command.add_argument("--output", required=True)
    command.add_argument("--evaluation", required=True)
    command.add_argument("--iteration", type=int, required=True)
    command.set_defaults(function=improve)

    command = commands.add_parser("collect-run")
    command.add_argument("--output", required=True)
    command.add_argument("--collector", required=True)
    command.add_argument("--episodes", type=int, required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--repo-id", required=True)
    command.set_defaults(function=collect_run)

    command = commands.add_parser("train-run")
    command.add_argument("--output", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--train-config", required=True)
    command.add_argument(
        "--base-checkpoint",
        default=str(ROOT / "checkpoints/task1_smolvla_v5_homereset_stage2_restartlr_pyav_2000steps_b32/checkpoints/002000/pretrained_model"),
    )
    command.add_argument(
        "--vlm-assets", default=str(ROOT / "models/smolvlm2_500m_video_instruct_assets")
    )
    command.add_argument("--repo-id", required=True)
    command.add_argument("--job-name", required=True)
    command.add_argument("--steps", type=int, required=True)
    command.add_argument("--batch-size", type=int, required=True)
    command.add_argument("--cuda-device", default="auto")
    command.set_defaults(function=train_run)

    command = commands.add_parser("evaluate-run")
    command.add_argument("--output", required=True)
    command.add_argument("--runner", required=True)
    command.add_argument("--checkpoint-root", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--suite", required=True)
    command.add_argument("--episodes", type=int, required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--perturbation-seed", type=int, required=True)
    command.add_argument("--cuda-device", default="auto")
    command.set_defaults(function=evaluate_run)

    command = commands.add_parser("correction-run")
    command.add_argument("--output", required=True)
    command.add_argument("--collector", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--failure-report", required=True)
    command.add_argument("--sampling-requests")
    command.add_argument("--repo-id", required=True)
    command.add_argument("--rollin-seconds", type=float, default=3.0)
    command.add_argument("--max-cases", type=int, default=2)
    command.add_argument("--cuda-device", default="auto")
    command.set_defaults(function=correction_run)

    command = commands.add_parser("merge-run")
    command.add_argument("--output", required=True)
    command.add_argument("--merger", required=True)
    command.add_argument("--base-dataset", required=True)
    command.add_argument("--augmentation-dataset", required=True)
    command.add_argument("--repo-id", required=True)
    command.add_argument("--base-role", default="nominal_success_filtered")
    command.add_argument(
        "--augmentation-role", default="failure_driven_policy_rollin_correction"
    )
    command.set_defaults(function=merge_run)

    command = commands.add_parser("cycle-correction-run")
    command.add_argument("--output", required=True)
    command.add_argument("--pipeline-run-dir", required=True)
    command.add_argument("--iteration", type=int, required=True)
    command.add_argument("--collector", required=True)
    command.add_argument("--initial-checkpoint", required=True)
    command.add_argument("--initial-dataset", required=True)
    command.add_argument("--initial-failure-report", required=True)
    command.add_argument("--initial-sampling-requests", required=True)
    command.add_argument("--repo-id-prefix", required=True)
    command.add_argument("--rollin-seconds", type=float, default=3.0)
    command.add_argument("--max-cases", type=int, default=1)
    command.add_argument("--cuda-device", default="auto")
    command.set_defaults(function=cycle_correction_run)

    command = commands.add_parser("cycle-merge-run")
    command.add_argument("--output", required=True)
    command.add_argument("--pipeline-run-dir", required=True)
    command.add_argument("--iteration", type=int, required=True)
    command.add_argument("--merger", required=True)
    command.add_argument("--initial-dataset", required=True)
    command.add_argument("--repo-id-prefix", required=True)
    command.set_defaults(function=cycle_merge_run)

    command = commands.add_parser("paired-evaluate-run")
    command.add_argument("--output", required=True)
    command.add_argument("--runner", required=True)
    command.add_argument("--baseline-checkpoint", required=True)
    command.add_argument("--candidate-checkpoint-root", required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--suite", required=True)
    command.add_argument("--episodes", type=int, required=True)
    command.add_argument("--minimum-qualification-episodes", type=int, default=20)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--perturbation-seed", type=int, required=True)
    command.add_argument("--iteration", type=int, default=0)
    command.add_argument("--seed-stride", type=int, default=0)
    command.add_argument("--cuda-device", default="auto")
    command.set_defaults(function=paired_evaluate_run)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
