#!/usr/bin/env python3
"""Closed-loop visual SmolVLA evaluation for the task1 MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
SOURCE_SCENE = ROOT / "vendor/gpt6-real2sim/real2sim_microphones/scene_portable.xml"
TASK = "抓取桌面上的水瓶并抬离桌面"


def _positive_range(values: list[float], name: str) -> tuple[float, float]:
    low, high = map(float, values)
    if low <= 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 < LOW <= HIGH, got {values}")
    return low, high


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two MuJoCo wxyz quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _rotation_vector_to_quat(rotation_vector_rad: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector_rad))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = rotation_vector_rad / angle
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))], dtype=np.float64)


def _apply_visual_perturbation(
    image: np.ndarray,
    brightness_scale: float,
    occlusion_fraction: float,
    occlusion_center_xy: tuple[float, float],
) -> np.ndarray:
    perturbed = np.clip(image.astype(np.float32) * brightness_scale, 0, 255).astype(np.uint8)
    if occlusion_fraction <= 0:
        return perturbed
    height, width = perturbed.shape[:2]
    side_fraction = np.sqrt(occlusion_fraction)
    box_width = max(1, round(width * side_fraction))
    box_height = max(1, round(height * side_fraction))
    center_x = round(occlusion_center_xy[0] * (width - 1))
    center_y = round(occlusion_center_xy[1] * (height - 1))
    x0 = int(np.clip(center_x - box_width // 2, 0, width - box_width))
    y0 = int(np.clip(center_y - box_height // 2, 0, height - box_height))
    perturbed[y0:y0 + box_height, x0:x0 + box_width] = 0
    return perturbed


def make_compatible_checkpoint_view(checkpoint: Path):
    """Create a lightweight view for configs saved by a nearby LeRobot revision."""
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    ignored_fields = {"pretrained_revision"}
    fields_to_remove = ignored_fields.intersection(config)
    fields_to_add = {"type": "smolvla"} if "type" not in config else {}
    if not fields_to_remove and not fields_to_add:
        return checkpoint, None, [], []

    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    overlay = tempfile.TemporaryDirectory(prefix="smolvla-checkpoint-", dir=temp_root)
    overlay_path = Path(overlay.name)
    for child in checkpoint.iterdir():
        if child.name != "config.json":
            os.symlink(child.resolve(), overlay_path / child.name, target_is_directory=child.is_dir())
    for field in fields_to_remove:
        config.pop(field)
    config.update(fields_to_add)
    (overlay_path / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return overlay_path, overlay, sorted(fields_to_remove), sorted(fields_to_add)


def resolve_tokenizer_path(checkpoint: Path, policy):
    bundled = checkpoint.resolve() / "tokenizer"
    if bundled.is_dir():
        return bundled

    preprocessor_path = checkpoint / "policy_preprocessor.json"
    if preprocessor_path.is_file():
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                saved = Path(step.get("config", {}).get("tokenizer_name", ""))
                if saved.is_dir():
                    return saved.resolve()

    vlm_assets = Path(policy.config.vlm_model_name)
    if vlm_assets.is_dir():
        return vlm_assets.resolve()
    raise FileNotFoundError(f"no local tokenizer found for checkpoint: {checkpoint}")


def hide_artifact_visuals(model):
    model.light_castshadow[:] = 0
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        mesh_name = None
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id]))
        if geom_name in {"back_wall", "left_wall", "right_wall", "task1_back_counter"} or mesh_name == "base_visual_gate":
            model.geom_group[geom_id] = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processor-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint from which to load the policy pre/post-processors. "
            "This is intended for controlled normalization-ablation experiments; model "
            "weights and config are always loaded from --checkpoint."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/task1_smolvla",
        help="Training dataset used by the checkpoint (report provenance only).",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/task1-smolvla-eval")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--control-fps", type=int, default=10)
    parser.add_argument("--scenario", default="nominal", help="Human-readable perturbation label.")
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=None,
        help="Independent seed for perturbations (default: --seed + 2000003).",
    )
    parser.add_argument(
        "--initial-joint-noise-std-rad",
        type=float,
        default=0.0,
        help="Per-episode Gaussian offset applied to the six active-arm joints at reset.",
    )
    parser.add_argument(
        "--camera-position-noise-std-m",
        type=float,
        default=0.0,
        help="Per-episode Gaussian XYZ offset applied to the policy camera.",
    )
    parser.add_argument(
        "--camera-rotation-noise-std-deg",
        type=float,
        default=0.0,
        help="Per-episode Gaussian local-axis rotation-vector perturbation for the policy camera.",
    )
    parser.add_argument("--bottle-mass-scale-range", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--bottle-friction-scale-range", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--brightness-scale-range", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument(
        "--occlusion-fraction",
        type=float,
        default=0.0,
        help="Fraction of the RGB frame covered by one episode-fixed black square.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Number of predicted actions to execute before replanning (default: checkpoint config).",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-episode", type=int, default=0)
    args = parser.parse_args()
    if args.initial_joint_noise_std_rad < 0:
        raise ValueError("initial-joint-noise-std-rad must be non-negative")
    if args.camera_position_noise_std_m < 0 or args.camera_rotation_noise_std_deg < 0:
        raise ValueError("camera perturbation standard deviations must be non-negative")
    if not 0.0 <= args.occlusion_fraction < 1.0:
        raise ValueError("occlusion-fraction must be in [0, 1)")
    mass_scale_range = _positive_range(args.bottle_mass_scale_range, "bottle-mass-scale-range")
    friction_scale_range = _positive_range(
        args.bottle_friction_scale_range, "bottle-friction-scale-range"
    )
    brightness_scale_range = _positive_range(args.brightness_scale_range, "brightness-scale-range")
    args.output.mkdir(parents=True, exist_ok=True)

    # Keep the source path constant with the collector; no target pose or time is
    # exposed to the policy below.
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE))
    data = mujoco.MjData(model)
    initial_q = source.key_qpos[0, :16].copy()
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task1_bottle_free")
    bottle_qpos = model.jnt_qposadr[bottle_joint]
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "task1_head_camera")
    bottle_geom_ids = np.flatnonzero(model.geom_bodyid == bottle_id)
    base_camera_pos = model.cam_pos[camera_id].copy()
    base_camera_quat = model.cam_quat[camera_id].copy()
    base_bottle_mass = float(model.body_mass[bottle_id])
    base_bottle_inertia = model.body_inertia[bottle_id].copy()
    base_bottle_friction = model.geom_friction[bottle_geom_ids].copy()
    hide_artifact_visuals(model)
    option = mujoco.MjvOption()
    option.geomgroup[5] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        checkpoint_view,
        checkpoint_overlay,
        ignored_config_fields,
        added_config_fields,
    ) = make_compatible_checkpoint_view(args.checkpoint)
    try:
        checkpoint_config = json.loads((checkpoint_view / "config.json").read_text(encoding="utf-8"))
        if checkpoint_config.get("use_peft", False):
            policy_config = PreTrainedConfig.from_pretrained(checkpoint_view)
            policy_config.pretrained_path = checkpoint_view
            policy_config.device = device.type
            dataset_meta = LeRobotDatasetMetadata(
                repo_id=f"autopolicy/{args.dataset_root.name}",
                root=args.dataset_root,
            )
            policy = make_policy(policy_config, ds_meta=dataset_meta)
        else:
            policy = SmolVLAPolicy.from_pretrained(checkpoint_view)
    finally:
        if checkpoint_overlay is not None:
            checkpoint_overlay.cleanup()
    if args.n_action_steps is not None:
        if not 1 <= args.n_action_steps <= policy.config.chunk_size:
            raise ValueError(
                f"n-action-steps must be in [1, {policy.config.chunk_size}], got {args.n_action_steps}"
            )
        policy.config.n_action_steps = args.n_action_steps
    policy.to(device).eval()
    processor_checkpoint = args.processor_checkpoint or args.checkpoint
    tokenizer_path = resolve_tokenizer_path(processor_checkpoint, policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=processor_checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    renderer = mujoco.Renderer(model, height=240, width=320)

    writer = None
    if not 0 <= args.render_episode < args.episodes:
        raise ValueError(f"render-episode must be in [0, {args.episodes - 1}]")
    video_path = args.output / f"smolvla_rollout_episode_{args.render_episode:03d}.mp4"
    if args.render:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 720))
        if not writer.isOpened():
            raise RuntimeError(f"failed to open video writer: {video_path}")

    dt = float(model.opt.timestep)
    control_stride = max(1, round(1.0 / (args.control_fps * dt)))
    video_stride = max(1, round(1.0 / (30.0 * dt)))
    horizon_steps = round(9.0 / dt)
    position_rng = np.random.default_rng(args.seed)
    perturbation_seed = args.perturbation_seed if args.perturbation_seed is not None else args.seed + 2_000_003
    perturbation_rng = np.random.default_rng(perturbation_seed)
    bottle_positions = position_rng.uniform([0.43, 0.12], [0.51, 0.20], size=(args.episodes, 2))
    rows = []
    try:
        for episode in range(args.episodes):
            bottle_xy = bottle_positions[episode].astype(np.float64)
            joint_offset = perturbation_rng.normal(0.0, args.initial_joint_noise_std_rad, size=6)
            camera_position_offset = perturbation_rng.normal(
                0.0, args.camera_position_noise_std_m, size=3
            )
            camera_rotation_vector_deg = perturbation_rng.normal(
                0.0, args.camera_rotation_noise_std_deg, size=3
            )
            mass_scale = float(perturbation_rng.uniform(*mass_scale_range))
            friction_scale = float(perturbation_rng.uniform(*friction_scale_range))
            brightness_scale = float(perturbation_rng.uniform(*brightness_scale_range))
            occlusion_center_xy = tuple(perturbation_rng.uniform(0.0, 1.0, size=2).tolist())

            episode_initial_q = initial_q.copy()
            episode_initial_q[:6] += joint_offset
            for joint_index in range(6):
                episode_initial_q[joint_index] = np.clip(
                    episode_initial_q[joint_index],
                    model.jnt_range[joint_index, 0],
                    model.jnt_range[joint_index, 1],
                )
            camera_delta = _rotation_vector_to_quat(np.deg2rad(camera_rotation_vector_deg))
            perturbed_camera_quat = _quat_multiply(base_camera_quat, camera_delta)
            model.cam_pos[camera_id] = base_camera_pos + camera_position_offset
            model.cam_quat[camera_id] = perturbed_camera_quat / np.linalg.norm(perturbed_camera_quat)
            model.body_mass[bottle_id] = base_bottle_mass * mass_scale
            model.body_inertia[bottle_id] = base_bottle_inertia * mass_scale
            model.geom_friction[bottle_geom_ids] = base_bottle_friction * friction_scale
            # Refresh MuJoCo constants that depend on mutable body mass/inertia.
            mujoco.mj_setConst(model, data)

            data.qpos[:16] = episode_initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array([*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0])
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            policy.reset()
            torch.manual_seed(args.seed + episode)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + episode)
            action = np.zeros(14, dtype=np.float32)
            finger_contacts = 0
            max_penetration = 0.0
            max_z = float(data.xpos[bottle_id, 2])
            action_calls = 0
            chunk_inferences = 0
            steps_above_lift_threshold = 0
            for step in range(horizon_steps):
                if step % control_stride == 0:
                    renderer.update_scene(data, camera="task1_head_camera", scene_option=option)
                    image = _apply_visual_perturbation(
                        renderer.render().copy(),
                        brightness_scale,
                        args.occlusion_fraction,
                        occlusion_center_xy,
                    )
                    observation = {
                        "observation.images.camera_00": image,
                        "observation.state": data.qpos[:16].astype(np.float32),
                    }
                    if len(policy._queues["action"]) == 0:
                        chunk_inferences += 1
                    predicted = predict_action(
                        observation=observation,
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=False,
                        task=TASK,
                        robot_type="yam_task1",
                    )
                    action = predicted.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    if not np.isfinite(action).all():
                        raise RuntimeError(f"non-finite action in episode {episode}, step {step}")
                    action = np.clip(action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                    action_calls += 1
                data.ctrl[:] = action
                mujoco.mj_step(model, data)
                max_z = max(max_z, float(data.xpos[bottle_id, 2]))
                steps_above_lift_threshold += int(float(data.xpos[bottle_id, 2]) > 0.83)
                for contact in data.contact:
                    max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                    if "task1_bottle" in g1 or "task1_bottle" in g2:
                        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or ""
                        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or ""
                        if b1.startswith(("left_lf", "left_rf", "right_lf", "right_rf")) or b2.startswith(("left_lf", "left_rf", "right_lf", "right_rf")):
                            finger_contacts += 1
                if episode == args.render_episode and writer is not None and step % video_stride == 0:
                    renderer.update_scene(data, camera="task1_head_camera", scene_option=option)
                    rgb = _apply_visual_perturbation(
                        renderer.render().copy(),
                        brightness_scale,
                        args.occlusion_fraction,
                        occlusion_center_xy,
                    )
                    frame = cv2.resize(rgb[:, :, ::-1], (960, 720), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(
                        frame,
                        f"SmolVLA {args.scenario} (replan={policy.config.n_action_steps})  t={step * dt:.2f}s",
                        (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (30, 220, 40),
                        2,
                        cv2.LINE_AA,
                    )
                    writer.write(frame)
            final_z = float(data.xpos[bottle_id, 2])
            success = bool(final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01)
            row = {
                "episode": episode,
                "bottle_xy_ground_truth_for_scoring_only": bottle_xy.tolist(),
                "perturbations": {
                    "initial_active_arm_joint_offset_rad": joint_offset.tolist(),
                    "camera_position_offset_m": camera_position_offset.tolist(),
                    "camera_rotation_vector_deg": camera_rotation_vector_deg.tolist(),
                    "bottle_mass_scale": mass_scale,
                    "bottle_friction_scale": friction_scale,
                    "brightness_scale": brightness_scale,
                    "occlusion_fraction": args.occlusion_fraction,
                    "occlusion_center_xy_fraction": list(occlusion_center_xy),
                },
                "final_bottle_z_m": final_z,
                "max_bottle_z_m": max_z,
                "finger_contacts": finger_contacts,
                "max_penetration_m": max_penetration,
                "control_actions": action_calls,
                "chunk_inferences": chunk_inferences,
                "time_above_lift_threshold_s": steps_above_lift_threshold * dt,
                "success": success,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        if writer is not None:
            writer.release()
        renderer.close()

    successes = sum(row["success"] for row in rows)
    report = {
        "schema": "autopolicy.task1_smolvla_evaluation/v2",
        "policy_type": "smolvla",
        "checkpoint": str(args.checkpoint.resolve()),
        "processor_checkpoint": str(processor_checkpoint.resolve()),
        "processor_override": args.processor_checkpoint is not None,
        "dataset": str(args.dataset_root.resolve()),
        "task": TASK,
        "scenario": args.scenario,
        "position_seed": args.seed,
        "perturbation_seed": perturbation_seed,
        "perturbation_distribution": {
            "initial_active_arm_joint_offset_rad": {
                "distribution": "normal",
                "std": args.initial_joint_noise_std_rad,
            },
            "camera_position_offset_m": {
                "distribution": "normal_xyz",
                "std": args.camera_position_noise_std_m,
            },
            "camera_local_rotation_vector_deg": {
                "distribution": "normal_xyz",
                "std": args.camera_rotation_noise_std_deg,
            },
            "bottle_mass_scale": {"distribution": "uniform", "range": list(mass_scale_range)},
            "bottle_friction_scale": {
                "distribution": "uniform",
                "range": list(friction_scale_range),
            },
            "brightness_scale": {
                "distribution": "uniform",
                "range": list(brightness_scale_range),
            },
            "occlusion_fraction": args.occlusion_fraction,
        },
        "policy_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "oracle_inputs_to_policy": [],
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "ignored_checkpoint_config_fields": ignored_config_fields,
        "added_checkpoint_config_fields": added_config_fields,
        "tokenizer_path": str(tokenizer_path),
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes if args.episodes else 0.0,
        "video": str(video_path.resolve()) if args.render else None,
        "episode_results": rows,
        "limitations": [
            "This is MuJoCo replica-sim evaluation, not real-robot evaluation.",
            "Success rate is based only on the requested deterministic seeds and episode count.",
        ],
    }
    report_path = args.output / "evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"success_rate": report["success_rate"], "report": str(report_path), "video": report["video"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
