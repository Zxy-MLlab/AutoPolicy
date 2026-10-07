#!/usr/bin/env python3
"""Collect expert corrections from actual SmolVLA roll-in failure states."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

from collect_task1_visual_vla import command_at, solve_position_ik
from evaluate_task1_smolvla import (
    ROOT,
    SCENE,
    SOURCE_SCENE,
    TASK,
    _apply_visual_perturbation,
    _quat_multiply,
    _rotation_vector_to_quat,
    hide_artifact_visuals,
    make_compatible_checkpoint_view,
    resolve_tokenizer_path,
)


CORRECTION_PLANNER = "orientation_aware_wrist_regrasp_v5"
RECOVERY_SECONDS = 1.5
CORRECTION_SECONDS = 8.5


def solve_position_ik_with_fixed_wrist(model, data, q, target, site_id, wrist_yaw):
    """Solve grasp-site position while keeping the final wrist joint fixed."""
    q = q.copy()
    arm_qpos = np.arange(5)
    q[5] = wrist_yaw
    for _ in range(500):
        data.qpos[:16] = q
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        if np.linalg.norm(error) < 0.0008:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = jacp[:, arm_qpos]
        damping = 0.035
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
        q[arm_qpos] += 0.45 * dq
        for joint_index in arm_qpos:
            q[joint_index] = np.clip(
                q[joint_index],
                model.jnt_range[joint_index, 0],
                model.jnt_range[joint_index, 1],
            )
        q[5] = wrist_yaw
    return q


def select_wrist_grasp_configuration(
    model,
    data,
    q,
    grasp_target,
    bottle_axis,
    site_id,
    left_finger_id,
    right_finger_id,
):
    """Find a reachable grasp whose jaw axis is horizontal and normal to the bottle."""
    bottle_axis = np.asarray(bottle_axis, dtype=np.float64)
    bottle_axis /= max(np.linalg.norm(bottle_axis), 1e-12)
    desired_jaw_axis = np.cross(np.array([0.0, 0.0, 1.0]), bottle_axis)
    if np.linalg.norm(desired_jaw_axis) < 1e-6:
        desired_jaw_axis = np.array([1.0, 0.0, 0.0])
    desired_jaw_axis /= np.linalg.norm(desired_jaw_axis)
    wrist_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_joint6")
    wrist_low, wrist_high = model.jnt_range[wrist_joint_id]
    candidates = np.linspace(wrist_low, wrist_high, 181)
    scored = []
    for wrist_yaw in candidates:
        candidate_q = solve_position_ik_with_fixed_wrist(
            model, data, q, grasp_target, site_id, wrist_yaw
        )
        data.qpos[:16] = candidate_q
        mujoco.mj_forward(model, data)
        jaw_axis = data.xpos[right_finger_id] - data.xpos[left_finger_id]
        jaw_norm = np.linalg.norm(jaw_axis)
        if jaw_norm < 1e-12:
            continue
        jaw_axis /= jaw_norm
        alignment_error = 1.0 - abs(float(np.dot(jaw_axis, desired_jaw_axis)))
        horizontal_error = abs(float(jaw_axis[2]))
        position_error = float(np.linalg.norm(data.site_xpos[site_id] - grasp_target))
        # A horizontal jaw is essential here: a large vertical component makes
        # one finger hit the tabletop while the other pushes the fallen bottle.
        score = alignment_error + 2.0 * horizontal_error + 10.0 * position_error
        scored.append(
            (
                score,
                position_error,
                abs(float(wrist_yaw - q[5])),
                wrist_yaw,
                candidate_q,
                jaw_axis.copy(),
                alignment_error,
                horizontal_error,
            )
        )
    if not scored:
        raise RuntimeError("unable to determine the left gripper jaw axis")
    (
        score,
        position_error,
        _,
        wrist_yaw,
        grasp_q,
        jaw_axis,
        alignment_error,
        horizontal_error,
    ) = min(scored, key=lambda row: (row[0], row[1], row[2]))
    return grasp_q, {
        "selected_wrist_yaw_rad": float(wrist_yaw),
        "bottle_axis": bottle_axis.tolist(),
        "desired_jaw_axis": desired_jaw_axis.tolist(),
        "actual_jaw_axis_at_grasp": jaw_axis.tolist(),
        "absolute_target_axis_dot": abs(float(np.dot(jaw_axis, desired_jaw_axis))),
        "alignment_error": alignment_error,
        "horizontal_error": horizontal_error,
        "grasp_site_position_error_m": position_error,
        "score": score,
        "candidate_count": len(scored),
    }


def correction_plan(
    model,
    data,
    current_q,
    grasp_point_xyz,
    bottle_axis,
    site_id,
    left_finger_id,
    right_finger_id,
    current_grip,
):
    """Plan a grasp after the recovery motion has allowed the object to settle."""
    grasp_target = np.asarray(grasp_point_xyz, dtype=np.float64)
    above_target = grasp_target + np.array([0.0, 0.0, 0.18])
    lift_target = grasp_target.copy()
    lift_target[2] = max(1.08, float(grasp_target[2] + 0.20))
    unconstrained_q_grasp = solve_position_ik(model, data, current_q, grasp_target, site_id)
    q_grasp, wrist_selection = select_wrist_grasp_configuration(
        model,
        data,
        unconstrained_q_grasp,
        grasp_target,
        bottle_axis,
        site_id,
        left_finger_id,
        right_finger_id,
    )
    wrist_yaw = wrist_selection["selected_wrist_yaw_rad"]
    q_above = solve_position_ik_with_fixed_wrist(
        model, data, q_grasp, above_target, site_id, wrist_yaw
    )
    q_lift = solve_position_ik_with_fixed_wrist(
        model, data, q_grasp, lift_target, site_id, wrist_yaw
    )
    plan = [
        (0.0, current_q.copy(), current_grip),
        (1.5, q_above, 0.0475),
        (3.0, q_grasp, 0.0475),
        (3.7, q_grasp, 0.0),
        (5.5, q_lift, 0.0),
        (CORRECTION_SECONDS - RECOVERY_SECONDS, q_lift, 0.0),
    ]
    return plan, wrist_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset metadata used to instantiate a PEFT checkpoint (defaults to the failure report dataset).",
    )
    parser.add_argument("--failure-report", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="autopolicy/task1_policy_rollin_corrections")
    parser.add_argument("--rollin-seconds", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--case-offset",
        type=int,
        default=0,
        help="Skip this many failed source episodes before applying --max-cases.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    allowed_root = ROOT.resolve()
    if root == allowed_root or allowed_root not in root.parents:
        raise ValueError(f"dataset root must be a child of {allowed_root}: {root}")
    if args.rollin_seconds <= 0:
        raise ValueError("rollin-seconds must be positive")
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"dataset exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)

    source_report = json.loads(args.failure_report.read_text(encoding="utf-8"))
    if source_report.get("schema") != "autopolicy.task1_smolvla_evaluation/v2":
        raise ValueError(f"unsupported source report schema: {source_report.get('schema')}")
    if Path(source_report["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("failure report checkpoint does not match --checkpoint")
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else Path(source_report.get("dataset", "")).resolve()
    )
    if not (dataset_root / "meta/info.json").is_file():
        raise FileNotFoundError(f"policy dataset metadata not found: {dataset_root}")
    all_failure_cases = [row for row in source_report["episode_results"] if not row["success"]]
    if args.case_offset < 0 or args.case_offset >= len(all_failure_cases):
        raise ValueError(
            f"case-offset must be in [0, {max(0, len(all_failure_cases) - 1)}]"
        )
    failure_cases = all_failure_cases[args.case_offset :]
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("max-cases must be positive")
        failure_cases = failure_cases[: args.max_cases]
    if not failure_cases:
        raise ValueError("failure report contains no failed episodes")

    state_names = [f"left_joint_{i}" for i in range(6)] + [
        "left_finger_left",
        "left_finger_right",
    ] + [f"right_joint_{i}" for i in range(6)] + ["right_finger_left", "right_finger_right"]
    action_names = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"] + [
        f"right_joint_{i}" for i in range(6)
    ] + ["right_gripper"]
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features={
            "observation.images.camera_00": {
                "dtype": "video",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": {"axes": state_names},
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": {"axes": action_names},
            },
        },
        robot_type="yam_task1",
        use_videos=True,
        image_writer_threads=2,
    )

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE))
    data = mujoco.MjData(model)
    initial_q = source.key_qpos[0, :16].copy()
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task1_bottle_free")
    bottle_qpos = model.jnt_qposadr[bottle_joint]
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_lf_down")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rf_down")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "task1_head_camera")
    bottle_geom_ids = np.flatnonzero(model.geom_bodyid == bottle_id)
    base_camera_pos = model.cam_pos[camera_id].copy()
    base_camera_quat = model.cam_quat[camera_id].copy()
    base_bottle_mass = float(model.body_mass[bottle_id])
    base_bottle_inertia = model.body_inertia[bottle_id].copy()
    base_bottle_friction = model.geom_friction[bottle_geom_ids].copy()
    hide_artifact_visuals(model)
    render_option = mujoco.MjvOption()
    render_option.geomgroup[5] = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_view, checkpoint_overlay, _, _ = make_compatible_checkpoint_view(args.checkpoint)
    try:
        checkpoint_config = json.loads((checkpoint_view / "config.json").read_text(encoding="utf-8"))
        if checkpoint_config.get("use_peft", False):
            policy_config = PreTrainedConfig.from_pretrained(checkpoint_view)
            policy_config.pretrained_path = checkpoint_view
            policy_config.device = device.type
            dataset_meta = LeRobotDatasetMetadata(
                repo_id=f"autopolicy/{dataset_root.name}",
                root=dataset_root,
            )
            policy = make_policy(policy_config, ds_meta=dataset_meta)
        else:
            policy = SmolVLAPolicy.from_pretrained(checkpoint_view)
    finally:
        if checkpoint_overlay is not None:
            checkpoint_overlay.cleanup()
    policy.to(device).eval()
    tokenizer_path = resolve_tokenizer_path(args.checkpoint, policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )

    renderer = mujoco.Renderer(model, height=240, width=320)
    dt = float(model.opt.timestep)
    control_stride = max(1, round(1.0 / (args.fps * dt)))
    rollin_steps = round(args.rollin_seconds / dt)
    correction_steps = round(CORRECTION_SECONDS / dt)
    saved_cases = []
    rejected_cases = []
    try:
        for source_row in failure_cases:
            source_episode = int(source_row["episode"])
            perturbations = source_row["perturbations"]
            bottle_xy = np.asarray(source_row["bottle_xy_ground_truth_for_scoring_only"], dtype=np.float64)
            joint_offset = np.asarray(
                perturbations["initial_active_arm_joint_offset_rad"], dtype=np.float64
            )
            camera_position_offset = np.asarray(
                perturbations["camera_position_offset_m"], dtype=np.float64
            )
            camera_rotation_vector_deg = np.asarray(
                perturbations["camera_rotation_vector_deg"], dtype=np.float64
            )
            mass_scale = float(perturbations["bottle_mass_scale"])
            friction_scale = float(perturbations["bottle_friction_scale"])
            brightness_scale = float(perturbations["brightness_scale"])
            occlusion_fraction = float(perturbations["occlusion_fraction"])
            occlusion_center_xy = tuple(perturbations["occlusion_center_xy_fraction"])

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
            mujoco.mj_setConst(model, data)

            data.qpos[:16] = episode_initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array(
                [*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0]
            )
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            policy.reset()
            policy_seed = int(source_report["position_seed"]) + source_episode
            torch.manual_seed(policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(policy_seed)
            action = np.zeros(14, dtype=np.float32)
            for step in range(rollin_steps):
                if step % control_stride == 0:
                    renderer.update_scene(data, camera="task1_head_camera", scene_option=render_option)
                    image = _apply_visual_perturbation(
                        renderer.render().copy(),
                        brightness_scale,
                        occlusion_fraction,
                        occlusion_center_xy,
                    )
                    predicted = predict_action(
                        observation={
                            "observation.images.camera_00": image,
                            "observation.state": data.qpos[:16].astype(np.float32),
                        },
                        policy=policy,
                        device=device,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=False,
                        task=TASK,
                        robot_type="yam_task1",
                    )
                    action = predicted.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    action = np.clip(action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                data.ctrl[:] = action
                mujoco.mj_step(model, data)

            rollin_qpos = data.qpos.copy()
            rollin_qvel = data.qvel.copy()
            rollin_ctrl = data.ctrl.copy()
            mujoco.mj_forward(model, data)
            bottle_xyz_at_correction = data.xpos[bottle_id].copy()
            current_q = data.qpos[:16].copy()
            # The source demonstrations keep the inactive right arm fixed and
            # leave the last active-arm wrist joint constant. Preserve that
            # action support exactly: tiny policy/solver drift on zero-variance
            # action dimensions would otherwise explode under MEAN_STD scaling.
            current_q[5] = initial_q[5]
            current_q[8:14] = initial_q[8:14]
            current_grip = float(np.clip(data.ctrl[6], 0.0, 0.0475))
            recovery_plan = [
                (0.0, current_q.copy(), current_grip),
                (RECOVERY_SECONDS, initial_q.copy(), 0.0475),
            ]
            plan = None
            bottle_xyz_after_recovery = None
            bottle_grasp_point_after_recovery = None
            bottle_axis_after_recovery = None
            wrist_selection = None

            max_penetration = 0.0
            finger_contacts = 0
            for step in range(correction_steps):
                time = step * dt
                if time < RECOVERY_SECONDS:
                    expert_action = command_at(recovery_plan, time)
                else:
                    if plan is None:
                        # The roll-in can leave the bottle moving. Re-estimate
                        # its pose only after the arm has opened/retracted and
                        # the object has had time to settle on the table.
                        mujoco.mj_forward(model, data)
                        bottle_xyz_after_recovery = data.xpos[bottle_id].copy()
                        bottle_rotation = data.xmat[bottle_id].reshape(3, 3).copy()
                        bottle_axis_after_recovery = bottle_rotation[:, 2].copy()
                        bottle_grasp_point_after_recovery = (
                            bottle_xyz_after_recovery
                            + bottle_rotation @ np.array([0.0, 0.0, 0.118])
                        )
                        replanning_qpos = data.qpos.copy()
                        replanning_qvel = data.qvel.copy()
                        replanning_ctrl = data.ctrl.copy()
                        replanning_q = data.qpos[:16].copy()
                        replanning_q[5] = initial_q[5]
                        replanning_q[8:14] = initial_q[8:14]
                        plan, wrist_selection = correction_plan(
                            model,
                            data,
                            replanning_q,
                            bottle_grasp_point_after_recovery,
                            bottle_axis_after_recovery,
                            site_id,
                            left_finger_id,
                            right_finger_id,
                            0.0475,
                        )
                        # IK mutates qpos; restore the exact dynamic state.
                        data.qpos[:] = replanning_qpos
                        data.qvel[:] = replanning_qvel
                        data.ctrl[:] = replanning_ctrl
                        mujoco.mj_forward(model, data)
                    expert_action = command_at(plan, time - RECOVERY_SECONDS)
                expert_action[7:13] = initial_q[8:14]
                expert_action[13] = 0.0475
                data.ctrl[:] = np.clip(
                    expert_action,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1],
                )
                mujoco.mj_step(model, data)
                # The right arm is outside this task and fixed in the source
                # data. Remove irrelevant solver drift before it reaches the
                # observation stream.
                data.qpos[8:14] = initial_q[8:14]
                data.qvel[8:14] = 0.0
                mujoco.mj_forward(model, data)
                for contact in data.contact:
                    max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                    if "task1_bottle" in g1 or "task1_bottle" in g2:
                        b1 = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]
                        ) or ""
                        b2 = mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]
                        ) or ""
                        if b1.startswith(("left_lf", "left_rf")) or b2.startswith(("left_lf", "left_rf")):
                            finger_contacts += 1
                if step % control_stride == 0:
                    renderer.update_scene(data, camera="task1_head_camera", scene_option=render_option)
                    image = _apply_visual_perturbation(
                        renderer.render().copy(),
                        brightness_scale,
                        occlusion_fraction,
                        occlusion_center_xy,
                    )
                    dataset.add_frame(
                        {
                            "observation.images.camera_00": image,
                            "observation.state": data.qpos[:16].astype(np.float32),
                            "action": expert_action.astype(np.float32),
                            "task": TASK,
                        }
                    )

            final_z = float(data.xpos[bottle_id, 2])
            success = bool(final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01)
            result = {
                "source_failure_episode": source_episode,
                "policy_seed": policy_seed,
                "bottle_xy_ground_truth_for_expert_and_scoring_only": bottle_xy.tolist(),
                "bottle_xyz_at_correction_m": bottle_xyz_at_correction.tolist(),
                "bottle_xyz_after_recovery_m": (
                    bottle_xyz_after_recovery.tolist()
                    if bottle_xyz_after_recovery is not None
                    else None
                ),
                "bottle_grasp_point_after_recovery_m": (
                    bottle_grasp_point_after_recovery.tolist()
                    if bottle_grasp_point_after_recovery is not None
                    else None
                ),
                "bottle_axis_after_recovery": (
                    bottle_axis_after_recovery.tolist()
                    if bottle_axis_after_recovery is not None
                    else None
                ),
                "wrist_selection": wrist_selection,
                "rollin_state_qpos": rollin_qpos[:16].tolist(),
                "rollin_bottle_qpos": rollin_qpos[bottle_qpos:bottle_qpos + 7].tolist(),
                "rollin_bottle_qvel": rollin_qvel[-6:].tolist(),
                "final_bottle_z_m": final_z,
                "finger_contacts": finger_contacts,
                "max_penetration_m": max_penetration,
                "expert_correction_success": success,
            }
            if success:
                dataset.save_episode()
                result["saved_episode"] = len(saved_cases)
                saved_cases.append(result)
            else:
                dataset.clear_episode_buffer()
                rejected_cases.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        renderer.close()
        dataset.finalize()

    report = {
        "schema": "autopolicy.task1_policy_rollin_corrections/v1",
        "method": "actual_smolvla_rollin_then_state_conditional_expert_replan",
        "correction_planner": CORRECTION_PLANNER,
        "source_failure_report": str(args.failure_report.resolve()),
        "rollin_checkpoint": str(args.checkpoint.resolve()),
        "rollin_policy_dataset": str(dataset_root),
        "task": TASK,
        "policy_rollin_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "correction_dataset_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "correction_dataset_target": "action",
        "oracle_inputs_to_policy_or_dataset_observation": [],
        "expert_label_oracles": ["bottle pose from MuJoCo state", "robot kinematics"],
        "constant_action_support": {
            "reason": "match zero-variance dimensions in the nominal dataset",
            "active_arm_joint_5": "orientation-aware expert target",
            "inactive_right_arm": "nominal home targets",
            "inactive_right_gripper": 0.0475,
        },
        "rollin_seconds": args.rollin_seconds,
        "correction_seconds": CORRECTION_SECONDS,
        "fps": args.fps,
        "source_failures": len(failure_cases),
        "total_source_failures": len(all_failure_cases),
        "case_offset": args.case_offset,
        "saved_expert_corrections": len(saved_cases),
        "rejected_expert_corrections": len(rejected_cases),
        "frames": len(saved_cases) * round(CORRECTION_SECONDS * args.fps),
        "dataset": str(root),
        "saved_cases": saved_cases,
        "rejected_cases": rejected_cases,
        "limitations": [
            "The roll-in and correction trajectories are MuJoCo replica-sim data, not real robot data.",
            "The expert uses simulator bottle pose only to generate labels; it is absent from policy observations.",
            "The correction planner searches wrist yaw but does not solve full 6D end-effector pose.",
        ],
    }
    (root / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset": str(root),
                "saved": len(saved_cases),
                "rejected": len(rejected_cases),
                "frames": report["frames"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
