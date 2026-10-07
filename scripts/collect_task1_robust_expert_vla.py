#!/usr/bin/env python3
"""Collect full-horizon expert demonstrations for recorded task1 perturbations.

Unlike policy-roll-in correction clips, every saved episode contains the complete
approach, grasp, and lift sequence.  Simulator object pose is used only by the
expert planner and success scorer; policy observations remain RGB, robot state,
and task text.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np

from lerobot.datasets import LeRobotDataset

from collect_task1_visual_vla import command_at, episode_plan
from evaluate_task1_smolvla import (
    ROOT,
    SCENE,
    SOURCE_SCENE,
    TASK,
    _apply_visual_perturbation,
    _quat_multiply,
    _rotation_vector_to_quat,
    hide_artifact_visuals,
)


def _bottle_finger_contact(model: mujoco.MjModel, contact: mujoco.MjContact) -> bool:
    geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
    geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
    if "task1_bottle" not in geom1 and "task1_bottle" not in geom2:
        return False
    body1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]
    ) or ""
    body2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]
    ) or ""
    return body1.startswith(("left_lf", "left_rf")) or body2.startswith(
        ("left_lf", "left_rf")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default="autopolicy/task1_robust_expert")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--source-episodes",
        type=int,
        nargs="+",
        default=None,
        help="Optional source-report episode ids to select before other filters.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Use only failed source-policy cases; default uses every recorded perturbation case.",
    )
    parser.add_argument(
        "--balance-regions",
        action="store_true",
        help="Select the same number of cases on each side of --region-threshold-x.",
    )
    parser.add_argument("--region-threshold-x", type=float, default=0.4839584794)
    parser.add_argument(
        "--max-per-region",
        type=int,
        default=None,
        help="Maximum cases selected from each region; requires --balance-regions.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.max_per_region is not None and args.max_per_region <= 0:
        raise ValueError("max-per-region must be positive")
    if args.max_per_region is not None and not args.balance_regions:
        raise ValueError("max-per-region requires --balance-regions")
    root = args.root.resolve()
    data_root = (ROOT / "data").resolve()
    if root == data_root or data_root not in root.parents:
        raise ValueError(f"dataset root must be a child of {data_root}: {root}")
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"dataset exists: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    if source_report.get("schema") != "autopolicy.task1_smolvla_evaluation/v2":
        raise ValueError(f"unsupported source report schema: {source_report.get('schema')}")
    cases = list(source_report["episode_results"])
    if args.source_episodes is not None:
        requested = set(args.source_episodes)
        available = {int(row["episode"]) for row in cases}
        missing = requested - available
        if missing:
            raise ValueError(f"source episode ids not present in report: {sorted(missing)}")
        cases = [row for row in cases if int(row["episode"]) in requested]
    if args.failed_only:
        cases = [row for row in cases if not row["success"]]
    candidate_cases = len(cases)
    candidate_region_counts = {
        "left": sum(
            row["bottle_xy_ground_truth_for_scoring_only"][0] < args.region_threshold_x
            for row in cases
        ),
        "right": sum(
            row["bottle_xy_ground_truth_for_scoring_only"][0] >= args.region_threshold_x
            for row in cases
        ),
    }
    if args.balance_regions:
        left = [
            row
            for row in cases
            if row["bottle_xy_ground_truth_for_scoring_only"][0] < args.region_threshold_x
        ]
        right = [
            row
            for row in cases
            if row["bottle_xy_ground_truth_for_scoring_only"][0] >= args.region_threshold_x
        ]
        per_region = min(len(left), len(right))
        if args.max_per_region is not None:
            per_region = min(per_region, args.max_per_region)
        cases = left[:per_region] + right[:per_region]
    if args.max_cases is not None:
        if args.max_cases <= 0:
            raise ValueError("max-cases must be positive")
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("no source cases selected")

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
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
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
    render_option = mujoco.MjvOption()
    render_option.geomgroup[5] = 0
    renderer = mujoco.Renderer(model, height=240, width=320)
    dt = float(model.opt.timestep)
    control_stride = max(1, round(1.0 / (args.fps * dt)))
    horizon_steps = round(9.0 / dt)

    saved_cases: list[dict] = []
    rejected_cases: list[dict] = []
    try:
        for source_row in cases:
            perturbations = source_row["perturbations"]
            bottle_xy = np.asarray(
                source_row["bottle_xy_ground_truth_for_scoring_only"], dtype=np.float64
            )
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
            camera_quat = _quat_multiply(base_camera_quat, camera_delta)
            model.cam_pos[camera_id] = base_camera_pos + camera_position_offset
            model.cam_quat[camera_id] = camera_quat / np.linalg.norm(camera_quat)
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
            plan = episode_plan(model, data, episode_initial_q, bottle_xy, site_id)

            # IK mutates qpos; restore the declared perturbed reset before rollout.
            data.qpos[:16] = episode_initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array(
                [*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0]
            )
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)

            max_penetration = 0.0
            finger_contacts = 0
            for step in range(horizon_steps):
                expert_action = command_at(plan, step * dt)
                # Match the zero-variance action support of the nominal dataset.
                expert_action[5] = initial_q[5]
                expert_action[7:13] = initial_q[8:14]
                expert_action[13] = 0.0475
                data.ctrl[:] = np.clip(
                    expert_action,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1],
                )
                mujoco.mj_step(model, data)
                for contact in data.contact:
                    max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                    finger_contacts += int(_bottle_finger_contact(model, contact))
                if step % control_stride == 0:
                    renderer.update_scene(
                        data, camera="task1_head_camera", scene_option=render_option
                    )
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
            result = {
                "source_episode": int(source_row["episode"]),
                "source_policy_success": bool(source_row["success"]),
                "bottle_xy_ground_truth_for_expert_and_scoring_only": bottle_xy.tolist(),
                "perturbations": perturbations,
                "final_bottle_z_m": final_z,
                "finger_contacts": finger_contacts,
                "max_penetration_m": max_penetration,
                "expert_success": bool(
                    final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01
                ),
            }
            if result["expert_success"]:
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
        "schema": "autopolicy.task1_robust_expert_vla/v1",
        "method": "full_horizon_expert_under_recorded_replica_sim_perturbations",
        "source_report": str(args.source_report.resolve()),
        "source_scenario": source_report.get("scenario"),
        "task": TASK,
        "policy_inputs": ["observation.images.camera_00", "observation.state", "task"],
        "policy_target": "action",
        "oracle_inputs_to_policy_or_dataset_observation": [],
        "expert_label_oracles": ["bottle pose from MuJoCo state", "robot kinematics"],
        "selection": "failed_only" if args.failed_only else "all_recorded_cases",
        "requested_source_episodes": args.source_episodes,
        "selection_balance": {
            "enabled": args.balance_regions,
            "region_threshold_x_m": args.region_threshold_x,
            "max_per_region": args.max_per_region,
            "candidate_cases_before_balancing": candidate_cases,
            "candidate_region_counts_before_balancing": candidate_region_counts,
            "selected_region_counts": {
                "left": sum(
                    row["bottle_xy_ground_truth_for_scoring_only"][0]
                    < args.region_threshold_x
                    for row in cases
                ),
                "right": sum(
                    row["bottle_xy_ground_truth_for_scoring_only"][0]
                    >= args.region_threshold_x
                    for row in cases
                ),
            },
        },
        "selected_cases": len(cases),
        "saved_expert_episodes": len(saved_cases),
        "rejected_expert_episodes": len(rejected_cases),
        "frames": len(saved_cases) * round(9.0 * args.fps),
        "fps": args.fps,
        "dataset": str(root),
        "saved_cases": saved_cases,
        "rejected_cases": rejected_cases,
        "limitations": [
            "These are MuJoCo replica-sim expert demonstrations, not real-robot data.",
            "Perturbations replay one finite engineering stress-test sample, not a calibrated uncertainty distribution.",
            "The expert uses simulator bottle pose only for labels and scoring; it is absent from policy observations.",
            "The IK expert controls position and does not optimize wrist orientation.",
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
