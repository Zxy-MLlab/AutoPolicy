#!/usr/bin/env python3
"""Run a hypothesis-driven single-video bottle grasp with the YAM model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path("/data/zxy/autopolicy").resolve()
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
SOURCE_SCENE = ROOT / "vendor/gpt6-real2sim/real2sim_microphones/scene_portable.xml"
OUTPUT = ROOT / "runs/task1-real2sim-modeling"


def solve_position_ik(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, target: np.ndarray, site_id: int) -> np.ndarray:
    q = q.copy()
    arm_qpos = np.arange(6)
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
            q[joint_index] = np.clip(q[joint_index], model.jnt_range[joint_index, 0], model.jnt_range[joint_index, 1])
    return q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE))
    data = mujoco.MjData(model)
    source_start = source.key_qpos[0, :16].copy()
    data.qpos[:16] = source_start
    data.qpos[16:23] = np.array([0.47, 0.16, 0.762, 1.0, 0.0, 0.0, 0.0])
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    if site_id < 0:
        raise RuntimeError("YAM left_grasp_site missing")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task1_bottle_free")
    bottle_qpos = model.jnt_qposadr[bottle_joint]

    initial_q = data.qpos[:16].copy()
    waypoints = [
        (0.0, np.array([0.47, 0.16, 1.06]), 0.0475),
        (2.5, np.array([0.47, 0.16, 0.88]), 0.0475),
        (4.0, np.array([0.47, 0.16, 0.88]), 0.0000),
        (5.0, np.array([0.47, 0.16, 1.08]), 0.0000),
        (8.0, np.array([0.47, 0.16, 1.08]), 0.0000),
    ]
    q_waypoints = [(time, solve_position_ik(model, data, initial_q, target, site_id), grip) for time, target, grip in waypoints]
    duration = 9.0
    dt = float(model.opt.timestep)
    qlog = []
    bottle_log = []
    site_log = []
    contacts = []
    contact_pair_counts = {}
    finger_contacts = 0
    first_finger_contact_time = None
    first_lift_time = None
    max_penetration = 0.0
    lifted = False
    renderer = None
    writer = None
    if args.render:
        # EGL shadow maps produce severe block artifacts for this overlapping CAD scene.
        # Disabling cast shadows changes only pixels, not contacts or dynamics.
        model.light_castshadow[:] = 0
        renderer = mujoco.Renderer(model, height=540, width=960)
        writer = cv2.VideoWriter(str(OUTPUT / "task1_yam_bottle.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
        # Hide environment-only visual meshes that overlap their collision proxies;
        # collision geoms remain active for physics and contact validation.
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if geom_name in {"back_wall", "left_wall", "right_wall", "task1_back_counter"} or (
                model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id])) == "base_visual_gate"
            ):
                model.geom_group[geom_id] = 5
        render_option = mujoco.MjvOption()
        render_option.geomgroup[5] = 0
    try:
        for step in range(int(duration / dt)):
            time = step * dt
            segment = max(0, min(len(q_waypoints) - 2, next((i for i in range(len(q_waypoints) - 1) if time <= q_waypoints[i + 1][0]), len(q_waypoints) - 2)))
            t0, q0, g0 = q_waypoints[segment]
            t1, q1, g1 = q_waypoints[segment + 1]
            alpha = 0.0 if t1 == t0 else np.clip((time - t0) / (t1 - t0), 0.0, 1.0)
            target_q = (1.0 - alpha) * q0 + alpha * q1
            target_grip = (1.0 - alpha) * g0 + alpha * g1
            # YAM qpos layout is [left arm (6), left fingers (2), right arm (6), right fingers (2)],
            # while actuators are [left arm (6), left gripper, right arm (6), right gripper].
            data.ctrl[0:6] = target_q[0:6]
            data.ctrl[6] = target_grip
            data.ctrl[7:13] = target_q[8:14]
            data.ctrl[13] = 0.0475
            mujoco.mj_step(model, data)
            bottle_pos = data.xpos[bottle_id].copy()
            site_pos = data.site_xpos[site_id].copy()
            if bottle_pos[2] > 0.83:
                lifted = True
                if first_lift_time is None:
                    first_lift_time = time
            for contact in data.contact:
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or str(contact.geom1)
                geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or str(contact.geom2)
                if "task1_bottle" in geom1 or "task1_bottle" in geom2:
                    contacts.append([round(time, 4), geom1, geom2, float(contact.dist)])
                    body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or str(model.geom_bodyid[contact.geom1])
                    body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or str(model.geom_bodyid[contact.geom2])
                    pair = "|".join(sorted((body1, body2)))
                    contact_pair_counts[pair] = contact_pair_counts.get(pair, 0) + 1
                    if body1.startswith(("left_lf", "left_rf", "right_lf", "right_rf")) or body2.startswith(("left_lf", "left_rf", "right_lf", "right_rf")):
                        finger_contacts += 1
                        if first_finger_contact_time is None:
                            first_finger_contact_time = time
            if step % 10 == 0:
                qlog.append([time, *data.qpos[:16]])
                bottle_log.append([time, *bottle_pos])
                site_log.append([time, *site_pos])
            if writer is not None and step % max(1, round(1.0 / (30.0 * dt))) == 0:
                renderer.update_scene(data, camera="task1_head_camera", scene_option=render_option)
                frame = renderer.render()[:, :, ::-1].copy()
                cv2.putText(frame, f"task1 YAM hypothesis  t={time:.2f}s", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 220, 40), 2, cv2.LINE_AA)
                writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
        if renderer is not None:
            renderer.close()

    qlog_array = np.asarray(qlog)
    bottle_array = np.asarray(bottle_log)
    site_array = np.asarray(site_log)
    np.savez_compressed(OUTPUT / "task1_replay.npz", time=qlog_array[:, 0], qpos=qlog_array[:, 1:], bottle_pos=bottle_array[:, 1:], grasp_site=site_array[:, 1:])
    report = {
        "schema": "autopolicy.task1_modeling/v1",
        "authenticity": "single_video_with_explicit_robot_and_scale_hypotheses",
        "task": "抓取桌面上面的水瓶",
        "robot_model": {
            "name": "YAM bimanual station (similar-model substitution)",
            "source": str(ROOT / "vendor/gpt6-real2sim/real2sim_microphones/robot_portable.xml"),
            "arms": 2,
            "arm_dof_each": 6,
            "gripper_state_model": "single scalar symmetric open/close actuator per arm",
            "exact_identity_confirmed": False,
        },
        "camera_model": {
            "name": "single fixed pinhole head camera hypothesis",
            "resolution": [1920, 1080],
            "fov_y_deg": 60.0,
            "intrinsics": {"fx_px": 935.3, "fy_px": 935.3, "cx_px": 960.0, "cy_px": 540.0},
            "extrinsics": {"position_m": [0.0842089, 0.0133344, 1.70907], "orientation_quat_wxyz": [-0.674230, -0.182618, 0.196570, 0.688059], "mount": "fixed high-mounted calibration-head camera hypothesis"},
            "calibrated_from_video": False,
            "rendering": {"cast_shadows": False, "reason": "avoid EGL shadow-map artifacts from overlapping CAD meshes"},
        },
        "scale_hypotheses": {
            "table_surface_z_m": 0.762,
            "bottle_height_m": 0.24,
            "bottle_diameter_m": 0.066,
            "bottle_mass_kg": 0.55,
            "source": "common 500 mL water-bottle prior; not measured from video",
        },
        "metrics": {
            "simulation_duration_s": duration,
            "simulation_timestep_s": dt,
            "simulation_steps": int(duration / dt),
            "max_collision_penetration_m": max_penetration,
            "bottle_height_max_m": float(np.max(bottle_array[:, 3])),
            "bottle_height_final_m": float(bottle_array[-1, 3]),
            "grasp_contacts": len(contacts),
            "finger_contacts": finger_contacts,
            "first_finger_contact_time_s": first_finger_contact_time,
            "first_lift_time_s": first_lift_time,
            "top_contact_pairs": sorted(contact_pair_counts.items(), key=lambda item: item[1], reverse=True)[:12],
            "bottle_lifted_above_0_83m": lifted,
        },
        "passed": bool(lifted and finger_contacts > 0 and max_penetration < 0.01),
        "limitations": [
            "Robot identity, camera intrinsics/extrinsics, joint trajectory, and bottle dimensions are hypotheses because the video has no calibration/state logs.",
            "The scripted trajectory is generated by position IK and is not the recorded robot trajectory.",
            "Visual similarity and physical task success are separate claims; this is a model feasibility test.",
        ],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "metrics": report["metrics"], "report": str(OUTPUT / "report.json")}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
