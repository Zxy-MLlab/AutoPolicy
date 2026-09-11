#!/usr/bin/env python3
"""Collect MuJoCo expert demonstrations in the visual SmolVLA schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from lerobot.datasets import LeRobotDataset


ROOT = Path("/data/zxy/autopolicy")
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
SOURCE_SCENE = ROOT / "vendor/gpt6-real2sim/real2sim_microphones/scene_portable.xml"
TASK = "抓取桌面上的水瓶并抬离桌面"


def solve_position_ik(model, data, q, target, site_id):
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


def episode_plan(model, data, initial_q, bottle_xy, site_id):
    x, y = bottle_xy
    waypoints = [
        (0.0, np.array([x, y, 1.06]), 0.0475),
        (2.5, np.array([x, y, 0.88]), 0.0475),
        (4.0, np.array([x, y, 0.88]), 0.0),
        (5.0, np.array([x, y, 1.08]), 0.0),
        (9.0, np.array([x, y, 1.08]), 0.0),
    ]
    return [(t, solve_position_ik(model, data, initial_q, target, site_id), grip) for t, target, grip in waypoints]


def command_at(plan, time):
    segment = max(0, min(len(plan) - 2, next((i for i in range(len(plan) - 1) if time <= plan[i + 1][0]), len(plan) - 2)))
    t0, q0, g0 = plan[segment]
    t1, q1, g1 = plan[segment + 1]
    alpha = np.clip((time - t0) / (t1 - t0), 0.0, 1.0) if t1 != t0 else 0.0
    action = np.zeros(14, dtype=np.float32)
    action[:6] = (1.0 - alpha) * q0[:6] + alpha * q1[:6]
    action[6] = (1.0 - alpha) * g0 + alpha * g1
    action[7:13] = q0[8:14]
    action[13] = 0.0475
    return action


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
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--root", type=Path, default=ROOT / "data/task1_smolvla")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"dataset exists: {args.root}; pass --overwrite to replace it")
        import shutil
        shutil.rmtree(args.root)
    state_names = [f"left_joint_{i}" for i in range(6)] + ["left_finger_left", "left_finger_right"] + [f"right_joint_{i}" for i in range(6)] + ["right_finger_left", "right_finger_right"]
    action_names = [f"left_joint_{i}" for i in range(6)] + ["left_gripper"] + [f"right_joint_{i}" for i in range(6)] + ["right_gripper"]
    features = {
        "observation.images.camera_00": {"dtype": "video", "shape": (args.height, args.width, 3), "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (16,), "names": {"axes": state_names}},
        "action": {"dtype": "float32", "shape": (14,), "names": {"axes": action_names}},
    }
    dataset = LeRobotDataset.create(
        repo_id="autopolicy/task1_smolvla",
        root=args.root,
        fps=args.fps,
        features=features,
        robot_type="yam_task1",
        use_videos=True,
        image_writer_threads=2,
    )

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE_SCENE))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    bottle_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "task1_bottle_free")
    bottle_qpos = model.jnt_qposadr[bottle_joint]
    initial_q = source.key_qpos[0, :16].copy()
    dt = float(model.opt.timestep)
    stride = max(1, round(1.0 / (args.fps * dt)))
    horizon_steps = round(9.0 / dt)
    rng = np.random.default_rng(args.seed)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    hide_artifact_visuals(model)
    render_option = mujoco.MjvOption()
    render_option.geomgroup[5] = 0
    successes = 0
    episode_metrics = []
    try:
        for episode in range(args.episodes):
            bottle_xy = rng.uniform([0.43, 0.12], [0.51, 0.20]).astype(np.float64)
            data.qpos[:16] = initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array([*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0])
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            plan = episode_plan(model, data, initial_q, bottle_xy, site_id)
            max_penetration = 0.0
            finger_contacts = 0
            for step in range(horizon_steps):
                time = step * dt
                action = command_at(plan, time)
                data.ctrl[:6] = action[:6]
                data.ctrl[6] = action[6]
                data.ctrl[7:13] = action[7:13]
                data.ctrl[13] = action[13]
                mujoco.mj_step(model, data)
                for contact in data.contact:
                    max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                    g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                    g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                    if "task1_bottle" in g1 or "task1_bottle" in g2:
                        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or ""
                        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or ""
                        if b1.startswith(("left_lf", "left_rf")) or b2.startswith(("left_lf", "left_rf")):
                            finger_contacts += 1
                if step % stride == 0:
                    renderer.update_scene(data, camera="task1_head_camera", scene_option=render_option)
                    image = renderer.render().copy()
                    dataset.add_frame({
                        "observation.images.camera_00": image,
                        "observation.state": data.qpos[:16].astype(np.float32),
                        "action": action,
                        "task": TASK,
                    })
            final_z = float(data.xpos[bottle_id, 2])
            success = bool(final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01)
            successes += int(success)
            episode_metrics.append({"episode": episode, "bottle_xy": bottle_xy.tolist(), "final_bottle_z_m": final_z, "finger_contacts": finger_contacts, "max_penetration_m": max_penetration, "success": success})
            dataset.save_episode()
            print(json.dumps({"episode": episode, "success": success, "final_z": final_z}, ensure_ascii=False), flush=True)
    finally:
        renderer.close()
        dataset.finalize()

    report = {
        "schema": "autopolicy.task1_visual_vla/v1",
        "task": TASK,
        "policy_input": ["observation.images.camera_00", "observation.state", "task"],
        "policy_input_excludes": ["bottle_xy", "normalized_time", "oracle_object_pose"],
        "action_shape": [14],
        "state_shape": [16],
        "camera_shape_hwc": [args.height, args.width, 3],
        "fps": args.fps,
        "episodes": args.episodes,
        "frames": args.episodes * round(9.0 * args.fps),
        "expert_successes": successes,
        "expert_success_rate": successes / args.episodes if args.episodes else 0.0,
        "dataset": str(args.root),
        "episode_metrics": episode_metrics,
    }
    (args.root / "collection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.root), "episodes": args.episodes, "frames": report["frames"], "expert_success_rate": report["expert_success_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
