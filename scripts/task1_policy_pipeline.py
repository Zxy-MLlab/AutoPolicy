#!/usr/bin/env python3
"""Collect, train, and evaluate an oracle-state policy for task1 in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path("/data/zxy/autopolicy")
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
SOURCE = ROOT / "vendor/gpt6-real2sim/real2sim_microphones/scene_portable.xml"
RUN = ROOT / "runs/task1-policy-pipeline"
RAW = RUN / "raw_rollouts.npz"
MODEL = RUN / "policy_model.npz"


def solve_ik(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, target: np.ndarray, site_id: int) -> np.ndarray:
    q = q.copy()
    for _ in range(500):
        data.qpos[:16] = q
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        if np.linalg.norm(error) < 0.0008:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = jacp[:, :6]
        damping = 0.035
        dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
        q[:6] += 0.45 * dq
        for joint_index in range(6):
            q[joint_index] = np.clip(q[joint_index], model.jnt_range[joint_index, 0], model.jnt_range[joint_index, 1])
    return q


def episode_plan(model: mujoco.MjModel, data: mujoco.MjData, initial_q: np.ndarray, bottle_xy: np.ndarray, site_id: int):
    x, y = bottle_xy
    waypoints = [
        (0.0, np.array([x, y, 1.06]), 0.0475),
        (2.5, np.array([x, y, 0.88]), 0.0475),
        (4.0, np.array([x, y, 0.88]), 0.0),
        (5.0, np.array([x, y, 1.08]), 0.0),
        (8.0, np.array([x, y, 1.08]), 0.0),
    ]
    return [(time, solve_ik(model, data, initial_q, target, site_id), grip) for time, target, grip in waypoints]


def command_at(plan, time: float) -> np.ndarray:
    segment = max(0, min(len(plan) - 2, next((i for i in range(len(plan) - 1) if time <= plan[i + 1][0]), len(plan) - 2)))
    t0, q0, g0 = plan[segment]
    t1, q1, g1 = plan[segment + 1]
    alpha = 0.0 if t1 == t0 else np.clip((time - t0) / (t1 - t0), 0.0, 1.0)
    command = np.zeros(7, dtype=np.float32)
    command[:6] = (1.0 - alpha) * q0[:6] + alpha * q1[:6]
    command[6] = (1.0 - alpha) * g0 + alpha * g1
    return command


def make_observation(target_xy: np.ndarray, time: float) -> np.ndarray:
    """Oracle policy input: latched bottle XY target and normalized episode time."""
    return np.asarray([target_xy[0], target_xy[1], time / 9.0], dtype=np.float32)


def collect(episodes: int, seed: int) -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    initial_q = source.key_qpos[0, :16].copy()
    rng = np.random.default_rng(seed)
    obs_rows, action_rows, episode_rows = [], [], []
    dt = float(model.opt.timestep)
    control_stride = 20
    horizon_steps = int(9.0 / dt)
    successes = 0
    for episode in range(episodes):
        bottle_xy = rng.uniform([0.43, 0.12], [0.51, 0.20]).astype(np.float64)
        data.qpos[:16] = initial_q
        data.qpos[16:23] = np.array([*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        plan = episode_plan(model, data, initial_q, bottle_xy, site_id)
        start = len(obs_rows)
        max_penetration = 0.0
        finger_contacts = 0
        for step in range(horizon_steps):
            time = step * dt
            command = command_at(plan, time)
            data.ctrl[:6] = command[:6]
            data.ctrl[6] = command[6]
            data.ctrl[7:13] = initial_q[8:14]
            data.ctrl[13] = 0.0475
            if step % control_stride == 0:
                obs_rows.append(make_observation(bottle_xy, time))
                action_rows.append(command)
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
        final_z = float(data.xpos[bottle_id, 2])
        success = bool(final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01)
        successes += int(success)
        episode_rows.append([episode, start, len(obs_rows) - start, *bottle_xy, final_z, int(success), finger_contacts, max_penetration])
    obs = np.asarray(obs_rows, dtype=np.float32)
    actions = np.asarray(action_rows, dtype=np.float32)
    episodes_array = np.asarray(episode_rows, dtype=np.float64)
    np.savez_compressed(RAW, observations=obs, actions=actions, episodes=episodes_array)
    summary = {
        "authenticity": "synthetic_mujoco_expert_rollouts",
        "policy_observation": "oracle_target (latched bottle XY and normalized time)",
        "episodes": episodes,
        "frames": int(len(obs)),
        "control_hz": int(round(1.0 / (dt * control_stride))),
        "successes": successes,
        "success_rate": successes / episodes if episodes else 0.0,
        "raw_rollouts": str(RAW),
    }
    (RUN / "collection_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def relu(x):
    return np.maximum(x, 0.0)


def train(seed: int, steps: int) -> dict:
    data = np.load(RAW)
    x = data["observations"].astype(np.float64)
    y = data["actions"].astype(np.float64)
    rng = np.random.default_rng(seed)
    mean_x, std_x = x.mean(0), x.std(0) + 1e-6
    mean_y, std_y = y.mean(0), y.std(0) + 1e-6
    xn, yn = (x - mean_x) / std_x, (y - mean_y) / std_y
    n, din, dout = len(xn), xn.shape[1], yn.shape[1]
    hidden = 64
    weights = {
        "w1": rng.normal(0, np.sqrt(2 / din), (din, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0, np.sqrt(2 / hidden), (hidden, hidden)),
        "b2": np.zeros(hidden),
        "w3": rng.normal(0, np.sqrt(2 / hidden), (hidden, dout)),
        "b3": np.zeros(dout),
    }
    moments = {key: np.zeros_like(value) for key, value in weights.items()}
    velocities = {key: np.zeros_like(value) for key, value in weights.items()}
    losses = []
    batch = min(512, n)
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        xb, yb = xn[idx], yn[idx]
        z1 = xb @ weights["w1"] + weights["b1"]
        h1 = relu(z1)
        z2 = h1 @ weights["w2"] + weights["b2"]
        h2 = relu(z2)
        pred = h2 @ weights["w3"] + weights["b3"]
        diff = (pred - yb) / batch
        grads = {
            "w3": h2.T @ diff,
            "b3": diff.sum(0),
        }
        dh2 = diff @ weights["w3"].T
        dz2 = dh2 * (z2 > 0)
        grads.update({"w2": h1.T @ dz2, "b2": dz2.sum(0)})
        dh1 = dz2 @ weights["w2"].T
        dz1 = dh1 * (z1 > 0)
        grads.update({"w1": xb.T @ dz1, "b1": dz1.sum(0)})
        for key in weights:
            moments[key] = 0.9 * moments[key] + 0.1 * grads[key]
            velocities[key] = 0.999 * velocities[key] + 0.001 * grads[key] ** 2
            mhat = moments[key] / (1 - 0.9**step)
            vhat = velocities[key] / (1 - 0.999**step)
            weights[key] -= 0.002 * mhat / (np.sqrt(vhat) + 1e-8)
        if step == 1 or step % 100 == 0:
            full_hidden1 = relu(xn @ weights["w1"] + weights["b1"])
            full_hidden2 = relu(full_hidden1 @ weights["w2"] + weights["b2"])
            full_prediction = full_hidden2 @ weights["w3"] + weights["b3"]
            full = ((full_prediction - yn) ** 2).mean()
            losses.append([step, float(full)])
    np.savez_compressed(MODEL, **weights, mean_x=mean_x, std_x=std_x, mean_y=mean_y, std_y=std_y)
    report = {"authenticity": "trained_from_synthetic_mujoco_expert_rollouts", "architecture": [din, hidden, hidden, dout], "steps": steps, "samples": n, "final_normalized_mse": losses[-1][1], "loss_curve": losses, "checkpoint": str(MODEL)}
    (RUN / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def predict(model_data, observation):
    x = (observation - model_data["mean_x"]) / model_data["std_x"]
    h1 = relu(x @ model_data["w1"] + model_data["b1"])
    h2 = relu(h1 @ model_data["w2"] + model_data["b2"])
    normalized = h2 @ model_data["w3"] + model_data["b3"]
    return normalized * model_data["std_y"] + model_data["mean_y"]


def evaluate(episodes: int, seed: int, render: bool = False) -> dict:
    model_data = dict(np.load(MODEL))
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    source = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_grasp_site")
    bottle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "task1_bottle")
    initial_q = source.key_qpos[0, :16].copy()
    rng = np.random.default_rng(seed)
    rows = []
    dt = float(model.opt.timestep)
    renderer = writer = render_option = None
    video_path = RUN / "trained_policy_rollout.mp4"
    if render:
        model.light_castshadow[:] = 0
        for geom_id in range(model.ngeom):
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom_id])) if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH else None
            if geom_name in {"back_wall", "left_wall", "right_wall", "task1_back_counter"} or mesh_name == "base_visual_gate":
                model.geom_group[geom_id] = 5
        render_option = mujoco.MjvOption()
        render_option.geomgroup[5] = 0
        renderer = mujoco.Renderer(model, height=540, width=960)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
    for episode in range(episodes):
        xy = rng.uniform([0.43, 0.12], [0.51, 0.20])
        data.qpos[:16] = initial_q
        data.qpos[16:23] = np.array([*xy, 0.762, 1.0, 0.0, 0.0, 0.0])
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        max_penetration, finger_contacts = 0.0, 0
        for step in range(int(9.0 / dt)):
            if step % 20 == 0:
                obs = make_observation(xy, step * dt)
                command = predict(model_data, obs)
                command[6] = 0.0475 if command[6] >= 0.024 else 0.0
                command[:6] = np.clip(command[:6], model.actuator_ctrlrange[:6, 0], model.actuator_ctrlrange[:6, 1])
            data.ctrl[:6] = command[:6]
            data.ctrl[6] = command[6]
            data.ctrl[7:13] = initial_q[8:14]
            data.ctrl[13] = 0.0475
            mujoco.mj_step(model, data)
            if episode == 0 and writer is not None and step % max(1, round(1.0 / (30.0 * dt))) == 0:
                renderer.update_scene(data, camera="task1_head_camera", scene_option=render_option)
                frame = renderer.render()[:, :, ::-1].copy()
                cv2.putText(frame, f"trained oracle-target policy  t={step * dt:.2f}s", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 220, 40), 2, cv2.LINE_AA)
                writer.write(frame)
            for contact in data.contact:
                max_penetration = max(max_penetration, max(0.0, -float(contact.dist)))
                g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
                g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
                if "task1_bottle" in g1 or "task1_bottle" in g2:
                    b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]) or ""
                    b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]) or ""
                    if b1.startswith(("left_lf", "left_rf")) or b2.startswith(("left_lf", "left_rf")):
                        finger_contacts += 1
        final_z = float(data.xpos[bottle_id, 2])
        rows.append({"episode": episode, "bottle_xy": [float(xy[0]), float(xy[1])], "final_bottle_z_m": final_z, "finger_contacts": finger_contacts, "max_penetration_m": max_penetration, "success": bool(final_z > 0.83 and finger_contacts > 0 and max_penetration < 0.01)})
    if writer is not None:
        writer.release()
        renderer.close()
    result = {"authenticity": "closed_loop_mujoco_policy_evaluation", "checkpoint": str(MODEL), "episodes": rows, "successes": sum(row["success"] for row in rows), "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else 0.0, "policy_input": "oracle target (latched bottle XY and normalized time); no RGB perception", "video": str(video_path) if render else None}
    (RUN / "evaluation_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect"); p_collect.add_argument("--episodes", type=int, default=80); p_collect.add_argument("--seed", type=int, default=7)
    p_train = sub.add_parser("train"); p_train.add_argument("--steps", type=int, default=2000); p_train.add_argument("--seed", type=int, default=11)
    p_eval = sub.add_parser("evaluate"); p_eval.add_argument("--episodes", type=int, default=20); p_eval.add_argument("--seed", type=int, default=23); p_eval.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.command == "collect": result = collect(args.episodes, args.seed)
    elif args.command == "train": result = train(args.seed, args.steps)
    else: result = evaluate(args.episodes, args.seed, args.render)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
