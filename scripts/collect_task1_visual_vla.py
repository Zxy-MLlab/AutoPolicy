#!/usr/bin/env python3
"""Collect MuJoCo expert demonstrations in the visual SmolVLA schema."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from lerobot.datasets import LeRobotDataset


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
SOURCE_SCENE = ROOT / "vendor/gpt6-real2sim/real2sim_microphones/scene_portable.xml"
TASK = "抓取桌面上的水瓶并抬离桌面"


def sample_positions(rng, count, low, high, sampling):
    if count == 0:
        return np.empty((0, 2), dtype=np.float64)
    if sampling == "random":
        return rng.uniform(low, high, size=(count, 2)).astype(np.float64)

    nx = math.ceil(math.sqrt(count))
    ny = math.ceil(count / nx)
    cell = (high - low) / np.array([nx, ny], dtype=np.float64)
    centers = np.array(
        [low + (np.array([ix, iy]) + 0.5) * cell for iy in range(ny) for ix in range(nx)]
    )
    rng.shuffle(centers)
    jitter = rng.uniform(-0.3, 0.3, size=centers.shape) * cell
    return np.clip(centers + jitter, low, high)[:count]


def derive_focus_range(args, low, high):
    if args.focus_episodes == 0:
        return None, None
    if args.focus_x_range is not None:
        focus_low = np.array([args.focus_x_range[0], low[1]], dtype=np.float64)
        focus_high = np.array([args.focus_x_range[1], high[1]], dtype=np.float64)
        return (focus_low, focus_high), {"method": "explicit"}
    if args.failure_report is None:
        raise ValueError("focus episodes require --failure-report or --focus-x-range")

    report = json.loads(args.failure_report.read_text(encoding="utf-8"))
    failures = [row for row in report.get("episode_results", []) if not row.get("success", False)]
    failure_x = [row["bottle_xy_ground_truth_for_scoring_only"][0] for row in failures]
    if not failure_x:
        raise ValueError(f"no failed episodes with bottle positions in {args.failure_report}")
    x_threshold = float(np.quantile(failure_x, args.failure_x_quantile))
    focus_low = np.array([max(low[0], x_threshold), low[1]], dtype=np.float64)
    focus_high = high.copy()
    return (focus_low, focus_high), {
        "method": "failure_x_quantile",
        "failure_report": str(args.failure_report.resolve()),
        "failure_count": len(failure_x),
        "x_quantile": args.failure_x_quantile,
        "derived_x_threshold_m": x_threshold,
    }


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
    parser.add_argument(
        "--execution-noise-std-rad",
        type=float,
        default=0.0,
        help=(
            "Stddev of zero-mean arm-joint target noise, refreshed at the dataset FPS. "
            "Observations reflect the perturbed rollout while labels remain clean expert actions."
        ),
    )
    parser.add_argument(
        "--execution-bias-std-rad",
        type=float,
        default=0.0,
        help="Stddev of an episode-constant arm-joint target bias used for recovery-tube data.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Independent execution-noise seed (default: --seed + 1000003).",
    )
    parser.add_argument("--x-range", type=float, nargs=2, default=[0.43, 0.51])
    parser.add_argument("--y-range", type=float, nargs=2, default=[0.12, 0.20])
    parser.add_argument(
        "--sampling",
        choices=("random", "stratified"),
        default="random",
        help="Stratified sampling covers the full XY workspace with a jittered grid.",
    )
    parser.add_argument(
        "--focus-episodes",
        type=int,
        default=0,
        help="Additional episodes sampled from a failure-focused subregion.",
    )
    parser.add_argument("--focus-x-range", type=float, nargs=2, default=None)
    parser.add_argument(
        "--failure-report",
        type=Path,
        default=None,
        help="Evaluation report used to derive the focus region from failed positions.",
    )
    parser.add_argument("--failure-x-quantile", type=float, default=0.5)
    parser.add_argument("--repo-id", default="autopolicy/task1_smolvla")
    parser.add_argument("--root", type=Path, default=ROOT / "data/task1_smolvla")
    parser.add_argument(
        "--successful-only",
        action="store_true",
        help="Discard failed episode buffers so only successful expert rollouts enter the dataset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.execution_noise_std_rad < 0 or args.execution_bias_std_rad < 0:
        raise ValueError("execution noise standard deviations must be non-negative")
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
        repo_id=args.repo_id,
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
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed + 1_000_003
    noise_rng = np.random.default_rng(noise_seed)
    low = np.array([args.x_range[0], args.y_range[0]], dtype=np.float64)
    high = np.array([args.x_range[1], args.y_range[1]], dtype=np.float64)
    if np.any(high <= low):
        raise ValueError(f"invalid workspace range: low={low.tolist()} high={high.tolist()}")
    if not 0 <= args.focus_episodes <= args.episodes:
        raise ValueError(f"focus episodes must be in [0, {args.episodes}]")
    if not 0.0 <= args.failure_x_quantile <= 1.0:
        raise ValueError("failure x quantile must be in [0, 1]")
    focus_range, focus_derivation = derive_focus_range(args, low, high)
    if focus_range is not None:
        focus_low, focus_high = focus_range
        if np.any(focus_high <= focus_low) or np.any(focus_low < low) or np.any(focus_high > high):
            raise ValueError(
                f"invalid focus range: low={focus_low.tolist()} high={focus_high.tolist()} "
                f"workspace={low.tolist()}..{high.tolist()}"
            )
    base_episodes = args.episodes - args.focus_episodes
    base_positions = sample_positions(rng, base_episodes, low, high, args.sampling)
    focused_positions = (
        sample_positions(rng, args.focus_episodes, focus_low, focus_high, args.sampling)
        if focus_range is not None
        else np.empty((0, 2), dtype=np.float64)
    )
    bottle_positions = np.concatenate([base_positions, focused_positions], axis=0)
    position_sources = np.array(
        ["global"] * base_episodes + ["failure_focus"] * args.focus_episodes, dtype=object
    )
    order = rng.permutation(args.episodes)
    bottle_positions = bottle_positions[order]
    position_sources = position_sources[order]
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    hide_artifact_visuals(model)
    render_option = mujoco.MjvOption()
    render_option.geomgroup[5] = 0
    successes = 0
    saved_episodes = 0
    episode_metrics = []
    try:
        for episode in range(args.episodes):
            bottle_xy = bottle_positions[episode].copy()
            data.qpos[:16] = initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array([*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0])
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            plan = episode_plan(model, data, initial_q, bottle_xy, site_id)
            # IK planning mutates data.qpos while evaluating candidate waypoints.
            # Restore the declared episode initial state before executing/recording
            # the plan so every demonstration starts from the same robot home pose.
            data.qpos[:16] = initial_q
            data.qpos[bottle_qpos:bottle_qpos + 7] = np.array(
                [*bottle_xy, 0.762, 1.0, 0.0, 0.0, 0.0]
            )
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)
            execution_bias = np.zeros(6, dtype=np.float64)
            if args.execution_bias_std_rad > 0:
                execution_bias = noise_rng.normal(0.0, args.execution_bias_std_rad, size=6)
            execution_noise = np.zeros(6, dtype=np.float64)
            max_penetration = 0.0
            finger_contacts = 0
            for step in range(horizon_steps):
                time = step * dt
                action = command_at(plan, time)
                if step % stride == 0 and args.execution_noise_std_rad > 0:
                    execution_noise = noise_rng.normal(0.0, args.execution_noise_std_rad, size=6)
                data.ctrl[:6] = np.clip(
                    action[:6] + execution_bias + execution_noise,
                    model.actuator_ctrlrange[:6, 0],
                    model.actuator_ctrlrange[:6, 1],
                )
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
            episode_metrics.append({"episode": episode, "position_source": str(position_sources[episode]), "bottle_xy": bottle_xy.tolist(), "final_bottle_z_m": final_z, "finger_contacts": finger_contacts, "max_penetration_m": max_penetration, "success": success})
            if success or not args.successful_only:
                dataset.save_episode()
                saved_episodes += 1
            else:
                dataset.clear_episode_buffer()
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
        "workspace_xy_m": {"low": low.tolist(), "high": high.tolist()},
        "position_sampling": args.sampling,
        "position_seed": args.seed,
        "robot_initial_state": {
            "source": str(SOURCE_SCENE),
            "keyframe_index": 0,
            "qpos_dimensions": 16,
            "restored_after_ik_planning": True,
        },
        "global_episodes": base_episodes,
        "failure_focus_episodes": args.focus_episodes,
        "failure_focus_xy_m": (
            {"low": focus_low.tolist(), "high": focus_high.tolist()}
            if focus_range is not None
            else None
        ),
        "failure_focus_derivation": focus_derivation,
        "recovery_tube": {
            "method": "perturbed_execution_with_clean_expert_labels",
            "arm_execution_noise_std_rad": args.execution_noise_std_rad,
            "arm_episode_bias_std_rad": args.execution_bias_std_rad,
            "noise_refresh_hz": args.fps,
            "noise_seed": noise_seed,
            "gripper_execution_perturbed": False,
            "label_is_clean_expert_action": True,
        },
        "attempted_episodes": args.episodes,
        "dataset_episodes": saved_episodes,
        "successful_only": args.successful_only,
        "frames": saved_episodes * round(9.0 * args.fps),
        "expert_successes": successes,
        "expert_success_rate": successes / args.episodes if args.episodes else 0.0,
        "dataset": str(args.root),
        "episode_metrics": episode_metrics,
    }
    (args.root / "collection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(args.root), "episodes": args.episodes, "frames": report["frames"], "expert_success_rate": report["expert_success_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
