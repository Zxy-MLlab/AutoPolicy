#!/usr/bin/env python3
"""Re-run and validate the published GPT6-real2sim DROID ep0 model.

This deliberately runs the simulator from the recorded robot trajectory. It does not
accept the archived validation JSON as proof of a successful re-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


ROOT = Path("/data/zxy/autopolicy").resolve()
EP0 = ROOT / "vendor/gpt6-real2sim/real2sim_ep0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "runs/real2sim-ep0-model-validation",
        help="Directory for the independent verification report and outputs.",
    )
    parser.add_argument("--render", action="store_true", help="Also create a contact-sheet render.")
    return parser.parse_args()


def project_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cache = ROOT / "cache"
    values = {
        "HOME": ROOT,
        "TMPDIR": ROOT / "tmp",
        "XDG_CACHE_HOME": cache / "xdg",
        "MPLCONFIGDIR": cache / "matplotlib",
        "PYTHONNOUSERSITE": "1",
        "MUJOCO_GL": "egl",
    }
    for key, value in values.items():
        environment[key] = str(value)
        if isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)
    return environment


def assert_model_and_replay(scene: Path, replay: Path) -> dict[str, Any]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    with np.load(replay) as stored:
        qpos = stored["qpos"]
        qvel = stored["qvel"]
        body_pos = stored["body_pos"]
        body_quat = stored["body_quat"]
        body_names = stored["body_names"].tolist()
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise RuntimeError(f"qpos shape {qpos.shape} does not match model nq={model.nq}")
        if qvel.shape[1] != model.nv:
            raise RuntimeError(f"qvel shape {qvel.shape} does not match model nv={model.nv}")
        if body_pos.shape[1] != model.nbody or body_quat.shape[1] != model.nbody:
            raise RuntimeError("stored body transforms do not match model body count")
        expected_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
        if body_names != expected_names:
            raise RuntimeError("stored body names do not match scene.xml")
        finite_samples = 0
        for index in np.unique(np.linspace(0, len(qpos) - 1, min(40, len(qpos)), dtype=int)):
            data.qpos[:] = qpos[index]
            data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.xpos).all() or not np.isfinite(data.xquat).all():
                raise RuntimeError(f"non-finite transform at replay sample {index}")
            finite_samples += 1
        if not np.isfinite(body_pos).all() or not np.isfinite(body_quat).all():
            raise RuntimeError("stored transforms contain non-finite values")
        if not np.allclose(np.linalg.norm(body_quat, axis=-1), 1.0, atol=1e-6):
            raise RuntimeError("stored body quaternions are not normalized")
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "njnt": model.njnt,
        "nu": model.nu,
        "replay_frames": int(len(qpos)),
        "finite_replay_samples_checked": finite_samples,
        "freejoint_marker": bool(model.joint("marker_free").type[0] == mujoco.mjtJoint.mjJNT_FREE),
        "scene_assets_loaded": True,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if run_dir != ROOT and ROOT not in run_dir.parents:
        raise SystemExit(f"run directory must be under {ROOT}")
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = project_environment()
    label = "autopolicy_ep0_verified"
    command = [sys.executable, "simulate.py", "--label", label, "--observed-gripper"]
    if args.render:
        (EP0 / "inspection").mkdir(parents=True, exist_ok=True)
        command.append("--render")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    completed = subprocess.run(
        command,
        cwd=EP0,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (run_dir / "simulation.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"simulation failed with exit code {completed.returncode}; see {run_dir / 'simulation.log'}")

    physics_path = EP0 / f"{label}_physics.json"
    replay_path = EP0 / f"{label}_replay.npz"
    if not physics_path.is_file() or not replay_path.is_file():
        raise SystemExit("simulation did not produce both physics JSON and replay NPZ")
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    model_report = assert_model_and_replay(EP0 / "scene.xml", replay_path)
    archived = json.loads((EP0 / "physics_physics.json").read_text(encoding="utf-8"))
    numeric_fields = (
        "marker_final_xyz",
        "marker_at_placement_xyz",
        "marker_max_height",
        "bilateral_finger_contact_seconds",
        "arm_tracking_rmse_rad",
        "arm_tracking_max_rad",
        "max_penetration_m",
        "penetration_p95_m",
    )
    drift: dict[str, Any] = {}
    for field in numeric_fields:
        current = np.asarray(physics[field], dtype=float)
        reference = np.asarray(archived[field], dtype=float)
        drift[field] = {
            "max_absolute_difference": float(np.max(np.abs(current - reference))),
            "match": bool(np.allclose(current, reference, rtol=1e-7, atol=1e-9)),
        }
    thresholds = {
        "max_penetration_m": 0.005,
        "max_trajectory_rmse_rad": 0.02,
        "max_arm_tracking_error_rad": 0.01,
    }
    gates = {
        "model_loads": model_report["scene_assets_loaded"],
        "replay_transforms_finite": model_report["finite_replay_samples_checked"] > 0,
        "marker_is_free_body": model_report["freejoint_marker"],
        "arm_tracking": float(physics["arm_tracking_rmse_rad"]) <= thresholds["max_trajectory_rmse_rad"]
        and float(physics["arm_tracking_max_rad"]) <= thresholds["max_arm_tracking_error_rad"],
        "collision": float(physics["max_penetration_m"]) <= thresholds["max_penetration_m"],
        "task_transfer": bool(
            physics["out_of_bowl_at_end"]
            and physics["on_counter_at_placement"]
            and physics["final_on_counter"]
        ),
        "deterministic_against_archived_replay": all(item["match"] for item in drift.values()),
    }
    outputs: list[dict[str, Any]] = []
    for source in (physics_path, replay_path):
        destination = run_dir / source.name
        shutil.copy2(source, destination)
        outputs.append({"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)})
    if args.render:
        image = EP0 / "inspection" / f"{label}_physics.jpg"
        if image.is_file():
            destination = run_dir / image.name
            shutil.copy2(image, destination)
            outputs.append({"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)})

    report = {
        "schema": "autopolicy.real2sim_validation/v1",
        "authenticity": "rerun_of_real_recorded_trajectory",
        "scene": "GPT6-real2sim DROID ep0",
        "source_repository": str(EP0),
        "source_commit": "fc74e18b35b40dcc440faf7470b0b4d6bff1fb6a",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": command,
        "model": model_report,
        "metrics": physics,
        "thresholds": thresholds,
        "gates": gates,
        "passed": all(gates.values()),
        "drift_against_archived_physics": drift,
        "limitations": [
            "The scene is an approximate reconstruction with manually fitted sparse camera landmarks.",
            "The recorded later marker roll/fall is not reproduced by this model.",
            "Object mass, friction, bowl thickness, and some background geometry are assumptions.",
            "This validates the simulator model and nominal replay, not hardware safety or sim-to-real transfer.",
        ],
        "outputs": outputs,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "gates": gates, "run_dir": str(run_dir)}, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
