#!/usr/bin/env python3
"""Render a verified Real2Sim replay as a side-by-side MP4."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1].resolve()
EP0 = ROOT / "vendor/gpt6-real2sim/real2sim_ep0"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        type=Path,
        default=ROOT / "runs/real2sim-ep0-model-validation/autopolicy_ep0_verified_replay.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/real2sim-ep0-model-validation/autopolicy_ep0_verified.mp4",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    replay_path = args.replay.resolve()
    output = args.output.resolve()
    if ROOT not in replay_path.parents or ROOT not in output.parents:
        raise SystemExit(f"paths must be under {ROOT}")

    model = mujoco.MjModel.from_xml_path(str(EP0 / "scene.xml"))
    data = mujoco.MjData(model)
    with np.load(replay_path) as replay:
        qpos = replay["qpos"]
        times = replay["time"]

    width, height = 640, 360
    renderer = mujoco.Renderer(model, height=height, width=width)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open video writer: {output}")
    try:
        for index, qpos_value in enumerate(qpos):
            data.qpos[:] = qpos_value
            mujoco.mj_forward(model, data)
            panels = []
            for camera in ("23404442", "29838012"):
                renderer.update_scene(data, camera=camera, scene_option=option)
                panel = renderer.render()[:, :, ::-1].copy()
                cv2.putText(
                    panel,
                    f"{camera}  t={float(times[index]):.2f}s",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 220, 40),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(panel)
            writer.write(np.hstack(panels))
    finally:
        writer.release()
        renderer.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
