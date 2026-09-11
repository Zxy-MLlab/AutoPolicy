#!/usr/bin/env python3
"""Convert task1 oracle-state rollouts into a state-only LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("/data/zxy/autopolicy")
RUN = ROOT / "runs/task1-policy-pipeline"
RAW = RUN / "raw_rollouts.npz"
DATASET = ROOT / "data/task1_oracle_state_lerobot"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def main() -> int:
    raw = np.load(RAW)
    observations = raw["observations"].astype(np.float32)
    actions = raw["actions"].astype(np.float32)
    episodes = raw["episodes"]
    if DATASET.exists():
        shutil.rmtree(DATASET)
    (DATASET / "meta").mkdir(parents=True)
    (DATASET / "data/chunk-000").mkdir(parents=True)
    episode_rows = []
    stats_rows = []
    task_text = "抓取桌面上的水瓶并抬离桌面"
    state_names = [f"state_{i:02d}" for i in range(observations.shape[1])]
    action_names = [f"left_joint_{i+1}" for i in range(6)] + ["left_gripper"]
    for row in episodes:
        episode_index, start, length = (int(row[0]), int(row[1]), int(row[2]))
        end = start + length
        table = pa.table(
            {
                "observation.state": pa.array(observations[start:end].tolist(), type=pa.list_(pa.float32(), observations.shape[1])),
                "action": pa.array(actions[start:end].tolist(), type=pa.list_(pa.float32(), actions.shape[1])),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / 50.0),
                "frame_index": pa.array(np.arange(length, dtype=np.int64)),
                "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(start, end, dtype=np.int64)),
                "task_index": pa.array(np.zeros(length, dtype=np.int64)),
            }
        )
        pq.write_table(table, DATASET / f"data/chunk-000/episode_{episode_index:06d}.parquet", compression="zstd")
        episode_rows.append({"episode_index": episode_index, "tasks": [task_text], "length": length})
        stats_rows.append({"episode_index": episode_index, "stats": {"observation.state": {"mean": observations[start:end].mean(0).tolist(), "std": (observations[start:end].std(0) + 1e-6).tolist()}, "action": {"mean": actions[start:end].mean(0).tolist(), "std": (actions[start:end].std(0) + 1e-6).tolist()}}})
    info = {
        "codebase_version": "v2.1",
        "robot_type": "yam_task1_oracle_state",
        "total_episodes": len(episode_rows),
        "total_frames": int(len(observations)),
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": 50,
        "splits": {"train": f"0:{len(episode_rows)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [int(observations.shape[1])], "names": [state_names]},
            "action": {"dtype": "float32", "shape": [int(actions.shape[1])], "names": [action_names]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (DATASET / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    write_jsonl(DATASET / "meta/episodes.jsonl", episode_rows)
    write_jsonl(DATASET / "meta/episodes_stats.jsonl", stats_rows)
    write_jsonl(DATASET / "meta/tasks.jsonl", [{"task_index": 0, "task": task_text}])
    report = {"authenticity": "synthetic_mujoco_expert_rollouts_converted_to_lerobot", "dataset": str(DATASET), "episodes": len(episode_rows), "frames": int(len(observations)), "observation_dim": int(observations.shape[1]), "action_dim": int(actions.shape[1]), "images": False, "note": "State-only dataset for oracle-state behavior cloning; not suitable for visual VLA training."}
    (RUN / "dataset_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
