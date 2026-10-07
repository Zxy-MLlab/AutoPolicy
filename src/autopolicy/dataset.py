from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import StageError
from .io import read_json, read_jsonl


REQUIRED_FEATURES = {"observation.state", "action"}


def _read_episode_index(dataset_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Read either the LeRobot v2 JSONL or v3 sharded Parquet episode index."""
    jsonl_path = dataset_root / "meta/episodes.jsonl"
    if jsonl_path.is_file():
        return read_jsonl(jsonl_path), "jsonl"

    parquet_paths = sorted((dataset_root / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(
            f"no episode index found at {jsonl_path} or meta/episodes/chunk-*/file-*.parquet"
        )
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - depends on the runtime environment
        raise RuntimeError(
            "auditing a LeRobot v3 dataset requires pyarrow to read its episode index"
        ) from exc

    rows: list[dict[str, Any]] = []
    for path in parquet_paths:
        rows.extend(parquet.read_table(path).to_pylist())
    return rows, "parquet"


def _format_episode_asset_path(
    pattern: str,
    row: dict[str, Any],
    kind: str,
    video_key: str | None = None,
    chunk_size: int = 1000,
) -> str:
    """Resolve asset paths for both v2 episode-per-file and v3 sharded layouts."""
    index = int(row["episode_index"])
    values: dict[str, Any] = {
        "episode_index": index,
        "episode_chunk": index // chunk_size,
    }
    prefix = kind if video_key is None else f"videos/{video_key}"
    if f"{prefix}/chunk_index" in row:
        values["chunk_index"] = int(row[f"{prefix}/chunk_index"])
    if f"{prefix}/file_index" in row:
        values["file_index"] = int(row[f"{prefix}/file_index"])
    return pattern.format(video_key=video_key, **values)


def audit_lerobot_dataset(root: Path, verify_files: bool = True) -> dict[str, Any]:
    if not root.is_dir():
        raise StageError(f"dataset root does not exist: {root}")
    candidates = [root] if (root / "meta/info.json").is_file() else sorted(
        path for path in root.iterdir() if (path / "meta/info.json").is_file()
    )
    if not candidates:
        raise StageError(f"no LeRobot dataset directories found under {root}")

    errors: list[str] = []
    datasets: list[dict[str, Any]] = []
    total_episodes = 0
    total_frames = 0
    for dataset_root in candidates:
        try:
            info = read_json(dataset_root / "meta/info.json")
            episodes, episode_index_format = _read_episode_index(dataset_root)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{dataset_root.name}: {exc}")
            continue
        features = set(info.get("features", {}))
        missing = REQUIRED_FEATURES - features
        if missing:
            errors.append(f"{dataset_root.name}: missing features {sorted(missing)}")
        declared_episodes = int(info.get("total_episodes", -1))
        declared_frames = int(info.get("total_frames", -1))
        actual_frames = sum(int(row.get("length", 0)) for row in episodes)
        if declared_episodes != len(episodes):
            errors.append(
                f"{dataset_root.name}: declared {declared_episodes} episodes, found {len(episodes)}"
            )
        if declared_frames != actual_frames:
            errors.append(
                f"{dataset_root.name}: declared {declared_frames} frames, indexed {actual_frames}"
            )
        missing_data = 0
        missing_videos = 0
        if verify_files:
            data_pattern = str(info.get("data_path", ""))
            video_pattern = str(info.get("video_path", ""))
            video_keys = [
                key
                for key, value in info.get("features", {}).items()
                if value.get("dtype") == "video"
            ]
            chunk_size = int(info.get("chunks_size", 1000))
            referenced_data: set[Path] = set()
            referenced_videos: set[Path] = set()
            for row in episodes:
                if data_pattern:
                    referenced_data.add(
                        dataset_root
                        / _format_episode_asset_path(
                            data_pattern, row, "data", chunk_size=chunk_size
                        )
                    )
                for video_key in video_keys:
                    if video_pattern:
                        referenced_videos.add(
                            dataset_root
                            / _format_episode_asset_path(
                                video_pattern,
                                row,
                                "video",
                                video_key=video_key,
                                chunk_size=chunk_size,
                            )
                        )
            missing_data = sum(not path.is_file() for path in referenced_data)
            missing_videos = sum(not path.is_file() for path in referenced_videos)
        if missing_data:
            errors.append(f"{dataset_root.name}: {missing_data} missing parquet files")
        if missing_videos:
            errors.append(f"{dataset_root.name}: {missing_videos} missing video files")
        total_episodes += max(declared_episodes, 0)
        total_frames += max(declared_frames, 0)
        datasets.append(
            {
                "name": dataset_root.name,
                "root": str(dataset_root),
                "codebase_version": info.get("codebase_version"),
                "robot_type": info.get("robot_type"),
                "fps": info.get("fps"),
                "episode_index_format": episode_index_format,
                "episodes": declared_episodes,
                "frames": declared_frames,
                "features": sorted(features),
                "missing_data_files": missing_data,
                "missing_video_files": missing_videos,
            }
        )
    return {
        "format": "LeRobotDataset",
        "roots": len(datasets),
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "valid": not errors and len(datasets) == len(candidates),
        "errors": errors,
        "datasets": datasets,
    }
