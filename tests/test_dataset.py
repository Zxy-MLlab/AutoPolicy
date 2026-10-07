import json
from pathlib import Path

import pytest

from autopolicy.dataset import audit_lerobot_dataset


def test_dataset_audit_detects_consistent_fixture(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test",
        "total_episodes": 1,
        "total_frames": 2,
        "fps": 10,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n', encoding="utf-8")
    report = audit_lerobot_dataset(root, verify_files=False)
    assert report["valid"]
    assert report["total_frames"] == 2


def test_dataset_audit_detects_bad_counts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    info = {
        "total_episodes": 2,
        "total_frames": 99,
        "features": {"observation.state": {}, "action": {}},
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/episodes.jsonl").write_text('{"episode_index": 0, "length": 2}\n', encoding="utf-8")
    report = audit_lerobot_dataset(root, verify_files=False)
    assert not report["valid"]
    assert len(report["errors"]) == 2


def test_dataset_audit_supports_v3_parquet_episode_index(tmp_path: Path) -> None:
    parquet = pytest.importorskip("pyarrow.parquet")
    pa = __import__("pyarrow")
    root = tmp_path / "dataset"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "videos/observation.images.camera_00/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "test",
        "total_episodes": 1,
        "total_frames": 2,
        "fps": 10,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.images.camera_00": {"dtype": "video", "shape": [2, 2, 3]},
            "observation.state": {"dtype": "float32", "shape": [2]},
            "action": {"dtype": "float32", "shape": [2]},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    index = pa.table(
        {
            "episode_index": [0],
            "length": [2],
            "data/chunk_index": [0],
            "data/file_index": [0],
            "videos/observation.images.camera_00/chunk_index": [0],
            "videos/observation.images.camera_00/file_index": [0],
        }
    )
    parquet.write_table(index, root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "data/chunk-000/file-000.parquet").touch()
    (root / "videos/observation.images.camera_00/chunk-000/file-000.mp4").touch()

    report = audit_lerobot_dataset(root, verify_files=True)

    assert report["valid"]
    assert report["datasets"][0]["episode_index_format"] == "parquet"
    assert report["total_frames"] == 2
