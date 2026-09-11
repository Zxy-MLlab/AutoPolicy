import json
from pathlib import Path

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
