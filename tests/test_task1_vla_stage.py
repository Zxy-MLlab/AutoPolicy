import pytest
import json
from pathlib import Path

from scripts import task1_vla_pipeline_stage as stage
from scripts.task1_vla_pipeline_stage import paired_outcomes, resolve_cuda_device


def report(successes, positions=None):
    positions = positions or [[0.45, 0.15], [0.49, 0.17]]
    return {
        "episode_results": [
            {
                "episode": index,
                "success": success,
                "bottle_xy_ground_truth_for_scoring_only": positions[index],
                "perturbations": {"seeded": index},
            }
            for index, success in enumerate(successes)
        ]
    }


def test_paired_outcomes_requires_and_counts_exact_pairing() -> None:
    result = paired_outcomes(report([True, False]), report([False, True]))
    assert result == {
        "both_success": 0,
        "both_failure": 0,
        "baseline_only_success": 1,
        "candidate_only_success": 1,
    }


def test_paired_outcomes_rejects_position_mismatch() -> None:
    with pytest.raises(ValueError, match="bottle positions differ"):
        paired_outcomes(
            report([True, False]),
            report([True, False], positions=[[0.45, 0.15], [0.50, 0.17]]),
        )


def test_explicit_cuda_device_is_preserved() -> None:
    assert resolve_cuda_device("6") == "6"


def test_archived_paths_relocate_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage, "ROOT", tmp_path)
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps({"old": "/data/zxy/autopolicy/data/a", "current": str(tmp_path / "data/a")}),
        encoding="utf-8",
    )
    report = stage.read_json(source)
    assert report["old"] == str(tmp_path / "data/a")
    assert report["current"] == str(tmp_path / "data/a")
