import json
from pathlib import Path

import pytest

from autopolicy.config import ConfigError, load_config, storage_environment


def test_rejects_storage_outside_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTOPOLICY_ROOT", raising=False)
    config = tmp_path / "bad.json"
    config.write_text(
        json.dumps({"root": str(tmp_path / "project"), "paths": {"datasets": str(tmp_path / "outside")}}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be under"):
        load_config(config)


def test_storage_environment_stays_in_workspace() -> None:
    config = load_config("configs/smoke.json")
    environment = storage_environment(config)
    assert environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["HOME"] == str(config.root / "cache/home")
    assert all(
        value.startswith("/data/zxy/autopolicy")
        for key, value in environment.items()
        if key not in {"PYTHONNOUSERSITE", "MUJOCO_GL"}
    )


def test_config_relocates_to_another_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relocated = tmp_path / "another-checkout"
    monkeypatch.setenv("AUTOPOLICY_ROOT", str(relocated))
    config = load_config("configs/task1_vla_execute_smoke.json")
    assert config.root == relocated
    assert config.stage("reconstruct").command[0] == str(relocated / "envs/lerobot/bin/python")
    assert config.stage("deploy").options["scene"] == str(
        relocated / "data/real/task1/model/task1_yam_bottle.xml"
    )


def test_real_task1_config_uses_only_non_mock_stages() -> None:
    config = load_config("configs/task1_vla_verified_artifacts.json")
    assert all(stage.backend != "mock" for stage in config.stages.values())
    assert config.stage("deploy").backend == "preparation"


def test_failure_cycle_config_uses_real_stage_adapters() -> None:
    config = load_config("configs/task1_vla_failure_cycle_smoke.json")
    assert config.stage("generate").backend == "external"
    assert "correction-run" in config.stage("generate").command
    assert "merge-run" in config.stage("dataset").command
    assert "paired-evaluate-run" in config.stage("evaluate").command
    assert all(stage.backend != "mock" for stage in config.stages.values())


def test_continuous_failure_loop_has_two_real_iterations() -> None:
    config = load_config("configs/task1_vla_continuous_failure_loop_smoke.json")
    assert config.max_iterations == 2
    assert "cycle-correction-run" in config.stage("generate").command
    assert "cycle-merge-run" in config.stage("dataset").command
    assert any("{iteration}" in argument for argument in config.stage("train").command)
