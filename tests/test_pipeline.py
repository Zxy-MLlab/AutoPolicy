import json
from pathlib import Path
import sys

import pytest

from autopolicy.config import load_config
from autopolicy.errors import ConfigError
from autopolicy.io import read_json
from autopolicy.pipeline import PipelineRunner, _git_revision, summarize_run


def test_upstream_archive_records_pinned_revision(tmp_path: Path) -> None:
    source = tmp_path / "upstream"
    source.mkdir()
    (source / ".upstream-commit").write_text("a" * 40 + "\n", encoding="utf-8")
    assert _git_revision(source) == "a" * 40


def test_smoke_pipeline_is_auditable(tmp_path: Path) -> None:
    config = load_config("configs/smoke.json")
    run_dir = tmp_path / "run"
    runner = PipelineRunner(config, run_dir)
    state = runner.run()
    assert state["status"] == "completed"
    assert state["iterations"]
    for iteration in state["iterations"]:
        assert list(iteration["stages"]) == [
            "reconstruct", "validate", "generate", "dataset", "train", "evaluate", "deploy", "improve"
        ]
        assert all(stage["status"] == "completed" for stage in iteration["stages"].values())
    manifest = read_json(run_dir / "iteration-000/validate/manifest.json")
    assert manifest["result"]["passed"]
    assert manifest["artifacts"][0]["sha256"]
    second_generation = read_json(run_dir / "iteration-001/generate/manifest.json")
    assert second_generation["result"]["feedback_requests_consumed"] > 0
    assert second_generation["result"]["attempted"] > 10
    assert summarize_run(run_dir)["status"] == "completed"


def test_completed_run_can_be_resumed(tmp_path: Path) -> None:
    config = load_config("configs/smoke.json")
    run_dir = tmp_path / "run"
    runner = PipelineRunner(config, run_dir)
    first = runner.run()
    second = runner.run(resume=True)
    assert second == first


def test_partial_run_can_continue(tmp_path: Path) -> None:
    config = load_config("configs/smoke.json")
    run_dir = tmp_path / "run"
    runner = PipelineRunner(config, run_dir)
    partial = runner.run(to_stage="validate")
    assert partial["status"] == "completed"
    assert "generate" not in partial["iterations"][0]["stages"]
    completed = runner.run(resume=True, from_stage="generate")
    assert completed["status"] == "completed"
    assert completed["iterations"][0]["stages"]["improve"]["status"] == "completed"


def test_dry_run_resolves_external_commands_without_side_effects(tmp_path: Path) -> None:
    config = load_config("configs/task1_vla_verified_artifacts.json")
    run_dir = tmp_path / "planned-run"

    plan = PipelineRunner(config, run_dir).dry_run(from_stage="train", to_stage="evaluate")

    assert plan["side_effects"] is False
    assert plan["selected_stages"] == ["train", "evaluate"]
    assert not run_dir.exists()
    train = plan["iterations"][0]["stages"][0]
    assert train["backend"] == "external"
    assert str(run_dir / "iteration-000/train") in train["command"]


def test_continuous_loop_dry_run_resolves_cross_iteration_context(tmp_path: Path) -> None:
    config = load_config("configs/task1_vla_continuous_failure_loop_smoke.json")
    run_dir = tmp_path / "continuous"

    plan = PipelineRunner(config, run_dir).dry_run(from_stage="generate", to_stage="dataset")

    assert len(plan["iterations"]) == 2
    second_generate = plan["iterations"][1]["stages"][0]["command"]
    assert "--pipeline-run-dir" in second_generate
    assert str(run_dir) in second_generate
    assert second_generate[second_generate.index("--iteration") + 1] == "1"
    assert not run_dir.exists()


def test_run_records_source_configuration_provenance(tmp_path: Path) -> None:
    config = load_config("configs/smoke.json")
    state = PipelineRunner(config, tmp_path / "run").run(to_stage="reconstruct")

    assert state["configuration"]["path"] == str(config.source)
    assert len(state["configuration"]["sha256"]) == 64


def test_resume_reexecutes_stage_when_recorded_artifact_was_modified(tmp_path: Path) -> None:
    config = load_config("configs/smoke.json")
    run_dir = tmp_path / "run"
    runner = PipelineRunner(config, run_dir)
    runner.run(to_stage="reconstruct")
    scene = run_dir / "iteration-000/reconstruct/scene.json"
    scene.write_text("tampered", encoding="utf-8")

    state = runner.run(resume=True, to_stage="reconstruct")

    assert state["status"] == "completed"
    assert scene.read_text(encoding="utf-8") != "tampered"


def test_resume_rejects_changed_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    raw = json.loads(Path("configs/smoke.json").read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    config = load_config(config_path)
    runner = PipelineRunner(config, tmp_path / "run")
    runner.run(to_stage="reconstruct")
    raw["seed"] += 1
    config_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ConfigError, match="configuration changed"):
        runner.run(resume=True, to_stage="reconstruct")


def test_external_stage_records_command_inputs_and_nested_outputs(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        """import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
(root / 'nested').mkdir(parents=True)
(root / 'nested/payload.txt').write_text('real output')
(root / 'result.json').write_text(json.dumps({'authenticity': 'test_external'}))
""",
        encoding="utf-8",
    )
    raw = json.loads(Path("configs/smoke.json").read_text(encoding="utf-8"))
    raw["max_iterations"] = 1
    raw["stages"]["reconstruct"] = {
        "backend": "external",
        "command": [sys.executable, str(adapter), "{stage_dir}"],
    }
    config_path = tmp_path / "external.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    run_dir = tmp_path / "run"

    PipelineRunner(load_config(config_path), run_dir).run(to_stage="reconstruct")

    manifest = read_json(run_dir / "iteration-000/reconstruct/manifest.json")
    artifact_paths = {record["path"] for record in manifest["artifacts"]}
    assert "iteration-000/reconstruct/nested/payload.txt" in artifact_paths
    assert str(adapter) in {record["path"] for record in manifest["command_inputs"]}
    assert manifest["execution_plan"]["command"][-1].endswith("/reconstruct")
