from pathlib import Path

from autopolicy.config import load_config
from autopolicy.io import read_json
from autopolicy.pipeline import PipelineRunner, summarize_run


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
