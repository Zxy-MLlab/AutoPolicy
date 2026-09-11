import json
from pathlib import Path

import pytest

from autopolicy.config import ConfigError, load_config, storage_environment


def test_rejects_storage_outside_data_zxy(tmp_path: Path) -> None:
    config = tmp_path / "bad.json"
    config.write_text(json.dumps({"root": "/tmp/autopolicy"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be under /data/zxy"):
        load_config(config)


def test_storage_environment_stays_in_workspace() -> None:
    config = load_config("configs/smoke.json")
    environment = storage_environment(config)
    assert environment
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert all(
        value.startswith("/data/zxy/autopolicy")
        for key, value in environment.items()
        if key != "PYTHONNOUSERSITE"
    )
