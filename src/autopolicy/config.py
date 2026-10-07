from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STAGE_NAMES = (
    "reconstruct",
    "validate",
    "generate",
    "dataset",
    "train",
    "evaluate",
    "deploy",
    "improve",
)
BACKENDS = {"mock", "artifact", "external", "catalog", "preparation", "disabled"}


@dataclass(frozen=True)
class StageConfig:
    backend: str
    command: tuple[str, ...] = ()
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    source: Path
    name: str
    root: Path
    seed: int
    max_iterations: int
    success_target: float
    stages: dict[str, StageConfig]
    paths: dict[str, Path]
    raw: dict[str, Any]

    def stage(self, name: str) -> StageConfig:
        return self.stages[name]


def _resolve_path(value: str, config_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _assert_storage_path(path: Path, label: str, root: Path) -> None:
    if path != root and root not in path.parents:
        raise ConfigError(f"{label} must be under {root}, got {path}")


def _expand_root(value: Any, root: Path) -> Any:
    """Resolve repository-relative config templates while preserving stage placeholders."""
    if isinstance(value, str):
        return value.replace("{root}", str(root))
    if isinstance(value, list):
        return [_expand_root(item, root) for item in value]
    if isinstance(value, dict):
        return {key: _expand_root(item, root) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> PipelineConfig:
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load config {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("pipeline config must be a JSON object")

    root_setting = os.environ.get("AUTOPOLICY_ROOT", raw.get("root", str(WORKSPACE_ROOT)))
    root = _resolve_path(str(root_setting), source.parent)
    raw = _expand_root(raw, root)
    raw["root"] = str(root)
    name = str(raw.get("name", "autopolicy"))
    seed = int(raw.get("seed", 7))
    max_iterations = int(raw.get("max_iterations", 1))
    success_target = float(raw.get("success_target", 0.8))
    if max_iterations < 1:
        raise ConfigError("max_iterations must be at least 1")
    if not 0.0 <= success_target <= 1.0:
        raise ConfigError("success_target must be between 0 and 1")

    raw_paths = raw.get("paths", {})
    if not isinstance(raw_paths, dict):
        raise ConfigError("paths must be an object")
    defaults = {
        "gpt6_real2sim": root / "vendor/gpt6-real2sim",
        "robotwin": root / "vendor/robotwin",
        "lerobot": root / "vendor/lerobot",
        "runs": root / "runs",
        "datasets": root / "data",
        "checkpoints": root / "checkpoints",
        "models": root / "models",
        "cache": root / "cache",
        "tmp": root / "tmp",
    }
    paths: dict[str, Path] = {}
    for key, default in defaults.items():
        paths[key] = _resolve_path(str(raw_paths.get(key, default)), source.parent)
        _assert_storage_path(paths[key], f"paths.{key}", root)

    raw_stages = raw.get("stages", {})
    if not isinstance(raw_stages, dict):
        raise ConfigError("stages must be an object")
    unknown = set(raw_stages) - set(STAGE_NAMES)
    if unknown:
        raise ConfigError(f"unknown stages: {', '.join(sorted(unknown))}")
    stages: dict[str, StageConfig] = {}
    for stage_name in STAGE_NAMES:
        value = raw_stages.get(stage_name, {"backend": "mock"})
        if not isinstance(value, dict):
            raise ConfigError(f"stages.{stage_name} must be an object")
        backend = str(value.get("backend", "mock"))
        if backend not in BACKENDS:
            raise ConfigError(f"unsupported backend for {stage_name}: {backend}")
        command_value = value.get("command", [])
        if not isinstance(command_value, list) or not all(isinstance(item, str) for item in command_value):
            raise ConfigError(f"stages.{stage_name}.command must be a string array")
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ConfigError(f"stages.{stage_name}.options must be an object")
        stages[stage_name] = StageConfig(backend, tuple(command_value), options)

    return PipelineConfig(source, name, root, seed, max_iterations, success_target, stages, paths, raw)


def storage_environment(config: PipelineConfig) -> dict[str, str]:
    cache = config.paths["cache"]
    environment = {
        "HOME": str(cache / "home"),
        "TMPDIR": str(config.paths["tmp"]),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "huggingface"),
        "HF_DATASETS_CACHE": str(cache / "huggingface/datasets"),
        "TRANSFORMERS_CACHE": str(cache / "huggingface/transformers"),
        "TORCH_HOME": str(cache / "torch"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "WANDB_DIR": str(cache / "wandb"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "CUDA_CACHE_PATH": str(cache / "nvidia"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "PYTORCH_KERNEL_CACHE_PATH": str(cache / "torch/kernels"),
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "PYTHONPATH": ":".join(
            str(path)
            for path in (
                config.root,
                config.paths["lerobot"] / "src",
                config.root / "scripts",
                config.root / "src",
            )
        ),
        "PYTHONNOUSERSITE": "1",
        "MUJOCO_GL": "egl",
        "AUTOPOLICY_ROOT": str(config.root),
    }
    directory_keys = {
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "TRANSFORMERS_CACHE",
        "TORCH_HOME",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "WANDB_DIR",
        "MPLCONFIGDIR",
        "CUDA_CACHE_PATH",
        "TRITON_CACHE_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
        "PYTHONPYCACHEPREFIX",
    }
    for key in directory_keys:
        Path(environment[key]).mkdir(parents=True, exist_ok=True)
    return environment
