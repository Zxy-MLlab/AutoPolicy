from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

from .errors import StageError


def expand_command(command: Sequence[str], variables: Mapping[str, object]) -> list[str]:
    expanded: list[str] = []
    safe = {key: str(value) for key, value in variables.items()}
    for argument in command:
        try:
            expanded.append(argument.format_map(safe))
        except KeyError as exc:
            raise StageError(f"unknown command placeholder {exc.args[0]!r}") from exc
    return expanded


def run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    variables: Mapping[str, object],
) -> list[str]:
    if not command:
        raise StageError("external backend requires a non-empty command")
    argv = expand_command(command, variables)
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            argv,
            cwd=cwd,
            env={**os.environ, **environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode != 0:
        raise StageError(f"command failed with exit code {process.returncode}; see {log_path}")
    return argv
