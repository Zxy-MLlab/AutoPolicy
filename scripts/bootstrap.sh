#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:-full}"
if [[ "$MODE" != "full" && "$MODE" != "--core-only" ]]; then
  echo "usage: scripts/bootstrap.sh [--core-only]" >&2
  exit 2
fi

export AUTOPOLICY_ROOT="$ROOT"
export HOME="$ROOT/cache/home"
export TMPDIR="$ROOT/tmp"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$ROOT/cache/torch"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$ROOT/envs/lerobot"
export PYTHONPYCACHEPREFIX="$ROOT/cache/pycache"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/src:$ROOT/scripts:$ROOT/vendor/lerobot/src"
mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$ROOT/envs" "$ROOT/vendor"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v python3.12 >/dev/null || { echo "Python 3.12 is required for pinned LeRobot" >&2; exit 1; }
if [[ ! -x "$ROOT/envs/orchestrator/bin/python" ]]; then
  python3.12 -m venv "$ROOT/envs/orchestrator"
fi
"$ROOT/envs/orchestrator/bin/python" -m pip install --disable-pip-version-check -e "$ROOT[dev]"

if [[ "$MODE" == "--core-only" ]]; then
  "$ROOT/envs/orchestrator/bin/python" -m pytest -q "$ROOT/tests"
  exit 0
fi

command -v nvidia-smi >/dev/null || { echo "nvidia-smi is required for the GPU setup" >&2; exit 1; }
"$ROOT/envs/orchestrator/bin/python" -m pip install --disable-pip-version-check 'uv==0.11.7'

"$ROOT/envs/orchestrator/bin/python" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ['AUTOPOLICY_ROOT'])
upstreams = json.loads((root / 'UPSTREAMS.json').read_text(encoding='utf-8'))
for name, spec in upstreams.items():
    destination = root / spec['path']
    commit = spec['commit']
    if not (destination / '.git').is_dir():
        if destination.exists():
            raise SystemExit(f'{destination} exists but is not a Git checkout')
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', spec['url'], str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'fetch', 'origin', commit], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise SystemExit(f'{name} is at {actual}, expected {commit}; refusing to change an existing checkout')
    print(f'{name}: {actual}')
PY

"$ROOT/envs/orchestrator/bin/uv" sync \
  --project "$ROOT/vendor/lerobot" --locked \
  --extra training --extra smolvla --extra fastwam

"$ROOT/envs/lerobot/bin/python" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ['AUTOPOLICY_ROOT'])
snapshot_download(
    repo_id='HuggingFaceTB/SmolVLM2-500M-Video-Instruct',
    local_dir=root / 'models/smolvlm2_500m_video_instruct_assets',
    allow_patterns=['*.json', '*.txt'],
)
PY

"$ROOT/envs/orchestrator/bin/python" "$ROOT/scripts/prepare_task1_scene.py"
"$ROOT/scripts/autopolicy.sh" doctor --config "$ROOT/configs/smoke.json"
echo "Base software is ready. Run scripts/install_task1_assets.py to fetch the project-owned task1 assets."
