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
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
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
import shutil
import tarfile
from pathlib import Path
from urllib.parse import urlparse

root = Path(os.environ['AUTOPOLICY_ROOT'])
upstreams = json.loads((root / 'UPSTREAMS.json').read_text(encoding='utf-8'))
for name, spec in upstreams.items():
    destination = root / spec['path']
    commit = spec['commit']
    if (destination / '.git').is_dir():
        actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    elif (destination / '.upstream-commit').is_file():
        actual = (destination / '.upstream-commit').read_text(encoding='utf-8').strip()
    else:
        if destination.exists():
            raise SystemExit(f'{destination} exists without a recorded upstream commit')
        parsed = urlparse(spec['url'])
        if parsed.hostname != 'github.com':
            raise SystemExit(f'unsupported upstream source: {spec["url"]}')
        repository = parsed.path.strip('/').removesuffix('.git')
        url = f'https://codeload.github.com/{repository}/tar.gz/{commit}'
        archive = root / 'cache/upstreams' / f'{name}-{commit}.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            partial = archive.with_name(archive.name + '.part')
            subprocess.run(['curl', '--silent', '--show-error', '--fail', '--location', '--retry', '5', '--continue-at', '-', '--output', str(partial), url], check=True)
            with tarfile.open(partial, 'r:gz') as downloaded:
                downloaded.getmembers()
            partial.replace(archive)
        staging = root / 'vendor' / f'.{name}-unpack'
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        with tarfile.open(archive, 'r:gz') as package:
            package.extractall(staging, filter='data')
        entries = list(staging.iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise SystemExit(f'unexpected upstream archive layout: {archive}')
        entries[0].rename(destination)
        staging.rmdir()
        (destination / '.upstream-commit').write_text(commit + '\n', encoding='utf-8')
        actual = commit
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
    revision='7b375e1b73b11138ff12fe22c8f2822d8fe03467',
    local_dir=root / 'models/smolvlm2_500m_video_instruct_assets',
    allow_patterns=['*.json', '*.txt'],
)
PY

"$ROOT/envs/orchestrator/bin/python" "$ROOT/scripts/prepare_task1_scene.py"
"$ROOT/scripts/autopolicy.sh" doctor --config "$ROOT/configs/smoke.json"
echo "Base software is ready. Run scripts/install_task1_assets.py to fetch the project-owned task1 assets."
