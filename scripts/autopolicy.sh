#!/usr/bin/env bash
set -euo pipefail

AUTOPOLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export AUTOPOLICY_ROOT
export HOME="$AUTOPOLICY_ROOT/cache/home"
export TMPDIR="$AUTOPOLICY_ROOT/tmp"
export XDG_CACHE_HOME="$AUTOPOLICY_ROOT/cache/xdg"
export HF_HOME="$AUTOPOLICY_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$AUTOPOLICY_ROOT/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$AUTOPOLICY_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$AUTOPOLICY_ROOT/cache/torch"
export PIP_CACHE_DIR="$AUTOPOLICY_ROOT/cache/pip"
export UV_CACHE_DIR="$AUTOPOLICY_ROOT/cache/uv"
export WANDB_DIR="$AUTOPOLICY_ROOT/cache/wandb"
export MPLCONFIGDIR="$AUTOPOLICY_ROOT/cache/matplotlib"
export CUDA_CACHE_PATH="$AUTOPOLICY_ROOT/cache/nvidia"
export TRITON_CACHE_DIR="$AUTOPOLICY_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$AUTOPOLICY_ROOT/cache/torch/kernels"
export PYTHONPYCACHEPREFIX="$AUTOPOLICY_ROOT/cache/pycache"
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$AUTOPOLICY_ROOT:$AUTOPOLICY_ROOT/vendor/lerobot/src:$AUTOPOLICY_ROOT/scripts:$AUTOPOLICY_ROOT/src"

mkdir -p "$HOME" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$WANDB_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH" "$PYTHONPYCACHEPREFIX"
PYTHON_BIN="$AUTOPOLICY_ROOT/envs/orchestrator/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python3
fi
exec "$PYTHON_BIN" -m autopolicy.cli "$@"
