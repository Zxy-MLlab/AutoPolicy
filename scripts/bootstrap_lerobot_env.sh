#!/usr/bin/env bash
set -euo pipefail

AUTOPOLICY_ROOT=/data/zxy/autopolicy
export HOME="$AUTOPOLICY_ROOT"
export TMPDIR="$AUTOPOLICY_ROOT/tmp"
export XDG_CACHE_HOME="$AUTOPOLICY_ROOT/cache/xdg"
export HF_HOME="$AUTOPOLICY_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$AUTOPOLICY_ROOT/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$AUTOPOLICY_ROOT/cache/huggingface/transformers"
export TORCH_HOME="$AUTOPOLICY_ROOT/cache/torch"
export PIP_CACHE_DIR="$AUTOPOLICY_ROOT/cache/pip"
export UV_CACHE_DIR="$AUTOPOLICY_ROOT/cache/uv"
export UV_PROJECT_ENVIRONMENT="$AUTOPOLICY_ROOT/envs/lerobot"
export CUDA_CACHE_PATH="$AUTOPOLICY_ROOT/cache/nvidia"
export TRITON_CACHE_DIR="$AUTOPOLICY_ROOT/cache/triton"
export PYTORCH_KERNEL_CACHE_PATH="$AUTOPOLICY_ROOT/cache/torch/kernels"
export PYTHONPYCACHEPREFIX="$AUTOPOLICY_ROOT/cache/pycache"
export PYTHONNOUSERSITE=1

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$PYTORCH_KERNEL_CACHE_PATH" "$PYTHONPYCACHEPREFIX"
exec /home/zhouxueyang/.local/bin/uv sync \
  --project "$AUTOPOLICY_ROOT/vendor/lerobot" \
  --locked \
  --extra training \
  --extra smolvla \
  --extra fastwam
