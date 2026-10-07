#!/usr/bin/env python3
"""Create an auditable checkpoint view with a scaled LoRA residual.

For a LoRA layer, the residual is proportional to ``B @ A``.  Scaling every
``lora_B`` tensor by ``factor`` therefore linearly interpolates between the
base policy (factor 0) and the trained adapter (factor 1) without changing the
base checkpoint or PEFT configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1].resolve()


def under_root(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT or ROOT in resolved.parents


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=float)
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not under_root(source) or not under_root(output):
        raise ValueError("input and output must be under {ROOT}")
    if not 0.0 <= args.factor <= 1.0:
        raise ValueError("factor must be in [0, 1]")
    weights_path = source / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if output.exists():
        raise FileExistsError(output)

    output.mkdir(parents=True)
    for child in source.iterdir():
        if child.name != "adapter_model.safetensors":
            os.symlink(child.resolve(), output / child.name, target_is_directory=child.is_dir())

    tensors = load_file(weights_path, device="cpu")
    scaled: dict[str, torch.Tensor] = {}
    scaled_keys: list[str] = []
    for name, tensor in tensors.items():
        if ".lora_B." in name:
            scaled[name] = tensor * args.factor
            scaled_keys.append(name)
        else:
            scaled[name] = tensor
    if not scaled_keys:
        raise ValueError("adapter contains no lora_B tensors")
    save_file(scaled, output / "adapter_model.safetensors")

    manifest = {
        "schema": "autopolicy.scaled_lora_adapter/v1",
        "source": str(source),
        "output": str(output),
        "factor": args.factor,
        "method": "multiply every lora_B tensor; leave lora_A unchanged",
        "tensor_count": len(tensors),
        "scaled_tensor_count": len(scaled_keys),
        "non_weight_checkpoint_files_are_symlinked": True,
        "source_modified": False,
    }
    (output / "scaled_adapter_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
