#!/usr/bin/env python3
"""Create a lightweight checkpoint view compatible with the active LeRobot checkout."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import fields
from pathlib import Path

from lerobot.policies.smolvla import SmolVLAConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config not found: {source / 'config.json'}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    modified_names = {"config.json", "policy_preprocessor.json"}
    for child in source.iterdir():
        if child.name not in modified_names:
            os.symlink(child, output / child.name, target_is_directory=child.is_dir())

    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    allowed_fields = {field.name for field in fields(SmolVLAConfig)} | {"type"}
    removed_fields = sorted(set(config) - allowed_fields)
    for name in removed_fields:
        config.pop(name)
    # Policy-specific ``from_pretrained`` does not need the registry discriminator,
    # but the generic LeRobot training CLI parses through PreTrainedConfig and does.
    config["type"] = "smolvla"
    config["pretrained_path"] = str(source)
    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    preprocessor_path = source / "policy_preprocessor.json"
    tokenizer_override = None
    tokenizer_source = None
    if preprocessor_path.is_file():
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        bundled_tokenizer = source / "tokenizer"
        if bundled_tokenizer.is_dir():
            # The tokenizer directory is already symlinked into the view. Keep the
            # declared artifact relative so LeRobot can validate and resolve it.
            tokenizer_override = "tokenizer"
            tokenizer_source = str(bundled_tokenizer.resolve())
        else:
            for step in preprocessor.get("steps", []):
                if step.get("registry_name") != "tokenizer_processor":
                    continue
                saved_tokenizer = Path(step.get("config", {}).get("tokenizer_name", ""))
                if saved_tokenizer.is_dir():
                    tokenizer_override = str(saved_tokenizer.resolve())
                    tokenizer_source = tokenizer_override
                    break
        if tokenizer_override is None:
            vlm_assets = Path(config.get("vlm_model_name", ""))
            if vlm_assets.is_dir():
                tokenizer_override = str(vlm_assets.resolve())
                tokenizer_source = tokenizer_override
        if tokenizer_override is None:
            raise FileNotFoundError(f"no local tokenizer found for checkpoint: {source}")
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                step.setdefault("config", {})["tokenizer_name"] = tokenizer_override
                if bundled_tokenizer.is_dir():
                    step.setdefault("artifacts", {})["tokenizer_name"] = "tokenizer"
                elif "artifacts" in step:
                    # An external absolute tokenizer path is valid as processor
                    # configuration but cannot be declared as a checkpoint artifact.
                    step["artifacts"].pop("tokenizer_name", None)
                    if not step["artifacts"]:
                        step.pop("artifacts")
        (output / "policy_preprocessor.json").write_text(
            json.dumps(preprocessor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    report = {
        "schema": "autopolicy.lerobot_checkpoint_compatibility_view/v1",
        "source_checkpoint": str(source),
        "view": str(output),
        "removed_config_fields": removed_fields,
        "pretrained_path_rewritten": str(source),
        "tokenizer_override": tokenizer_override,
        "tokenizer_source": tokenizer_source,
        "weights_are_symlinked": True,
        "source_checkpoint_modified": False,
    }
    (output / "compatibility_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
