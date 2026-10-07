#!/usr/bin/env python3
"""Measure SmolVLA action-chunk error on recorded task1 demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from lerobot.common.control_utils import prepare_observation_for_inference
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy


ROOT = Path(__file__).resolve().parents[1]


def make_compatible_checkpoint_view(checkpoint: Path):
    """Create a temporary, symlink-backed view for nearby LeRobot revisions."""
    checkpoint = checkpoint.resolve()
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    fields_to_remove = {"pretrained_revision"}.intersection(config)
    fields_to_add = {"type": "smolvla"} if "type" not in config else {}
    if not fields_to_remove and not fields_to_add:
        return checkpoint, None, [], []

    temp_root = ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    overlay = tempfile.TemporaryDirectory(prefix="smolvla-diagnostic-checkpoint-", dir=temp_root)
    overlay_path = Path(overlay.name)
    for child in checkpoint.iterdir():
        if child.name != "config.json":
            os.symlink(child, overlay_path / child.name, target_is_directory=child.is_dir())
    for field in fields_to_remove:
        config.pop(field)
    config.update(fields_to_add)
    (overlay_path / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return overlay_path, overlay, sorted(fields_to_remove), sorted(fields_to_add)


def resolve_tokenizer_path(checkpoint: Path, policy) -> Path:
    checkpoint = checkpoint.resolve()
    bundled = checkpoint / "tokenizer"
    if bundled.is_dir():
        return bundled
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    if preprocessor_path.is_file():
        preprocessor = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "tokenizer_processor":
                saved = Path(step.get("config", {}).get("tokenizer_name", ""))
                if saved.is_dir():
                    return saved.resolve()
    vlm_assets = Path(policy.config.vlm_model_name)
    if vlm_assets.is_dir():
        return vlm_assets.resolve()
    raise FileNotFoundError(f"no local tokenizer found for checkpoint: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processor-checkpoint",
        type=Path,
        default=None,
        help="Optional source for normalization and tokenizer processors.",
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/task1_smolvla")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frame-offsets",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 40, 50, 60, 70, 80],
        help="Frame offsets within each 90-frame episode.",
    )
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 20, 29])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_view, checkpoint_overlay, ignored_fields, added_fields = (
        make_compatible_checkpoint_view(args.checkpoint)
    )
    try:
        policy = SmolVLAPolicy.from_pretrained(checkpoint_view)
    finally:
        if checkpoint_overlay is not None:
            checkpoint_overlay.cleanup()
    policy.to(device).eval()
    processor_checkpoint = (args.processor_checkpoint or args.checkpoint).resolve()
    tokenizer_path = resolve_tokenizer_path(processor_checkpoint, policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=processor_checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    dataset = LeRobotDataset(
        repo_id="autopolicy/task1_smolvla",
        root=args.dataset,
        delta_timestamps={"action": [i / 10 for i in range(policy.config.chunk_size)]},
        video_backend="pyav",
    )

    rows: list[dict] = []
    all_abs_errors: list[np.ndarray] = []
    predicted_first_actions: list[np.ndarray] = []
    target_first_actions: list[np.ndarray] = []
    for episode in args.episodes:
        for frame_offset in args.frame_offsets:
            index = episode * 90 + frame_offset
            sample = dataset[index]
            image = (
                sample["observation.images.camera_00"].permute(1, 2, 0).mul(255).round().byte().numpy()
            )
            observation = {
                "observation.images.camera_00": image,
                "observation.state": sample["observation.state"].numpy(),
            }
            batch = prepare_observation_for_inference(
                observation,
                device,
                task=sample["task"],
                robot_type="yam_task1",
            )
            batch = preprocessor(batch)
            torch.manual_seed(args.seed + index)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed + index)
            with torch.inference_mode():
                predicted = postprocessor(policy.predict_action_chunk(batch)).squeeze(0).cpu().numpy()

            target = sample["action"].numpy()
            valid = ~sample["action_is_pad"].numpy()
            predicted = predicted[valid]
            target = target[valid]
            abs_error = np.abs(predicted - target)
            all_abs_errors.append(abs_error)
            predicted_first_actions.append(predicted[0])
            target_first_actions.append(target[0])
            left_dynamic = abs_error[:, :7]
            row = {
                "episode": episode,
                "frame_offset": frame_offset,
                "valid_horizon": int(valid.sum()),
                "mae_all_14": float(abs_error.mean()),
                "mae_left_7": float(left_dynamic.mean()),
                "first_action_pred": predicted[0].tolist(),
                "first_action_target": target[0].tolist(),
                "gripper_pred_start_end": [float(predicted[0, 6]), float(predicted[-1, 6])],
                "gripper_target_start_end": [float(target[0, 6]), float(target[-1, 6])],
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    concatenated = np.concatenate(all_abs_errors, axis=0)
    predicted_first = np.stack(predicted_first_actions)
    target_first = np.stack(target_first_actions)
    correlations = []
    for dim in range(predicted_first.shape[1]):
        if predicted_first[:, dim].std() < 1e-12 or target_first[:, dim].std() < 1e-12:
            correlations.append(None)
        else:
            correlations.append(float(np.corrcoef(predicted_first[:, dim], target_first[:, dim])[0, 1]))
    report = {
        "schema": "autopolicy.task1_smolvla_offline_diagnostic/v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "processor_checkpoint": str(processor_checkpoint),
        "processor_override": args.processor_checkpoint is not None,
        "ignored_checkpoint_config_fields": ignored_fields,
        "added_checkpoint_config_fields": added_fields,
        "tokenizer_path": str(tokenizer_path),
        "dataset": str(args.dataset.resolve()),
        "samples": len(rows),
        "mae_all_14": float(concatenated.mean()),
        "mae_left_7": float(concatenated[:, :7].mean()),
        "mae_per_action_dim": concatenated.mean(axis=0).tolist(),
        "predicted_first_action_std_per_dim": predicted_first.std(axis=0).tolist(),
        "target_first_action_std_per_dim": target_first.std(axis=0).tolist(),
        "predicted_target_first_action_correlation_per_dim": correlations,
        "rows": rows,
        "note": "Metrics are on training demonstrations, not held-out or closed-loop success metrics.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("samples", "mae_all_14", "mae_left_7")}, indent=2))


if __name__ == "__main__":
    main()
