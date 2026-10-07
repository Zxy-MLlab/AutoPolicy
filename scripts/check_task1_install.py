#!/usr/bin/env python3
"""Check that the portable task1 simulation and policy inputs are usable."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import torch

from autopolicy.dataset import audit_lerobot_dataset


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints/task1_smolvla_v5_homereset_stage2_restartlr_pyav_2000steps_b32/checkpoints/002000/pretrained_model"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    scene = ROOT / "data/real/task1/model/task1_yam_bottle.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    dataset = audit_lerobot_dataset(ROOT / "data/task1_smolvla_v5_homereset_success99")
    metadata = json.loads((ROOT / "data/real/task1/metadata.json").read_text(encoding="utf-8"))
    videos = {
        "source": ROOT / "data/real/task1/VID_20260611_160742_1000810407.mp4",
        "normalized": ROOT / "data/real/task1/videos/camera_00.mp4",
    }
    checks = {
        "nvidia_cuda_available": torch.cuda.is_available(),
        "scene_executable": model.nu == 14 and model.nq >= 16,
        "dataset_valid": dataset["valid"] and dataset["total_episodes"] == 99,
        "checkpoint_complete": all((BASE / name).is_file() for name in (
            "model.safetensors", "config.json", "train_config.json",
            "policy_preprocessor.json", "policy_postprocessor.json",
        )),
        "vlm_processor_available": (ROOT / "models/smolvlm2_500m_video_instruct_assets/tokenizer.json").is_file(),
        "source_video_hash": sha256(videos["source"]) == metadata["source"]["sha256"],
        "normalized_video_hash": sha256(videos["normalized"]) == metadata["normalized"]["sha256"],
        "modeling_report": (ROOT / "runs/task1-real2sim-modeling/report.json").is_file(),
        "robustness_report": (ROOT / "runs/task1-smolvla-v5-robustness-20seed8080/robustness_report.json").is_file(),
    }
    print(json.dumps({"ok": all(checks.values()), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
