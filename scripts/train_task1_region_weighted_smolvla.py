#!/usr/bin/env python3
"""Run LeRobot training with auditable task1 per-episode loss weights.

The adapter monkey-patches only LeRobot's sample-weighter factory in memory. It
does not modify the LeRobot checkout, and the upstream training loop, optimizer,
checkpoint writer, and SmolVLA implementation remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from lerobot.utils import sample_weighting


class EpisodeRegionWeighter(sample_weighting.SampleWeighter):
    def __init__(self, weights_path: Path, device: torch.device):
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        weights = {int(key): float(value) for key, value in payload["episode_weights"].items()}
        if not weights or min(weights) < 0 or min(weights.values()) <= 0:
            raise ValueError("episode weight manifest must contain positive weights")
        expected = list(range(max(weights) + 1))
        if sorted(weights) != expected:
            raise ValueError("episode weights must cover contiguous indices starting at zero")
        self.device = device
        self.weights_path = weights_path.resolve()
        self.lookup = torch.tensor([weights[index] for index in expected], device=device)

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        episode_index = batch.get("episode_index")
        if not isinstance(episode_index, torch.Tensor):
            raise KeyError("region weighting requires tensor batch['episode_index']")
        episode_index = episode_index.to(self.device, dtype=torch.long).reshape(-1)
        if episode_index.numel() == 0 or int(episode_index.min()) < 0:
            raise ValueError("invalid episode_index batch")
        if int(episode_index.max()) >= self.lookup.numel():
            raise IndexError("episode_index is not covered by the weight manifest")
        raw = self.lookup[episode_index]
        weights = raw / raw.mean().clamp_min(1e-12)
        return weights, {
            "mean_weight": float(weights.mean()),
            "min_weight": float(weights.min()),
            "max_weight": float(weights.max()),
            "type": "episode_region",
        }

    def get_stats(self) -> dict:
        return {
            "type": "episode_region",
            "weights_path": str(self.weights_path),
            "num_episodes": self.lookup.numel(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--episode-weights", type=Path, required=True)
    adapter_args, lerobot_args = parser.parse_known_args()
    weights_path = adapter_args.episode_weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)

    original_factory = sample_weighting.make_sample_weighter

    def make_sample_weighter(config, policy, device, dataset_root=None, dataset_repo_id=None):
        if config is not None and config.type == "episode_region":
            configured = Path(config.progress_path).resolve() if config.progress_path else weights_path
            if configured != weights_path:
                raise ValueError(
                    f"sample weighting path mismatch: adapter={weights_path}, config={configured}"
                )
            return EpisodeRegionWeighter(weights_path, device)
        return original_factory(config, policy, device, dataset_root, dataset_repo_id)

    sample_weighting.make_sample_weighter = make_sample_weighter
    sys.argv = [
        sys.argv[0],
        *lerobot_args,
        "--sample_weighting.type=episode_region",
        f"--sample_weighting.progress_path={weights_path}",
    ]
    from lerobot.scripts.lerobot_train import train

    train()


if __name__ == "__main__":
    main()
