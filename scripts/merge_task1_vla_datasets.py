#!/usr/bin/env python3
"""Merge nominal and policy-roll-in correction LeRobot datasets with provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, default=None)
    parser.add_argument(
        "--augmentation-roots",
        type=Path,
        nargs="+",
        default=None,
        help="One or more augmentation datasets; mutually exclusive with --correction-root.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--base-role",
        default="nominal_success_filtered",
        help="Provenance label for the base dataset.",
    )
    parser.add_argument(
        "--augmentation-role",
        default="policy_rollin_expert_corrections",
        help="Provenance label for the second dataset.",
    )
    args = parser.parse_args()

    if (args.correction_root is None) == (args.augmentation_roots is None):
        raise ValueError("provide exactly one of --correction-root or --augmentation-roots")
    augmentation_roots = (
        [args.correction_root] if args.correction_root is not None else args.augmentation_roots
    )

    data_root = ROOT.resolve()
    output_root = args.output_root.resolve()
    if output_root == data_root or data_root not in output_root.parents:
        raise ValueError(f"output-root must be a child of {data_root}: {output_root}")
    source_roots = {args.base_root.resolve(), *(path.resolve() for path in augmentation_roots)}
    if output_root in source_roots:
        raise ValueError("output-root must differ from every source dataset")
    if output_root.exists():
        raise SystemExit(f"output dataset already exists: {output_root}")

    base = LeRobotDataset(
        repo_id=f"autopolicy/{args.base_root.name}",
        root=args.base_root,
        video_backend="pyav",
    )
    augmentations = [
        LeRobotDataset(
            repo_id=f"autopolicy/{augmentation_root.name}",
            root=augmentation_root,
            video_backend="pyav",
        )
        for augmentation_root in augmentation_roots
    ]
    merged = merge_datasets(
        [base, *augmentations],
        output_repo_id=args.repo_id,
        output_dir=output_root,
        concatenate_videos=True,
        concatenate_data=True,
    )
    manifest = {
        "schema": "autopolicy.task1_vla_dataset_merge/v1",
        "method": "lerobot.datasets.dataset_tools.merge_datasets",
        "sources": [
            {
                "role": args.base_role,
                "root": str(args.base_root.resolve()),
                "episodes": base.num_episodes,
                "frames": len(base),
            },
            *[
                {
                    "role": args.augmentation_role,
                    "root": str(augmentation_root.resolve()),
                    "episodes": augmentation.num_episodes,
                    "frames": len(augmentation),
                }
                for augmentation_root, augmentation in zip(
                    augmentation_roots, augmentations, strict=True
                )
            ],
        ],
        "output": str(output_root),
        "repo_id": args.repo_id,
        "episodes": merged.num_episodes,
        "frames": len(merged),
        "augmentation_episode_index_range": [base.num_episodes, merged.num_episodes],
        "augmentation_frame_fraction": sum(len(dataset) for dataset in augmentations) / len(merged),
    }
    if args.augmentation_role == "policy_rollin_expert_corrections":
        # Backward-compatible aliases consumed by prepare_task1_rollin_weights.py.
        manifest["correction_episode_index_range"] = manifest[
            "augmentation_episode_index_range"
        ]
        manifest["correction_frame_fraction"] = manifest["augmentation_frame_fraction"]
    (output_root / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
