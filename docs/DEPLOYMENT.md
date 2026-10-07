# Deploy AutoPolicy on another Linux/NVIDIA server

This checkout can be placed anywhere on the target server. Runtime data, Python environments, downloaded
source, models, and caches stay under that checkout. The documented task1 deployment is a MuJoCo simulation
and SmolVLA research workflow; no real-robot adapter is provided.

## 1. Host prerequisites

- Linux x86_64, Python 3.12 with `venv`, Git, curl, and an NVIDIA GPU/driver that can run the CUDA 12.8
  PyTorch wheels pinned by the LeRobot `uv.lock`.
- Network access to GitHub, PyPI, PyTorch's CUDA wheel index, and Hugging Face.
- Disk space for CUDA dependencies and the downloaded task1 release asset. The release archive is about
  840 MB and expands to about 1.25 GB. Keep additional space for environments and generated runs.
- A working `nvidia-smi`; check GPU memory before starting training.

## 2. Clone and install software

```bash
git clone https://github.com/Zxy-MLlab/AutoPolicy.git
cd AutoPolicy
scripts/bootstrap.sh
```

`bootstrap.sh` creates `envs/orchestrator`, downloads the exact upstream commits in `UPSTREAMS.json` into
`vendor/`, installs LeRobot with its locked training, SmolVLA, and FastWAM extras into `envs/lerobot`,
downloads the public SmolVLM2 processor files, and renders the task1 MJCF for this checkout path. It is
idempotent and refuses to alter an existing upstream snapshot at a different commit. On a server without
the training dependencies, `scripts/bootstrap.sh --core-only` installs and tests the lightweight runner.

The public SmolVLM2 processor is pinned to revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` from
[`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).
The trained task1 checkpoint is a separate project-owned artifact.

## 3. Install the project-owned task1 assets

```bash
scripts/autopolicy.sh-env envs/orchestrator/bin/python scripts/install_task1_assets.py
scripts/autopolicy.sh-env envs/lerobot/bin/python scripts/check_task1_install.py
```

The installer downloads `task1-assets-v1.tar.gz` from this repository's `task1-assets-v1` GitHub Release,
checks its byte size and SHA-256 against `assets/task1-assets-v1.json`, safely extracts only the listed files,
and adjusts archived JSON paths to the new checkout. The release contains:

- The task1 source and normalized videos.
- The 99-episode success-filtered LeRobot expert dataset and its collection record.
- The v5 SmolVLA inference checkpoint at step 2000, without optimizer states or earlier checkpoints.
- The task1 modeling and paired simulation evaluation records needed for provenance checks.

Other experimental datasets, checkpoints, caches, environments, and public upstream assets are excluded.
The release is public, as is this repository.
The optional 50-task RoboTwin catalog audit additionally needs `data/robotwin_clean_50`; that separate
38 GB workspace dataset is not part of the task1 release.

## 4. Verify and run task1

Revalidate the archived result with all eight pipeline stages:

```bash
scripts/autopolicy.sh run --config configs/task1_vla_verified_artifacts.json
```

Run the new two-episode collection, two-step SmolVLA training, and two-scenario MuJoCo execution smoke test:

```bash
scripts/autopolicy.sh run --config configs/task1_vla_execute_smoke.json
```

This test proves the execution path. Its one-episode scenario results are not a policy-quality estimate.
The feedback cycle can be exercised with `configs/task1_vla_failure_cycle_smoke.json`. Inspect each run's
`pipeline_state.json` and stage manifests under `runs/`; a completed run can still have `target_met: false`.

For a longer training job using the bundled expert dataset:

```bash
scripts/autopolicy.sh-env envs/lerobot/bin/python scripts/train_lerobot_policies.py \
  --lerobot-root "$PWD/vendor/lerobot" \
  --dataset-root "$PWD/data/task1_smolvla_v5_homereset_success99" \
  --output-dir "$PWD/checkpoints/new-training" \
  --profiles smolvla --steps 100000
```

Choose batch size, steps, and evaluation protocol for the target GPU. The FastWAM launcher is installed but
has no task1 policy-quality claim in this repository. Real-robot deployment remains disabled because the
exact robot mapping, measured camera calibration, safety limits, emergency stop, and hardware interface are
not supplied.

## Release maintenance

On the original server, `scripts/package_task1_assets.py` builds the release archive and manifest from an
explicit allowlist. It omits caches and all unselected experiment assets. After reviewing the generated
`assets/task1-assets-v1.json`, upload the archive as the matching GitHub Release asset. New asset contents
need a new release tag and manifest version; replacing an existing release asset would invalidate SHA-256
verification for previous clones.
