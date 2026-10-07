# AutoPolicy

AutoPolicy is an auditable orchestration layer for a closed-loop robotics workflow:

```text
real observations -> Real2Sim -> validation gate -> task/expert rollouts
                  -> LeRobotDataset -> VLA/WAM -> simulation evaluation
                  -> targeted new data -> retraining -> approved robot rollout
```

The repository keeps orchestration, upstream source, datasets, caches, environments, models, checkpoints,
and run artifacts below its checkout directory. Every stage writes a manifest with its backend, provenance,
result, timestamps, artifacts, and SHA-256 hashes. The pipeline stops on a failed validation gate or command.

## What is implemented

- Pinned GPT6-real2sim, RoboTwin, and LeRobot revisions, installed into `vendor/` by the bootstrap script.
- A resumable eight-stage state machine: reconstruction, validation, generation, dataset conversion/audit,
  training, evaluation, deployment, and feedback-driven improvement.
- Interchangeable backends: deterministic `mock` contracts, archived/existing `artifact` and `catalog`
  adapters, arbitrary argv-based `external` adapters, and a no-hardware deployment-preparation backend.
- Validation gates for camera alignment, collision penetration, joint/velocity limits, physics tracking,
  placement, and task executability.
- LeRobot v2 JSONL and v3 sharded-Parquet validation, including metadata/frame counts and the existence of
  every referenced parquet and video file.
- Feedback requests ranked by failed OOD scenario; the next generation iteration consumes these requests and
  adds targeted rollouts.
- A real LeRobot launcher for SmolVLA and FastWAM. It intentionally does not auto-launch expensive training.
- A hardware safety boundary: external deployment requires explicit operator approval and an emergency-stop
  description. The included configurations never command a robot.

The upstream GPT6-real2sim repository is a strong artifact study but explicitly describes itself as
semi-manual, not a general learned one-click reconstruction system. AutoPolicy preserves that distinction:
archived results are labeled `archived_real2sim_artifact`, smoke outputs are labeled
`synthetic_smoke_test`, and external results are labeled `external`.

## Deploy on another Linux/NVIDIA server

See [the full deployment guide](docs/DEPLOYMENT.md) for prerequisites, asset provenance, and validation.
The source repository stays small: project-owned task1 videos, the 99-episode expert dataset, and the trained
v5 checkpoint are downloaded from this repository's `task1-assets-v1` GitHub Release after cloning. Public
upstream source and model processor files are fetched from their original hosts. Caches and unrelated
experimental checkpoints are not published.

```bash
git clone https://github.com/Zxy-MLlab/AutoPolicy.git
cd AutoPolicy
scripts/bootstrap.sh
scripts/autopolicy.sh-env envs/orchestrator/bin/python scripts/install_task1_assets.py
scripts/autopolicy.sh-env envs/lerobot/bin/python scripts/check_task1_install.py
scripts/autopolicy.sh run --config configs/task1_vla_verified_artifacts.json
```

Python 3.12, Git, curl, an NVIDIA driver compatible with the pinned LeRobot CUDA wheels, and sufficient disk
space are required. The last command validates an archived task1 result; it does not claim new training or
real-robot execution.

## Orchestrator-only quick start

No third-party Python dependency is needed for orchestration itself.

```bash
cd AutoPolicy
scripts/bootstrap.sh --core-only
scripts/autopolicy.sh doctor --config configs/smoke.json
scripts/autopolicy.sh run --config configs/smoke.json
scripts/autopolicy.sh status "$PWD/runs/<run-id>"
```

Run the acceptance checks:

```bash
scripts/autopolicy.sh-env envs/orchestrator/bin/pytest -q
```

Audit the imported RoboTwin data:

```bash
scripts/autopolicy.sh audit-dataset "$PWD/data/robotwin_clean_50"
```

Run the integration audit over a GPT6-real2sim artifact and all real RoboTwin datasets:

```bash
scripts/autopolicy.sh run --config configs/artifact_catalog.json
```

## Configurations

- `configs/smoke.json`: two closed-loop synthetic iterations, safe and fast.
- `configs/artifact_catalog.json`: GPT6-real2sim ep0 metric gates plus the 50-task RoboTwin dataset audit;
  training and deployment are disabled.
- `configs/external.template.json`: contract template for real reconstruction, simulation, conversion,
  training, and evaluation commands.
- `configs/task1_vla_verified_artifacts.json`: revalidates the real task1 scene, dataset, SmolVLA checkpoint,
  robustness report, failure requests, and deployment-readiness blockers in one auditable run.
- `configs/task1_vla_execute_smoke.json`: actually collects two new MuJoCo demonstrations, trains a two-step
  SmolVLA LoRA smoke checkpoint, and loads it in a two-scenario closed-loop MuJoCo evaluation. This proves
  execution wiring only and is intentionally not a policy-quality benchmark.
- `configs/task1_vla_failure_cycle_smoke.json`: consumes a recorded OOD failure request, collects expert
  corrections after actual policy roll-in, merges them with nominal replay, trains a candidate LoRA, and runs
  a paired baseline/candidate promotion gate. Its one-episode scenarios are execution smoke tests only.

Preview a real task1 run without creating files or invoking a stage:

```bash
scripts/autopolicy.sh run \
  --config configs/task1_vla_execute_smoke.json \
  --run-dir "$PWD/runs/task1-preview" \
  --dry-run
```

External commands are executed as argv arrays without a shell. Available placeholders include `{root}`,
`{run_dir}`, `{stage_dir}`, `{iteration}`, `{gpt6_real2sim}`, `{robotwin}`, `{lerobot}`, `{datasets}`,
`{models}`, and `{checkpoints}`. An external stage should write `result.json` (or the filename configured by
`options.result`) into its stage directory. Validation results must contain `"passed": true`.

See [architecture.md](docs/architecture.md) for artifact contracts, closure behavior, and production bring-up.

## Real training

The included launcher uses standard LeRobot policy types `smolvla` and `fastwam`:

```bash
scripts/autopolicy.sh-env envs/lerobot/bin/python scripts/train_lerobot_policies.py \
  --lerobot-root "$PWD/vendor/lerobot" \
  --dataset-root "$PWD/data/robotwin_clean_50/<one-task-dataset>" \
  --output-dir "$PWD/checkpoints/first-run" \
  --steps 100000
```

This requires a LeRobot environment at `envs/lerobot` with the matching SmolVLA/FastWAM extras and the
chosen training dataset. Training is a deliberate operator action because it is expensive and model/dataset
choices materially affect the experiment.

Create that environment, including all caches, inside this project with:

```bash
scripts/bootstrap.sh
```

The install is intentionally separate from the lightweight orchestrator because CUDA wheels and the two model
families require substantial disk space.

## Scope boundaries

The repository supplies executable integration contracts and verified local artifacts. A new physical scene
still requires RGB/RGB-D/video, calibration evidence, a robot model, and a configured MuJoCo or Isaac Sim
adapter. Real deployment additionally requires the robot interface, workspace limits, tested emergency stop,
and operator authorization. No smoke or archived-artifact result is a hardware safety certificate.
