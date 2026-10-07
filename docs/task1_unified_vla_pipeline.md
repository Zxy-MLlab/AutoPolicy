# Task1 unified VLA pipeline verification

Date: 2026-09-12 UTC

## Scope

This task uses the source video:

```text
/data/zxy/autopolicy/data/real/task1/VID_20260611_160742_1000810407.mp4
```

and the instruction `抓取桌面上的水瓶并抬离桌面`.

The task1 replica is a single-video, task-focused MuJoCo approximation. Robot identity, metric scale, and
camera calibration are engineering hypotheses. It is not a full-scene reconstruction and not an automatic
metric digital twin.

## Two verified operating modes

### Existing-artifact verification

Configuration:

```text
/data/zxy/autopolicy/configs/task1_vla_verified_artifacts.json
```

Run:

```text
/data/zxy/autopolicy/runs/task1-vla-unified-artifact-verification-20260912
```

The unified runner completed all eight stages. It rechecked source-video hashes, loaded the MuJoCo XML,
audited the complete 99-episode/8,910-frame LeRobotDataset, verified the SmolVLA model and training
configuration, normalized the existing 120 paired MuJoCo evaluations, wrote deployment blockers, and derived
90 requested failure-focused episodes.

The policy gate intentionally failed because nominal was 18/20 (90%) but combined OOD was 4/20 (20%), below
the configured 50% combined threshold. The pipeline therefore produced improvement requests instead of
promoting the checkpoint.

### New execution smoke test

Configuration:

```text
/data/zxy/autopolicy/configs/task1_vla_execute_smoke.json
```

Run:

```text
/data/zxy/autopolicy/runs/task1-vla-unified-execute-smoke-20260912
```

This run performed new downstream work rather than only referencing history:

| Stage | Evidence |
|---|---|
| reconstruct | Reused the explicitly labeled existing task-fitted replica |
| validate | Loaded and checked the MuJoCo scene in the current process |
| generate | Collected 2/2 successful expert episodes |
| dataset | Audited a new LeRobotDataset containing 180 frames |
| train | Trained a new two-step SmolVLA LoRA adapter |
| evaluate | Loaded the new checkpoint and executed nominal and combined-OOD closed-loop rollouts |
| deploy | Wrote readiness blockers; sent zero hardware commands |
| improve | Converted the smoke failures into 16 requested episodes |

The new adapter is 2,684,600 bytes and has SHA-256
`a6c7353d46967efd108140aa18acbcd719125f063a70b2bb2ec0e95ab2f59d2b`.

Both one-episode scenarios failed. A two-step checkpoint trained on two episodes is intentionally incapable of
supporting a performance claim; this run proves command wiring, VLA input/output compatibility, checkpoint
serialization/loading, and MuJoCo closed-loop execution only.

Evaluation video:

```text
/data/zxy/autopolicy/runs/task1-vla-unified-execute-smoke-20260912/iteration-000/evaluate/robustness/combined_ood/smolvla_rollout_episode_000.mp4
```

### Failure-driven correction cycle

Configuration:

```text
/data/zxy/autopolicy/configs/task1_vla_failure_cycle_smoke.json
```

Run:

```text
/data/zxy/autopolicy/runs/task1-vla-unified-failure-cycle-smoke-20260912
```

This run consumed the previous combined-OOD sampling request (`priority=16`, 32 requested episodes) and used
the v5 policy to roll into two recorded combined-OOD failure states. The state-conditional MuJoCo expert then
successfully produced two correction episodes (120 frames, zero rejected). The dataset stage merged those
corrections with all 99 nominal replay episodes, producing 101 episodes and 9,030 frames.

A new two-step SmolVLA LoRA adapter was trained and then compared with v5 on exactly paired bottle positions
and perturbations. In the one-episode nominal and combined-OOD smoke scenarios, both baseline and candidate
failed. The gate correctly rejected promotion because there was no combined-OOD improvement and the sample
count was below the hard minimum of 20 episodes per scenario. This result validates the feedback wiring but
does not estimate candidate quality.

Candidate evaluation video:

```text
/data/zxy/autopolicy/runs/task1-vla-unified-failure-cycle-smoke-20260912/iteration-000/evaluate/candidate/combined_ood/smolvla_rollout_episode_000.mp4
```

## Commands

Dry-run (no run directory or artifacts are created):

```bash
scripts/autopolicy.sh run \
  --config configs/task1_vla_execute_smoke.json \
  --run-dir /data/zxy/autopolicy/runs/task1-vla-preview \
  --dry-run
```

Execute or resume:

```bash
scripts/autopolicy.sh run \
  --config configs/task1_vla_execute_smoke.json \
  --run-dir /data/zxy/autopolicy/runs/task1-vla-run

scripts/autopolicy.sh run \
  --config configs/task1_vla_execute_smoke.json \
  --run-dir /data/zxy/autopolicy/runs/task1-vla-run \
  --resume
```

Resume rejects a changed configuration and re-executes a stage if a recorded output file is missing, has a
different size, or has a different SHA-256 hash. External commands run as argv arrays without a shell.

## Current automation boundary

The following task1 steps now run under one command after a task-fitted scene exists: simulation validation,
expert collection with successful-only filtering, LeRobotDataset audit, SmolVLA training, MuJoCo evaluation,
failure-request generation, policy-roll-in correction collection, nominal replay merge, candidate retraining,
paired promotion gating, and deployment-readiness reporting.

The first reconstruction remains artifact-backed and semi-automatic. Converting an arbitrary new MP4 into a
calibrated full-scene simulator still requires measured camera calibration, an exact robot model/state log,
metric references, object inventory/assets, and reconstruction logic beyond the current task1 adapter.
Real-robot deployment also remains disabled until an operator provides and validates the robot interface,
joint mapping, workspace/velocity/force limits, and emergency stop.
