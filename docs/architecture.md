# Architecture and contracts

## Run layout

Each run is append-like and independently inspectable:

```text
runs/<run-id>/
  resolved_config.json
  pipeline_state.json
  iteration-000/
    reconstruct/manifest.json
    validate/manifest.json
    generate/manifest.json
    dataset/manifest.json
    train/manifest.json
    evaluate/manifest.json
    deploy/manifest.json
    improve/manifest.json
```

Stage manifests transition from `running` to `completed` or `failed`. Completed manifests list hashes for
every direct output file. `--resume` skips completed stages and restarts the first incomplete stage.

## Reconstruction closure

An external reconstruction adapter consumes synchronized RGB/RGB-D/video, robot states, calibration priors,
and task context. It should emit a metric digital twin (MJCF/USD, collision assets, articulation, calibrated
cameras, physics parameters, and provenance). The validation adapter measures visual and geometry agreement,
robot-camera alignment, collision/articulation behavior, physics/trajectory agreement, and task execution.
It writes `passed: false` when any configured gate fails. AutoPolicy then stops before trajectory generation;
the adapter or a controlling agent can edit the model and resume the validation stage.

The `artifact` adapter normalizes the published GPT6-real2sim validation structure into these gates. It does
not claim the archived scene was reconstructed from a new user video.

## Data-policy closure

Generation produces attempted rollouts and a success-filtered index. A production adapter should record at
least task text, observations, actions, timestamps, simulator seed/domain parameters, terminal success, and
failure reason. Conversion emits a canonical LeRobotDataset. The catalog auditor enforces feature and episode
metadata consistency and checks every declared parquet/video path.

Evaluation aggregates nominal, perturbation, and OOD scenarios. Improvement converts failures into ranked
sampling requests. On the next iteration the generator consumes those requests; production generators should
map each requested scenario to domain randomization, initial-state sampling, task wording, and expert-policy
rollouts. The loop stops when the configured simulation success target is met or the iteration budget is used.

## Upstream integration

| Component | AutoPolicy role | Production entry point |
| --- | --- | --- |
| GPT6-real2sim | Reconstruction and validation artifact/reference | external reconstruction and validation adapters |
| RoboTwin | Task simulation, expert rollouts, perturbation/OOD evaluation | `run_task.sh`, task configs, policy eval adapter |
| LeRobot | Dataset schema, VLA/WAM training, policy/hardware interfaces | `lerobot-train`, `lerobot-eval`, `lerobot-rollout` |
| SmolVLA | VLA profile | `policy.type=smolvla` |
| FastWAM | World Action Model profile | `policy.type=fastwam` |

Upstream repositories are isolated snapshots so existing dirty worktrees elsewhere in `/data/zxy` are not
modified. Their exact commits are recorded in every run state.

## Storage and execution safety

External stages receive `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `HF_HOME`, `HF_DATASETS_CACHE`,
`TRANSFORMERS_CACHE`, `TORCH_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `WANDB_DIR`, and `MPLCONFIGDIR`
redirected below the checkout root. Configuration rejects managed output paths outside that root.

External commands are passed directly to `subprocess.run`; shell expansion is not used. Real deployment is
disabled unless configuration explicitly selects `external`, supplies `operator_approved: true`, and names an
emergency-stop mechanism. The deployment adapter remains responsible for robot-specific joint, velocity,
force, collision, and workspace interlocks.

## Production bring-up order

1. Put raw synchronized observations under `data/real/<scene>` and write a reconstruction adapter for the
   selected simulator.
2. Calibrate quantitative thresholds from held-out observations and make validation pass.
3. Map the digital twin/task into RoboTwin or a replica MuJoCo/Isaac environment; verify scripted expert task
   success before collection.
4. Convert successful rollouts and pass `audit-dataset`.
5. Run small overfit tests for one VLA and FastWAM before full training.
6. Evaluate nominal and perturbation matrices, inspect failure logs, and run at least one feedback iteration.
7. Only after simulation acceptance, configure a low-speed shadow/dry run and review hardware safety with an
   operator before enabling action commands.
