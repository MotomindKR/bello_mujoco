# SuperMimic for Bello Mini

SuperMimic alternates simulator collection, learned dynamics fitting, and
actor/critic updates through imagined trajectories. Policy gradients pass through
the learned model. See the [paper](https://arxiv.org/pdf/2610.28480) and
[project](https://supermimic.github.io/) for the method.

## Prepare inputs

Training consumes Mini `MotionClip` files and a verified `sonic-corpus.json`
manifest containing checksums, source identities, and train/evaluation splits.
Keep segments and mirrors of the same source in one split. `evaluation_clips`
must list every held-out clip in the manifest. Existing Mini clips can be indexed
with `python -m bello_mujoco.sonic.corpus prepare DIRECTORY --splits SPLITS.json`.

LAFAN, BONES and AMASS use the pinned stock
[Mini GMR profile](../deps/GMR/general_motion_retargeting/ik_configs/smplx_to_bello_mini.json).
Each frame receives one GMR call; output poses are preserved. Shared reconstruction
computes central-difference velocities, native link-origin FK/twists, geometric
foot-contact labels, position-envelope validation, and penetration diagnostics.
Source datasets remain read-only. Build a new corpus after changing preparation
settings; existing datasets and checkpoints retain their original identities.

### LAFAN1

Download the original BVHs using the
[dataset's instructions](https://github.com/ubisoft/ubisoft-laforge-animation-dataset),
then prepare the sequences selected in
[prepare_lafan.yaml](../configs/commands/prepare_lafan.yaml):

```bash
uv run --group soma bello-prepare-lafan source_directory=/data/lafan1/bvh \
  output_directory=outputs/lafan-mini \
  soma.asset_directory=/data/soma-assets \
  smplx_model_directory=/data/body-models/SMPLX
```

The default path fits the LAFAN body landmarks to a SOMA rig with fixed source
bone lengths, uses pinned [SOMA-X](https://github.com/NVlabs/SOMA-X) pose transfer
to native neutral SMPL-X, then runs stock Mini GMR. `human/` retains both human
representations. Fitting batches preserve the full source clock; no windows are
discarded. The YAML owns batch/fit settings and the 40 training / 9 evaluation
split. Licensed SMPL-X models and SOMA assets must already exist locally.

To consume existing native exports, select `human_source=smplx` and
`smplx_directory`; files must be named `SEQUENCE.npz` and cover their BVHs' clocks.
Both native routes use the reader's shape-dependent GMR height estimate. Direct
BVH retargeting remains explicit with `human_source=bvh` and `human_height_m`,
without SOMA assets or a body-model directory.

### BONES-SEED

Prepare SOMA Uniform BVHs while retaining an existing verified LAFAN corpus:

```bash
uv run bello-prepare-bones source_directory=/data/bones-seed \
  output_directory=outputs/bones-mini \
  lafan_corpus=outputs/lafan-mini/corpus
```

[prepare_bones.yaml](../configs/commands/prepare_bones.yaml) owns selection budgets
and workers. The current command selects a duration-budgeted pilot, not the
paper's full filtered dataset. Selection balances skills and keeps motion families
split-disjoint. SOMA bind rotations are removed before GMR. `settings.json`,
`selection.json`, and `preparation.jsonl` support resumption with identical inputs.
An explicit `max_retarget_failure_fraction` permits bounded IK/position failures;
all exclusions are recorded. Other failures prevent publication.

For AMASS, use the [AMASS preparation workflow](sonic_transfer.md#amass-research-corpus).

## Train and resume

Use a clean checkout with pinned submodules on Linux x86 CUDA. The current trainer
and Vast launcher support one GPU. Start with the smoke profile:

```bash
nix develop .#training
uv run --group training bello-train experiment=bello_mini_supermimic_smoke \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' wandb_enabled=false

uv run --group training bello-train experiment=bello_mini_supermimic \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' \
  motion_dataset_artifact=ENTITY/PROJECT/mini:v0 \
  checkpoint_name=mini_supermimic_seed0 run_name=mini_supermimic_seed0
```

Edit the [base recipe](../configs/profiles/train/bello_mini_supermimic.yaml) for
learning, control, networks, sampling, memory budgets, and checkpoints. The
[pilot](../configs/profiles/train/bello_mini_supermimic_pilot.yaml),
[AMASS](../configs/profiles/train/bello_mini_supermimic_amass.yaml), and
[LAFAN](../configs/profiles/train/bello_mini_supermimic_lafan.yaml) files contain
only their overrides. Use public CLI fields for per-run overrides; see
[configuration](training_configuration.md#supermimic-configuration).

Resume with `resume_path=checkpoints/RUN/update_NNNNNN`. Checkpoints include replay,
optimizers, normalizers, RNG, collector and sampler state. Resume requires the
same learning contract, corpus, exact clean source commit and dependencies.
Operational wall-time/checkpoint-evaluation limits may change. Use a historical
checkout for historical policies; checkpoint migration is not supported.
Local checkpoint saves complete before training continues. Artifact registration
and uploads run serially in a background worker; failures stop the run, and
shutdown drains pending uploads before closing W&B.

## Runtime contract

- The actor receives proprioceptive history and future reference goals. State
  uses metres, radians, WXYZ quaternions and logical joint coordinates. Six-value
  rotations use `[R00, R01, R10, R11, R20, R21]`; export records the layout.
- Simulation uses direct PD targets, scaled by effort/stiffness and transformed
  through the ankle transmission. Exploration clipping applies only during
  collection. Imagination and deterministic inference retain raw actions.
- Physical randomization is explicit context for the model and critic. Command
  delay is rejected. Replay windows and actor history do not cross resets.
- Collection episode caps are bootstrapped timeouts. Full-clip evaluation retains
  the original recording length. Height/tilt errors and nonfinite states cause
  hard failures; horizontal drift and joint RMS are reported separately.
- Precision, gains, rewards, termination thresholds and memory guards belong in
  the recipe. A changed recipe does not alter saved policies. Physical hardware
  commissioning remains pending.

## Evaluate and export

Evaluate on Linux or Lima with native CPU MuJoCo and Mesa llvmpipe rendering:

```bash
uv run bello-eval task=supermimic checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' \
  output_path=outputs/supermimic-evaluation stop_on_done=true

uv run bello-export-policy checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini reference_clips='[walk,held_out_walk]' \
  output_path=outputs/supermimic-policy.zip
```

Evaluation starts each clip at frame zero with deterministic actions and frozen
normalizers. `summary.json` records completion, first failure, normalized return,
pose errors and motor interventions, weighting clips equally. Normalized return
uses the full planned clip duration, including after early failure. Select
`evaluation_split=training` explicitly to score training clips; those reports
are coverage measurements rather than held-out generalization.

Export bundles the actor, normalization, physics, selected references and runtime
metadata. The world model and critic are not deployed. The Rust runtime bounds
motor targets separately from raw policy actions:

```bash
nix develop .#rust-runtime --command cargo run --locked -p bello-policy -- \
  --policy outputs/supermimic-policy.zip --source . --clip walk --check
```

Use `--telemetry` for JSONL inference. Inputs are `motor_position`, `motor_velocity`,
`previous_motor_target`, `orientation_xyzw`, `angular_velocity_local`,
`reference_time`, and `initialize`, in bundle coordinate order. Initialize on
entry/restart and advance one control period per sample. Hardwared owns feedback
filtering. Both modes are read-only; `--hardware-config` is rejected for
SuperMimic pending commissioning.

### Reference and world-model audits

```bash
uv run bello-audit-reference motion_dataset=/data/mini \
  output_path=outputs/reference-dynamics.json
uv run bello-audit-world-model checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini output_path=outputs/model-errors.json \
  windows='[{clip:jumps1_subject1,start_frame:11613,max_steps:400}]'
```

The [reference audit](../configs/commands/audit_reference.yaml) reports inverse
motor loads, root wrench and geometric support diagnostics without editing clips.
These are conditional diagnostics, not feasibility certificates. The
[model audit](../configs/commands/audit_world_model.yaml) measures autoregressive
prediction error against native rollouts using recorded actions; it requires
`dynamics.eqx`. These errors differ from reference-tracking errors.

### Automatic checkpoint checks

`checkpoint_evaluation` queues explicit training windows to an isolated CPU worker.
The LAFAN recipe enables these checks. Reports go to
`checkpoint-evaluation/results/UPDATE/` and a separate W&B run keyed by checkpoint
update. Worker failures fail the job; training drains queued checks before exit.
These window checks do not replace full-clip corpus evaluation.

## Remote preflight and validation

Commit and push the source branch, and pin a private immutable W&B dataset version.
Inspect the offer without renting:

```bash
uv run bello-vast-train experiment=bello_mini_supermimic \
  branch=feat/supermimic max_price=1.0 duration_hours=12 \
  motion_dataset_artifact=ENTITY/PROJECT/mini:v0 \
  evaluation_clips='[held_out_walk]' dry_run=true
```

The launcher first checks compiled learning, checkpoint restoration, native
inference and export. Successful training is followed by corpus evaluation,
artifact upload and verified teardown. The deadline includes setup and final
processing; see [Vast.ai](vast_ai.md). Run the
[repository checks](../CONTRIBUTION.md#validation) before launch. Native/CUDA
integration tests are opt-in. Experiment reports and videos belong in W&B and
ignored `outputs/`, rather than this workflow guide.
