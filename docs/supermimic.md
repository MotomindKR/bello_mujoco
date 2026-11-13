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
Offline IK retains the stock profile and adds floor inequalities from the
runtime collision surfaces. Penetrating poses receive bounded convergence passes,
reported as `floor_ik_extra_passes`, followed by a nonnegative root lift for any
remaining penetration, reported as `max_floor_projection_m`.
Before recording the first frame, repeated stock GMR solves settle the initial
pose until consecutive solves differ by at most 1 mm root translation,
5 mrad root rotation and 10 mrad per joint. Failure to settle within the bounded
pass budget rejects the clip explicitly. Diagnostics record
`initial_pose_passes` and the maximum displacement/tolerance ratio
`initial_pose_residual`; source frames and clocks are preserved.
Airborne targets retain their height. A clip-wide height shift is insufficient:
a low hand pose can otherwise lift every walking frame above the floor.
Shared reconstruction
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

Cache the full SOMA Uniform archive through SOMA-X, native neutral SMPL-X, and
stock Mini GMR:

```bash
uv run --group soma bello-prepare-bones source_directory=/data/bones-seed \
  output_directory=/data/derived/bones-mini \
  staging_directory=/local-ssd/bones-staging \
  soma.asset_directory=/data/soma-assets \
  smplx_model_directory=/data/body-models/SMPLX
```

[prepare_bones.yaml](../configs/commands/prepare_bones.yaml) owns name exclusions
and resource limits. There is no duration cap or skill quota. Its exclusions use
the examples in paper Appendix E, whose complete keyword list is not published
there; this does not reproduce the authors' exact 129,514-motion subset.
Robot-specific feasibility screening and training/evaluation corpus assembly
follow this cache step.

The command streams the compressed archive without extracting raw BVH copies.
Use a local SSD for `staging_directory`: conversion, validation and checksums run
there. The coordinator publishes completed clips while workers process the next
recordings; the bounded queue limits local disk use. Files reach a temporary NAS
directory before one atomic rename, avoiding intermediate NAS reads and ZIP seeks.
Workers reuse their body and robot models. Each atomically completed
`clips/SHARD/SEQUENCE/` retains the SOMA pose, native `human.npz`, Mini `mini.npz`,
and a checksum/diagnostics receipt. Physical IK or position-limit failures retain
human exports with an explicit rejection receipt; other errors fail the job.
Resuming with the same clean source commit, settings, metadata, bind rig and body
assets skips completed recordings. Source datasets remain read-only.

For AMASS, use the [AMASS preparation workflow](sonic_transfer.md#amass-research-corpus).

## Train and resume

Use a clean checkout with pinned submodules on Linux x86 CUDA. Start with the smoke profile:

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
Uploads read the immutable checkpoint files directly and skip W&B's local cache,
so retaining a recovery checkpoint does not require extra staging and cache copies.

### Multiple GPUs

Multi-GPU collection separates policy preparation, physics stepping, replay and
curriculum updates, physics reset, and final metrics into compiled phases. Device
fences keep Warp's CUDA graph cleanup from overlapping cross-device collectives.
Validate the alternating collect-fit-collect-policy-collect loop on the target
CUDA host before starting a full run.

`num_devices` selects GPUs within one host process. Environments, replay and sampled
learning batches are partitioned; batch settings remain global. Physics receives
distinct slices of the randomized worlds. Normalization and the return curriculum
aggregate all devices, and global learning losses synchronize parameter gradients.
Collection uses `pmap` with the pinned MJX-Warp runtime; replay sampling and learning
use sharded JAX arrays.
Use the matched JAX, MuJoCo and Warp releases in `uv.lock`; upgrading JAX alone
does not establish CUDA compatibility. Keep XLA's default kernel selection.
The CUDA check exercises randomized and nominal physics, including collection
after both model fitting and policy updates.
Reference arrays are replicated, so each GPU must fit the whole reference bank plus
its training state. Checkpoints preserve the partitioned replay and require the same
device count when resuming. Other training tasks remain single-GPU.

The [BONES profile](../configs/profiles/train/bello_mini_supermimic_bones.yaml) selects
one device and a full-corpus array budget, inheriting the base learning recipe.
Vast forwards `offer.num_gpus` to the trainer.
On a local CUDA host:

```bash
uv run --group training bello-train experiment=bello_mini_supermimic_bones \
  motion_dataset=/data/bones-mini evaluation_clips='[held_out_clip]'
```

Whole-frame collection clocks use int32 indices, retaining every frame in corpora
larger than the float32 integer range. Fractional power-of-two control steps retain
the bounded float32 clock. GPU integration is checked with
`JAX_PLATFORMS=cuda uv run --group training pytest -m integration tests/test_supermimic_parallel.py`.

## Runtime contract

The learner follows the original release at
[`abc124c`](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55),
with its inspected mjlab dependency at `dd51a9f`. This is an executable-code
baseline, not a claim to reproduce the exact unpublished headline-run settings.

- Generalized-coordinate dynamics predicts body-frame root and logical joint
  velocity changes. Pose integration accounts for physics substeps; world-frame
  root twists and WXYZ quaternions remain the physical-state contract.
- State and processed PD-command channels use running normalization. Simulation
  context uses its configured physical ranges. Model loss scores integrated root
  position/rotation and joint position/velocity using fixed nominal error scales
  from `config.model_*_error`; no output-delta normalizer is trained.
- Dynamics, goal encoder, decoder and critic use LayerNorm/GELU hidden layers.
  The actor receives gravity, local angular velocity, relative joint positions,
  joint velocities and previous action, with history newest last. Its goal
  concatenates windowed joint positions/velocities and future root orientations
  relative to the current robot root. Only the critic also sees windowed root
  translation, generalized state and physical context.
- Collection stores raw actions; the model normalizes their scaled logical PD
  commands. Differential motor transmission remains Mini-specific. Simulation
  setpoints are unprojected; hardware command limiting is separate.
- A model window counts stored state frames, supervising one fewer transitions.
  Sampling excludes reset/invalid interiors and selects environments uniformly
  among those with complete windows. Actor history never crosses resets.
- Reward and termination score the post-step state against the reference held
  before stepping; goals then advance and clamp at clip endings. Collection time
  caps bootstrap as timeouts. Full-clip evaluation retains recording duration.
- Imagined termination is a sigmoid of the maximum normalized height/tilt error.
  Backpropagation recomputes each imagined step's network activations instead of
  retaining them across the full horizon, preserving the loss and batch sizes.
  Actor bootstraps are upper-bounded by the attainable discounted reward; critic
  GAE uses raw values and clamps final targets. Five critic epochs reuse one
  shuffled partition. Fully terminated imagination becomes absorbing to prevent
  unreachable numerical overflow, while every finite horizon state is fitted.
- Hidden Linear weights between the first and last Linear of each MLP use Muon
  with RMS learning-rate scaling and explicit weight decay. Other parameters use
  Adam. The released-code baseline uses constant learning rate and FP32 networks
  with TF32 matrix products; BF16 is an explicit alternative. The paper's Adam
  and cosine schedule differ from these released defaults.
- Muon orthogonalization uses BF16 internally, matching PyTorch. Reference banks
  retain root/joint trajectories and contact labels; tracked bodies are
  reconstructed through FK instead of cached across the full corpus.
- The buffered return curriculum updates every visited temporal bin, handles
  terminal/timeout boundaries, and retains bounded per-bin samples. Goal stride,
  curriculum settings, reset noise and the training cap belong in YAML.
- Invalid physics is excluded from replay and statistics. Invalid resets,
  capacity overflows, invalid learner updates and configured critic-loss
  violations stop training with diagnostics. Checkpoint uploads remain asynchronous.

The input, model and optimizer contracts changed. Historical checkpoints require
historical checkouts; there is no conversion or compatibility fallback. Corrected
stock-GMR motion files keep their existing identities. Physical commissioning
remains pending.

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
uses the full planned clip duration, including after early failure.
Failure events identify the reference frame scored before clock advancement. Select
`evaluation_split=training` explicitly to score training clips; those reports
are coverage measurements rather than held-out generalization.

Export bundles the actor, normalization, physics, selected references and runtime
metadata. Scalar actuator input labels are omitted for MuJoCo 3.12 compatibility;
their force parameters and limits are preserved. The world model and critic are
not deployed. The Rust runtime bounds
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

`checkpoint_evaluation` queues training windows and full clips to an isolated CPU worker.
`training_sample_size` selects a fixed training sample using the run seed;
`evaluate_held_out` evaluates the separate evaluation split. The BONES recipe
enables both, and LAFAN retains its model windows alongside full held-out clips.
Reports include full-duration return, completion and pose errors; per-step
`collection/step_reward` and `imagination/step_reward` are step averages, not
full-motion success metrics. Reports go to
`checkpoint-evaluation/results/UPDATE/` and a separate W&B run keyed by checkpoint
update. Worker failures fail the job; training drains queued checks before exit.
Samples do not replace an exhaustive corpus evaluation.

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
