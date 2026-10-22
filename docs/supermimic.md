# SuperMimic for Bello Mini

SuperMimic trains a Mini motion tracker from scratch by alternating MJX-Warp
collection, fitting a differentiable residual dynamics model, and actor/critic
updates through imagined trajectories. The actor receives proprioceptive history
and future reference frames. Gradients pass through the learned model, not the
simulator. Method references are the [project](https://supermimic.github.io/) and
[paper](https://arxiv.org/pdf/2610.28480). The [port plan](supermimic_plan.md)
records the initial choices and later experiments; this page describes the
implemented workflow.

Synthetic JAX tests, native Mini physics/FK tests, and ONNX/Rust checks validate
numerical behavior. CUDA training and physical commissioning have not been run
on the development macOS host. There is no trained checkpoint or measured
tracking performance yet. The small profile checks the compiled loop; it is
not a learning-performance recipe.

## Prepare inputs

Use Mini `motion.clip.MotionClip` files from the
[offline AMASS workflow](sonic_transfer.md#amass-research-corpus) or existing
retargets with source provenance. SuperMimic reuses the checked corpus and
reference-bank code under `sonic/`, without the G1 adapter, tokenizer, pretrained
weights, or LoRA model.

The directory must contain `sonic-corpus.json`, clip checksums, source sequence
identities, and train/evaluation splits. Prepare an existing set of Mini clips:

```bash
uv run python -m bello_mujoco.sonic.corpus prepare /data/mini \
  --splits /data/mini-splits.json
```

The splits JSON maps clip stems to objects such as
`{"split":"train","source_sequence":"walk-001"}` or the `evaluation` split.
Keep segments and mirrors of a source sequence in the same split. Supply every
held-out stem in `evaluation_clips` when training; the manifest must agree.
Joint/body names, generalized coordinates, and FPS are checked against the
compiled Mini. At least one training clip must cover the imagination horizon.

For a local pilot, the checked directory suffices. An optional
`motion_dataset_artifact` must be an immutable W&B dataset `:vN` version or
digest, verified against the local bytes. Publishing a corpus is a separate
explicit operation. Remote jobs require the pinned artifact and download it
before the preflight.

## Train and resume

Use a clean checkout and initialized pinned dependencies on a Linux x86 CUDA
host. Generated assets and checkpoints remain outside Git.

```bash
git submodule update --init --recursive
nix develop .#training

uv run --group training bello-train \
  experiment=bello_mini_supermimic_smoke \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' \
  checkpoint_name=mini_supermimic_smoke wandb_enabled=false

uv run --group training bello-train \
  experiment=bello_mini_supermimic \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' \
  motion_dataset_artifact=ENTITY/PROJECT/mini:v0 \
  checkpoint_name=mini_supermimic_seed0 run_name=mini_supermimic_seed0
```

Edit the [full profile](../configs/profiles/train/bello_mini_supermimic.yaml)
or [smoke overrides](../configs/profiles/train/bello_mini_supermimic_smoke.yaml)
for nested settings. Public overrides include seed, environment count, total
timesteps, learning rate, contact budget, logging, and checkpoint destinations.
Training rejects single-motion paths, pretrained inputs, payload overrides, and
`video_enabled`. Evaluate checkpoints separately for videos.

`bello_mini_supermimic_pilot` uses the full actor/dynamics architectures and
32-step imagination with 128 real environments, smaller optimizer/replay batches,
and 2,097,152 real transitions. Use it after the smoke check for initial GPU
memory, throughput, stability, and tracking measurements. It is a pilot recipe,
not a demonstrated convergence budget.

The full recipe uses 50 Hz control, four 5 ms substeps, 10 ms command delay,
friction/calibration/torso-COM context, noisy feedback, low-pass filtering, and
0-2 control steps of observation latency. These are initial simulation choices;
their ranges have not been established by Mini system identification.

Collection adds Gaussian action noise and stores the actual clipped action,
previous/new accepted motor targets, physical successor before reset, measured
observation, filter state, reference clock/alignment, episode identity, context,
reward, and distinct terminal/timeout flags. Dynamics windows cannot cross
episode, clip, or context boundaries. Imagination reconstructs causal actor and
feedback histories. Ineligible starts skip policy improvement with a metric.

At the default update-to-data ratio, each outer update collects alternating
two/three vector steps, fits ten dynamics batches, then updates actor/critic
after model warmup. Dynamics fitting autoregresses eight steps. Actor improvement
differentiates 32 imagined steps through fixed world-model and critic parameters,
preserving input derivatives. The critic learns detached lambda-return targets.
Invalid optimization updates are rejected and stop the run. The learning-rate
schedule uses the outer-update clock. Total timesteps count real transitions.

Metrics include model/actor/critic losses, gradient norm, imagined reward,
completed episodes, failures, skipped policy updates, and
`collection/limited_fraction` for projection/slew intervention. Logging uses the
existing local/W&B path. Reset sampling mixes uniform/difficult/historical
weighting over one-second training bins. Difficulty uses an EMA of actual
discounted episode returns attributed to the starting bin; historical weights
accumulate previous difficulty distributions. Held-out clips never enter
learning or the curriculum.

Checkpoints include full learner/optimizer state, normalizers, replay, collector
and feedback memory, PRNG, sampling statistics, counters, corpus manifest, and a
smaller actor-only state. Files are hashed, staged, fsynced, and published
atomically. Resume requires the same clean source commit, corpus identity, and
resolved learning/environment/network settings. Only the total timestep budget
and operational logging/checkpoint destinations may change; the dataset directory
may move if its identity is preserved.

```bash
uv run --group training bello-train \
  experiment=bello_mini_supermimic motion_dataset=/data/mini \
  motion_dataset_artifact=ENTITY/PROJECT/mini:v0 \
  evaluation_clips='[held_out_walk]' \
  resume_path=checkpoints/RUN/update_001000 total_timesteps=6553600000
```

Use the same profile, seed, and original learning overrides on resume. Corpus
arrays are verified separately, not embedded in the checkpoint. Replay's actual
footprint is checked against `config.max_replay_bytes` before allocation.
Reference arrays have a separate `dataset.max_array_bytes` guard. Network,
gradient, and simulator allocations require additional VRAM beyond both guards.

## State, observations, and commands

Resolve joints and bodies by name. The reference root is `pelvis_link`, including
its fixed transform from the Mini free body. Physical state contains root world
position, WXYZ orientation, world-frame link-origin linear/angular velocities,
and logical joint positions/velocities. Native free-joint angular velocity is
explicitly converted between local and world conventions. Analytic FK includes
joint pivots, reference positions, fixed bodies, and link-origin twists. Units
are metres, radians, and seconds.
Heading is the projected forward-axis azimuth, defined as zero with zero
heading derivative when that axis is vertical (squared horizontal norm below
`1e-12`), consistently in JAX and Rust.

For `N` joints, the default actor sample is `3N+12` values:

1. Joint position minus neutral and calibration offset (`N`).
2. Joint velocity (`N`).
3. Root-local angular velocity (`3`).
4. Root-local projected gravity (`3`).
5. Previous clipped policy action (`N`).
6. Current aligned reference orientation relative to measured root (`6`).

History contains ten samples, oldest first; resets repeat the first valid
sample. Disable `observations.reference_orientation` explicitly for its ablation.
The separate goal concatenates ten future frames at 1-10 control steps. Each
frame contains displacement and orientation in the current reference heading
frame, future root-local linear/angular velocities, absolute logical joint
positions, and scaled joint velocities. Its size is `K*(15+2N)`. Futures clamp
to the selected clip end. Six-dimensional rotations concatenate the first two
rotation-matrix columns, separately from SONIC's historical reshape convention.

Persisted running moments normalize history and goals. The actor's ELU motion
encoder produces a tanh latent, concatenated with normalized history for the ELU
decoder. The SiLU/LayerNorm world model consumes heading-invariant physical
state, previous/new motor targets, and context, predicting normalized root-local
increments. The privileged critic additionally receives reference error, goal,
context, and actor history. The collector's single-sample critic protocol is
separate from the model-based learner's history-bearing critic. The SuperMimic
profile does not route through shared PPO.

Commands use `neutral_joint + calibration + scale*clipped_action`, with
`scale=0.25*effort/stiffness` in logical joint order. Transform to motors,
ray-project into the joint/motor envelope, then apply connected motor step
limits from the previous accepted target. Collection, imagination, native
evaluation, and Rust telemetry share this order and real piecewise derivatives.
Previous action means clipped policy action, not an accepted motor target.

Tracking reward uses analytic FK: four body pose/twist exponential terms of
weight one, root pose terms of weight one-half, action-rate regularization, and
a soft joint-limit penalty. The sum is scaled by control timestep. Hard failures
use root height error, projected-gravity Z error, or nonfinite state. Imagination
uses smooth survival factors and stops bootstrapping at clip end. These settings
live in `tracking`; editing a profile does not change saved policies.

## Evaluate and export

On Linux or Lima, select names in the recorded held-out split. Evaluation
disables noise, sampled latency, and physical randomization, preserving trained
control delay and filtering. Normalizers and curriculum are frozen. Each clip
starts at frame zero and runs deterministic actions.
Video evaluation loads the canonical visual model; training and export retain
the physics model. Robot collision geometry in the physics model is transparent
and is unsuitable as the sole source of a policy video.
CPU evaluation compiles goal assembly, command diagnostics, and tracking-error
measurements. SuperMimic uses fixed native contact/constraint metadata capacities
of 4,096 contacts and 65,536 force rows to prevent contact-count changes from
recompiling the complete outcome function. Every actual contact and force is
retained; exceeding either capacity raises an error. MuJoCo physics and control
timing retain their recorded settings.

```bash
uv run bello-eval task=supermimic checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini evaluation_clips='[held_out_walk]' \
  output_path=outputs/supermimic-evaluation stop_on_done=true

uv run bello-export-policy checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini reference_clips='[walk,held_out_walk]' \
  output_path=outputs/supermimic-policy.zip
```

The output directory contains one video per clip and `summary.json`: return,
normalized return, completion, failure/timeout, root pose error, joint RMS error,
peak actuator effort ratio, and limiting frequency. Summary means weight clips
equally. Normalized return uses the maximum tracking reward over the full planned
clip; early failure cannot improve its denominator. A `steps` override retains
that denominator. Foot-slip/thermal diagnostics, multi-seed learning curves,
and a matched PPO comparison remain experiments from the plan.

Export creates actor-only ONNX with frozen normalization, self-contained physics,
selected clips, named coordinate/IMU transforms, and history/goal/rotation/timing
metadata. ONNX emits raw action and proposed motor target; the runtime applies
final projection and slew limits. No world model or critic is deployed.

```bash
nix develop .#rust-runtime --command cargo run --locked -p bello-policy -- \
  --policy outputs/supermimic-policy.zip --source . --clip walk --check
nix develop .#rust-runtime --command cargo run --locked -p bello-policy -- \
  --policy outputs/supermimic-policy.zip --source . --clip walk --telemetry
```

Telemetry consumes one JSON object per control step: `motor_position`,
`motor_velocity`, `previous_motor_target` (the actually accepted command),
`orientation_xyzw`, `angular_velocity_local`, `reference_time` in seconds, and
`initialize`. Motor arrays follow bundle coordinate order. Orientation/angular
velocity are in the bundled hardware IMU frame, converted to the trained pelvis
frame. Hardwared must own the matching feedback filter; the codec does not apply
it again. Initialize on entry/restart. Subsequent timestamps must advance one
control period; clip-end restart is explicit.

`--check` and `--telemetry` perform no hardware writes. SuperMimic rejects
`--hardware-config` pending commissioning. Arming and live PICO following require
later integration and trained-policy validation.

## Remote preflight and validation

All SuperMimic profiles support `bello-vast-train` with a pinned dataset and no pretrained
revision. Commit/push the reviewed branch for remote cloning. A dry run inspects
selection without renting:

```bash
uv run bello-vast-train experiment=bello_mini_supermimic \
  branch=feat/supermimic max_price=1.0 duration_hours=12 \
  motion_dataset_artifact=ENTITY/PROJECT/mini:v0 \
  evaluation_clips='[held_out_walk]' dry_run=true
```

Before selected training, the launcher runs the small profile, verifies actor
updates, restores native MuJoCo inference, checks finite stepping, and exports
a bundle. No GEAR-SONIC weight download occurs. The source-only `gear_sonic`
submodule is still initialized because the repository's exact-source contract
requires its recorded gitlink for every checkpoint operation. Deadlines and teardown rules
apply; see [Vast.ai](vast_ai.md).

After successful training, the host evaluates all selected held-out clips in
native CPU MuJoCo, exports an actor bundle, and uploads the report, videos, bundle,
and checkpoint metadata as a W&B evaluation artifact. It waits for upload before
marking the job complete. With no held-out clips, it exports a selected training
clip and records no generalization evaluation. The job deadline includes setup,
preflight, training, evaluation, and upload; leave time for those final steps.

Run the [repository gate](../CONTRIBUTION.md#validation). Default tests cover
invariance, FK Jacobians, constraints, replay masks/history, dynamics fitting,
actor gradients, full-state checkpoint restoration, export, and CLI/remote routes.
Opt-in checks:

```bash
uv run pytest -m integration tests/test_supermimic_kinematics.py
uv run pytest -m integration tests/test_supermimic_environment.py
JAX_PLATFORMS=cuda uv run --group training pytest -m integration \
  tests/test_supermimic_environment.py
uv run pytest -m integration tests/test_supermimic_deployment.py
```

The final command needs `BELLO_POLICY_BIN` and `ORT_DYLIB_PATH`; see
[native parity setup](../CONTRIBUTION.md#validation). macOS parity was checked
with ONNX Runtime 1.23.2. The locally tested 1.22.1 library aborted during shutdown
in existing and new inference tests. The project lock is unchanged.

Validation recorded on 2026-10-22: the complete CPU gate passed 894 tests, with
five expected skips. Lockfile, protobuf generation, Ruff formatting/lint, Rust
formatting, and the locked Rust build passed. Seven SuperMimic native tests
passed, including Mini checkpoint/export/evaluation and Rust observation/action
parity at upright and vertical headings. Its CUDA collection test skipped on
this CPU-only host. The broader native runtime regression run also passed.
The local gate preceded the paid CUDA pilot below. Physical commissioning
remains separate.

## First CUDA pilot

The broader follow-up uses `bello_mini_supermimic_amass` with immutable dataset
`gyungmin/bello-mujoco/mini-supermimic-amass-8h-20261022:v0`, manifest SHA-256
`f83d855b410b245bb3e86e28bc14bd3969838f71a336a3f79bfe3266d615c3fc`.
All 4,710 parent clips passed current Mini joint/motor, speed, and terrain checks.
The bank retains 4,166 training clips (28,807.76 s) and 544 held-out clips
(2,880.62 s), with disjoint inherited subject and source identities. Canonical
tracked link-origin twists were recomputed for every frame. This is a fresh
training run; its corpus differs from the six-clip pilot below.

The 2026-10-22 [Vast pilot run](https://wandb.ai/gyungmin/bello-mujoco/runs/5rd94jy0)
uses source commit `2b4f5550302a6fed447222444d0301e87d69cdd2`, stack
`29d4d9bc59e2999dcd623a32bf893d0c1735d686`, and the
`bello_mini_supermimic_pilot` profile with seed zero. It keeps the full actor and
world-model architecture, 32-step imagination, 128 environments, and 512 replay
steps. The authorized budget is $0.75/hour and two hours, with host-owned teardown;
the selected RTX 4090 has 48 GiB and costs $0.6963/hour.
The run completed 2,097,152 real transitions and 6,528 training iterations,
including 13,056 world-model optimizer steps and 6,028 actor/critic optimization
rounds, in 49 minutes of training. All update validity checks
passed. Automatic teardown completed, and Vast's instance inventory confirmed
that pilot instance `57179009` was absent afterward. The unrelated stopped
instance was left untouched.

The immutable dataset is
`gyungmin/bello-mujoco/mini-supermimic-amass-pilot-20261022:v0`, manifest SHA-256
`8e4a6bf0f1f70e580bd63f8192d18bfcf159ccfee9d77340f7e947f319825ea6`.
Five 50 Hz clips cover standing, swaying, walking, slow walking, and clapping,
totalling 26.86 seconds. The held-out clip is
`amass__BMLrub__rub016__0005_normal_walk1_stageii` (3.24 seconds), from a different
subject than the training walk. Splits and source identities are inherited from
the existing AMASS corpus. Preparation recomputes canonical tracked link-origin
twists and screens joint/motor bounds, velocity, and terrain penetration without
relaxing the robot constraints.

Native CPU MuJoCo evaluation starts at frame zero with deterministic actions:

| Checkpoint update | Real transitions | Completion | Normalized return | Root position RMS | Root orientation RMS | Joint RMS |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | 328,192 | 51.2% | 0.2194 | 0.256 m | 24.31 deg | 0.352 rad |
| 2,000 | 648,192 | 100% | 0.5929 | 0.203 m | 7.83 deg | 0.286 rad |
| 4,000 | 1,288,192 | 100% | 0.6171 | 0.251 m | 7.17 deg | 0.276 rad |
| Final (6,528) | 2,097,152 | 100% | 0.6385 | 0.178 m | 6.67 deg | 0.276 rad |

The final clip reached its normal timeout without a terminal failure. Command
projection or slew limits intervened on 9.26% of steps; peak actuator effort
reached the configured limit. These are measured constraints, not evidence of
hardware commissioning.

The CUDA smoke completed collection, model fitting, imagined actor/critic updates,
checkpoint restoration, native stepping, and export. A separate diagnostic
restored the full update-4,000 runner, including replay and optimizer/controller
state, and advanced collection, model fitting, and actor/critic optimization.
The trained update-2,000 actor passed the Rust bundle check and 64-frame telemetry
parity: maximum action difference `1.87e-6`, accepted motor-target difference
`5.97e-7 rad`. The final actor also passed: action difference `4.07e-6` and accepted
motor-target difference `9.84e-7 rad`. Training alone used about 4.72 GiB of device
memory, measured before the separate resume diagnostic.

Persisted W&B artifacts are:

- Full final checkpoint:
  `gyungmin/bello-mujoco/supermimic-amass-pilot-20261022-seed0_20261022_075614_795649:v6`.
- Visible video, report, and self-contained ONNX bundle:
  `gyungmin/bello-mujoco/supermimic-amass-pilot-20261022-seed0-final-visible-evaluation:v0`,
  from the [visible evaluation run](https://wandb.ai/gyungmin/bello-mujoco/runs/wzpgbxn4).

Both artifacts were downloaded locally; actor, full-state, and corpus-manifest
checksums passed. The standard and visible final evaluations produced identical
numeric reports.

This run predates the visual-model evaluation fix. Its visible diagnostic video
changes canonical collision-geometry RGBA and headlight only; physics, inference,
weights, and source identity remain unchanged. Checkpoints require a separate
clean checkout at their recorded commit and `uv sync --frozen`; advancing this
branch does not migrate historical checkpoints.

These measurements cover one short held-out walking clip and one seed. They do
not establish long-horizon recovery, corpus-wide generalization, target-device
50 Hz timing, or physical operation. Larger screened corpora, longer held-out
motions, multiple seeds, and matched PPO comparisons are the next experiments.
