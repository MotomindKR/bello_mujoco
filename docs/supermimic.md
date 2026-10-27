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
numerical behavior. Bounded CUDA training has run on rented Vast RTX 4090 hosts;
the measurements and immutable artifacts are recorded below. Physical
commissioning has not run. The small profile checks the compiled loop; it is
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

LAFAN1 is a separate human BVH dataset. Download the original archive from
[Ubisoft La Forge](https://github.com/ubisoft/ubisoft-laforge-animation-dataset),
which documents its license and Git LFS workflow:

```bash
mkdir -p outputs/lafan1-source
curl -fL --retry 3 \
  https://media.githubusercontent.com/media/ubisoft/ubisoft-laforge-animation-dataset/master/lafan1/lafan1.zip \
  -o outputs/lafan1-source/lafan1.zip
unzip -tq outputs/lafan1-source/lafan1.zip
unzip -q outputs/lafan1-source/lafan1.zip -d outputs/lafan1-source/bvh
```

The checked archive SHA-256 is
`ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`.
It contains 77 sequences (4.5981 hours); every whole sequence exceeds 30 seconds.
The [unofficial reproduction](https://github.com/zita-ch/supermimic_vibed)
uses 40 G1 retargets from
[this dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
All 40 have matching original BVHs, totaling 2.4506 hours. G1 CSVs cannot be
loaded as Mini references: retarget the human source to Mini and validate the
resulting motion clips. Do not apply the AMASS preparation's 2-30 second whole
sequence filter to LAFAN1.

Prepare the original human BVHs directly for Mini:

```bash
uv run bello-prepare-lafan \
  source_directory=outputs/lafan1-source/bvh \
  output_directory=outputs/lafan1-mini
```

The [preparation config](../configs/commands/prepare_lafan.yaml) lists the same
40 whole training sequences as the unofficial retargeting cohort: dance,
fall/get-up, fighting/sports, jumps, running, sprinting and walking. Nine separate
ground/multiple-action recordings form an additional evaluation split. These
are sequence-disjoint, not subject-disjoint. Local rotations are resampled at
50 Hz before forward kinematics. `human_source=bvh` converts named LAFAN
rotations to the pinned Mini SMPL-X frame convention. Source toe transforms
provide conservative planted-foot labels based on height, linear/angular speed
and interval duration. These are geometric proxies, not measured contact forces.

Offline preparation fits source orientations to Mini with canonical coupled
motor/joint bounds, then constructs position targets from that pose's native
forward kinematics. Human hip rotations target the final hip yaw link after all
three hip hinges. This replaces independent human landmark scaling and offsets
as limb position targets: anatomical hip centers are not Mini hip roll origins.
Root translation goals and orientation-frame offsets use the pinned GMR convention.
Contact refinement tracks a fixed sole material point during each labelled
stationary interval. Its position objective has explicit `plant_position_cost`,
and touchdown height blends over `contact_blend_frames`. Foot labels are geometric
proxies, so this objective remains soft when it conflicts with physical limits;
per-sequence planted-point residuals are reported.
Refinement also applies each joint's canonical URDF speed bound and linearized
floor/self-contact constraints. A final minimum-displacement IK projection
repairs nonlinear contact residuals.

Ground alignment is one constant translation per sequence, estimated from
calibrated sole heights on source-planted frames. When no stationary foot labels
exist, the whole-body lower envelope supplies the plane alignment; the chosen
method and translation are reported. No sequence is rejected for absent labels,
and alignment preserves all flight-phase height differences. Calibration and
refinement start from the robot's configured posture and fitted pose, respectively;
they do not require an older reference corpus.
The initial fitted pose is projected through the physical constraints before
any planted-foot anchors are captured, without advancing the source clock.
Settings live under `retargeting.*`, including iteration counts, contact blending,
`contacts.*` distance bounds and
`source_contacts.*` proxy thresholds.

To use full-sequence SOMA-to-native-SMPL-X exports instead of the BVH rotation
adapter, select the input explicitly:

```bash
uv run bello-prepare-lafan source_directory=/data/lafan1/bvh \
  output_directory=outputs/lafan-soma-mini \
  human_source=smplx smplx_directory=/data/lafan-smplx \
  smplx_model_directory=/data/body_models/smplx
```

Each native file must be named `SEQUENCE.npz` and contain `surface_model_type=smplx`,
`gender`, `mocap_frame_rate`, `root_orient` (frames by 3), `pose_body` (frames by 63),
`trans` (frames by 3) and constant `betas` (up to 16). The licensed gender-specific
`SMPLX_GENDER.npz` supplies the shape-dependent rest joints. Native root rotations
already express the Z-up source world; `trans` is added to the shaped rest pelvis.
After resampling, the full sequence must have the same clock length as its BVH;
partial pilot windows are rejected. BVH contact labels remain the common source
for either input mode. Both source hashes and the native body-model hash are
recorded. Supplying SMPL-X directories with `human_source=bvh` is an error.
This command consumes native exports; it does not install or invoke SOMA-X.

Live GMR behavior and the pinned vendor files are unchanged. The pinned GMR
solver's six-argument call is translated explicitly to Mink's keyword `limits`
at the module boundary. Cached body twists use `mjOBJ_XBODY` link origins.
Velocities use central pose differences, with one-sided differences at clip
endpoints. Free-joint angular velocities are transformed into the current pose's
local frame before FK, matching the unofficial world-frame differentiation.

Every requested sequence must succeed before the immutable corpus manifest is
published. The new output directory contains `settings.json`, per-sequence
`preparation.jsonl` diagnostics, and the checked `corpus/`. Source datasets stay
read-only. Passing these kinematic checks does not establish dynamic trackability.

## BONES-SEED pilot preparation

Prepare human SOMA Uniform sources from the read-only BONES-SEED release,
retaining a previously verified LAFAN corpus unchanged:

```bash
uv run bello-prepare-bones source_directory=/data/bones-seed \
  output_directory=outputs/bones-mini-7h \
  lafan_corpus=outputs/lafan-mini/corpus \
  max_retarget_failure_fraction=0.01
```

The [public config](../configs/commands/prepare_bones.yaml) selects approximately
seven training hours and half an evaluation hour. Whole recordings are sampled
with equal duration budgets for ground transitions, jumps, dance, sports/combat,
locomotion, and other gestures/actions. Within each bucket, selection visits
different motion families before adding more takes of a family. The seed fixes
both selection and the train/evaluation assignment. Original and mirrored takes,
including different actors performing the same named content, share one split.
No upper duration cap, walking-only restriction, or blanket exclusion of floor,
roll, kneel, or recovery motions is applied. Explicit unsupported support objects
and terrain are excluded by recorded metadata terms; this is a Mini pilot
selection, not the paper's exact filtered G1 benchmark.

The command reads the configured metadata CSV and streams `soma_uniform.tar.gz`,
extracting only selected regular BVH members into the output directory. It
preserves the original source clock, resamples local transforms to 50 Hz, and
uses the shared canonical Mini calibration/contact refinement. SOMA joint axes
are removed using the release's `soma_base_skel_minimal.bvh` bind rig. They are
not assumed to match LAFAN axes: in particular the leg and toe frames differ.
BONES uses `retargeting.initialization_mode=constrained_neutral`: it fits the
first frame under contact constraints from the canonical neutral joint pose,
before capturing plantar anchors. This avoids asking a local projection to
untangle an already intersecting crossed-leg fit. Initialization does not
advance the source clock. LAFAN's default `fitted_projection` mode is unchanged.
Source BVH hashes, metadata and bind-rig hashes, selection reasons, contact
residuals, and physical validation reports are recorded. G1 joint CSVs are not
Mini targets. Kinematic validation does not certify dynamic feasibility.

`settings.json`, `selection.json`, and `preparation.jsonl` allow interruption and
resumption with the same configuration and source identities. Failed sequences
are retried. By default all must succeed before publishing the manifest.
An explicit `max_retarget_failure_fraction` can permit a small quarantine of
physical IK/contact failures, limited by both clip count and source duration
separately for training and evaluation. Other errors always abort preparation.
`quarantine.json` and manifest provenance retain every failure, source hash and
reason; no recording is silently dropped or trimmed. Previously successful
clips must still exist. The seven-hour pilot uses a 0.01 cap.
The resulting `corpus/` copies all original LAFAN clips and splits byte for byte,
adds `bones__` clips with family-based source identities, and verifies every
checksum. All its evaluation stems must be passed to training. Pin and publish
this corpus as a private immutable W&B dataset before a remote job. Inspect its
uncompressed array bytes against `dataset.max_array_bytes` and available device
memory before scaling beyond this pilot.

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

The [LAFAN profile](../configs/profiles/train/bello_mini_supermimic_lafan.yaml)
samples starts across the full recordings and caps collection episodes at 500
control steps (ten seconds). The cap is a timeout with a preserved successor and
curriculum bootstrap; failures remain terminal. Full-clip native evaluation
does not use this training cap. The current LAFAN profile uses network-only
`matmul_precision: bfloat16`; geometry and controller math keep their existing
precision. Parameters and replay stay float32. The `tensorfloat32` option
requests JAX `Precision.HIGH`; `float32` uses `Precision.HIGHEST`.
Per-update collection, dynamics fitting, and actor/critic timings are logged.
The LAFAN comparison profile uses the full unofficial batch sizes: 4,096
environments, 1,000 replay steps, ten dynamics batches of 2,048 windows, and
4,096 imagined starts. Its 143,622,144-transition target covers 64 warmup vector
steps and exactly 14,000 update cycles, including 500 dynamics-only cycles.
There is no profile wall-time cutoff. This matches the unofficial successful
checkpoint's update budget, not the paper's much larger BONES run.

For this comparison, command delay, feedback filtering, and observation latency
are disabled. Reset perturbations, calibration resampling, friction range,
ten-frame history/goals, optimizer schedule, and gradient clipping follow the
unofficial recipe. Mini geometry, actuator gains, the physical ankle transmission,
and the Mini-scale torso-COM range remain explicit departures.
This simulation baseline requires separate hardware robustness validation.

Native evaluation defaults to `evaluation_split=held_out`. To measure tracking
on training motions, explicitly select `evaluation_split=training`; selected
names must belong to the immutable manifest's training split. Reports label the
split and whether they measure held-out generalization. On Vast,
`evaluate_training_split=true` adds a separate numerical report for every
training motion under `results/training/`, alongside the usual held-out report.
Training rejects single-motion paths, pretrained inputs, payload overrides, and
`video_enabled`. Evaluate checkpoints separately for videos.

`bello_mini_supermimic_pilot` uses the full actor/dynamics architectures and
32-step imagination with 128 real environments, smaller optimizer/replay batches,
and 2,097,152 real transitions. Use it after the smoke check for initial GPU
memory, throughput, stability, and tracking measurements. It is a pilot recipe,
not a demonstrated convergence budget.

The full recipe uses 50 Hz control, four 5 ms substeps, zero command delay,
friction/calibration/torso-COM context and noisy feedback. Command delay is
rejected because the raw-action dynamics state does not include delayed motor
commands. Physical randomization ranges have not been established by Mini
system identification.

Simulation PD targets follow the reference controller:
`motor_from_joint @ (neutral_joint + calibration_offset + action_scale * action)`.
SuperMimic does not project these setpoints into position envelopes or slew-limit
them. MuJoCo joint constraints, actuator force limits and the ankle transmission
remain enabled. Reset poses still satisfy the canonical target limits.
`environment.control.action_clip` bounds exploration actions during collection
only. Imagination, deterministic evaluation and ONNX proposals preserve raw
policy actions, including their gradients outside that collection bound.

Collection adds Gaussian action noise and stores the actual clipped action,
previous/new motor targets for diagnostics, physical successor before reset, measured
observation before residual noise, filter state, reference clock/alignment, episode identity, context,
reward, and distinct terminal/timeout flags. Dynamics windows cannot cross
episode, clip, or context boundaries. Imagination reconstructs causal actor and
feedback histories and draws fresh residual sensor noise for imagined histories.
Missing history before the oldest replay record is padded with that record;
history never crosses a reset. Ineligible starts skip policy improvement with
a metric. Reset motor targets start at the constrained reset pose rather than
the neutral pose in collection and native evaluation.

At the default update-to-data ratio, each outer update collects alternating
two/three vector steps, fits ten dynamics batches, then updates actor/critic
after model warmup. Dynamics fitting autoregresses eight steps. Actor improvement
differentiates 32 imagined steps through fixed world-model and critic parameters,
preserving input derivatives. The dynamics inputs are invariant physical state,
raw action and normalized physical context; previous motor targets and sensor
latency are not dynamics inputs. The critic learns detached lambda-return targets.
Invalid optimization updates are rejected and stop the run. The learning-rate
schedule uses the outer-update clock. Total timesteps count real transitions.
Observation/goal moments update before collection actions. Dynamics moments
update once from each real transition, never from repeated replay samples.
Moments use `sqrt(variance + epsilon**2)` and freeze after their count reaches
200 million. The dynamics objective sums masked errors over the prediction
window and averages over sampled windows; logged loss is divided by window
length. This preserves the reference gradient scale before clipping.

Metrics include model/actor/critic losses, gradient norm, imagined reward,
completed episodes, failures, skipped policy updates, and
`collection/limited_fraction` for projection/slew intervention. Logging uses the
existing local/W&B path. Separate `collection/position_projected_fraction`,
`collection/rate_limited_fraction`, and `collection/torque_saturated_fraction`
count control steps with any affected motor. Per-motor fractions use physical
coordinate names under `motor/` and are emitted on update 1 and every 50 updates.
Projection compares requested and projected targets; rate limiting compares
projected and accepted targets. Saturation means sampled actuator force reaches
99% of the configured peak. Forces are sampled at the end of the control step;
these metrics do not capture every physics-substep peak or certify thermal safety.
Position-projection and slew-intervention fractions are zero for the direct PD
simulation controller; they do not measure physical joint-limit contacts.
Collection also logs `physics/max_constraints` and
`physics/contacts`; an `invalid/constraint_capacity` or
`invalid/contact_capacity` result stops training before further optimization.
Nonfinite actions, feedback, replay transitions, and normalization statistics
are also rejected during collection, before model fitting.
Reset sampling mixes uniform/difficult/historical
weighting over one-second training bins. Difficulty uses an EMA of actual
discounted episode returns attributed to the starting bin. At a clip timeout,
the episode's mean step reward estimates the discounted continuation; terminal
failures receive no continuation, including failures on the final frame. This
keeps successful short clip tails from appearing harder solely because they
contain fewer steps. With gamma equal to one, continuation uses the bank's
maximum episode length. Return estimates use alpha 0.2 starting from zero.
Historical weights use a 0.001 EMA on reset events. Simultaneous returns for the
same bin are averaged before the EMA, avoiding the unofficial duplicate-index
assignment ambiguity. Held-out clips never enter
learning or the curriculum.

Checkpoints include full learner/optimizer state, normalizers, replay, collector
and feedback memory, PRNG, sampling statistics, counters, corpus manifest, and a
smaller actor-only state. `dynamics.eqx` separately stores the world model and its
input/output normalizers, with its own SHA-256. This permits model audits without
allocating replay or restoring optimizers; it is not a continuation checkpoint.
Files are hashed, staged, fsynced, and published
atomically. Resume requires the same clean source commit, corpus identity, and
resolved learning/environment/network settings. Only the total timestep budget
and operational logging/checkpoint evaluation settings may change; the dataset directory
may move if its identity is preserved.
Historical checkpoints require their recorded runtime, including versions that
predate the dynamics snapshot. New code does not rewrite their identities.

Large checkpoint leaves transfer in row chunks of at most 16 MiB (or one row
when a row is larger), retaining Equinox's NPY stream format. Full replay and
collector continuation remain included. This bounds staging memory without
replacing an exact continuation checkpoint with a model-only snapshot.

The imagined recurrent loop computes only policy actions, learned dynamics, and
causal feedback. Goals/reference samples and independent goal encodings are
prepared before recurrence; reward
and critic heads run over the flattened trajectory afterward, sharing one FK
pass for predicted states. Reference body poses/twists use the verified corpus
cache on integer collection clocks, with the same XY/yaw alignment as physical
references. Sub-frame collection clocks compute FK after joint interpolation.
This preserves derivatives through predicted states and the critic bootstrap.
The short dynamics scan unrolls at most eight steps per loop iteration, exposing
the reference-sized prediction window to the compiler without changing its loss.
Checkpoint start/completion records include save and publication times so a
slow save is visible separately from training throughput.

All maintained SuperMimic recipes disable robot self-collisions, preserving
robot-ground contacts, to match the unofficial reproduction. This choice is
recorded in `asset_config.self_collisions` and applies to both generated models.
They inherit the Mini motion-profile stiffnesses with reduced damping for
hip roll and the ankle differential motors:

| Motor group | Kp | SuperMimic Kd | Motion-profile Kd |
| --- | ---: | ---: | ---: |
| Hip roll, each side | 140 | 6 | 9 |
| Ankle differential, each motor | 20 | 1 | 1.5 |

These task-specific overrides live in `asset_config.joint_gains` in the common
SuperMimic profile. Stiffness and its action scaling, torque limits, and all
other gains retain their inherited values. Position/rate command limits apply
to reset targets and the separate Rust hardware controller, not simulation PD
setpoints.
With the current unit-reduction differential, the effective ankle gains are
Kp 40 and Kd 2 per logical axis before saturation; pitch and roll still share
the two motors' torque budget. The physical reduction magnitudes require
calibration. The damping adjustment is a simulation training default, not a
hardware commissioning result. Native frozen-policy sensitivity checks do not
establish quality after retraining. Saved checkpoints retain their recorded gains;
training with changed gains requires a fresh run rather than resuming an old
world model and replay buffer under different dynamics.

They also disable command transport delay and feedback
filtering/latency, matching the reference comparison. The generic smoke now
exercises that same control path. A CUDA audit found nonfinite feedback with the
optional filtered hardware path; that configuration remains unvalidated for
hardware commissioning and is rejected at GPU environment initialization.
Its filter equations and the existing FK implementation
are unchanged. No global Linux memory policy or third-party engine changes are
part of the training implementation.

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
5. Previous transition action (`N`): exploration-clipped in collection and raw
   in imagination/evaluation.
6. Current aligned reference orientation relative to measured root (`6`).

History contains ten samples, oldest first; resets repeat the first valid
sample. Disable `observations.reference_orientation` explicitly for its ablation.
The separate goal concatenates ten future frames at 1-10 control steps. Each
frame contains displacement and orientation in the current reference heading
frame, future root-local linear/angular velocities, absolute logical joint
positions, and scaled joint velocities. Its size is `K*(15+2N)`. Futures clamp
to the selected clip end. Six-dimensional rotations flatten the first two
rotation-matrix columns in row-major order, matching the unofficial code:
`[R00, R01, R10, R11, R20, R21]`. Rust uses the same ordering; bundle metadata
requires `rotation_6d_layout: first_two_columns_row_major`. Historical policies
with the previous ordering require their original checkout.

Persisted running moments normalize history and goals. The actor's ELU motion
encoder produces a tanh latent, concatenated with normalized history for the ELU
decoder. The SiLU/LayerNorm world model consumes heading-invariant physical
state, raw actions, and physical context, predicting normalized root-local
increments. The privileged critic receives invariant state, joint-position
errors, heading-local re-anchored body-position errors, root position/orientation
errors, normalized context, and future goals. Actor history and previous motor
targets are not critic inputs. Collection and imagined rollouts share this
feature codec. The SuperMimic
profile does not route through shared PPO.

Commands use `neutral_joint + calibration + scale*action`, with
`scale=0.25*effort/stiffness` in logical joint order, transformed to motor
coordinates. Collection, imagination and native evaluation share this direct
mapping. The separate hardware controller ray-projects proposals into the
canonical envelope and applies connected motor step limits from the previous
accepted target. Previous action is the action used for the transition:
exploration-clipped during collection and raw during imagination/evaluation.
The Rust codec retains raw previous actions separately from bounded motor targets.

Tracking reward uses analytic FK: four body pose/twist exponential terms of
weight one, root pose terms of weight one-half, action-rate regularization, and
a soft joint-limit penalty. The sum is scaled by control timestep. Hard failures
use root height error, projected-gravity Z error, or nonfinite state. Imagination
uses smooth survival factors and stops bootstrapping at clip end. These settings
live in `tracking`; editing a profile does not change saved policies.

The comparison defaults match Appendix A of the paper and the pinned unofficial
implementation: absolute pelvis-height error above 0.25 m or an absolute
difference above 0.8 between the reference and actual root-frame gravity Z
components. The latter is dimensionless, corresponding to about 78 degrees of
tilt for an upright reference. End-effector error, horizontal root drift, and
joint RMS error are not termination gates. A full-clip evaluation fails at its
first hard termination; reaching the clip end is a successful timeout. The
10-second collection cap does not shorten benchmark recordings. See
[the reference termination code](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/mdp.py#L148-L162)
and [the reference evaluation loop](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/evaluate.py#L47-L83).

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
Video lighting brightens the floor and headlight so the canonical dark Mini
meshes remain visible. Native evaluation workers use at most four Linux CPUs
each to bound thread-pool contention during small-batch inference.

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
that denominator. Reports separate position projection, rate limiting, and
sampled torque saturation, including fractions for each physical motor.
`failure_event` records the first failed control step, reference frame, height
and projected-gravity errors, and triggered gates. Multi-seed learning curves,
and a matched PPO comparison remain experiments from the plan.

### Reference and world-model audits

These CPU commands write reports without modifying or rejecting dataset clips:

```bash
uv run bello-audit-reference motion_dataset=/data/mini \
  output_path=outputs/reference-dynamics.json

uv run bello-audit-world-model checkpoint_path=checkpoints/RUN/final \
  motion_dataset=/data/mini output_path=outputs/model-errors.json \
  windows='[{clip:jumps1_subject1,start_frame:11613,max_steps:400}]'
```

The reference audit uses `experiment=bello_mini_supermimic_lafan` by default,
verifies the manifest, and samples all clips every ten source frames. Set
`clips` or `frame_stride` explicitly for another selection. It reports physical
motor inverse torque against peak/catalog ratings, differentiated speed and
acceleration, unexplained free-root wrench, near-ground foot-point speed, and
COM distance to the available near-foot polygon. Proximity uses
`contact_distance_m=0.03`; sliding uses `slip_speed_m_s=0.2`.
Inverse forces include the configured MuJoCo passive and soft-contact law at the
literal reference pose. Small foot-clearance or acceleration changes can change
these estimates substantially. A nonzero root wrench or torque exceedance is a
diagnostic, not proof of infeasibility. The geometric polygon assumes quasi-static
support, and rated torque depends on duty and cooling. Missing polygons are
reported explicitly. The report records the clean source commit, verified corpus
hash, and audit settings.

World-model validation resets a frozen actor at each explicit source-frame
window and records deterministic native trajectories. It autoregresses 1, 8,
and 32 control steps using the recorded raw actions and physical context, with no
policy actions recomputed from predicted states. Reports include root pose/twist
and joint position/velocity errors in physical units, with mean, p95, maximum,
and coverage. Strata distinguish stable ground-contact signatures, any contact
change within the horizon, and wholly airborne windows. Missing coverage and
nonfinite predictions are explicit. `anchor_stride=32` and
`max_windows_per_horizon=32` bound computation. The same trajectories provide
tracking, motor intervention, and first-failure diagnostics. Window returns use
the planned window duration as denominator; they are distinct from full-clip
benchmark scores. Reports label source frames and recorded training/held-out
membership. This command requires a checkpoint containing `dynamics.eqx`.

### Automatic checkpoint checks

The LAFAN profile enables `checkpoint_evaluation` for four fixed 400-step dance,
jump, get-up, and sports windows from training recordings. The common and smoke
profiles leave it disabled. Enabled checks require explicit, unique windows and
checkpoint saving. They do not evaluate held-out motions during training.

Each saved checkpoint is queued to a separate CPU process, which measures window
tracking and model errors from the same trajectories without rendering video.
On Linux, `cpu_cores` (default 1, maximum 4) bounds affinity before JAX imports;
the worker also lowers its own scheduling priority. Checkpoints are processed
in update order; an identical periodic/final pair is evaluated once. Source,
corpus, actor, dynamics, and metadata identities are checked before execution.
Reports live under `checkpoint-evaluation/results/UPDATE/`. W&B uses a separate
linked run whose step is the checkpoint update, preserving delayed score timing.
Full-clip corpus evaluation remains a separate benchmark.

Training polls worker health every 50 updates and drains all queued checks after
the final checkpoint. A worker failure fails the run. Draining is bounded by
`finish_timeout_seconds=900`; timeout terminates the worker and fails explicitly.
A training failure cancels the worker. External Vast deadlines still apply to
the entire job, including evaluation. These are operational settings and may
change when resuming an otherwise identical training contract.

See the [measured motor and world-model audit](supermimic_implementation_audit.md)
for results from the existing trained checkpoint and all 49 reference clips.

### Reference learning checks

The optional `reference-validation` group supplies PyTorch for differential
checks. `tools/check_supermimic_reference.py` requires a clean vibed checkout at
`a1508d42c9d1699a52e8233ab16d08c488441ab9`, its prepared G1 MJCF and motion CSVs,
a matching `evaluate.py --record` trajectory, and a G1 checkpoint:

```bash
uv run --group reference-validation python tools/check_supermimic_reference.py \
  --reference-checkout /path/to/supermimic_vibed \
  --recording /path/to/heldout-trajectories.npz \
  --checkpoint /path/to/ckpt_014000.pt \
  --output outputs/reference-update-check.json
uv run --group reference-validation pytest -q tests/test_supermimic_precision.py
```

The checker uses recorded poses with differentiated velocities and recomputed
checkpoint actions. Each of three independent full-width cases starts from
identical weights/normalizers and fresh Adam states, then fits two dynamics
batches, differentiates 32 imagined steps, and trains the critic for five epochs
of four matched minibatches. Noise is disabled. It checks losses, network outputs,
Adam moments/counts and normalization statistics. Actor-output tolerance is
0.002 radians in G1 PD-target coordinates. Parameter differences are reported
separately because tiny gradient differences can be amplified by Adam near its
epsilon. The unit test also compares repeated optimizer steps using identical
supplied gradients, including tiny and zero gradients.

These are learning-arithmetic checks; they do not establish simulator equivalence
or training quality. Source/data hashes, versions and tolerances are recorded.

Export creates actor-only ONNX with frozen normalization, self-contained physics,
selected clips, named coordinate/IMU transforms, and history/goal/rotation/timing
metadata. ONNX emits raw action and proposed motor target; the runtime applies
final projection and slew limits. The exported codec retains raw previous
actions, while accepted hardware targets remain bounded. Bundle metadata uses
`simulation_control: unprojected_pd` and `collection_action_clip`; the collection
bound does not clip deterministic actor outputs. No world model or critic is
deployed. Evaluation in the training simulator and bounded hardware playback are
different control contracts and require separate commissioning.

Raw-action dynamics changes the model shape and learning contract. Start a fresh
training run; historical checkpoints require their recorded clean checkout.
Existing checkpoint files are not converted or overwritten.

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

## Full AMASS run

The broader follow-up uses `bello_mini_supermimic_amass` with immutable dataset
`gyungmin/bello-mujoco/mini-supermimic-amass-8h-20261022:v0`, manifest SHA-256
`f83d855b410b245bb3e86e28bc14bd3969838f71a336a3f79bfe3266d615c3fc`.
All 4,710 parent clips passed current Mini joint/motor, speed, and terrain checks.
The bank retains 4,166 training clips (28,807.76 s) and 544 held-out clips
(2,880.62 s), with disjoint inherited subject and source identities. Canonical
tracked link-origin twists were recomputed for every frame. This is a fresh
training run; its corpus differs from the six-clip pilot below.

The [full-bank training run](https://wandb.ai/gyungmin/bello-mujoco/runs/0n2msvmr)
uses source `913d5a6417c294d0086b7774ccc17ebd527f810f` and seed zero. It retains
the full networks and 32-step imagination, increases collection to 1,024
environments, and uses two model batches of 512 and imagined batches of 512.
The RTX 4090 has 24,564 MiB of memory and a quoted rate of $0.6024/hour, below
the authorized $0.75/hour cap. The entire remote job has a two-hour deadline.

Training stopped at its explicit 4,500-second wall budget, saving the complete
final state after 6,088 outer updates, 12,176 dynamics optimizer steps, 5,588
actor/critic updates, and 15,650,816 real transitions. The transition ceiling was
16,777,216; the earlier wall-time cutoff preserves time for the full benchmark
and artifact upload. W&B records 4,503.34 seconds including finalization. All
update validity checks passed. The full final checkpoint is
`gyungmin/bello-mujoco/supermimic-amass-8h-20261022-seed0_20261022_095807_004219:v6`.

The final deterministic native benchmark covered every held-out clip from frame
zero: all 544 completed their full duration, totalling 144,031 control steps.
Equal-per-clip normalized return was 0.7713, root position RMS 0.325 m, root
orientation RMS 6.70 deg, and joint position RMS 0.178 rad. Projection or slew
limits intervened on 5.55% of steps on average. Mean peak actuator effort ratio
was 0.9943. Completion measures reaching the clip end; return and pose errors
measure tracking quality.

| Held-out collection | Completed clips | Normalized return | Root position RMS |
| --- | --- | --- | --- |
| BMLmovi | 57/57 | 0.8123 | 0.200 m |
| BMLrub | 43/43 | 0.6981 | 0.309 m |
| CMU | 26/26 | 0.7525 | 0.483 m |
| EKUT | 81/81 | 0.8451 | 0.185 m |
| KIT | 69/69 | 0.7798 | 0.347 m |
| MoSh | 3/3 | 0.7674 | 0.136 m |
| WEIZMANN | 265/265 | 0.7514 | 0.377 m |

The eight-clip suite was fixed before training and covers walking, circular
walking, jogging, turning, bending, one-leg motion, and two other full-body
sequences. Its matched learning measurements are:

| Outer update | Real transitions | Mean completion | Normalized return |
| --- | --- | --- | --- |
| 1,000 | 2,625,536 | 26.3% | 0.0813 |
| 3,000 | 7,745,536 | 100% | 0.6638 |
| Final (6,088) | 15,650,816 | 100% | 0.7252 |

The [full evaluation run](https://wandb.ai/gyungmin/bello-mujoco/runs/svthtghw)
persisted reports for all 544 clips, the eight selected videos, the self-contained
ONNX bundle, checkpoint metadata, and evaluation provenance in
`gyungmin/bello-mujoco/supermimic-amass-8h-20261022-seed0-compiled-evaluation:v0`.
The checkpoint, actor, full state, corpus manifest, driver, and bundle identity
checks passed after download. The final bundle passed Rust validation and
64-frame JAX/Rust telemetry parity: maximum observation/action difference
`2.87e-6`, accepted motor-target difference `9.54e-7 rad`, using ONNX Runtime
1.23.2.
The [analysis run](https://wandb.ai/gyungmin/bello-mujoco/runs/tzkhttm1) preserves
learning curves, the captioned eight-motion overview, collection-level results,
parity checks, integrity checks, and teardown evidence in
`gyungmin/bello-mujoco/supermimic-amass-8h-20261022-seed0-analysis:v0`.

The recorded training source predates the CPU evaluation optimization. Its
benchmark used a separately recorded compiled driver against the unchanged
source APIs. Two complete clips matched the original scorer's episode counters
and failure flags, with maximum return difference `1.40e-7` and root-position
RMS difference `2.86e-8 m`. The driver SHA-256 is
`7009bd711dac08bbe7b200700ddc8f86657153aeb79c631cfae54fd189b02057`.
The duplicate slow scorer was paused; the compiled benchmark, export, and upload
finished in 1,148.61 seconds. The source commit and checkpoint metadata were
preserved throughout. Restore this historical policy in a clean checkout at
`913d5a6` with its frozen dependencies; the artifact includes its benchmark driver.

Training peak device memory was 9,099 MiB. Instance `57190867` was manually
destroyed after both artifacts were collected and verified; Vast inventory
confirmed its absence. The two-hour job deadline and host teardown supervisor
remained armed. Rental wall time was 111.19 minutes; the quoted-rate estimate
was $1.12, excluding any provider billing adjustments.

The final CPU gate passed 902 tests with five expected skips. Lockfile, protobuf,
Ruff, native visual/headless parity, and fixed-contact-buffer physics checks
passed. This is one seed with nominal evaluation dynamics. Longer training,
multi-seed robustness, a matched PPO comparison, target-device timing, and
physical commissioning remain unmeasured. The lowest-return held-out clip,
`amass__CMU__144__144_05_stageii`, scored 0.444 with 0.840 m root-position RMS;
the aggregate completion rate leaves meaningful tracking errors to improve.

## Tracking audit

The October 23 audit found two implementation discrepancies relative to the
[unofficial source at a1508d4](https://github.com/zita-ch/supermimic_vibed/tree/a1508d42c9d1699a52e8233ab16d08c488441ab9).
The original paper does not supply official implementation code; matching the
unofficial source is a separate check from matching the paper.

First, the return curriculum compared raw discounted returns from variable-length
episodes with an infinite-horizon maximum of 10. Even perfect one-step, one-second,
and ten-second episodes scored 0.1, 3.9499, and 9.9343 respectively. Normal clip
endings therefore looked difficult. In the final checkpoint, the last-second
bins comprised 13.52% of eligible bins but received 43.53% of reset probability
and 76.83% of the current difficulty component. The collector now continues
timeouts at their mean step reward when scoring the curriculum, matching the
unofficial implementation's bootstrap. Actual failures receive no continuation.
This changes curriculum statistics, not recorded rewards or terminal flags.

Second, tracking rewards aligned reference bodies using the difference of two
independently extracted headings. The existing BeyondMimic path and unofficial
implementation instead extract the yaw of the relative root rotation. These
operations differ when the two roots have different tilts. SuperMimic now uses
the latter expression, with a regression case covering tilted roots.

The historical checkpoint and normalizers restored exactly in their unchanged
`913d5a6` source checkout. Native/model probes on three walking states showed
reasonable short-horizon predictions away from reset, but mixed action-gradient
agreement that depended on the finite-difference scale and contact state. These
probes do not establish a model-gradient implementation bug. A separate native
counterfactual changing only the initial motor target from neutral to reference
also had mixed results, so reset control initialization remains unchanged.

Training scale remains a separate concern. The bounded AMASS profile used
1,024 model windows per outer update versus 20,480 in the full recipe, and 512
imagined starts versus 4,096. Its learning-rate decay used 10,000 updates instead
of 250,000. The completed run contains 6,088 outer updates. These differences
and the curriculum defect are plausible contributors to weak tracking; their
individual effect requires retraining and ablation. Completion alone remains
insufficient evidence of good tracking.

Broader source preselection on the NAS found 13,991 AMASS candidates (48.75 h)
when removing the upper duration cap and blanket activity exclusions, compared
with 11,657 (25.50 h) under the previous source rules. This is source selection,
not a validated retarget corpus. Explicit terrain/object-supported sources remain
excluded from this flat-ground candidate pass. BONES-SEED human SOMA BVHs are
available separately from its G1 CSVs. Exploratory Mini retargets covered six
LAFAN sequences and eight BONES sequences. Some BONES probes contain solver
startup spikes, sustained high speeds, or floor penetration; they are diagnostic
outputs, not training-ready references. Source archives remain unchanged.

Local evidence is under `outputs/supermimic-audit-20261022/`: checkpoint and
curriculum audits, native/model and reset counterfactuals, LAFAN archive inventory
and exact 40-clip source cohort, broader AMASS candidate list, and exploratory
BVH retarget reports. Validation passed 906 default CPU tests plus the native
Mini rollout integration case; the CUDA case skipped on this CPU host. Lint,
formatting, lock, and protobuf checks passed. The retraining below measures the
resulting policy; separate ablations would be needed to isolate each fix.

## Corrected-source AMASS retraining

The [retraining run](https://wandb.ai/gyungmin/bello-mujoco/runs/0hr2s0md)
started from scratch at source `d3b3bb7a96a062cb3c21b08f73edfc415fe59eac`,
using the same immutable eight-hour AMASS corpus, held-out identities, seed zero,
and bounded learning profile as the previous run. All learning, environment,
observation, reward-configuration, network, sampling-configuration, Warp, and asset
settings compare equal. The two audited implementation fixes change behavior.
LAFAN/BONES motions were not added to this controlled comparison.

Training reached its 16,777,216-transition ceiling after 6,528 outer updates,
13,056 dynamics optimizer steps, and 6,028 actor/critic updates. W&B records
3,770.77 seconds, below the 4,500-second training limit. The prior run stopped at
6,088 updates on its wall budget, so the new final policy also has 7.2% more real
transitions. All update-validity checks passed.

The complete native benchmark evaluated the same 544 held-out clips from their
first frame. All completed in both runs; metrics are equal-per-clip means.

| Metric | Previous | Corrected-source retraining |
| --- | --- | --- |
| Normalized return | 0.7713 | 0.7868 |
| Root position RMS | 0.325 m | 0.295 m |
| Root orientation RMS | 6.70 deg | 6.59 deg |
| Joint position RMS | 0.1780 rad | 0.1700 rad |
| Projection/slew intervention | 5.55% | 4.44% |

Joint RMS improved on 441/544 clips and root position RMS on 334/544. These are
modest aggregate improvements. The fixed eight-motion suite remains weak:
normalized return changed from 0.7252 to 0.7377, root RMS from 0.360 to 0.359 m,
and joint RMS from 0.2342 to 0.2348 rad. Its final pose errors are essentially
unchanged. At the matched 3,000-update checkpoint, both policies completed all
eight motions; root RMS improved from 0.472 to 0.354 m and joint RMS from 0.2811
to 0.2751 rad. Normalized-return comparisons also include the yaw-alignment
reward correction; pose-error definitions are unchanged. This is one seed, not
an isolated ablation or a reproduction of the paper's G1/BONES experiment.

The curriculum defect is measurably removed: final reset probability assigned
to last-second bins fell from 43.53% to 7.26%, against their 13.52% uniform mass.
Their share of the difficulty component fell from 76.83% to 0.71%. Every training
bin was visited. Both saved replay buffers contain zero held-out transitions.
Removing this sampling bias did not resolve the poor tracking on the hard suite.

The final checkpoint is
`gyungmin/bello-mujoco/supermimic-amass-fixed-20261023-seed0_20261023_070415_582836:v6`;
the complete report, eight videos, and deployment bundle are
`gyungmin/bello-mujoco/supermimic-amass-fixed-20261023-seed0-evaluation:v0`.
Actor/full-state/corpus hashes, checkpoint source, all evaluation identities,
video presence, and deployment-bundle source were verified after download.
The bundle passed Rust validation and 64-frame JAX/Rust telemetry parity using
ONNX Runtime 1.23.2: maximum observation/action difference `3.15e-6`, and accepted
motor-target difference `1.07e-6 rad`.

Local comparison outputs are under `outputs/supermimic-retrain-20261023/`.
`reference-before-after.mp4` synchronizes exact kinematic target poses, the prior
policy, and the retrained policy for all eight preselected motions. It starts
each column at reference frame one, matching the first simulated policy frame.
`before-after-comparison.png` covers all 544 clips; `learning-comparison.png`
covers the fixed suite at matched intermediate updates and each final policy.
The [comparison analysis](https://wandb.ai/gyungmin/bello-mujoco/runs/bojicw1z)
is archived as
`gyungmin/bello-mujoco/supermimic-amass-fixed-20261023-seed0-analysis:v1`,
including the verified 50 Hz comparison video, plots, sampler audit, configuration
comparison, native parity, and teardown evidence.

Vast instance `57335092` used an RTX 4090 at $0.4770/hour and completed under the
two-hour job limit. Automatic host teardown was verified by instance absence at
08:22:03 UTC on October 23. The observed rental interval was 90.1 minutes,
approximately $0.72 at the quoted rate, excluding transfer/billing adjustments.
The historical `913d5a6` checkout and a separate clean `d3b3bb7` evaluation
checkout remain available; checkpoint compatibility guards were preserved.

## First CUDA pilot

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

## Validated LAFAN corpus

The complete 40-motion cohort was prepared at 50 Hz without activity or duration
exclusions: 2.4504 training hours, plus nine sequence-disjoint evaluation motions
(0.4043 hours). All 49 passed floor, self-contact, coupled position, joint speed,
and clip-format checks. Worst floor penetration was zero; worst self-penetration
was 10.278 mm, within the explicit 10 mm allowance plus 1 mm numerical tolerance.
The maximum joint speed/URDF-bound ratio was 1.0000015.

Immutable dataset: `gyungmin/bello-mujoco/mini-supermimic-lafan40-20261024:v0`.
Manifest SHA-256:
`ed5a5c36cf1408ba16695c95ed98bb89688a91e73800da7842416e47812228c3`.
The corpus contains retargeted references, not evidence of learned skill.

## Broader LAFAN retraining

The bounded LAFAN run used clean source
`3f029897a2a695a08be2a43ba7b2659099dd480f`, the dataset above, and seed zero.
An RTX 5090 completed 8,210 dynamics/critic update cycles, 7,710 actor updates,
and 42,166,272 real simulation transitions in 4,506.42 seconds. This used 2,048
environments, five dynamics batches of 1,024 windows, 1,024 imagined starts,
32-step imagination, ten-second collection episodes, and network-only TF32.
The [training run](https://wandb.ai/gyungmin/bello-mujoco/runs/fbf7it95) records
the full configuration. This has different data and training settings from the
earlier AMASS runs and is not a controlled before/after performance comparison.

Every requested full recording received deterministic native CPU evaluation.
The training-set and held-out reports remain separate; the latter holds out
source sequences, not subjects.

| Full recordings | Training (40) | Held out (9) |
| --- | ---: | ---: |
| Mean normalized return | 0.4526 | 0.1184 |
| Mean completion fraction | 0.7190 | 0.2387 |
| Complete recordings | 19/40 | 0/9 |
| Mean root-position RMS | 4.841 m | 1.455 m |
| Mean joint-position RMS | 16.35 deg | 28.75 deg |

Four eight-second training windows were selected before actor training, using
maximum wrist speed, maximum root height, or the largest two-second root rise
from below 0.3 m. Their results expose failures hidden by aggregate completion:

| Active window | Evaluated duration | Joint RMS | Root RMS |
| --- | ---: | ---: | ---: |
| Dance | 3.32/8 s; failed | 14.4 deg | 0.69 m |
| Jump | 2.92/8 s; failed | 17.1 deg | 0.32 m |
| Get up | 8/8 s | 29.4 deg | 1.45 m |
| Fight/sports | 8/8 s | 19.0 deg | 1.85 m |

Completion means avoiding the height/tilt termination thresholds. It does not
establish accurate imitation. Complex tracking remains poor in this run.
The 32-second comparison shows the entire target windows and labels failed
policy segments while holding their last simulated frame.

All 8,842 eligible training time bins were visited, with 254,071 completed
collection episodes. None of the 1,048,576 saved replay transitions came from
the held-out split. All 513,900 reference frames fit the static action range;
the largest required normalized action was 15.88 against a clip of 20.
At exact reference poses, the soft joint-limit cost reduces normalized score
by approximately 2-5% before action-rate penalties. These checks do not prove
dynamic feasibility. Both that learner and the unofficial implementation gave
future goals to the critic. That run's critic additionally received actor history and
previous motor targets, while omitting the unofficial critic's 42 per-body
position-error features. This run did not isolate the effect of those differences.

All 932 CPU tests, formatting, lint, lock, and generated-protocol checks passed.
The exported four-reference bundle passed Rust validation and 256-frame
prerecorded inference parity: maximum observation/action error `3.64e-6`,
accepted motor-target error `1.67e-6 rad`, with ONNX Runtime 1.23.2.

Immutable artifacts:

- Checkpoint: `gyungmin/bello-mujoco/supermimic-lafan-20261024-seed0_20261024_054340_125916:v8`.
- Full evaluations: `gyungmin/bello-mujoco/supermimic-lafan-20261024-seed0-evaluation:v0`.
- Active windows: `gyungmin/bello-mujoco/supermimic-lafan-20261024-seed0-active-evaluation:v0`.
- [Analysis, graphs, video, and deployment bundle](https://wandb.ai/gyungmin/bello-mujoco/runs/7vokuael):
  `gyungmin/bello-mujoco/supermimic-lafan-20261024-seed0-analysis:v0`.

Instance `57471104` was automatically destroyed; Vast inventory confirmed its
absence at 07:19:08 UTC on 2026-10-24. The observed rental interval was 113.71
minutes at $0.72222/hour, approximately $1.37 before billing adjustments.
Restore that checkpoint from a clean checkout at `3f02989`; the local
`outputs/supermimic-lafan-20261023/training-runtime` preserves that exact source.

## Reference alignment after the LAFAN result

The poor broader-motion result above prompted a fresh comparison against
unofficial source `a1508d42c9d1699a52e8233ab16d08c488441ab9`. The learner now
aligns window-loss reduction, real-transition normalization timing, normalizer
epsilon/count limits, critic body errors, context scaling, reset targets and
perturbations, fresh imagined-history noise, replay padding, rotation layout,
and return/historical sampling EMAs. The LAFAN profile restores the full batches
and 14,000-update target described above. These are verified implementation
differences; their individual effects on learned tracking have not been isolated.

The replacement dataset is
`gyungmin/bello-mujoco/mini-supermimic-lafan40-central-reference-20261024:v0`,
with manifest SHA-256
`be67dbd47dc6b3535c96f2743c2ae7fe2c77f217e5ca619ccac49d249d3b738b`.
All 513,900 poses across the same 40 training and nine additional evaluation
recordings are byte-identical to the previous corpus. Velocities and cached
link twists were regenerated with central differences. The prior artifact
remains unchanged.

A differential CPU fixture executes the pinned upstream goal/observation,
reward/termination, critic-feature, dynamics-loss, and imagined-step functions.
It uses the same synthetic 29-joint/14-body skeleton on both sides, reduced
network widths, copied weights, and disabled observation noise. Extra Mini
motor-memory/latency inputs have zero weights for this comparison. Across 22
checks, maximum absolute differences were `1.19e-7` for critic features,
`7.63e-6` for dynamics loss, `2.87e-6` for a dynamics weight gradient, `6.68e-6`
for the 32-step actor objective, and `2.73e-12` for an encoder weight gradient.
This checks shared mathematical behavior, not simulator equivalence or tracking
quality on Mini.

Validation passed 941 default CPU tests, seven native/ONNX/Rust integration
checks, the Rust rotation-layout unit test, formatting, lint, lock and generated
protocol checks. The CUDA integration case remains pending a GPU host. The new
full-budget run has not started; the results above belong to the earlier code.
