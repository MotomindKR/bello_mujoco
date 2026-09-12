# Training and evaluation

The complete public fields are in the
[command reference](commands.md#bello-train). Nested PPO, environment, reward,
asset, and output fields are in the
[training configuration reference](training_configuration.md).

## Profiles

`bello-train` exposes eight maintained experiments:

```text
smoke
bello_locomotion
bello_mini_locomotion
bello_mini_standing
motion
twist_teacher
twist_student
twist2
```

`smoke` runs a real, minimal MJX-Warp PPO update. `bello_locomotion` and
`bello_mini_locomotion` select their respective robot contracts after composing
the same production PPO, command, randomization, and observation recipe. The
Mini profile changes its asset, rounded height thresholds, exact-pose reset,
tracking tolerances, gait shaping, and output names,
adds non-saturating moving linear and yaw
velocity-error shaping, and raises the healthy-survival weight from `1.5` to
`2.25` so the dense penalties do not erase the early survival margin. Force and
impact terms scale automatically by robot weight or actuator limit. The selected
robot contract supplies its planar tracking frame and crouched joint defaults.
Their periodic and final evaluation videos command 0.5 m/s forward motion.
Mini actor feedback uses the physical IMU position and the published `imu_link`
axes; command and reward velocity remain pelvis-relative. New Mini runs must be
trained with this convention. Existing pelvis-origin checkpoints retain their
saved convention and are not repaired by changing the training profile.
`motion` is bounded BeyondMimic tracking; the TWIST experiments are documented
separately. The
shared `configs/train/locomotion.yaml` layer is internal and is not a public
experiment name.

`bello_mini_standing` adds a standing-only overlay to the complete Mini
locomotion profile. It keeps the same action, command, and gait-observation
layout, but fixes every command range to zero and sets standing probability to
one, making gait probability exactly zero. Moving-only and gait-pattern reward
weights are disabled; zero-velocity tracking, standing balance, smoothness,
termination, randomization, and pushes remain active.

Train the Mini production profile with:

```bash
uv run bello-train experiment=bello_mini_locomotion
```

Train the compatible standing-only policy with:

```bash
uv run bello-train experiment=bello_mini_standing
```

The public command can override seed, total timesteps, environment count,
motion path, resume path, checkpoint destination/name, W&B enablement/run name,
and video enablement. Other task numerics are maintained in
`configs/profiles/train/`; arbitrary nested profile keys are not part of the
public command.

## Locomotion recipe

Native locomotion evaluation and periodic/final training videos preserve the
saved observation low-pass filter, noise-before-filter and residual noise, and
per-episode observation delay. Feedback advances once per policy step and is
reseeded on episode reset. Re-reading a frame to update a gamepad command does
not advance the filter or redraw noise. Observation randomness uses the saved
`env_config.model_randomization.seed`, independently of `action_seed`; deterministic
actions do not disable configured observation noise. This evaluates the saved
feedback model. Mini also applies its saved 10 ms motor-command transport delay
at physics-substep resolution; this is not a complete CAN scheduling simulation.

The Bello profile uses 13 policy joints and holds 12 arm joints. Bello Mini uses
14 policy joints (waist yaw, waist roll, and 12 leg joints) and holds its 14 arm
joints. Commands contain pelvis-frame `[lin_vel_x, lin_vel_y, ang_vel_yaw]`.
Bello forward is pelvis `+Y`; Bello Mini forward is pelvis `+X`. Gamepad and
hardware-remote inputs remain semantic forward/right/yaw intent and are mapped
through that robot-specific axis contract. Twenty-five percent of batched
environments form an exact-zero standing cohort; moving samples are conditioned
outside the shared standing deadbands.
The standing-only Mini profile instead assigns every environment to the exact-zero
cohort. Its command and gait-phase observation slots remain present and are all
zero, including after startup, periodic resampling, and failure reset.
Failure resets preserve the exact command request. Command resampling uses a
separate control-step clock that survives episode resets, so repeated early
failures cannot trap an environment in one mode. Initial clock ages are evenly
staggered across the batch, so only a small fraction resamples on each control
step instead of changing the whole cohort every ten seconds. Each resample can
train a transition in either direction between standing and moving commands.

Moving commands activate a per-leg gait clock: 0.90 Hz for Bello and
`(1.6 + 0.35 * norm(command_xy)) / 2` Hz for Mini. The legs are offset by half a cycle and use a 60%
stance fraction, which provides short double-support intervals. The actor
receives sine/cosine phase and a gait-active bit. Moving training resets sample
uniformly across the full gait cycle to keep parallel environments phase-diverse.
Mini native evaluation and hardware starts select the same-side swing leg for
lateral commands, or alternate starting preference for straight commands.
In baseline Bello, once gait is active, zero is a
standing request rather than an immediate policy command. The effective moving
command remains active until the next scheduled half-cycle boundary, even when
the request arrives during double support. The step that reaches the boundary
still receives gait rewards; the following policy step observes a zero command
and inactive gait component and begins receiving standing rewards. Environments
initialized in standing start there directly.

Mini extends that baseline stop sequence by one more physical step. Its command
decelerates while the clock remains active through two half-cycle boundaries,
then standing rewards ramp in over 0.2 seconds. Its new standing rewards align
sole fore/aft position and nominal width; signed separation discourages crossing
in every mode. Phase-shaped toe/heel clearance and unexpected mid-swing contact
penalties address scuffing. Alignment/stillness pressure relaxes during recovery.
Recovery also relaxes soft hip/knee/waist posture and hip-roll thermal pressure;
single-support corrective steps retain standing velocity-tracking rewards.
Hard motor limits are never relaxed. Mini retains 25% of commands for standing
and ramps random pushes over the first approximately 1B transitions. Paired
walking and standing-push evaluation videos expose recovery failures separately
from aggregate training reward. These changes require a newly trained policy;
they do not establish push robustness on their own.
See [the formulation and configuration](training_configuration.md#locomotion-reward-reward_config).
Mini's reference-aligned profile uses `0.5 m/s` planar and `0.5 rad/s` yaw
tracking tolerances in both standing and moving components. It starts with zero
floating-base velocity and retains torso COM randomization at `+/-0.15 m` per
axis, with the existing curriculum retained. Reward weights, 25% standing
sampling, sensor noise/filter/latency, and hard limits are unchanged. The
[reference comparison](training_configuration.md#mini-reference-alignment)
documents which Unitree/Booster choices were adopted and which were not.
The Mini profile halves horizontal command ranges to +/-0.5 m/s per axis and
enforces the configured actuator effort limits. These changes require fresh
training; they do not repair existing policy weights. Validate phase-swept stops,
left/right starts, reversals, turns, and pushes before a long training run or
deployment. No new trained-policy performance is implied by unit-test success.

The canonical reward combines healthy survival, command tracking, base
stability, normalized torque costs, moving-only body-weight-normalized lower-body
joint reaction-force peaks, standing foot motion, phase contact matching,
stance-foot slip, swing clearance, bounded substep-integrated touchdown impact,
and termination cost. Uprightness and orientation termination are measured on
`torso_link`; pelvis velocity and base height remain separate measurements.
During healthy double support, standing frames also apply a bounded physical
foot-heading cost relative to the robot command frame. Its 5-degree
deadband and `0.01` per-step cap discourage learned toe-in without overwhelming
balance acquisition or constraining commanded turns. Both maintained robot
profiles disable the optional left/right actor-consistency and action
acceleration losses; action smoothness uses the original uniform action-rate
cost rather than a stateful command filter.
Yaw-rate tracking uses a 5 Hz low-pass-filtered measurement. Exact terms and
weights are defined in
[`configs/train/default.yaml`](../configs/train/default.yaml), with maintained
overrides in
[`configs/train/locomotion.yaml`](../configs/train/locomotion.yaml), and are explained in
the [training configuration reference](training_configuration.md#locomotion-reward-reward_config).
Mini keeps its coordinated flat-sole joint pose fixed during reset instead of
scaling each default joint independently. Bello Mini's canonical 43.50 kg
inertials automatically feed the body-weight-normalized force and impact terms;
those reward weights are not manually rescaled. Mini separately uses midpoint
gait shaping (`gait_contact: -2.5`, `gait_swing_clearance: -1.0`, and a `0.075` m
swing-clearance target) to discourage persistent double support. It also adds a
moving-only smooth-L1 planar velocity-error penalty with weight `-1.0` and a
`0.25 m/s` Huber delta. Unlike the Gaussian tracker, this retains a signed,
nonzero corrective gradient when velocity is far from or opposite the command.
The rounded height settings are a 0.87 m pelvis target and 0.56 m termination
floor, versus 0.86 m and 0.55 m for Bello. Bello preserves the baseline's
unclipped policy-effort dynamics; Mini saturates torque at its configured motor
limits during training. The over-peak torque term still uses the
reconstructed servo demand to expose infeasible policy requests. Emulator and
hardware execution continue to enforce physical motor effort limits.

## Evaluation

`bello-eval` always uses native MuJoCo. Locomotion streams the scene, commands,
actions, rewards, foot-heading error, and termination diagnostics to Rerun.
Motion evaluation renders a deterministic video and reports completion and RMS
tracking errors.

Locomotion gamepad input is enabled by default. With a connected controller it
replaces the configured fixed command; with a disconnected controller the
command is zero. Disable gamepad input to apply a fixed command. An unbounded
`steps=null` rollout continues until controller exit or interruption, and
completed episodes reset unless `stop_on_done=true`.

Motion evaluation rejects non-default locomotion command/gamepad/stochastic
fields. Its default step horizon is the checkpoint environment's maximum
episode length.

There is no public MJX evaluation backend. This keeps evaluation behavior
aligned with the native emulator and physical deployment.

## Checkpoints

Checkpoints store:

- the full Git commit that created it;
- resolved task and training metadata;
- policy architecture/distribution;
- actor, critic, and action layouts;
- motion content and its named data layout when applicable;
- normalization, optimizer, environment continuation, RNG, and sampler state.

There is no general automatic cross-revision adaptation. With an explicit
commit-mismatch override, evaluation and hardware loading remove the retired
generated-hand field and disabled selectors for joints now fixed by the
canonical URDF. Enabled retired selectors are rejected, and resume remains
strict. Checkpoint-producing training requires a clean `HEAD` contained by a
local `origin/*` tracking
branch and rechecks that revision immediately before every checkpoint is
published. The root commit pins `deps/bello-stack`; its recursive submodules
must be initialized, clean, and at their recorded commits. Fetch or push before
a local run so that tracking state is current. Evaluation, resume, and hardware
execution reject a dirty root or submodule worktree. They also reject a
different `HEAD` unless `allow_git_commit_mismatch=true` is explicitly set. That
override bypasses root-commit equality and provenance-only source-revision
identity; metadata semantics, state layout, task, motion content, submodule
integrity, and runtime validation remain active. It cannot load older
checkpoint envelopes that do not contain `git_commit`.

Locomotion checkpoints created before the gait-phase observation contract are
also rejected explicitly. Their historical 51-value Bello actor frame cannot
be adapted to the current 54-value Bello frame without retraining. Bello Mini's
14 policy joints produce a separate 57-value actor frame and checkpoint layout.

Checkpoint output must be outside the source repository or covered by a Git
ignore rule. The default `checkpoints/` directory is ignored. A checkpoint name
is one path component, so output cannot escape the validated parent.

Checkpointed motion training additionally requires the selected archive to be
tracked in `HEAD`; its checkpoint path is stored relative to the repository so
the exact checkout is portable across machines.

The default generated-asset cache is content-addressed by the preparation
configuration, generator code and runtime versions, and canonical URDF and mesh
content. Relevant source changes therefore receive a different cache key. The
manifest verifies the canonical generated XML and referenced mesh content, and
runtime terrain scenes are regenerated deterministically instead of trusted
across launches.

Regardless of whether the commit override is used, evaluation validates
document integrity, metadata and state-layout hashes, task and
actor/critic observation and action sizes/layouts, normalization, saved update,
motion content, and restored state. Resume additionally requires environment
continuation, observation stacks, RNG, current learning rate, sampler state,
environment layout, and the full resume contract. A revised `total_timesteps`
value is accepted when its resulting update budget still covers the saved
update; output destinations may change.

Locomotion final checkpoints are timestamped checkpoint roots. Motion runs also
maintain a `final` child below their timestamped root.
TWIST teacher and student runs likewise write `final` children, but their stage
types are not interchangeable. TWIST metadata additionally binds the packed
dataset, Bello model, joint/body order, observation contract, and source
revision. Student metadata binds the exact teacher metadata fingerprint and
corrected KL schedule. Public TWIST evaluation accepts only the student and
reports deterministic global, per-source, and per-clip held-out metrics.

Read the recorded revision without loading policy state:

```bash
jq -r .git_commit checkpoints/<checkpoint>/metadata.json
```

The recommended workflow is to run it from a separate checkout:

```bash
CHECKPOINT="$(cd checkpoints/<checkpoint> && pwd -P)"
COMMIT="$(jq -r .git_commit "$CHECKPOINT/metadata.json")"
REPOSITORY="$(git remote get-url origin)"
CHECKOUT="$(cd .. && pwd -P)/bello_mujoco-${COMMIT:0:12}"
git clone --recurse-submodules "$REPOSITORY" "$CHECKOUT"
git -C "$CHECKOUT" switch --detach "$COMMIT"
git -C "$CHECKOUT" submodule update --init --recursive
cd "$CHECKOUT" && uv sync --frozen
```

Reference the absolute `$CHECKPOINT` path and use that revision's documented
`bello-eval` or `bello-run-hardware` command. Keep each checkpoint commit
reachable through a protected remote ref or retain a source bundle beside the
artifact; a deleted or force-pushed commit cannot be reconstructed from the
checkpoint alone.

The Git commit is provenance, not a cryptographic signature over policy
weights. Checkpoint state is shape-checked but is not authenticated, so only
load artifacts from a trusted store.

Checkpoints created before embedded Git provenance remain tied to their
historical source and must be run from the revision identified by their
training run or artifact name; they are not converted or stamped
retroactively.

`bello-train` disables JAX GPU-memory preallocation and selects JAX's
`cuda_malloc_async` allocator unless the operator has explicitly configured
`XLA_PYTHON_CLIENT_PREALLOCATE` or `TF_GPU_ALLOCATOR`. MJX-Warp and JAX allocate
device memory independently; the async allocator lets unused JAX allocations
be reclaimed instead of leaving Warp's transient collision workspace starved
even when monitoring still reports apparent free memory.
