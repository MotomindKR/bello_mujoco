# Training and evaluation

The complete public fields are in the
[command reference](commands.md#bello-train). Nested PPO, environment, reward,
asset, and output fields are in the
[training configuration reference](training_configuration.md).

## Profiles

`bello-train` exposes four profiles:

```text
smoke
locomotion
hardware_locomotion
motion
```

`smoke` runs a real, minimal MJX-Warp PPO update. `locomotion` is the canonical
gait baseline. `hardware_locomotion` uses the same reward and policy contract
with the longer credit horizon used by the physical policy. `motion` is bounded
BeyondMimic tracking.

The public command can override seed, total timesteps, environment count,
motion path, resume path, checkpoint destination/name, W&B enablement/run name,
and video enablement. Other task numerics are maintained in
`configs/profiles/train/`; arbitrary nested profile keys are not part of the
public command.

## Locomotion baseline

The physical profile is a 60 kg sphere-hand robot with 13 policy joints and 12
held arm joints. Commands contain forward, lateral, and yaw velocity. Ten
percent of sampled commands are exactly zero; moving samples are conditioned
outside the shared standing deadbands.

The canonical reward combines command tracking, base stability, normalized
torque costs, standing foot motion, contact timing, touchdown air time, and
termination cost. Exact terms and weights are defined in
[`configs/train/default.yaml`](../configs/train/default.yaml) and explained in
the [training configuration reference](training_configuration.md#locomotion-reward-reward_config).
Training leaves policy effort unclipped so the over-peak torque penalty observes
excess demand; the emulator and hardware enforce physical limits.

## Evaluation

`bello-eval` always uses native MuJoCo. Locomotion streams the scene, commands,
actions, rewards, and termination diagnostics to Rerun. Motion evaluation
renders a deterministic video and reports completion and RMS tracking errors.

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

There is no automatic cross-revision adaptation. Checkpoint-producing training
requires a clean `HEAD` contained by a local `origin/*` tracking
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
