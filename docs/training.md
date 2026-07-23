# Training

`bello-train-ppo` is the only supported training entry point. It builds the
prepared Bello model, a MuJoCo/MJX-Warp environment, an Equinox actor-critic,
and the local PPO update loop from one typed Hydra configuration.

## Maintained recipes

Training recipes live under `experiments/train/`. The maintained set is:

```text
smoke.yaml
locomotion.yaml
```

Each recipe inherits `configs/train/default.yaml` directly. This keeps package
defaults in one place and makes every experiment-specific change visible in
the selected recipe.

Run the lightweight end-to-end smoke first:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name smoke
```

Run the maintained locomotion baseline with:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion
```

The standard recipe uses one seeded sloped plane per run and the validated
conservative PPO learning-rate and exploration settings. Randomized root yaw
varies the incline direction in the robot frame. Every
prepared Bello model uses the physical differential ankle transmission. The
policy interface remains logical joint space; the control stack maps ankle
pitch and roll targets into the tendon motor coordinates.

Hydra overrides follow the recipe name:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion \
  config.seed=2 \
  video_config.enabled=false
```

Use `--help` and `--hydra-help` for the current config tree. Unknown config keys
are rejected during typed config conversion.

## Dry run and execution

The package default has `dry_run=true`. A dry run prepares the asset, builds the
environment and policy, resets a vectorized batch, checks shapes and config
contracts, and exits before optimization. Maintained experiment recipes set
their intended `dry_run` value explicitly.

Set `dry_run=false` to optimize. `jit=true` compiles the training function with
JAX. The real training path uses batched MJX-Warp stepping; native CPU MuJoCo is
used only by optional video and evaluation paths.

## Environment contract

The actor emits normalized logical joint-position requests. Action zero is the
configured nominal target including its per-world target offset, constrained to
the logical soft limits. Negative and positive actions are scaled independently
so `-1` reaches the lower logical soft limit and `+1` reaches the upper logical
soft limit. For each control step, the environment:

1. bounds the policy request according to the environment action contract;
2. maps each side of zero to the corresponding logical soft-limit interval;
3. clamps logical targets defensively to that configured range;
4. applies transmission transforms such as the differential ankle mapping;
5. clamps actuator-coordinate targets defensively; and
6. advances the configured number of physics substeps.

Policy and held actuators are disjoint. Held joints remain impedance-controlled
at their configured defaults but do not appear in the actor action vector.
Per-world target-offset randomization applies to all policy and held joints by
default. Set
`env_config.model_randomization.joint_angle_offset_scope=policy` to perturb only
policy-controlled targets; joint-origin and inertia randomization remain
physical model variations independent of that target scope.

The policy distribution bound and environment action bound are one contract.
Training rejects configurations where `config.policy.action_limit` differs from
`env_config.action_clip`; otherwise PPO would assign probabilities to requests
different from the actions actually applied by the environment.

This limit mapping happens before transmission transforms. In particular,
ankle pitch and roll actions span their logical joint soft limits, then the
differential transform maps those targets to the two tendon motor coordinates.
Generated tendon actuator ranges cover the full logical pitch/roll rectangle.

The environment can expose separate actor and critic observations. The actor
stream should contain only deployable observations for the intended robot
runtime; the critic can receive additional clean or privileged state during
training. Actor observation noise is optional, while privileged critic noise is
rejected so the critic stream stays explicitly clean. PPO stacks each stream
according to the policy configuration and resets an environment's complete
stack when that environment resets.

Episode-length timeouts are truncations. They reset the environment but PPO
bootstraps from the pre-reset observation. Physical failure terminations cut
bootstrapping.

The reward surface is the canonical locomotion set exposed by `RewardConfig`.
Every term has a numeric `weight`; a zero weight is the only disabled state.
Recipes must not add parallel `enabled` flags. This keeps the resolved reward
definition auditable from one value per term.

Every weighted reward component is multiplied by the control timestep,
including contact-timing, touchdown air-time, and terminal events. Reward
weights therefore have one consistent per-second convention. The scalar reward
remains signed; penalties are never hidden by clamping the total to zero.
Locomotion is phase-free: neither actor/critic observations nor rewards contain
an externally prescribed gait phase. Alternation must emerge from velocity,
contact, air-time, and regularization terms.

Command sampling is an explicit mixture rather than an unconditioned uniform
box. Ten percent of new commands are exactly `[0, 0, 0]`. Every remaining
sample is conditioned so at least one linear or yaw component reaches its
configured minimum moving speed, `0.2 m/s` or `0.2 rad/s` by default. This
prevents near-zero draws from silently increasing the effective standing
fraction. The shared command definition classifies a command as standing only
when both linear axes are inside `linear_deadband` and yaw is inside
`yaw_deadband`; tracking and gait rewards use that same classification.

The default locomotion reward penalizes squared vertical base velocity with
weight `-2.0`. A phase-free `flight` penalty with weight `-2.0` applies while a
moving command or measured horizontal/yaw motion is active and both filtered
foot contacts are absent. Touchdown air-time credit is retained at weight
`0.5`, low enough that synchronized hopping cannot dominate the flight and
vertical-motion costs.

Linear and yaw velocity tracking both use weight `1.0`. Each command axis below
its command-config deadband tracks zero with a tighter standard deviation:
`0.25` for planar velocity and `0.2` for yaw rate. Commanded axes use standard
deviations `0.75` and `0.5`, respectively. This per-axis formulation strongly
penalizes lateral drift during forward walking and residual motion while
standing without narrowing the commanded-speed objective. Giving yaw the same
maximum per-step credit as planar translation prevents the optimizer from
cheaply sacrificing commanded rotation for translational tracking.

The general nominal-pose penalty has weight `-0.1`. A separate
`stand_foot_motion` penalty with weight `-0.25` activates only for a standing
command. For each foot it applies
`tanh(||linear_velocity||^2 / 0.15^2)` plus `0.25 *
tanh(||angular_velocity||^2 / 1.0^2)`, then averages over the feet. It uses full
3D foot velocity, has no contact mask, and is bounded, so it discourages
stepping and foot jitter without making an emergency recovery step more
expensive than a fall. There is no dual-contact reward and no joint-velocity
standing penalty.

Contact-timing and touchdown air-time are enabled only by a moving command.
Measured residual motion cannot turn these gait-shaping terms on at zero
command. Contact timing penalizes touchdown before `0.2 s` of air time and
liftoff before `0.3 s` of contact; touchdown air-time gives capped credit for a
swing longer than `0.2 s`. Flight remains responsive to measured motion, and
foot slip remains active for every command, preserving safety pressure during a
zero-command recovery without paying the policy to continue stepping.

Policy torque regularization is normalized per actuator by the configured peak
effort, rather than summing raw newton-metres across unequal motors. The
`torque_l1` value is `mean(abs(torque) / peak_torque)` with weight `-1.0`. A
separate `torque_peak` value is
`mean(max(abs(torque) / peak_torque - 1, 0)^2)` with weight `-10.0`, making an
over-peak excursion much more expensive than ordinary sub-peak effort. The
training asset keeps each configured `forcerange` as peak-torque metadata but
sets `forcelimited=false` on policy actuators, so the optimizer observes and is
penalized for requested over-peak force instead of having it hidden by physics
clipping. Non-policy hold actuators remain force-limited.

Dense stability feedback includes pelvis-height error with weight `-10.0` and
target height `0.86`, plus torso orientation with weight `-2.0`. Physical failure
termination has weight `-100.0`, which contributes `-2.0` at the default 50 Hz
control rate. These values use the same timestep scaling as every other term.
Reset placement evaluates the local terrain and collision geometry after pose
randomization, then aligns the lowest sole to `spawn.clearance`.

## PPO loop

The optimizer is a local PureJaxRL-style implementation using Equinox and
Optax. The JAX hot path includes:

- vectorized reset and MJX-Warp stepping;
- policy sampling and rollout collection with `jax.lax.scan`;
- generalized advantage estimation;
- clipped policy and value losses;
- entropy regularization;
- shuffled minibatch updates; and
- guarded optimizer state updates.

The loop uses a bounded tanh-Gaussian policy, explicit environment observation
scales, fixed or scheduled learning rates, adaptive learning rate from the
maximum measured minibatch KL, and a hard target-KL guard for the remaining
minibatches in an update. It does not update observation or value-normalization
statistics online, because doing so would change the represented policy or
critic outside PPO's trust-region update.

PPO retains the exact sample needed for policy-ratio calculations while the
environment separately reports the bounded action and applied control. This
keeps optimization probability calculations distinct from actuator safety.

Nonfinite training data, gradients, optimizer updates, or parameters are not
applied. Configured magnitude guards additionally reject implausibly large
observations, rewards, values, targets, policy samples, and gradients. The
training command raises instead of continuing with a poisoned optimizer state.
Environment-side bounds also terminate finite physics excursions in position,
velocity, acceleration, actuator force, or generalized constraint force before
their state can continue into another rollout step.

## Metrics

Training emits one metric row per PPO rollout update. W&B uses environment
steps as its step axis. Metric groups cover:

- actor, critic, entropy, KL, and optimizer behavior;
- policy samples, applied actions, and clipping;
- logical-target and actuator-coordinate saturation;
- reward components and termination reasons;
- completed-episode return and length;
- command sampling and resampling behavior; and
- contact, stance, and gait transitions.

`command/standing_fraction` and `command/moving_fraction` expose the observed
rollout occupancy. Gait metrics also include conditional `gait/standing_*` and
`gait/moving_*` values, including footfalls per second, so zero-command stepping
can be distinguished from commanded locomotion.

Compare treatments using the metrics that describe the intended behavior, not
total return alone. Reward totals are not comparable when reward definitions or
weights differ.

## Checkpoints

Checkpointing writes one directory containing:

```text
metadata.json
state/
continuation/
```

`state/` contains policy, optimizer, and completed-update state for evaluation.
`continuation/` contains the environment batch, actor and critic observation
stacks, rollout RNG, and adaptive learning rate needed for an exact
update-boundary resume. Metadata records the complete training, asset,
environment, task, observation, and action contracts. The checkpoint also
fingerprints the repository source assets, prepared URDF/MJCF dependency
closure and referenced meshes, exact simulation/viewer terrain scenes, Python
implementation, `uv.lock`, and the active transitive runtime distribution
versions. Training resume rejects every mismatch. Locomotion evaluation and
hardware deployment may use a newer runner implementation while retaining all
other runtime checks. Training captures this contract before its first update,
and every checkpoint save rejects source,
asset, lock, or environment drift that occurred while the process was running.
The project and lock files required for verification are included in built
wheels, so checkpoint operations do not require a source checkout.
The host-specific asset `output_root` is normalized out of metadata.
Checkpoints are versioned and strict; stale schemas and partially matching
training configs are rejected.

Set `checkpoint_config.every_updates` to write atomic periodic snapshots. They
are immutable siblings of the final checkpoint with an `_update_XXXXXX`
suffix. Resume one explicitly with the identical recipe and overrides:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion \
  checkpoint_config.resume_path=checkpoints/<snapshot> \
  checkpoint_config.every_updates=512
```

Resume continues at the next PPO update; it does not replay the completed
rollout. A checkpoint whose update already equals the configured run length is
complete and cannot be resumed.

When checkpoint artifact upload and W&B are both enabled, periodic snapshots
and the successful final checkpoint are uploaded as W&B artifacts. Local
checkpoint and W&B output are generated artifacts and must not be committed.

## W&B and video

W&B integration records the resolved run config and update metrics. It is a
no-op when disabled. Vast.ai training forces online W&B, checkpointing, and
checkpoint artifact upload so the rented instance can be destroyed without
losing the successful result.

Optional training video runs outside the JAX optimization path. It evaluates
the current in-memory policy in a single environment and renders with MuJoCo.
Video can run periodically and at the end of training, write an MP4, and log it
to W&B. The native renderer does not load MJX, and FFmpeg is launched with
`posix_spawn` so video encoding cannot fork the multithreaded JAX process.

## Validation

Training configuration is validated before optimizer construction. Validation
includes positive rollout dimensions, minibatch divisibility, valid schedules
and scalar ranges, policy architecture and distribution settings, actuator
partitioning, model metadata, observation/action dimensions, named links,
command ranges, terrain/contact requirements, and checkpoint/video settings.

Use the smoke recipe for any change that affects assets, simulation, controls,
observations, rewards, or PPO. Then run the focused automated tests and the full
gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

For remote training, follow [vast_ai.md](vast_ai.md). For checkpoint playback,
follow [evaluation.md](evaluation.md).
