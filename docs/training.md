# Training

`bello-train-ppo` is the only supported training entry point. It builds the
prepared Bello model, a MuJoCo/MJX-Warp environment, an Equinox actor-critic,
and the local PPO update loop from one typed Hydra configuration.

## Maintained recipes

Training recipes live under `experiments/train/`. The maintained set is:

```text
smoke.yaml
locomotion.yaml
locomotion_plane_slope.yaml
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

The standard recipe uses independently randomized plane slopes. Use
`locomotion_plane_slope` for the same terrain treatment with the longer-run
optimizer settings. Every prepared Bello model uses the physical differential
ankle transmission. The policy interface remains logical joint space; the
control stack maps ankle pitch and roll targets into the tendon motor
coordinates.

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

This limit mapping happens before transmission transforms. In particular,
ankle pitch and roll actions span their logical joint soft limits, then the
differential transform maps those targets to the two tendon motor coordinates.
Generated tendon actuator ranges cover the full logical pitch/roll rectangle.

The environment can expose separate actor and critic observations. The actor
stream should contain only deployable observations for the intended robot
runtime; the critic can receive additional clean or privileged state during
training. PPO stacks each stream according to the policy configuration and
resets an environment's complete stack when that environment resets.

Episode-length timeouts are truncations. They reset the environment but PPO
bootstraps from the pre-reset observation. Physical failure terminations cut
bootstrapping.

The reward surface is the canonical locomotion set exposed by `RewardConfig`.
Every term has a numeric `weight`; a zero weight is the only disabled state.
Recipes must not add parallel `enabled` flags. This keeps the resolved reward
definition auditable from one value per term.

## PPO loop

The optimizer is a local PureJaxRL-style implementation using Equinox and
Optax. The JAX hot path includes:

- vectorized reset and MJX-Warp stepping;
- policy sampling and rollout collection with `jax.lax.scan`;
- generalized advantage estimation;
- clipped policy and value losses;
- entropy regularization;
- shuffled minibatch updates; and
- optimizer and normalization state updates.

The loop uses a bounded tanh-Gaussian policy and supports actor and critic
observation normalization, optional value-target normalization, fixed or
scheduled learning rates, adaptive learning rate from measured KL, and a hard
target-KL guard for the remaining minibatches in an update.

PPO retains the exact sample needed for policy-ratio calculations while the
environment separately reports the bounded action and applied control. This
keeps optimization probability calculations distinct from actuator safety.

Nonfinite losses, gradients, updates, or parameters are not applied. The
training command raises instead of continuing with a poisoned optimizer state.

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
stacks, rollout RNG, adaptive learning rate, and value-normalization state
needed for an exact update-boundary resume. Metadata records the complete
training, asset, environment, task, observation, and action contracts.
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
to W&B.

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
uv run pytest
```

For remote training, follow [vast_ai.md](vast_ai.md). For checkpoint playback,
follow [evaluation.md](evaluation.md).
