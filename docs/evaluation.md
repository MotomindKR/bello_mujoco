# Evaluation Details

This document covers live policy evaluation through MuJoCo's GUI viewer. For
the compact repository reference, see [index.md](index.md). For checkpoint
creation and PPO training behavior, see [training.md](training.md).

## Entry Point

`bello-eval-policy` is the live evaluation entry point. It restores a training
checkpoint, rebuilds a single-environment Bello runtime, and runs the policy in
MuJoCo's interactive viewer.

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

Evaluation is interactive inspection only:

- it opens the MuJoCo GUI viewer
- it runs one environment
- it does not write MP4 files
- it does not log W&B artifacts
- it can show reward values in a viewer text overlay

## Full Config Tree

Hydra maps nested dataclass fields to dotted overrides. For example:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  'eval_config.command=[0.1,0.0,0.0]' \
  eval_config.camera_config.mode=follow \
  eval_config.reward_overlay.position=top_left
```

The complete `bello-eval-policy` root tree is:

```yaml
checkpoint_path: Path
eval_config: EvalPolicyConfig
```

### EvalPolicyConfig

```yaml
eval_config:
  command: [0.0, 0.0, 0.0]
  steps: null
  realtime: true
  stop_on_done: false
  camera_config: CameraConfig
  reward_overlay: RewardOverlayConfig
  runtime: EvalRuntimeConfig
```

`command` is `[lin_vel_x, lin_vel_y, yaw_vel]` and is passed to the initial
single-environment reset.

`steps=null` means run until the viewer window closes. Setting an integer stops
after that many control steps.

`realtime=true` sleeps between control steps using `env.control_timestep`.
`realtime=false` runs as fast as the viewer loop allows.

`stop_on_done=false` keeps the viewer open after terminal states. Set it true
when debugging done conditions and you want evaluation to stop immediately on
the first done.

### CameraConfig

```yaml
camera_config:
  mode: follow  # follow | fixed
  distance: 3.0
  azimuth: 135.0
  elevation: -18.0
  lookat_offset: [0.0, 0.0, 0.35]
```

`follow` mode tracks body `bello_root`. `fixed` mode keeps the configured free
camera orientation without updating `lookat`.

### RewardOverlayConfig

```yaml
reward_overlay:
  enabled: true
  position: top_right  # top_left | top_right | bottom_left | bottom_right
  font_scale: small  # small | normal | large
  precision: 3
  max_reward_terms: 12
  show_components: true
```

The overlay uses MuJoCo viewer text overlays and updates every displayed control
step. It shows:

```text
step
reward
done
episode_return
episode_length
base_height
reward components
```

Reward components are weighted `reward/*` terms from the environment info dict.
`max_reward_terms` limits how many component rows are shown.

Disable the overlay:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.reward_overlay.enabled=false
```

### EvalRuntimeConfig

```yaml
runtime:
  source: checkpoint  # checkpoint | override
  asset_config: AssetPrepConfig
  env_config: BelloEnvConfig
  reward_config: RewardConfig
  command_config: CommandConfig
  done_config: DoneConfig
  warp: WarpConfig
```

`source=checkpoint` is the default. It reconstructs the runtime from checkpoint
metadata:

```text
asset config
reward config
command config
done config
env config
warp config
obs_size
action_size
```

`source=override` merges nested runtime configs from the eval CLI onto the
checkpoint metadata. Fields that still equal the repo defaults keep their
checkpoint values; fields that differ from defaults replace the checkpoint
values. This is useful for inspection changes such as terrain, done behavior,
or reward overlay debugging.

Example terrain override:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.runtime.source=override \
  eval_config.runtime.env_config.terrain.mode=rough \
  eval_config.runtime.env_config.terrain.seed=3
```

Override mode must preserve checkpoint policy compatibility. If the rebuilt env
changes policy observation size or `action_size`, evaluation fails before
opening the viewer. Policy observation size is
`env.observation_size * metadata.ppo.policy.obs_stack_size`. Because overrides
are detected by comparing against repo defaults, use a new matching checkpoint
when you need to change a runtime field back to its default value from a
non-default checkpoint value.

## Runtime Behavior

Each viewer loop iteration:

1. Syncs the current MJX state into a MuJoCo `MjData`.
2. Updates the configured viewer camera.
3. Updates the reward overlay text.
4. Calls `viewer.sync()`.
5. Computes the deterministic policy mean action.
6. Steps the single environment once.
7. Optionally sleeps to match real time.

The policy uses the Gaussian mean action for deterministic inspection rather
than sampling stochastic actions.

## Checkpoint Compatibility

`bello-eval-policy` restores both policy parameters and checkpoint metadata.
The policy shape is recreated from:

```text
metadata.ppo
metadata.obs_size
metadata.action_size
```

After the eval runtime is rebuilt, it validates:

```text
env.observation_size * metadata.ppo.policy.obs_stack_size == metadata.obs_size
env.action_size == metadata.action_size
```

This means config changes that alter observations, held/policy actuators, asset
body selection, or action dimensions need a matching checkpoint.

## Validation And Assert Conditions

Evaluation runtime loading inherits environment/model validation:

- configured reward/done link names must exist
- enabled foot rewards require foot bodies and foot collision geoms
- enabled terrain-contact rewards require a runtime terrain geom when terrain is enabled
- policy and held actuator indices must be in range and disjoint
- actuator qpos/qvel metadata must match MuJoCo `nu`
- default qpos/qvel/ctrl shapes must match the MuJoCo model

Eval-specific validation:

```text
runtime.source in {"checkpoint", "override"}
command has exactly 3 values
steps is null or positive
camera distance is positive
reward overlay precision is nonnegative
reward overlay max_reward_terms is nonnegative
policy observation_size matches checkpoint metadata
action_size matches checkpoint metadata
follow camera requires body "bello_root"
```

## Required Smoke Commands

The GUI viewer is interactive, so it is not part of the headless test gate.
For eval code changes, run the non-GUI checks:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-eval-policy --help
uv run pytest tests/test_eval_policy.py -q
```

When a local display is available, manually inspect a checkpoint:

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```
