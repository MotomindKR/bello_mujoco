# Evaluation Details

This document covers live policy evaluation through the custom MuJoCo/ImGui
viewer. For the compact repository reference, see [index.md](index.md). For
checkpoint creation and PPO training behavior, see [training.md](training.md).

## Entry Point

`bello-eval-policy` is the live evaluation entry point. It restores a training
checkpoint, rebuilds a single-environment Bello runtime, and runs the policy in
an interactive CPU MuJoCo viewer.

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

Evaluation is interactive inspection only:

- it opens the MuJoCo/ImGui policy viewer
- it runs one environment
- it does not write MP4 files
- it does not log W&B artifacts
- it can show status, command, reward, action, and control panels

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
  stochastic_actions: false
  action_seed: 0
  camera_config: CameraConfig
  reward_overlay: RewardOverlayConfig
  action_overlay: ActionOverlayConfig
  joystick: JoystickConfig
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

`stochastic_actions=false` evaluates the actor mean. Set it true to sample from
the PPO Gaussian policy and clip the sampled action to the env action range.
`action_seed` makes sampled eval actions reproducible.

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.stochastic_actions=true \
  eval_config.action_seed=123
```

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

The viewer draws ImGui status and reward panels over the MuJoCo render and
updates them every displayed control step. It shows:

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
`position` and `font_scale` place and scale the status/reward panels.
`max_reward_terms` limits how many component rows are shown.

Disable the overlay:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.reward_overlay.enabled=false
```

### ActionOverlayConfig

```yaml
action_overlay:
  enabled: true
  max_actions: 13
  bar_width: 12
```

The live viewer action panel shows policy-action diagnostics as text bar
graphs. Each action row shows `m` for the raw actor mean and `a` for the clipped
action applied to the environment. Bars are scaled to the env action clip.

Summary rows:

```text
action_mean_abs_max
action_abs_max
ctrl_saturation
```

`ctrl_saturation` is the fraction of policy-controlled actuator targets at or
beyond their MuJoCo control-range limits. Set `max_actions` lower if the overlay
is too tall.

### JoystickConfig

```yaml
joystick:
  enabled: true
  lin_vel_x_step: 0.05
  lin_vel_y_step: 0.05
  yaw_vel_step: 0.05
  lin_vel_x_limit: [-1.0, 1.0]
  lin_vel_y_limit: [-1.0, 1.0]
  yaw_vel_limit: [-1.0, 1.0]
```

The live viewer includes a keyboard joystick for command-conditioned policies.
The command panel shows sliders for `cmd_x`, `cmd_y`, and `cmd_yaw`; the same
limits are used by the keyboard joystick.

Controls:

- `W` / `S`, or arrow up / down: increase/decrease `lin_vel_y`
- `A` / `D`, or arrow left / right: decrease/increase `lin_vel_x`
- `Q` / `E`: increase/decrease yaw velocity
- `Space` or `0`: reset command to zero
- `P`: pause/resume rollout stepping
- `N`: single-step while paused
- `R`: reset the current episode

For Bello locomotion, forward walking commands usually use `lin_vel_y`.
Disable keyboard command control with:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.joystick.enabled=false
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
critic_obs_size
action_size
```

Live eval does not allocate the MJX/Warp model, and the CLI defaults
`JAX_PLATFORMS=cpu` for policy inference unless the environment already sets a
different JAX platform. `warp config` is retained here only because it is part
of checkpoint metadata and shared runtime config.

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
changes actor observation size, critic observation size, or `action_size`,
evaluation fails before opening the viewer. Actor observation size is
`env.observation_size * metadata.ppo.policy.obs_stack_size`; critic observation
size uses `env_config.critic_observation` when configured and otherwise matches
the actor size. Because overrides are detected by comparing against repo
defaults, use a new matching checkpoint when you need to change a runtime field
back to its default value from a non-default checkpoint value.

## Runtime Behavior

Interactive evaluation validates checkpoint compatibility against the simulation
MJCF, then launches a separate display environment from the viewer MJCF. Both
models are loaded without MJX because live eval is an interactive CPU MuJoCo
rollout. The display env has the same policy-facing joints and actuators, plus
visual meshes and transparent collision geoms for display.

Each viewer loop iteration:

1. Reads the current CPU MuJoCo `MjData`.
2. Computes the deterministic policy mean action and clipped env action.
3. Updates the configured MuJoCo camera.
4. Draws red foot overlays when that foot has raw terrain contact.
5. Handles keyboard joystick, mouse camera, and ImGui panel input.
6. Renders the MuJoCo scene and ImGui panels through GLFW/OpenGL.
7. Steps the single CPU MuJoCo environment once with `mujoco.mj_step`, unless paused.
8. Optionally sleeps to match real time.

By default the policy uses the Gaussian mean action for deterministic
inspection. With `eval_config.stochastic_actions=true`, or with the
`Stochastic` checkbox enabled in the Controls panel, the viewer samples from the
actor Gaussian each control step before action clipping.

Rendered rollout videos use the same red foot-contact overlays before each
frame is written.

For stacked-observation policies, the viewer uses the checkpoint policy stack
size. When an episode ends, evaluation resets the env and rebuilds the stack
from the reset observation. With `stop_on_done=false`, the viewer keeps running
into the next episode; with `stop_on_done=true`, the viewer exits after the
first done.

## Checkpoint Compatibility

`bello-eval-policy` restores both policy parameters and checkpoint metadata.
The policy shape is recreated from:

```text
metadata.ppo
metadata.obs_size
metadata.critic_obs_size
metadata.action_size
```

After the eval runtime is rebuilt, it validates:

```text
env.observation_size * metadata.ppo.policy.obs_stack_size == metadata.obs_size
critic observation stack size == metadata.critic_obs_size, when present
env.action_size == metadata.action_size
```

This means config changes that alter observations, held/policy actuators, asset
body selection, or action dimensions need a matching checkpoint.

## Validation And Assert Conditions

Evaluation runtime loading inherits environment/model validation:

- `env_config.root_link` and configured reward/done link names must exist
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
