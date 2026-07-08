# Training Details

This document covers training behavior and invariants. For the compact config
tree, see [index.md](index.md).

## Entry Points

`bello-train-ppo` is the training entry point. It builds:

- `AssetPrepConfig`
- `BelloEnvConfig`
- `RewardConfig`
- `CommandConfig`
- `DoneConfig`
- `WarpConfig`
- `PPOConfig`
- optional `CheckpointConfig`
- optional `WandbConfig`
- optional `TrainingVideoConfig`

`dry_run=true` is the default. In dry-run mode, the command constructs the env,
initializes the policy, resets a vectorized batch, prints shape/config metadata,
and exits without optimization.

`dry_run=false` runs PPO. `jit=true` is the default and wraps the train function
with `jax.jit`.

## Hydra Configs

Training configs live under `configs/train/`. The default config is used when no
Hydra config name is provided:

```bash
bello-train-ppo
```

Use the measured RTX 5090 preset with:

```bash
bello-train-ppo --config-name rtx5090_4096x256
```

That preset uses:

```yaml
config.num_envs: 4096
config.num_steps: 256
config.num_minibatches: 8
config.update_epochs: 4
```

The resulting rollout batch is `1048576` samples and the PPO minibatch size is
`131072` samples. Override any field using Hydra dotlist syntax:

```bash
bello-train-ppo \
  --config-name rtx5090_4096x256 \
  dry_run=true \
  config.total_timesteps=1048576 \
  config.seed=1
```

## Full Config Tree

Hydra maps nested dataclass fields to dotted overrides. For example:

```bash
bello-train-ppo \
  config.num_envs=512 \
  env_config.terrain.mode=rough \
  reward_config.track_lin_vel_xy.weight=1.2 \
  done_config.base_height.min_height=0.75
```

The complete `bello-train-ppo` root tree is:

```yaml
config: PPOConfig
asset_config: AssetPrepConfig
reward_config: RewardConfig
command_config: CommandConfig
done_config: DoneConfig
env_config: BelloEnvConfig
checkpoint_config: CheckpointConfig
wandb_config: WandbConfig
metric_config: TrainingMetricConfig
video_config: TrainingVideoConfig
warp: WarpConfig
dry_run: true
jit: true
```

### PPOConfig

```yaml
config:
  seed: 0
  total_timesteps: 1000000
  num_envs: 64
  num_steps: 128
  update_epochs: 4
  num_minibatches: 4
  learning_rate: 0.0003
  anneal_lr: true
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  policy:
    hidden_size: 256
    hidden_depth: 2
    activation: tanh  # tanh | relu
    init_log_std: -0.5
    obs_stack_size: 1
```

Derived values:

```text
batch_size = num_envs * num_steps
num_updates = total_timesteps // batch_size
minibatch_size = batch_size // num_minibatches
```

### AssetPrepConfig

```yaml
asset_config:
  asset: full_body  # full_body | lower_body | upper_body
  output_root: null
  mesh_face_budget: 1000
  foot_collision_mesh_face_budget: 1000
  visual_mesh_face_budget: 200000
  collision_mode: mesh  # mesh | foot-mesh | box
  end_effector_sphere_mode: none  # none | both | right | left
  neighbor_exclusion_depth: 2
  actuator_config:
    default_action_scale_ratio: 0.25
    joints:
      <joint_name>:
        mode: policy  # policy | hold
        default_pos: 0.0
        action_scale: null
        soft_limit: null  # [lower, upper] | null
        gains: [40.0, 2.0]  # [kp, kv]
```

The default actuator config contains every tracked actuated Bello joint. Setting
a joint to `mode: hold` removes it from the policy action vector while keeping
its default control target in the environment.

### BelloEnvConfig

```yaml
env_config:
  decimation: 10
  max_episode_steps: 1000
  terrain:
    mode: plane  # none | plane | rough
    name: ground
    pos: [0.0, 0.0, 0.0]
    size: [20.0, 20.0, 0.05]
    rgba: [0.45, 0.45, 0.45, 1.0]
    friction: [1.0, 0.005, 0.0001]
    condim: 3
    seed: 0
    hfield_resolution: 65
    hfield_amplitude: 0.04
    hfield_base_thickness: 0.01
    hfield_frequency: 2.5
  gait:
    cycle_time: 0.80
    duty_factor: 0.55
    phase_offsets: [0.0, 0.5]
    standing_command_threshold: 0.05
  observation:
    joint_mode: all_actuated  # all_actuated | policy
    base_ang_vel: true
    projected_gravity: true
    gait_phase: true
    command: true
    joint_pos: true
    joint_vel: true
    actuator_force: true
    last_action: true
```

`joint_mode` controls `joint_pos`, `joint_vel`, and `actuator_force`. The command
observation is emitted only when both `env_config.observation.command` and
`command_config.enabled` are true.

`gait` defines a deterministic two-foot contact schedule. `cycle_time` is the
period in seconds, `duty_factor` is the stance fraction of each period,
`phase_offsets` are per-foot phase offsets, and
`standing_command_threshold` forces all feet scheduled in contact when command
speed is near zero. `observation.gait_phase=true` appends `[sin(phase),
cos(phase)]` to the policy observation.

### CommandConfig

```yaml
command_config:
  enabled: true
  lin_vel_x: [-0.1, 0.1]
  lin_vel_y: [-0.1, 0.1]
  ang_vel_yaw: [-0.1, 0.1]
  standing_probability: 0.02
  resample_seconds: 10.0
```

Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`. In vectorized stepping,
`resample_seconds` is rounded to control-step count and clamped to at least one
step.

### RewardConfig

Every reward term has `enabled` and `weight`. Link-aware terms also have `link`.

```yaml
reward_config:
  track_lin_vel_xy:
    enabled: true
    weight: 1.0
    link: bello_root
    sigma: 0.25
  track_ang_vel_z:
    enabled: true
    weight: 0.5
    link: bello_root
    sigma: 0.25
  alive:
    enabled: true
    weight: 0.15
  lin_vel_z:
    enabled: true
    weight: -2.0
    link: bello_root
  ang_vel_xy:
    enabled: true
    weight: -0.05
    link: bello_root
  flat_orientation:
    enabled: true
    weight: -5.0
    link: bello_root
  base_height:
    enabled: true
    weight: -10.0
    link: bello_root
    target: 0.86
  joint_vel:
    enabled: true
    weight: -0.001
  action_rate:
    enabled: true
    weight: -0.05
  energy:
    enabled: true
    weight: -0.00002
  joint_limit:
    enabled: true
    weight: -5.0
  hold_joint_deviation:
    enabled: true
    weight: -0.1
  foot_contact:
    enabled: true
    weight: 0.05
  foot_slip:
    enabled: true
    weight: -0.2
  foot_clearance:
    enabled: true
    weight: -2.0
    target: 0.08
  gait_contact:
    enabled: true
    weight: -0.1
  feet_air_time:
    enabled: true
    weight: 0.2
    min_air_time: 0.20
    max_air_time: 0.60
    command_speed_threshold: 0.05
  undesired_contact:
    enabled: true
    weight: -1.0
  termination:
    enabled: true
    weight: -1.0
```

Foot-dependent terms are `foot_contact`, `foot_slip`, `foot_clearance`,
`gait_contact`, and `feet_air_time`. Terrain-contact metadata is required for
those terms and for `undesired_contact` when terrain is enabled.

`gait_contact` penalizes mismatch between detected foot contacts and the
schedule from `env_config.gait`. `feet_air_time` rewards the first contact after
a swing interval, clipped between `min_air_time` and `max_air_time`, and is
disabled for near-standing commands by `command_speed_threshold`.

### DoneConfig

```yaml
done_config:
  base_height:
    enabled: true
    min_height: 0.75
    link: bello_root
  roll_pitch:
    enabled: true
    max_abs_roll: 1.0
    max_abs_pitch: 1.0
    link: bello_root
  timeout:
    enabled: true
  nonfinite:
    enabled: true
```

`base_height` and `roll_pitch` validate their configured links during env
initialization.

### WarpConfig

```yaml
warp:
  contact_budget_per_env: 32
  naconmax: null
  naccdmax: null
  njmax: 256
```

If `naconmax` is null:

```text
naconmax = batch_size * contact_budget_per_env
```

### CheckpointConfig

```yaml
checkpoint_config:
  enabled: false
  directory: checkpoints
  name: null
  overwrite: true
```

If `name` is null, the checkpoint name is:

```text
ppo_seed_<seed>
```

### WandbConfig

```yaml
wandb_config:
  enabled: false
  project: bello-mujoco
  entity: null
  group: null
  job_type: train
  run_name: null
  mode: online  # online | offline | disabled
  dir: .wandb
  tags: []
```

### TrainingMetricConfig

```yaml
metric_config:
  stream_updates: true
```

When `wandb_config.enabled=true` and `metric_config.stream_updates=true`,
training runs as a Python loop over JIT-compiled PPO updates and logs scalar
metrics after every PPO update. With the `rtx5090_4096x256` preset, that means
one W&B metric point every `4096 * 256 = 1048576` environment steps.

If `stream_updates=false`, the fully JIT-compiled training function runs first
and metrics are uploaded to W&B only after training returns.

### TrainingVideoConfig

Training videos are rendered from the training loop after selected PPO updates.
Rendering is synchronous and uses the current policy checkpoint held in memory.

```yaml
video_config:
  enabled: false
  every_updates: 10
  render_initial: false
  render_final: true
  simulation_backend: mujoco  # mujoco | mjx
  output_dir: outputs/train_videos
  filename_prefix: rollout
  command: [0.0, 0.0, 0.0]
  steps: 250
  width: 640
  height: 480
  fps: 50
  stop_on_done: true
  log_wandb: true
  wandb_name: train/rollout
  camera_config:
    mode: follow  # follow | fixed
    distance: 3.0
    azimuth: 135.0
    elevation: -18.0
    lookat_offset: [0.0, 0.0, 0.35]
```

Schedule semantics:

```text
render_initial=true renders update 0 before optimization.
every_updates=N renders updates where update % N == 0.
render_final=true always renders the final PPO update.
```

Videos are written as:

```text
<output_dir>/<filename_prefix>_<update:06d>.mp4
```

If `wandb_config.enabled` and `video_config.log_wandb` are both true, each
rendered file is also logged to W&B under `video_config.wandb_name`.

`simulation_backend=mujoco` renders the rollout by stepping a single CPU MuJoCo
`MjData` while still evaluating the current policy mean action. This is the
default for training videos because it avoids allocating extra MJX-Warp contact
buffers during an already memory-heavy PPO update loop. Use
`simulation_backend=mjx` only when the video rollout must match the MJX-Warp
step path exactly and the GPU has enough memory headroom.

### Root Runtime Flags

```yaml
dry_run: true
jit: true
```

`dry_run=true` initializes the env and policy, prints shape metadata, and exits.
`jit=true` compiles the full train function before running.

## PPO Loop

The implementation is a local PureJaxRL-style PPO loop using Equinox and Optax.
The hot path is JAX:

- vectorized env reset
- batched MJX-Warp stepping
- policy sampling
- rollout collection with `jax.lax.scan`
- GAE
- PPO clipped actor loss
- clipped value loss
- entropy regularization
- shuffled minibatch updates

Per PPO update:

1. Collect `num_steps` transitions across `num_envs`.
2. Bootstrap value from the last observation.
3. Compute GAE and value targets.
4. Flatten `[num_steps, num_envs]` into one batch.
5. Shuffle and split into `num_minibatches`.
6. Run `update_epochs` passes over minibatches.
7. Emit scalar metrics for the update.

`TrainState.update` counts PPO rollout updates, not optimizer minibatch steps.
W&B scalar logging uses environment steps as the logging step:

```text
env_steps = update_index * num_envs * num_steps
```

## Action Semantics

The policy samples Gaussian actions. PPO clips sampled actions before storing
log-probs and before passing them to the environment, so the action in the
transition is the action that affects dynamics.

The locomotion env also clips input actions defensively with `ACTION_CLIP` and
then clips actuator controls to MuJoCo ctrl ranges.

CartPole exposes `action_clip=1.0`; Bello exposes `action_clip=10.0`.

## Metrics

Training returns `out["metrics"]`, with one row per PPO update.

Core metrics:

```text
loss
actor_loss
value_loss
entropy
approx_kl
mean_reward
mean_done
mean_done/any
mean_done/base_height
mean_done/roll_pitch
mean_done/timeout
mean_done/nonfinite
mean_episode_return
max_episode_return
```

Reward component metrics are emitted as:

```text
mean_reward/<term_name>
```

Done reason metrics are emitted as:

```text
mean_done/<reason_name>
```

These show the fraction of rollout steps terminated by each done condition.
`mean_done/any` is the per-reason equivalent of `mean_done`.

`mean_episode_return` averages only completed episodes in the rollout. If no
episode completed, it returns `0.0`.

`max_episode_return` considers only completed episodes. If no episode completed
in the rollout, it returns `NaN`; this avoids reporting a fake `0` when real
returns are negative or absent.

## Checkpoints

Checkpointing is controlled by `CheckpointConfig`:

```yaml
enabled: false
directory: checkpoints
name: null
overwrite: true
```

When enabled, training writes:

```text
<directory>/<name-or-ppo_seed_N>/
  metadata.json
  state/
```

`state/` stores:

```text
policy params
opt_state
update
```

`metadata.json` stores:

```text
ppo config
asset config
reward config
command config
done config
env config
warp config
obs_size  # policy input size, env.observation_size * obs_stack_size
action_size
```

Evaluation reconstructs the env and policy from checkpoint metadata. The
checkpoint must match the code schema used to restore it.

## W&B

W&B is controlled by `WandbConfig`:

```yaml
enabled: false
project: bello-mujoco
entity: null
group: null
job_type: train
run_name: null
mode: online | offline | disabled
dir: .wandb
tags: []
```

If W&B is disabled, tracking helpers are no-ops.

When enabled:

- the full run config is logged to `run.config`
- training metrics are logged under `phase=train`
- metrics stream after every PPO update when `metric_config.stream_updates=true`
- the W&B logging step is environment steps
- training videos can be logged when `video_config.enabled` and
  `video_config.log_wandb` are true

## Training Videos

`bello-train-ppo` can periodically render policy rollouts during training. This
path intentionally runs outside the JAX training hot path because it uses
MuJoCo's renderer and writes MP4 files.

Root flags:

```yaml
video_config:
  enabled: false
  every_updates: 10
  render_initial: false
  render_final: true
  output_dir: outputs/train_videos
  filename_prefix: rollout
  command: [0.0, 0.0, 0.0]
  steps: 250
  width: 640
  height: 480
  fps: 50
  stop_on_done: true
  log_wandb: true
  wandb_name: train/rollout
  camera_config: CameraConfig
```

Camera config:

```yaml
mode: follow | fixed = follow
distance: 3.0
azimuth: 135.0
elevation: -18.0
lookat_offset: [0.0, 0.0, 0.35]
```

Follow mode requires body `bello_root`. Rendering uses a single-env rollout
with the current in-memory policy.

## Validation And Assert Conditions

`PPOConfig` is validated before training and optimizer construction.

Positive integer conditions:

```text
total_timesteps > 0
num_envs > 0
num_steps > 0
update_epochs > 0
num_minibatches > 0
```

Rollout coverage:

```text
total_timesteps // (num_envs * num_steps) > 0
```

Minibatch divisibility:

```text
(num_envs * num_steps) % num_minibatches == 0
```

Scalar range conditions:

```text
learning_rate > 0
0 <= gamma <= 1
0 <= gae_lambda <= 1
clip_eps > 0
ent_coef >= 0
vf_coef >= 0
max_grad_norm > 0
```

Policy config conditions:

```text
activation in {"tanh", "relu"}
hidden_size > 0
hidden_depth >= 0
init_log_std is finite
obs_stack_size > 0
```

`obs_stack_size` controls policy-side observation stacking. The environment
still emits one raw observation per step, but PPO repeats the initial
observation on reset, shifts in each new observation, resets each completed
env's stack to the reset observation, and feeds the flattened stack to the
actor-critic. The policy input size is therefore
`env.observation_size * config.policy.obs_stack_size`.

Environment/model initialization also validates:

- policy and held actuator indices are in range
- policy and held actuator indices are disjoint
- policy and held actuator indices cover every actuator exactly once
- actuator qpos/qvel metadata matches MuJoCo `nu`
- configured reward/done link names exist
- explicit commands have shape `(3,)`
- batch commands have shape `(3,)` or `(batch_size, 3)`
- command ranges are finite and ordered
- command standing probability is in `[0, 1]`
- command resample seconds is positive
- enabled tracking reward sigmas are positive
- foot-dependent rewards have foot bodies and foot collision geoms
- terrain contact rewards have a runtime terrain geom when terrain is enabled
- default qpos/qvel/ctrl shapes match the MuJoCo model

Training video validation applies only when `video_config.enabled=true`:

```text
video_config.every_updates > 0
video_config.steps > 0
video_config.width > 0
video_config.height > 0
video_config.fps > 0
```

## Required Smoke Commands

For training changes:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  dry_run=false \
  config.total_timesteps=4 \
  config.num_envs=2 \
  config.num_steps=2 \
  config.num_minibatches=1 \
  config.update_epochs=1
```

For env changes that affect training:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env step=true
```
