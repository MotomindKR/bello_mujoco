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
  num_steps: 24
  update_epochs: 5
  num_minibatches: 4
  learning_rate: 0.001
  anneal_lr: false
  schedule: adaptive  # null uses anneal_lr; or constant | linear | adaptive
  desired_kl: 0.01
  min_learning_rate: 0.00001
  max_learning_rate: 0.01
  adaptive_lr_factor: 1.5
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  ent_coef: 0.003
  vf_coef: 1.0
  max_grad_norm: 1.0
  normalize_value: true
  value_norm_eps: 0.00000001
  policy:
    hidden_size: 256
    hidden_depth: 2
    activation: tanh  # tanh | relu
    init_log_std: 0.0
    min_log_std: -5.0
    max_log_std: 1.0
    obs_stack_size: 10
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

Notable default full-body position-actuator gains are:

```text
hip_pitch: [200, 5]
hip_roll:  [150, 5]
hip_yaw:   [150, 5]
knee:      [220, 6]
ankle:     [22, 1]
waist_yaw: [180, 12]
```

These are MuJoCo position-actuator `[kp, kv]` values. They affect the torque
generated from target-position error and joint velocity; they are separate from
reward weights and action scales.

### BelloEnvConfig

```yaml
env_config:
  decimation: 10
  max_episode_steps: 1000
  root_link: bello_root
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
  contact:
    foot_contact_on_seconds: 0.0
    foot_contact_off_seconds: 0.0
  observation:
    joint_mode: all_actuated  # all_actuated | policy
    base_lin_vel: false
    base_ang_vel: true
    projected_gravity: true
    base_height: false
    gait_phase: true
    command: true
    joint_pos: true
    joint_vel: true
    actuator_force: true
    last_action: true
```

`root_link` is the default measurement body for base observations and
link-aware reward/done terms whose `link` is null. The package default is the
generated floating wrapper body `bello_root`; standing/balance recipes can set
it to a physical body such as `torso_link`.

`joint_mode` controls `joint_pos`, `joint_vel`, and `actuator_force`.
`base_lin_vel`, `base_ang_vel`, `projected_gravity`, and `base_height` are
measured on `root_link`. The command observation is emitted only when both
`env_config.observation.command` and `command_config.enabled` are true.

`gait` defines a deterministic two-foot contact schedule. `cycle_time` is the
period in seconds, `duty_factor` is the stance fraction of each period,
`phase_offsets` are per-foot phase offsets, and
`standing_command_threshold` forces all feet scheduled in contact when command
speed is near zero. `observation.gait_phase=true` appends `[sin(phase),
cos(phase)]` to the policy observation.

`contact` debounces raw foot-ground contacts before they feed contact-dependent
reward terms and foot-air-time memory. `foot_contact_on_seconds` is the required
duration before a raw non-contact foot becomes filtered contact, and
`foot_contact_off_seconds` is the required duration before a raw contact foot
becomes filtered non-contact. `0.0` disables debounce for that transition.

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

Every reward term has `enabled` and `weight`. Link-aware terms also have `link`;
`link: null` means use `env_config.root_link`.
Continuous reward components are multiplied by the environment control timestep
before they are returned or logged under `reward/*`, so their weights are
interpreted as per-second scales. Event-driven terms such as `feet_air_time` and
`termination` keep their configured event magnitude. The `termination` reward is
applied only for terminal failure conditions, not for episode-length timeouts.

```yaml
reward_config:
  track_lin_vel_xy:
    enabled: true
    weight: 1.0
    link: null
    sigma: 0.25
  track_ang_vel_z:
    enabled: true
    weight: 0.5
    link: null
    sigma: 0.25
  alive:
    enabled: true
    weight: 0.15
  lin_vel_z:
    enabled: true
    weight: -2.0
    link: null
  ang_vel_xy:
    enabled: true
    weight: -0.05
    link: null
  flat_orientation:
    enabled: true
    weight: -5.0
    link: null
  base_height:
    enabled: true
    weight: -10.0
    link: null
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
  torque:
    enabled: false
    weight: -0.00001
  joint_limit:
    enabled: true
    weight: -5.0
  hold_joint_deviation:
    enabled: true
    weight: -0.1
  policy_joint_deviation:
    enabled: false
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
  foot_distance:
    enabled: false
    weight: 1.0
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
`foot_distance`, `gait_contact`, and `feet_air_time`. `foot_distance` rewards
the horizontal distance between the two foot bodies. Terrain-contact metadata is
required for the contact-dependent foot terms and for `undesired_contact` when
terrain is enabled. Contact-dependent terms use the debounced foot-contact
signal from `env_config.contact`.

`joint_vel`, `action_rate`, `energy`, and `torque` are smoothness and effort
regularizers. `energy` penalizes absolute actuator force times joint velocity;
`torque` penalizes actuator-force magnitude directly.

`hold_joint_deviation` penalizes default-pose deviation for held actuators.
`policy_joint_deviation` does the same for policy-controlled actuators and is
disabled by default so velocity-tracking presets keep their previous reward
shape unless they opt into the pose regularizer.

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
    link: null
  roll_pitch:
    enabled: true
    max_abs_roll: 1.0
    max_abs_pitch: 1.0
    link: null
  timeout:
    enabled: true
  nonfinite:
    enabled: true
```

`done/any` is true for either terminal failure or timeout and drives env resets
and completed-episode logging. `done/terminal` is true only for failure
conditions, currently base height, roll/pitch, or nonfinite state. Timeout is
treated as truncation: it resets the env and emits `done/timeout`, but does not
receive the `reward/termination` penalty.

`base_height` and `roll_pitch` validate their configured links during env
initialization. When their `link` is null, `env_config.root_link` is validated
and used.

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
  upload_wandb_artifact: false
  wandb_artifact_name: null
  wandb_artifact_type: checkpoint
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

For stacked-observation policies, video rollouts use the checkpoint policy stack
size. With `stop_on_done=false`, the rollout continues after done and keeps
shifting the observation stack with subsequent observations; it does not replace
the stack with repeated terminal observations. With `stop_on_done=true`, the
rollout stops at the first done frame.

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
3. Compute GAE and raw value targets.
4. If `normalize_value=true`, update the running target-return statistics.
5. Flatten `[num_steps, num_envs]` into one batch.
6. Shuffle and split into `num_minibatches`.
7. Run `update_epochs` passes over minibatches.
8. Emit scalar metrics for the update.

When `normalize_value=true`, the critic predicts normalized values. Rollout and
GAE denormalize those predictions back into raw reward units, while the value
loss compares normalized critic outputs against normalized raw targets. Actor
advantages remain based on raw rewards and are still normalized per minibatch.

Timeouts are handled as time-limit truncations in PPO. They end the episode and
reset the env, but GAE bootstraps from the pre-reset timeout observation instead
of treating the transition as a zero-value terminal failure. Terminal failures
still cut bootstrapping.

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
value_loss_raw
entropy
approx_kl
value_norm_mean
value_norm_std
value_norm_count
mean_reward
mean_done
mean_done/any
mean_done/base_height
mean_done/terminal
mean_done/roll_pitch
mean_done/timeout
mean_done/nonfinite
mean_episode_return
max_episode_return
mean_episode_length
max_episode_length
action_rms
action_abs_max
action_mean_rms
action_mean_abs_max
action_std_mean
action_std_min
action_std_max
```

Reward component metrics are emitted as:

```text
mean_reward/<term_name>
```

Done reason metrics are emitted as:

```text
mean_done/<reason_name>
```

These show the fraction of rollout steps ended by each done condition.
`mean_done/any` is the per-reason equivalent of `mean_done`.
`mean_done/terminal` excludes timeouts.

`mean_episode_return` averages only completed episodes in the rollout. If no
episode completed, it returns `0.0`.

`max_episode_return` considers only completed episodes. If no episode completed
in the rollout, it returns `NaN`; this avoids reporting a fake `0` when real
returns are negative or absent.

`mean_episode_length` and `max_episode_length` use the same completed-episode
mask as the return metrics. They are useful for standing and balance tasks where
survival time is the main diagnostic signal.

Bello locomotion runs also emit actuator-control diagnostics:

```text
ctrl_saturation_fraction
ctrl_margin_min
```

`ctrl_saturation_fraction` is the fraction of policy controls exactly at their
MuJoCo ctrl-range bounds. `ctrl_margin_min` is the smallest remaining distance
to either bound across sampled policy controls in the rollout.

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

If `upload_wandb_artifact=true` and W&B is enabled, the final checkpoint
directory is also logged as a W&B artifact after the local checkpoint save
completes. `wandb_artifact_name=null` uses the checkpoint directory name.

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
- training metrics are logged with a `train/` metric-name prefix
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
  simulation_backend: mujoco
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
schedule in {null, "constant", "linear", "adaptive"}
desired_kl > 0
min_learning_rate > 0
max_learning_rate > 0
min_learning_rate <= max_learning_rate
adaptive_lr_factor > 1
0 <= gamma <= 1
0 <= gae_lambda <= 1
clip_eps > 0
ent_coef >= 0
vf_coef >= 0
max_grad_norm > 0
value_norm_eps > 0
```

Policy config conditions:

```text
activation in {"tanh", "relu"}
hidden_size > 0
hidden_depth >= 0
init_log_std is finite
min_log_std is finite
max_log_std is finite
min_log_std <= max_log_std
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
- `env_config.root_link` and configured reward/done link names exist
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
