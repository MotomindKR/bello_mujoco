# Bello MuJoCo Reference

This is the single user-facing reference for the repository. `README.md` is the
quick entry point, `CONTRIBUTION.md` is the contributor workflow, and
`AGENTS.md` is the coding-agent operating guide.

## Project Shape

`bello_mujoco` is a MuJoCo/MJX-Warp robot-learning scaffold for Bello.

The current stack is:

- Simulator: `mujoco`, `mujoco-mjx[warp]`
- JAX: `jax[cuda13]` from Python wheels
- Policy/model: Equinox actor-critic
- PPO: local PureJaxRL-style implementation
- RL utilities: RLax dependency available for future reward/math work
- Optimizer: Optax
- Checkpointing: Orbax
- Tracking/video: W&B, MuJoCo renderer, `imageio[ffmpeg]`
- Config CLI: Hydra over dataclasses
- Package/env management: Nix shell plus uv-managed Python

The simulation and PPO hot path is JAX-based: vectorized reset/step, reward and
done computation, rollout collection, GAE, PPO loss, and optimizer updates.
Host-side code handles asset generation, XML rewriting, checkpoints, W&B,
rendering, and CLI entry points.

## Setup

Enter the environment:

```bash
nix develop
```

With direnv:

```bash
direnv allow
```

The shell hook installs uv-managed Python 3.12 into `.uv-python`, syncs
dependencies into `.venv`, activates `.venv`, exports `MUJOCO_GL=egl` for
MuJoCo rendering, and keeps uv/Warp caches repo-local:

```text
.uv-cache/
.warp-cache/
```

Run commands through uv when outside the activated shell:

```bash
uv run bello-smoke-env step=true
```

Generated local artifacts should not be committed:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
wandb/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/
/tmp/bello-mujoco-test-assets-*
```

## Assets

Tracked robot sources live only under:

```text
assets/source/bello/
```

Valid asset names:

```text
full_body
lower_body
upper_body
```

Do not add external robot asset paths or asset aliases.

Prepare assets:

```bash
bello-prepare-assets asset=full_body
```

Generated assets default to a deterministic cache under:

```text
/tmp/bello-mujoco-assets/<config-hash>/
```

You may provide an explicit output root, but the manifest still validates the
asset config and source asset fingerprint before reusing cached output.

Asset prep supports:

- collision mesh simplification
- visual mesh simplification
- `mesh`, `foot-mesh`, and `box` collision modes
- neighbor collision exclusions
- optional hand/end-effector sphere replacement for `full_body` and `upper_body`
- MuJoCo training MJCF and viewer MJCF generation

Runtime terrain is added after robot asset generation. Source URDF/MJCF files do
not include ground.

## Environment

Main API:

```python
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv

env = BelloLocomotionEnv.load(AssetPrepConfig(asset="full_body"))
state, obs = env.reset()
state, obs, reward, done, info = env.step(state, action)
```

Vectorized API:

```python
state, obs = env.reset_batch(batch_size=512)
state, obs, reward, done, info = env.step_batch(state, actions)
```

`step_batch` steps a batched MJX-Warp data object and auto-resets done envs.
Contact reward terms filter Warp's shared contact buffer by world id.

Default full-body contract:

```text
terrain: plane
mjx_impl: warp
nu: 35
action_size: 35
observation_size: 149
```

Actions are residual position targets for policy actuators:

```text
ctrl[policy_ctrl_indices] = default_ctrl + policy_action_scale * action
```

Input actions are clipped by `ACTION_CLIP = 10.0`, and resulting controls are
clipped to MuJoCo actuator ranges. Joints configured as `mode="hold"` are
removed from the policy action vector but still use default actuator controls.
Default position-actuator gains are runtime `[kp, kv]` values. The full-body
lower-body defaults are hip pitch `[200, 5]`, hip roll/yaw `[150, 5]`, knee
`[220, 6]`, ankle `[22, 1]`, and waist yaw `[180, 12]`.

Default observation layout:

```text
base_ang_vel[3]
projected_gravity[3]
command[3]
joint_pos[35]
joint_vel[35]
actuator_force[35]
last_action[35]
```

`ObservationConfig` controls component inclusion. `base_lin_vel[3]` and
`base_height[1]` are available as opt-in proprioceptive state for balance tasks.
Base velocity, projected gravity, and base height are measured on
`BelloEnvConfig.root_link`, which defaults to the generated floating body
`bello_root`. Link-aware reward/done terms also use `root_link` when their
own `link` is null. `joint_mode=policy` makes joint position, joint velocity,
and actuator force observations use only policy actuator indices. Command
observation is present only when both
`ObservationConfig.command` and `CommandConfig.enabled` are true. The training
default YAML enables `ObservationConfig.gait_phase`, adding `[sin(phase),
cos(phase)]` and raising the default full-body training observation size to 151.

Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`. With command generation enabled,
`step_batch(..., rng=key)` periodically resamples according to
`CommandConfig.resample_seconds`. If no RNG is passed, commands remain fixed
until reset.

Default reward terms:

```text
track_lin_vel_xy
track_ang_vel_z
alive
lin_vel_z
ang_vel_xy
flat_orientation
base_height
joint_vel
action_rate
energy
torque
joint_limit
hold_joint_deviation
policy_joint_deviation
foot_contact
foot_slip
foot_clearance
gait_contact
feet_air_time
undesired_contact
termination
```

Reward and done terms have separate dataclass configs. Link-aware reward/done
terms validate body names at env initialization. Upper-body assets require
foot-dependent rewards and lower-body/base done terms to be disabled or changed
to valid links. Episode-length timeouts are truncations: they reset envs and
emit `done/timeout`, but they do not receive the `termination` reward penalty.

## Configuration

All tools use Hydra over dataclasses. Nested fields map to dotted overrides:

```bash
bello-train-ppo \
  env_config.observation.actuator_force=false \
  reward_config.track_lin_vel_xy.weight=1.2 \
  done_config.base_height.min_height=0.75
```

Important config roots:

```text
AssetPrepConfig
ActuatorConfig
BelloEnvConfig
ObservationConfig
TerrainConfig
GaitConfig
ContactConfig
CommandConfig
RewardConfig
DoneConfig
WarpConfig
PPOConfig
ActorCriticConfig
CheckpointConfig
WandbConfig
TrainingVideoConfig
EvalPolicyConfig
EvalRuntimeConfig
CameraConfig
```

Common defaults:

```yaml
AssetPrepConfig:
  asset: full_body
  mesh_face_budget: 1000
  foot_collision_mesh_face_budget: 1000
  visual_mesh_face_budget: 200000
  collision_mode: mesh
  end_effector_sphere_mode: none
  neighbor_exclusion_depth: 2

BelloEnvConfig:
  decimation: 10
  max_episode_steps: 1000
  root_link: bello_root
  terrain.mode: plane
  gait.cycle_time: 0.80
  gait.duty_factor: 0.55
  gait.phase_offsets: [0.0, 0.5]
  gait.standing_command_threshold: 0.05
  contact.foot_contact_on_seconds: 0.0
  contact.foot_contact_off_seconds: 0.0

WarpConfig:
  contact_budget_per_env: 32
  naconmax: null
  naccdmax: null
  njmax: 256

PPOConfig:
  total_timesteps: 1000000
  num_envs: 64
  num_steps: 24
  update_epochs: 5
  num_minibatches: 4
  learning_rate: 0.001
  schedule: adaptive
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
```

Use `--help` and `--hydra-help` on each entry point for the current exhaustive CLI tree.

## Training

Training entry point:

```text
bello-train-ppo
```

Root flags/configs:

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

Primary PPO flags:

```yaml
config.seed: 0
config.total_timesteps: 1000000
config.num_envs: 64
config.num_steps: 24
config.update_epochs: 5
config.num_minibatches: 4
config.learning_rate: 0.001
config.anneal_lr: false
config.schedule: null | constant | linear | adaptive
config.desired_kl: 0.01
config.min_learning_rate: 0.00001
config.max_learning_rate: 0.01
config.adaptive_lr_factor: 1.5
config.gamma: 0.99
config.gae_lambda: 0.95
config.clip_eps: 0.2
config.ent_coef: 0.003
config.vf_coef: 1.0
config.max_grad_norm: 1.0
config.normalize_value: true
config.value_norm_eps: 0.00000001
config.policy.hidden_size: 256
config.policy.hidden_depth: 2
config.policy.activation: tanh | relu
config.policy.init_log_std: 0.0
config.policy.min_log_std: -5.0
config.policy.max_log_std: 1.0
config.policy.obs_stack_size: 10
```

Training-adjacent flags:

```yaml
checkpoint_config.enabled: false
checkpoint_config.directory: checkpoints
checkpoint_config.name: null
checkpoint_config.overwrite: true
checkpoint_config.upload_wandb_artifact: false
checkpoint_config.wandb_artifact_name: null
checkpoint_config.wandb_artifact_type: checkpoint

wandb_config.enabled: false
wandb_config.project: bello-mujoco
wandb_config.mode: online | offline | disabled
wandb_config.dir: .wandb

metric_config.stream_updates: true

video_config.enabled: false
video_config.every_updates: 100
video_config.render_initial: false
video_config.render_final: true
video_config.simulation_backend: mujoco | mjx
video_config.output_dir: outputs/train_videos
video_config.command: [0.0, 0.0, 0.0]
video_config.steps: 250
video_config.width: 640
video_config.height: 480
video_config.fps: 50
video_config.camera_config.mode: follow | fixed

warp.contact_budget_per_env: 32
warp.naconmax: null
warp.naccdmax: null
warp.njmax: 256
```

Live evaluation entry point:

```text
bello-eval-policy
```

Evaluation root flags:

```yaml
checkpoint_path: Path
eval_config: EvalPolicyConfig
```

Primary eval flags:

```yaml
eval_config.command: [0.0, 0.0, 0.0]
eval_config.steps: null
eval_config.realtime: true
eval_config.stop_on_done: false
eval_config.camera_config.mode: follow | fixed
eval_config.reward_overlay.enabled: true
eval_config.reward_overlay.position: top_right
eval_config.runtime.source: checkpoint | override
```

`eval_config.runtime.source=checkpoint` rebuilds the env from checkpoint
metadata. Use `override` only when the replacement runtime config keeps the
checkpoint observation and action sizes unchanged.

Detailed training behavior is documented in [training.md](training.md). Detailed
live evaluation behavior is documented in [evaluation.md](evaluation.md).

## Testing

Standard gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

Simulation/training smoke:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env step=true
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  dry_run=false \
  config.total_timesteps=4 \
  config.num_envs=2 \
  config.num_steps=2 \
  config.num_minibatches=1 \
  config.update_epochs=1
```

Focused test sets:

```bash
uv run pytest tests/test_assets_pipeline.py tests/test_model.py -q
uv run pytest tests/test_actuators.py tests/test_env.py tests/test_rewards.py -q
uv run pytest tests/test_ppo.py tests/test_cartpole.py tests/test_cartpole_ppo_integration.py -q
uv run pytest tests/test_checkpoint.py tests/test_tracking.py -q
uv run pytest tests/test_eval_policy.py tests/test_train_ppo_tool.py -q
uv run bello-ppo-cartpole-smoke
```

Benchmark vectorized stepping:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-benchmark-env \
  batch_size=512 \
  step_count=5
```

Expected smoke output on a CUDA-capable host includes:

```text
backend: gpu
mjx_impl: warp
action_size: 35
observation_size: 149
```
