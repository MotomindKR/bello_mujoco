# Bello MuJoCo Reference

This is the single user-facing reference for the repository. `README.md` is the
quick entry point, `CONTRIBUTION.md` is the contributor workflow, and
`AGENTS.md` is the coding-agent operating guide.

## Project Shape

`bello_mujoco` is a MuJoCo/MJX-Warp robot-learning scaffold for Bello.

The current stack is:

- Simulator: `mujoco`, `mujoco-mjx[warp]`
- JAX: `jax[cuda13]` on Linux, CPU `jax` on Apple Silicon macOS
- Policy/model: Equinox actor-critic
- PPO: local PureJaxRL-style implementation
- RL utilities: RLax dependency available for future reward/math work
- Optimizer: Optax
- Checkpointing: Orbax
- Tracking/video: W&B, Rerun evaluation, MuJoCo training renderer, `imageio[ffmpeg]`
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
dependencies into `.venv`, activates `.venv`, exports `MUJOCO_GL=egl` on Linux
and `MUJOCO_GL=glfw` on Apple Silicon macOS, and keeps uv/Warp caches
repo-local:

```text
.uv-cache/
.warp-cache/
```

Run commands through uv when outside the activated shell:

```bash
uv run bello-smoke-env step=true
```

Console scripts can run from an editable checkout or from an installed wheel.
The wheel bundles `configs/` and `assets/source/` as package data so Hydra
configs and robot source assets are available without walking back to a repo
root.

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
motions/local/
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
- configurable URDF joints rewritten as fixed while retaining their child bodies
- MuJoCo training MJCF and viewer MJCF generation

The training MJCF keeps the selected collision geoms as the robot geometry. The
viewer MJCF injects visual mesh geoms and makes collision geoms transparent, so
`collision_mode: box` still renders the visual robot surface.
Asset conversion disables MuJoCo's URDF static-body fusion so named fixed links,
including the root `pelvis_link`, remain available to link-aware observations,
rewards, and done terms. The floating wrapper remains `bello_root`.

Foot collision is defined directly in the source URDFs with five primitives per
ankle-roll link: three boxes and two cylinders. The STL foot meshes are visual
geometry only.

Asset generation rebases both ankle-pitch coordinates by 15 degrees. Joint
position zero therefore places the large sole box horizontally; hard limits,
soft limits, actuator defaults, and home keyframes are shifted together so the
physical range and home pose do not change.

The fixed-waist, fixed-head whole-body task and GMR import contract are documented in
[motion_tracking.md](motion_tracking.md).

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

During PPO training, Bello stores the raw Gaussian action for policy-ratio
calculations and passes that request to the environment. The environment uses
the pre-clip value for `invalid_action`, then bounds the applied control with
the configured `env_config.action_clip`, which defaults to `5.0`. Generated
MuJoCo position actuators use
`ctrllimited="false"` and `forcelimited="true"`: policy target commands are not
actuator-clamped, but actuator force remains limited. Joints configured as
`mode="hold"` are removed from the policy action vector but still use default
actuator controls.
By default, each policy action scale is `0.25 * actuator_ctrlrange_width`,
where generated actuator `ctrlrange` comes from the configured soft limit when
present. Zero action still requests the configured default pose.
The knee soft target range starts at the hard lower joint limit, `0.0` rad,
with the upper soft limit at `1.98968` rad.
Default position-actuator gains are runtime `[kp, kv]` values. The full-body
lower-body defaults are hip pitch `[200, 5]`, hip roll/yaw `[150, 5]`, knee
`[220, 6]`, ankle `[22, 1]`, and waist yaw `[180, 12]`.
Generated position actuators are force-limited from URDF effort limits by
default, with per-joint `ActuatorConfig.effort_limit` available as an explicit
override. The current full-body effort limits use the Robstride peak-torque
groups: hip pitch and knee `120 Nm`, hip roll/yaw `60 Nm`, ankle `36 Nm`,
waist yaw `120 Nm`, shoulder pitch `120 Nm`, shoulder roll `60 Nm`, arm 3/4
`36 Nm`, arm 5/6 `14 Nm`, and arm 7 `5.5 Nm`. Known Bello joint overrides must
keep a positive `effort_limit`; this is validated so YAML recipes cannot
accidentally generate uncapped actuators.

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
Observation term noise and scaling are configurable under
`ObservationConfig.noise` and `ObservationConfig.scale`; noise is only applied
when training supplies an observation RNG. `BelloEnvConfig.random_push` can set
the floating-root XY velocity at a fixed interval during vectorized training,
matching Unitree's push event shape.
`BelloEnvConfig.critic_observation` can optionally define a separate privileged
critic stream. When absent, the critic uses the same stacked observations as the
actor. The Unitree-aligned locomotion recipe uses a clean critic observation
that adds `base_lin_vel[3]` while keeping actor observations noisy and
proprioceptive.
`BelloEnvConfig.spawn` controls reset randomization and optional initial
step-count staggering for vectorized PPO runs. `ContactConfig` debounces foot
contacts and sets the normal-force threshold for undesired non-foot contacts.

Commands are `[lin_vel_x, lin_vel_y, yaw_vel]` in the robot yaw frame. The
Bello source asset is Y-forward anatomically, so the current locomotion recipes
put forward walking commands on `lin_vel_y`; `lin_vel_x` is lateral for this
robot. With command generation enabled, `step_batch(..., rng=key)` periodically
resamples according to `CommandConfig.resample_seconds`. If no RNG is passed,
commands remain fixed until reset. PPO training can optionally enable
`CommandConfig.curriculum` to expand the active linear command ranges when
velocity tracking is good enough. Active ranges are logged under
`train/command_range/*`, and curriculum decision state such as returned episode
count, normalized tracking reward, threshold, period boundary, and expansion
flags is logged under `train/command_curriculum/*`.

Reward terms:

```text
track_lin_vel_xy
track_ang_vel_z
alive
lin_vel_z
ang_vel_xy
flat_orientation
base_height
joint_vel
joint_acc
action_rate
energy
torque
joint_limit
invalid_action
hold_joint_deviation
policy_joint_deviation
foot_contact
foot_slip
foot_clearance
swing_foot_velocity
foot_distance
stance_width
gait_contact
feet_air_time
standing_foot_vel
standing_foot_air_time
undesired_contact
self_collision
termination
```

Reward and done terms have separate dataclass configs. Link-aware reward/done
terms validate body names at env initialization. Upper-body assets require
foot-dependent rewards and lower-body/base done terms to be disabled or changed
to valid links. `track_lin_vel_xy.link` selects the velocity body, and optional
`track_lin_vel_xy.frame_link` selects the body yaw frame for command comparison.
Episode-length timeouts are truncations: they reset envs and emit
`done/timeout`, but they do not receive the `termination` reward penalty.
Bad-orientation termination uses an IsaacLab/Unitree-style projected-gravity
tilt angle, not independent roll and pitch thresholds.
`self_collision` penalizes active robot-robot contact slots above
`contact.self_collision_force_threshold`. `feet_air_time` continuously rewards
single support using the shorter of the stance-contact and swing-air durations;
double support and flight score zero, durations are capped, and near-zero
commands are gated out. `swing_foot_velocity` tracks a bounded, command-scaled
scheduled-swing velocity relative to a configured body.
`standing_foot_vel` and
`standing_foot_air_time` add command-gated stand-still penalties for foot motion
and accumulated foot air time. `invalid_action` penalizes pre-clip actions
outside the environment bound and requested actuator targets outside configured
soft ranges as squared violations in policy action units, so small-scale joints
receive comparable pressure to large-scale joints.
`stance_width` penalizes only lateral foot separation outside a configured
pelvis/root-frame band, avoiding the sagittal-step coupling of `foot_distance`.
Training reports footfall/liftoff rates, stance width, and contact-mode fractions
under `train/gait/*` and `train/mean_gait/*`.

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
  decimation: 4
  max_episode_steps: 1000
  action_clip: 5.0
  root_link: bello_root
  terrain.mode: plane
  gait.cycle_time: 0.80
  gait.command_speed_frequency_gain: 0.55
  gait.min_frequency: 1.20
  gait.max_frequency: 2.20
  gait.duty_factor: 0.55
  gait.phase_offsets: [0.0, 0.5]
  gait.standing_command_threshold: 0.05
  spawn.clearance: 0.01
  spawn.root_xy_range: [0.0, 0.0]
  spawn.root_yaw_range: [0.0, 0.0]
  spawn.joint_pos_noise: [0.0, 0.0]
  spawn.joint_vel_range: [0.0, 0.0]
  spawn.stagger_initial_step_count: false
  contact.foot_contact_on_seconds: 0.0
  contact.foot_contact_off_seconds: 0.0
  contact.foot_contact_force_threshold: 1.0
  contact.undesired_contact_force_threshold: 1.0
  contact.self_collision_force_threshold: 1.0

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
  target_kl: null
  min_learning_rate: 0.00001
  max_learning_rate: 0.01
  adaptive_lr_factor: 1.5
  gamma: 0.99
  gae_lambda: 0.95
  clip_eps: 0.2
  ent_coef: 0.01
  vf_coef: 1.0
  max_grad_norm: 1.0
  normalize_value: true
  value_norm_eps: 0.00000001
```

Generated MJCF assets use a `0.005s` MuJoCo timestep. With the default
`decimation: 4`, one policy action still covers `0.02s`.

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
config.target_kl: null
config.min_learning_rate: 0.00001
config.max_learning_rate: 0.01
config.adaptive_lr_factor: 1.5
config.gamma: 0.99
config.gae_lambda: 0.95
config.clip_eps: 0.2
config.ent_coef: 0.01
config.vf_coef: 1.0
config.max_grad_norm: 1.0
config.normalize_value: true
config.value_norm_eps: 0.00000001
config.policy.hidden_size: 256
config.policy.hidden_depth: 2
config.policy.hidden_dims: null
config.policy.activation: tanh | relu | elu
config.policy.init_log_std: 0.0
config.policy.min_log_std: -5.0
config.policy.max_log_std: 1.0
config.policy.obs_stack_size: 10
```

Command flags:

```yaml
command_config.enabled: true
command_config.lin_vel_x: [-0.1, 0.1]
command_config.lin_vel_y: [-0.1, 0.1]
command_config.ang_vel_yaw: [-0.1, 0.1]
command_config.standing_probability: 0.10
command_config.resample_seconds: 10.0
command_config.curriculum.enabled: false
command_config.curriculum.limit_lin_vel_x: [-0.5, 1.0]
command_config.curriculum.limit_lin_vel_y: [-0.3, 0.3]
command_config.curriculum.limit_ang_vel_yaw: [-0.2, 0.2]
command_config.curriculum.expand_ang_vel_yaw: false
command_config.curriculum.increment: 0.1
command_config.curriculum.success_threshold: 0.8
```

Training-adjacent flags:

```yaml
checkpoint_config.enabled: false
checkpoint_config.directory: checkpoints
checkpoint_config.name: null
checkpoint_config.upload_wandb_artifact: true
checkpoint_config.wandb_artifact_name: null
checkpoint_config.wandb_artifact_type: checkpoint

wandb_config.enabled: false
wandb_config.project: bello-mujoco
wandb_config.mode: online | offline | disabled
wandb_config.dir: .wandb

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
video_config.stochastic_actions: false
video_config.action_seed: 0
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

The command spawns a Rerun Viewer and reads live commands from an
SDL-compatible game controller.

Evaluation root flags:

```yaml
checkpoint_path: Path
eval_config: EvalPolicyConfig
```

Primary eval flags:

```yaml
eval_config.command: [0.0, 0.0, 0.0]
eval_config.steps: null
eval_config.stop_on_done: false
eval_config.simulation_backend: mujoco | mjx
eval_config.gamepad.enabled: true
eval_config.gamepad.controller_index: 0
eval_config.gamepad.deadzone: 0.1
eval_config.runtime.source: checkpoint | override
```

`eval_config.runtime.source=checkpoint` rebuilds the env from checkpoint
metadata. Use `override` only when the replacement runtime config keeps the
checkpoint observation and action sizes unchanged.

Detailed training behavior is documented in [training.md](training.md). Detailed
live evaluation behavior is documented in [evaluation.md](evaluation.md). The
branch-pinned cloud workflow is documented in [vast_ai.md](vast_ai.md).
The controlled clean-reward experiment matrix is documented in
[locomotion_ablations.md](locomotion_ablations.md).

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
uv run pytest tests/test_vast.py -q
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
