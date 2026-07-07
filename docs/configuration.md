# Configuration Tree

All command-line tools use tyro over dataclasses. Nested dataclass fields map to dotted CLI flags.

Example:

```bash
bello-train-ppo \
  --env-config.observation.no-actuator-force \
  --reward-config.track-lin-vel-xy.weight 1.2 \
  --done-config.base-height.min-height 0.75
```

## AssetPrepConfig

```yaml
asset: full_body | lower_body | upper_body = full_body
output_root: Path | null = null
mesh_face_budget: 1000
foot_collision_mesh_face_budget: 1000
visual_mesh_face_budget: 200000
collision_mode: mesh | foot-mesh | box = mesh
end_effector_sphere_mode: none | both | right | left = none
neighbor_exclusion_depth: 2
actuator_config: ActuatorConfig
```

## ActuatorConfig

```yaml
default_action_scale_ratio: 0.25
joints:
  <joint_name>:
    mode: policy | hold = policy
    default_pos: 0.0
    action_scale: float | null = null
    soft_limit: [lower, upper] | null = null
    gains: [kp, kv] = [40.0, 2.0]
```

The default config includes all tracked actuated joints. Full-body default exposes all 35 actuators to policy.

## BelloEnvConfig

```yaml
decimation: 10
max_episode_steps: 1000
terrain: TerrainConfig
observation: ObservationConfig
```

## ObservationConfig

```yaml
joint_mode: all_actuated | policy = all_actuated
base_ang_vel: true
projected_gravity: true
command: true
joint_pos: true
joint_vel: true
actuator_force: true
last_action: true
```

## TerrainConfig

```yaml
mode: none | plane | rough = none
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
```

Named defaults:

```text
ROBOT_ONLY_TERRAIN = TerrainConfig(mode="none")
FLAT_TERRAIN = TerrainConfig(mode="plane")
ROUGH_TERRAIN = TerrainConfig(mode="rough")
```

The locomotion env uses `FLAT_TERRAIN` by default.

## CommandConfig

```yaml
enabled: true
lin_vel_x: [-0.1, 0.1]
lin_vel_y: [-0.1, 0.1]
ang_vel_yaw: [-0.1, 0.1]
limit_lin_vel_x: [-0.5, 1.0]
limit_lin_vel_y: [-0.3, 0.3]
limit_ang_vel_yaw: [-0.2, 0.2]
standing_probability: 0.02
resample_seconds: 10.0
```

## RewardConfig

```yaml
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
  target: 0.91
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
undesired_contact:
  enabled: true
  weight: -1.0
termination:
  enabled: true
  weight: -1.0
```

## DoneConfig

```yaml
base_height:
  enabled: true
  min_height: 0.80
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

## WarpConfig

```yaml
contact_budget_per_env: 32
naconmax: null
naccdmax: null
njmax: 256
```

If `naconmax` is null, it resolves to:

```text
batch_size * contact_budget_per_env
```

## PPOConfig

```yaml
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
policy: ActorCriticConfig
```

## ActorCriticConfig

```yaml
hidden_size: 256
hidden_depth: 2
activation: tanh
init_log_std: -0.5
```

Valid activation values are effectively:

```text
tanh
relu
```

Any value other than `relu` falls back to `tanh`.

## CheckpointConfig

```yaml
enabled: false
directory: checkpoints
name: null
overwrite: true
```

## WandbConfig

```yaml
enabled: false
project: bello-mujoco
entity: null
group: null
job_type: train
run_name: null
mode: online | offline | disabled = online
dir: .wandb
tags: []
```

## CameraConfig

```yaml
mode: follow | fixed = follow
distance: 3.0
azimuth: 135.0
elevation: -18.0
lookat_offset: [0.0, 0.0, 0.35]
```

## CLI Root Configs

### bello-train-ppo

```yaml
config: PPOConfig
asset_config: AssetPrepConfig
reward_config: RewardConfig
command_config: CommandConfig
done_config: DoneConfig
env_config: BelloEnvConfig
checkpoint_config: CheckpointConfig
wandb_config: WandbConfig
dry_run: true
jit: true
contact_budget_per_env: 32
warp_naconmax: null
warp_naccdmax: null
warp_njmax: 256
```

### bello-smoke-env

```yaml
asset_config: AssetPrepConfig
env_config: BelloEnvConfig
reward_config: RewardConfig
command_config: CommandConfig
done_config: DoneConfig
step: false
contact_budget_per_env: 32
warp_njmax: 256
```

### bello-benchmark-env

```yaml
asset_config: AssetPrepConfig
env_config: BelloEnvConfig
reward_config: RewardConfig
command_config: CommandConfig
done_config: DoneConfig
batch_size: 512
step_count: 5
contact_budget_per_env: 32
warp_naconmax: null
warp_naccdmax: null
warp_njmax: 256
```

### bello-eval-policy

```yaml
checkpoint_path: Path
output_path: outputs/eval_rollout.mp4
command: [0.0, 0.0, 0.0]
steps: 250
width: 640
height: 480
fps: 50
stop_on_done: true
camera_config: CameraConfig
log_wandb: false
wandb_config: WandbConfig
```

### bello-prepare-assets

Uses `AssetPrepConfig` directly.

### bello-ppo-cartpole-smoke

```yaml
seed: 0
total_timesteps: 16384
num_envs: 64
num_steps: 64
update_epochs: 4
num_minibatches: 4
learning_rate: 0.0003
hidden_size: 64
hidden_depth: 2
ent_coef: 0.0
init_log_std: -0.5
```
