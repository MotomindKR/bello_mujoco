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

`experiments/train/locomotion_berkeley.yaml` is the Bello morphology port of
MuJoCo Playground's Berkeley Humanoid recipe. It adds a randomized 1.25--1.5 Hz
per-foot phase clock, touchdown-gated air time, phase-height tracking, L1 torque,
weighted pose, hip/knee deviation, and nonnegative clipped reward total. The
actor receives noisy proprioception and the four-value per-foot phase encoding;
the critic receives a clean observation with additional height and actuator
force channels.
It follows Brax PPO's bounded tanh-normal policy, Swish activation, and separate
running normalization for actor and critic observations. A target-KL guard is
enabled as an additional safety measure for this repository's PPO optimizer.

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

Unknown config keys are rejected during dataclass conversion. Use Hydra's `+`
syntax only for keys that are intentionally consumed by Hydra itself; training
runtime config fields must exist in the dataclass tree.

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
  target_kl: null  # set a positive value to enable same-update target-KL stopping
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
  policy:
    hidden_size: 256
    hidden_depth: 2
    hidden_dims: null  # optional explicit layer sizes, e.g. [512, 256, 128]
    activation: tanh  # tanh | relu | elu | swish
    init_log_std: 0.0
    min_log_std: -5.0
    max_log_std: 1.0
    obs_stack_size: 10
    action_distribution: gaussian  # gaussian | tanh_gaussian
    action_limit: 1.0
    normalize_observations: false
    observation_norm_eps: 0.00000001
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
        effort_limit: null
```

Asset preparation writes a simulation MJCF and a viewer MJCF. The simulation
MJCF uses the selected collision representation directly. The viewer MJCF
injects visual meshes and makes collision geoms transparent, so box collision
mode still displays the visual robot surface rather than opaque collision boxes.
Training uses the simulation MJCF; training videos use a separate viewer env
loaded from the viewer MJCF.

The ankle-roll foot collision in the source URDFs is represented by primitive
geometry: three boxes and two cylinders per foot. The STL feet remain visual
meshes, but foot contact uses the primitive collision shapes in every collision
mode.

The default actuator config contains every tracked actuated Bello joint. Setting
a joint to `mode: hold` removes it from the policy action vector while keeping
its default control target in the environment. If a training YAML replaces a
known Bello joint entry, it must include `effort_limit`; otherwise config
validation fails before asset generation. This prevents partial joint overrides
from silently producing uncapped MuJoCo actuators.

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

Policy action scales convert normalized residual actions to joint target
radians. When a joint does not set `action_scale` explicitly, the generated
model uses `default_action_scale_ratio * actuator_ctrlrange_width`; the default
ratio is `0.25`, and generated actuator `ctrlrange` comes from the configured
soft limit when present.
The knee soft target range starts at the hard lower joint limit, `0.0` rad,
and keeps the upper soft limit at `1.98968` rad.

Default full-body actuator effort limits follow the Robstride peak-torque
groups:

```text
waist_yaw:      120 Nm
hip_pitch:      120 Nm
hip_roll/yaw:    60 Nm
knee:           120 Nm
ankle:           36 Nm
shoulder_pitch: 120 Nm
shoulder_roll:   60 Nm
arm_3/arm_4:     36 Nm
arm_5/arm_6:     14 Nm
arm_7:          5.5 Nm
```

Generated position actuators are force-limited. By default the generator copies
the compiled joint `actuatorfrcrange` from URDF effort limits into each MuJoCo
actuator `forcerange`; `effort_limit` overrides that per joint when hardware
runtime limits differ from the URDF. A composed config should produce
`forcerange` on generated position actuators for all known Bello joints.

### BelloEnvConfig

```yaml
env_config:
  decimation: 4
  max_episode_steps: 1000
  action_clip: 5.0
  root_link: bello_root
  spawn:
    clearance: 0.01
    root_xy_range: [0.0, 0.0]
    root_yaw_range: [0.0, 0.0]
    joint_pos_noise: [0.0, 0.0]
    joint_vel_range: [0.0, 0.0]
    stagger_initial_step_count: false
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
    command_speed_frequency_gain: 0.55
    min_frequency: 1.20
    max_frequency: 2.20
    duty_factor: 0.55
    phase_offsets: [0.0, 0.5]
    standing_command_threshold: 0.05
  contact:
    foot_contact_on_seconds: 0.0
    foot_contact_off_seconds: 0.0
    foot_contact_force_threshold: 1.0
    undesired_contact_force_threshold: 1.0
    self_collision_force_threshold: 1.0
  random_push:
    enabled: false
    interval_seconds: 5.0
    lin_vel_x: [-0.5, 0.5]
    lin_vel_y: [-0.5, 0.5]
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
    scale:
      base_lin_vel: 1.0
      base_ang_vel: 1.0
      projected_gravity: 1.0
      base_height: 1.0
      gait_phase: 1.0
      command: 1.0
      joint_pos: 1.0
      joint_vel: 1.0
      actuator_force: 1.0
      last_action: 1.0
    noise:
      enabled: false
      base_lin_vel: [0.0, 0.0]
      base_ang_vel: [-0.2, 0.2]
      projected_gravity: [-0.05, 0.05]
      base_height: [0.0, 0.0]
      gait_phase: [0.0, 0.0]
      command: [0.0, 0.0]
      joint_pos: [-0.01, 0.01]
      joint_vel: [-1.5, 1.5]
      actuator_force: [0.0, 0.0]
      last_action: [0.0, 0.0]
```

The generated MuJoCo model timestep is `0.005s`, so the default
`decimation: 4` gives a `0.02s` control timestep.

`root_link` is the default measurement body for base observations and
link-aware reward/done terms whose `link` is null. The package default is the
generated floating wrapper body `bello_root`; standing/balance recipes can set
it to a physical body such as `torso_link`.
Named-body linear and angular velocities are measured at the selected MuJoCo
xbody origin. Base observations use that body's local axes, while reward terms
explicitly requesting world or yaw-frame motion transform the same
origin-centered velocity. MuJoCo's raw `cvel` is a subtree-COM-centered spatial
quantity and is not exposed directly as a body-local observation.
Reset runs a MuJoCo kinematics pass after qpos/qvel randomization, so named-body
position/orientation terms such as `torso_link` height and projected gravity are
valid in the initial observation stack before the first environment step.

`spawn` controls reset randomization. `root_xy_range` adds uniform world-frame
XY root offsets, `root_yaw_range` samples the initial root yaw, `joint_pos_noise`
adds uniform actuator-joint position noise clipped to hard joint limits, and
`joint_vel_range` samples initial actuator-joint velocities.
`stagger_initial_step_count` is used at PPO runner initialization to start
vectorized envs at random episode step counts, desynchronizing timeouts, gait
phase, command resampling, and random push events.

`joint_mode` controls `joint_pos`, `joint_vel`, and `actuator_force`.
`base_lin_vel`, `base_ang_vel`, `projected_gravity`, and `base_height` are
measured on `root_link`. The command observation is emitted only when both
`env_config.observation.command` and `command_config.enabled` are true.

`observation.noise` applies additive uniform noise to enabled observation terms
when an observation RNG is supplied by training. Noise is applied before
`observation.scale`, matching IsaacLab's observation manager order. Evaluation
and rendering call reset/step without an observation RNG by default, so they
remain deterministic unless a caller explicitly passes one.

`random_push` is a Unitree-style interval event. When enabled, vectorized
training sets the floating-root XY velocity to a sampled value every
`interval_seconds` per environment after the first step of an episode. Push
diagnostics are logged under `train/mean_random_push/*`.

`gait` defines a deterministic two-foot contact schedule. `cycle_time` defines
the zero-speed baseline period. Runtime gait frequency is
`clip(1 / cycle_time + command_speed_frequency_gain * norm(command[:2]),
min_frequency, max_frequency)`, so the default is
`clip(1.25 + 0.55 v, 1.2, 2.2)` Hz. `duty_factor` is the stance fraction of
each period, `phase_offsets` are per-foot phase offsets, and
`standing_command_threshold` forces all feet scheduled in contact when command
speed is near zero. The phase is also frozen at zero under that threshold, so a
standing policy observes the constant `[sin(phase), cos(phase)] = [0, 1]`
instead of a moving gait clock. `observation.gait_phase=true` appends that pair
to the policy observation.

The `locomotion_hard` recipe enables actor `base_lin_vel` in addition to the
privileged critic copy. This gives the policy direct feedback about stand-still
drift; checkpoints trained with older observation dimensions are not compatible
with that recipe change.

`contact` debounces raw foot-ground contacts before they feed contact-dependent
reward terms and foot-air-time memory. `foot_contact_on_seconds` is the required
duration before a raw non-contact foot becomes filtered contact, and
`foot_contact_off_seconds` is the required duration before a raw contact foot
becomes filtered non-contact. `0.0` disables debounce for that transition.
`foot_contact_force_threshold` is the minimum normal contact force before a
foot-ground contact contributes to foot-contact, gait-contact, slip, clearance,
and air-time terms.
`undesired_contact_force_threshold` is the minimum normal contact force before a
non-foot terrain contact contributes to `reward/undesired_contact`.
`self_collision_force_threshold` is the minimum normal contact force before a
robot-robot contact slot contributes to `reward/self_collision`.

### CommandConfig

```yaml
command_config:
  enabled: true
  lin_vel_x: [-0.1, 0.1]
  lin_vel_y: [-0.1, 0.1]
  ang_vel_yaw: [-0.1, 0.1]
  standing_probability: 0.10
  resample_seconds: 10.0
  curriculum:
    enabled: false
    limit_lin_vel_x: [-0.5, 1.0]
    limit_lin_vel_y: [-0.3, 0.3]
    limit_ang_vel_yaw: [-0.2, 0.2]
    expand_ang_vel_yaw: false
    increment: 0.1
    success_threshold: 0.8
```

Commands are `[lin_vel_x, lin_vel_y, yaw_vel]` in the robot yaw frame. The
Bello source asset is Y-forward anatomically, so current forward-walking recipes
use `lin_vel_y` for forward velocity and keep `lin_vel_x` as the lateral axis.
In vectorized stepping, `resample_seconds` is rounded to control-step count and
clamped to at least one step.
`standing_probability` samples an exact zero command instead of a velocity
command, so command-conditioned policies train on explicit stand-still episodes
instead of relying on the uniform range to land near zero.
The `locomotion_hard` recipe sets it to `0.35`, up from `0.20`, to give
stand-still recovery materially more training coverage while retaining a
majority of moving commands.

`command_config.curriculum` can expand the active command ranges during PPO
training while leaving the configured initial ranges as the reset value. When
enabled, the runner checks completed-episode `track_lin_vel_xy` reward sums and
divides by full episode duration, matching Unitree's
`episode_sums[track_lin_vel_xy] / max_episode_length_s` criterion. Ranges expand
by `increment` when that normalized episode score exceeds
`success_threshold * reward_config.track_lin_vel_xy.weight`. Expansion is
evaluated when rollout collection crosses an episode-length boundary. Linear X/Y
ranges expand toward their limits; yaw expansion is opt-in with
`expand_ang_vel_yaw`.

### RewardConfig

Every reward term has `enabled` and `weight`. Link-aware terms also have `link`;
`link: null` means use `env_config.root_link`.
Continuous reward components are multiplied by the environment control timestep
before they are returned or logged under `reward/*`, so their weights are
interpreted as per-second scales. Event-driven `contact_timing` and `termination`
keep their configured event magnitudes. Termination is applied only for terminal
failure conditions, not for episode-length timeouts.

```yaml
reward_config:
  track_lin_vel_xy:
    enabled: true
    weight: 1.0
    link: null
    frame_link: null
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
  joint_acc:
    enabled: true
    weight: -0.00000025
  action_rate:
    enabled: true
    weight: -0.05
  action_limit:
    enabled: false
    weight: -1.0
    soft_limit_ratio: 0.8
  energy:
    enabled: true
    weight: -0.00002
  torque:
    enabled: false
    weight: -0.00001
  joint_limit:
    enabled: true
    weight: -5.0
    soft_limit_factor: 0.9
  invalid_action:
    enabled: false
    weight: -1.0
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
    angular_speed_scale: 0.05
  foot_clearance:
    enabled: true
    weight: 1.0
    target: 0.10
    std: 0.05
    tanh_mult: 2.0
    command_speed_threshold: 0.05
  swing_foot_velocity:
    enabled: false
    weight: 0.5
    link: null
    std: 0.25
    max_target_speed: 2.0
    command_speed_threshold: 0.10
  foot_distance:
    enabled: false
    weight: 1.0
  stance_width:
    enabled: false
    weight: -2.0
    link: null
    min_width: 0.24
    max_width: 0.34
  gait_contact:
    enabled: true
    weight: 0.5
    command_speed_threshold: 0.10
  contact_timing:
    enabled: false
    weight: -0.10
    min_air_time: 0.20
    min_contact_time: 0.30
    command_speed_threshold: 0.10
  feet_air_time:
    enabled: false
    weight: 0.25
    max_duration: 0.40
    command_speed_threshold: 0.10
  standing_foot_vel:
    enabled: false
    weight: -1.0
    command_speed_threshold: 0.05
  standing_foot_air_time:
    enabled: false
    weight: -1.0
    command_speed_threshold: 0.05
  undesired_contact:
    enabled: true
    weight: -1.0
  self_collision:
    enabled: true
    weight: -1.0
  termination:
    enabled: true
    weight: -1.0
```

Foot-dependent terms are `foot_contact`, `foot_slip`, `foot_clearance`,
`swing_foot_velocity`, `foot_distance`, `stance_width`, `gait_contact`,
`contact_timing`, `feet_air_time`,
`standing_foot_vel`, and `standing_foot_air_time`. `foot_distance` rewards the
horizontal distance between the two foot bodies. Terrain-contact metadata is
required for the contact-dependent foot terms and for `undesired_contact` when
terrain is enabled. Robot collision-geom metadata is required for
`self_collision`. Contact-dependent foot terms use the debounced foot-contact
signal from `env_config.contact`.

`stance_width` is a zero-inside-band penalty on lateral foot separation in the
yaw frame of `link` (or `env_config.root_link` when null). It deliberately
ignores sagittal separation, unlike `foot_distance`, so it discourages a
spread-leg support-polygon exploit without penalizing useful forward step
length.

`track_lin_vel_xy` uses yaw-frame linear velocity, matching the Unitree
locomotion reward shape. `link` selects the body whose linear velocity is
tracked; `frame_link` selects the body whose yaw defines the command frame. When
`frame_link` is null, the tracked body also defines the yaw frame. Because the
Bello asset's anatomical forward axis is local Y, a forward Bello command is
currently expressed through `lin_vel_y`, not `lin_vel_x`. The locomotion easy
and hard recipes track `pelvis_link` linear velocity in the pelvis yaw frame and
track `pelvis_link` yaw angular velocity. Their torso-linked height,
orientation, and vertical/lateral stabilization terms remain unchanged.
`joint_vel`,
`joint_acc`, `action_rate`, `action_limit`, `energy`, and `torque` are
smoothness and effort regularizers. `joint_vel`, `joint_acc`, `action_rate`,
`energy`, and `torque` sum over their selected dimensions;
`joint_acc` penalizes finite-difference actuator joint acceleration. `energy`
penalizes absolute actuator force times joint velocity; `torque` penalizes
actuator-force magnitude directly. `action_limit` is zero inside
`soft_limit_ratio` of the environment action bound and applies a mean normalized
quadratic penalty between that margin and the bound. `joint_limit` contracts
each physical MuJoCo joint range about its center by `soft_limit_factor`, then
sums absolute joint-position violation of that soft interval. Its threshold is
independent of actuator `ctrlrange`. `invalid_action` sums squared pre-clip action-bound and
requested-target violations in action units using the same affine action scale
as the action-to-control mapper. Bello clips the request before sending its
control target to MuJoCo/MJX, but retains the original request for this penalty.
Generated
position actuators leave control targets unclamped (`ctrllimited="false"`) while
keeping actuator force limits enabled.
`foot_slip` sums contacted-foot lateral speed plus
`angular_speed_scale * angular_speed`, so rotating a planted foot is treated as
slip.

`hold_joint_deviation` penalizes default-pose deviation for held actuators.
`policy_joint_deviation` does the same for policy-controlled actuators and is
disabled by default so velocity-tracking presets keep their previous reward
shape unless they opt into the pose regularizer.

`gait_contact` rewards feet whose detected contact state matches the schedule
from `env_config.gait`. It is disabled for near-standing commands by
`command_speed_threshold`, matching the command-gated Unitree G1 gait reward
shape. The `locomotion_hard` recipe raises its weight from the package default
of `0.5` to `2.0`; this increases the per-step advantage of matching both
scheduled contacts over planted double support from `0.01` to `0.04` without
applying gait pressure to stand-still commands. `foot_clearance` is a positive
target-height reward for non-contact moving feet: lateral foot speed gates the
term with `tanh_mult`, contacted feet are ignored, and `std` controls the
exponential falloff around `target`. If all feet are contacted, or no
non-contact foot is moving, the clearance term is zero. It is also disabled for
near-standing commands by
`command_speed_threshold`, so stand-still samples do not receive swing-clearance
pressure.

`contact_timing` penalizes accepted touchdown events whose preceding air time is
shorter than `min_air_time`, and liftoff events whose preceding contact time is
shorter than `min_contact_time`. Shortfall is normalized by the configured
minimum and squared. Mature transitions score zero. Because this is an event
term, its weight is not multiplied by the control timestep. Linear commands at
or below `command_speed_threshold` are gated out.

`swing_foot_velocity` rewards the scheduled swing foot for matching a
directional XY velocity relative to `link`, measured in that link's yaw frame.
The target is `duty_factor / (1 - duty_factor) * command[:2]`, capped by
`max_target_speed`, and the score is `exp(-squared_error / std)`. It is disabled
when linear command speed is at or below `command_speed_threshold`; pure yaw and
stand-still commands therefore receive no swing-velocity pressure. The easy and
hard locomotion recipes enable the term at weight `0.5` with `pelvis_link` as
the reference body.

`feet_air_time` is a continuous positive-biped reward. While exactly one foot
is in debounced contact, it takes the shorter of the stance foot's current
contact duration and the swing foot's current air duration, capped by
`max_duration`. Double support and flight receive zero reward. The term is
disabled for near-standing commands by `command_speed_threshold` and, like
other continuous terms, is multiplied by the control timestep.

`standing_foot_vel` and `standing_foot_air_time` are near-zero-command
penalties. They are active only when command speed is at or below their
`command_speed_threshold`, and discourage stepping in place by penalizing
horizontal foot speed and accumulated foot air time during stand-still samples.

`self_collision` counts active robot-robot contact slots whose normal force is
above `env_config.contact.self_collision_force_threshold`. It is timestep
scaled like the continuous contact and smoothness terms.

### DoneConfig

```yaml
done_config:
  base_height:
    enabled: true
    min_height: 0.75
    link: null
  bad_orientation:
    enabled: true
    limit_angle: 1.0
    link: null
  timeout:
    enabled: true
  nonfinite:
    enabled: true
```

`done/any` is true for either terminal failure or timeout and drives env resets
and completed-episode logging. `done/terminal` is true only for failure
conditions, currently base height, bad orientation, or nonfinite state. Timeout
is treated as truncation: it resets the env and emits `done/timeout`, but does
not receive the `reward/termination` penalty.

`bad_orientation` matches IsaacLab/Unitree-style tilt termination: it computes
the angle between the configured link's projected gravity vector and the local
z-axis as `acos(-projected_gravity.z)`, then terminates when that angle exceeds
`limit_angle`. `base_height` and `bad_orientation` validate their configured
links during env initialization. When their `link` is null, `env_config.root_link`
is validated and used.

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
  upload_wandb_artifact: true
  wandb_artifact_name: null
  wandb_artifact_type: checkpoint
```

If `name` is null, the checkpoint directory base name is:

```text
ppo_seed_<seed>
```

Training appends a local timestamp to the base name before saving, for example
`ppo_seed_0_20260712_013045_123456`. Checkpoints are never overwritten.

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

Training always runs as a Python loop over JIT-compiled PPO updates. When
`wandb_config.enabled=true`, scalar metrics are logged after every PPO update.
The per-update loop is wrapped with a tqdm `PPO updates` progress bar. The loop
keeps only the latest metrics in host memory, so long runs do not accumulate a
full metric history locally.

### TrainingVideoConfig

Training videos are rendered from the training loop after selected PPO updates.
Rendering is synchronous and uses the current policy checkpoint held in memory.

```yaml
video_config:
  enabled: false
  every_updates: 100
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
  stochastic_actions: false
  action_seed: 0
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
`MjData` while still evaluating the current policy action path. This is the
default for training videos because it avoids allocating extra MJX-Warp contact
buffers during an already memory-heavy PPO update loop.

`simulation_backend=mjx` renders through the same MJX/Warp stepping path used by
training. Use it for pipeline-parity diagnostics when the question is whether
the policy, observation stack, action clipping, and reset behavior match the
training rollout. It may allocate additional MJX/Warp buffers while training is
running.

`stochastic_actions=true` samples video actions from the actor's Gaussian
distribution using the learned `log_std`, then clips them to the env action
range before stepping. `action_seed` makes that rollout reproducible. Leave
`stochastic_actions=false` when you want deterministic actor-mean inspection;
enable it when comparing videos to PPO's stochastic training rollouts.

When videos are logged to W&B, the video caption includes the rendered rollout's
own `first_done_step`, `first_done_seconds`, completed episode count, and done
cause counts. The training loop does not emit these single-rollout diagnostics as
scalar metric series; normal `train/mean_*` metrics still describe the batched
PPO training rollout.

For stacked-observation policies, video rollouts use the checkpoint policy stack
size. When an episode ends, video rollouts reset the env and rebuild the stack
from the reset observation. With `stop_on_done=false`, recording continues into
the next episode; with `stop_on_done=true`, the rollout stops at the first done
frame.

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
4. Normalize advantages across the full rollout batch.
5. If `normalize_value=true`, update the running target-return statistics.
6. Flatten `[num_steps, num_envs]` into one batch.
7. Shuffle and split into `num_minibatches`.
8. Run `update_epochs` passes over minibatches, optionally stopping remaining
   minibatches in the current PPO update when `target_kl` is configured and
   `approx_kl > target_kl`.
9. Emit scalar metrics for the update.

When `normalize_value=true`, the critic predicts normalized values. Rollout and
GAE denormalize those predictions back into raw reward units, while the value
loss compares normalized critic outputs against normalized raw targets. Actor
advantages remain based on raw rewards and are normalized once across the full
rollout batch before minibatches are shuffled, matching the rsl_rl default.

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
log-probs only when the policy helper is called with an explicit action clip.
The PPO training loop stores the raw sampled Gaussian action/log-prob for PPO
ratios and passes the raw request to the environment. Environments own their
action bounds so they can distinguish the requested action from the control
actually applied.

The Bello locomotion env exposes configurable `env_config.action_clip`, which
defaults to `5.0`. It clips the action for control and observations while using
the pre-clip request for `invalid_action`.
Bello maps the clipped action linearly to position-actuator control targets with
`default_ctrl + policy_action_scale * clipped_action`, matching the common
legged-gym/IsaacLab residual-position convention. Generated position actuators
use `ctrllimited="false"` so MuJoCo/MJX does not introduce actuator-layer target
saturation. Motor force limits remain active through generated `forcerange` /
`forcelimited="true"` actuator attributes.

CartPole exposes `action_clip=1.0`; Bello defaults to `action_clip=5.0`.

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
nonfinite_update
target_kl_exceeded
ppo_update_fraction
value_norm_mean
value_norm_std
value_norm_count
mean_reward
mean_done
mean_done/any
mean_done/base_height
mean_done/terminal
mean_done/bad_orientation
mean_done/timeout
mean_done/nonfinite
mean_episode_return
max_episode_return
mean_episode_length
max_episode_length
action_rms
action_abs_max
env_action_abs_max
action_mean_rms
action_mean_abs_max
action_std_mean
action_std_min
action_std_max
action_clip_fraction
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

`target_kl` is a same-update guardrail. When it is not null and a minibatch
reports `approx_kl > target_kl`, PPO skips the remaining minibatch updates in
that PPO update. `target_kl_exceeded` reports the fraction of minibatches whose
KL exceeded the threshold, and `ppo_update_fraction` reports the fraction of
minibatches that actually applied optimizer updates.
`nonfinite_update` reports the fraction of minibatches whose loss, metrics,
gradients, updates, or next parameters became nonfinite. Such minibatches are
not applied, and the training CLI raises immediately instead of continuing with
an invalid policy state.

PPO stores both the policy action and its exact pre-transform Gaussian sample.
Log-prob ratios for tanh policies use the stored latent sample instead of trying
to invert a float32 action that may have saturated at exactly `-1` or `1`.
`env_action_abs_max` reports the bounded action that was actually applied, and
`action_clip_fraction` reports the fraction of sampled action dimensions changed
by the environment bound.
For Bello, `ctrl_saturation_fraction` reports the fraction of requested policy
control targets at or beyond their configured MuJoCo ctrl-range bounds.
`ctrl_margin_min` is the smallest signed distance to
either bound across sampled policy controls in the rollout; it is negative when
requested targets exceed the soft range.

`policy.action_distribution=tanh_gaussian` samples a latent diagonal Gaussian
and maps it through `action_limit * tanh(x)`. PPO includes the tanh Jacobian in
the transformed log probability, while deterministic evaluation uses the
transformed actor mean. The default `gaussian` preserves existing experiment
behavior.

`policy.normalize_observations=true` maintains independent running mean and
variance statistics for stacked actor and critic observations. The statistics
are updated from each completed rollout for use by the next rollout, included
in checkpoints as part of the policy, and automatically applied during
deterministic evaluation.

Locomotion training also logs `gait/footfalls_per_second`,
`gait/liftoffs_per_second`, `gait/premature_touchdowns_per_second`,
`gait/premature_liftoffs_per_second`, `mean_gait/stance_width`, and mean
single-support, double-support, and flight fractions. These diagnostics are
emitted even when gait shaping rewards are disabled, so ablations remain
directly comparable.

## Checkpoints

Checkpointing is controlled by `CheckpointConfig`:

```yaml
enabled: false
directory: checkpoints
name: null
```

When enabled, training writes:

```text
<directory>/<name-or-ppo_seed_N>_<YYYYMMDD_HHMMSS_microseconds>/
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
critic_obs_size  # critic input size when privileged critic observations are configured
action_size
value_norm  # mean, variance, and count for normalized value targets
```

Evaluation reconstructs the env and policy from checkpoint metadata. The
checkpoint must match the code schema used to restore it.

If `upload_wandb_artifact=true` and W&B is enabled, the final checkpoint
directory is also logged as a W&B artifact after the local checkpoint save
completes. This is the default for training configs; it is inert unless both
checkpointing and W&B are enabled. `wandb_artifact_name=null` uses the
checkpoint directory name.

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

For training runs, W&B is initialized before MuJoCo/MJX environment setup and
uses W&B's thread start method. This avoids Python `os.fork()` warnings after
JAX has initialized worker threads.

When enabled:

- the full run config is logged to `run.config`
- training metrics are logged with a `train/` metric-name prefix
- metrics stream after every PPO update
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
  every_updates: 100
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
target_kl is null, or target_kl > 0
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
activation in {"tanh", "relu", "elu"}
hidden_size > 0
hidden_depth >= 0
hidden_dims is null, or every hidden_dims entry is > 0
init_log_std is finite
min_log_std is finite
max_log_std is finite
min_log_std <= max_log_std
obs_stack_size > 0
```

`obs_stack_size` controls actor and critic observation stacking. The environment
still emits one raw actor observation per step, and can optionally expose a
separate `env_config.critic_observation` stream for privileged value-function
inputs. PPO repeats initial observations on reset, shifts in each new
observation, resets each completed env's stacks to reset observations, and feeds
flattened stacks to the actor and critic. Actor input size is
`env.observation_size * config.policy.obs_stack_size`; critic input size is
`env.critic_observation_size * config.policy.obs_stack_size` when
`critic_observation` is configured, otherwise it matches the actor input size.

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
