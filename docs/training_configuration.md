# Training configuration

`bello-train` deliberately exposes a small command surface. Maintained YAML
profiles and immutable Python dataclass defaults jointly define the complete
training contract; YAML may omit fields that intentionally inherit their
dataclass defaults. The resolved mapping is decoded and validated before model
construction.

This document describes the nested configuration. The checked-in YAML remains
the source of maintained experiment overrides:

- [Locomotion defaults](../configs/train/default.yaml)
- [Motion defaults](../configs/train_motion/default.yaml)
- [Smoke profile](../configs/profiles/train/smoke.yaml)
- [Locomotion profile](../configs/profiles/train/locomotion.yaml)
- [Hardware-locomotion profile](../configs/profiles/train/hardware_locomotion.yaml)
- [Motion profile](../configs/profiles/train/motion.yaml)

## Resolution and ownership

Profile layers are merged from task defaults to experiment-specific overrides.
Hydra resolves interpolations, then strict dataclass decoding rejects unknown
keys. The narrow fields documented under
[`bello-train`](commands.md#bello-train) are applied last.

Do not add a second recipe in a script or command wrapper. Change:

- `configs/train/default.yaml` for shared locomotion contracts;
- `configs/train_motion/default.yaml` for shared motion contracts;
- `configs/profiles/train/*.yaml` for maintained experiment numerics;
- Python dataclasses only when the configuration schema or validation changes.

## PPO: `config`

The `config` section is shared by locomotion and motion tracking.

| Field | Locomotion base | Meaning and constraint |
| --- | --- | --- |
| `seed` | `0` | Root PRNG seed for policy initialization, rollout, and minibatch ordering. |
| `total_timesteps` | `1,000,000` | Transition budget. It must cover at least one `num_envs * num_steps` rollout batch. |
| `num_envs` | `64` | Parallel environment count. |
| `num_steps` | `24` | Rollout horizon per environment and PPO update. |
| `update_epochs` | `5` | Optimizer passes over each rollout batch. |
| `num_minibatches` | `4` | Minibatches per epoch. `num_envs * num_steps` must be divisible by this value. |
| `learning_rate` | `0.001` | Initial Adam learning rate; positive and finite. |
| `schedule` | `adaptive` | `constant`, `linear`, or KL-adaptive scheduling. |
| `desired_kl` | `0.01` | KL target used by the adaptive scheduler. |
| `target_kl` | `null` | Optional hard early-stop threshold for an update epoch. |
| `min_learning_rate` | `0.00001` | Lower adaptive-rate bound. |
| `max_learning_rate` | `0.01` | Upper adaptive-rate bound. |
| `adaptive_lr_factor` | `1.5` | Multiplicative adaptive-rate step; must exceed one. |
| `adaptive_kl_statistic` | `max` | Use `mean` or `max` minibatch KL for adaptation. Motion profiles use `mean`. |
| `gamma` | `0.99` | Reward discount in `[0,1]`. |
| `gae_lambda` | `0.95` | GAE trace parameter in `[0,1]`. |
| `clip_eps` | `0.2` | Positive PPO probability-ratio clip radius. |
| `ent_coef` | `0.01` | Nonnegative entropy coefficient. |
| `vf_coef` | `1.0` | Nonnegative value-loss coefficient. |
| `max_grad_norm` | `1.0` | Positive optimizer gradient clipping norm. |
| `normalize_value` | `false` | Track return moments and train the critic in normalized value space. |
| `value_norm_eps` | `1e-8` | Positive numerical floor for value normalization. |
| `normalize_observations` | `false` | Track and apply observation moments before the actor and critic. |
| `observation_norm_eps` | `1e-5` | Positive observation-normalization variance floor. |
| `observation_norm_clip` | `10` | Positive absolute clip after observation normalization. |
| `rollout_observation_max_abs` | `10,000` | Fail an update when rollout observations exceed this finite bound. |
| `rollout_reward_max_abs` | `1,000` | Fail an update when rewards exceed this bound. |
| `rollout_value_max_abs` | `10,000` | Fail an update when critic outputs exceed this bound. |
| `target_max_abs` | `10,000` | Fail an update when value targets exceed this bound. |
| `gradient_norm_max` | `10,000` | Independent nonfinite/explosion guard beyond optimizer clipping. |

The effective update count is
`total_timesteps // (num_envs * num_steps)`. Extra transitions below one full
rollout batch are not sampled.

### Policy: `config.policy`

| Field | Base default | Meaning |
| --- | --- | --- |
| `hidden_dims` | `[256,256]` | Actor and critic hidden widths. |
| `activation` | `tanh` | `tanh`, `relu`, `elu`, or `swish`. |
| `init_log_std` | `0` | Initial per-action Gaussian log standard deviation. |
| `min_log_std` | `-5` | Lower learned log-standard-deviation clamp. |
| `max_log_std` | `1` | Upper learned log-standard-deviation clamp. |
| `obs_stack_size` | `10` | Number of consecutive observations flattened into policy input. |
| `action_distribution` | `tanh_gaussian` | Bounded `tanh_gaussian` or unbounded `gaussian`. Deployment profiles use the bounded form. |
| `action_limit` | `1` | Squashed action magnitude. For bounded policies it must equal the environment `action_clip`. |

## Assets: `asset_config`

The asset section selects a deterministic prepared morphology.

| Field | Locomotion baseline | Meaning |
| --- | --- | --- |
| `asset` | `full_body` | The only supported prepared asset. |
| `output_root` | `null` | Optional prepared-asset destination for standalone preparation. Training requires `null` so an exact source revision regenerates through the canonical content-addressed cache. |
| `collision_mode` | `box` | `box` or simplified `mesh` collisions. Maintained policies use boxes. |
| `mesh_face_budget` | `1000` | Per-mesh collision simplification budget. |
| `visual_mesh_face_budget` | `200000` | Per-mesh viewer simplification budget. |
| `end_effector_sphere_mode` | `both` | Replace both wrist/gripper subtrees with fixed sphere hands. |
| `target_total_mass_kg` | `60` | Uniformly rescale all link masses and inertias to the target total. |
| `neighbor_exclusion_depth` | `2` | Exclude nearby kinematic neighbors from generated self-collision pairs. |
| `enforce_policy_effort_limits` | `false` | When false, training can observe and penalize over-peak requested force. The native emulator and hardware still enforce physical limits. |
| `imu_config` | disabled | Optional generated IMU site/sensors. Its `enabled` field controls generation and `body_name` selects the instrumented body. Training observations derive simulated IMU features without requiring this native sensor block. |
| `ankle_differential_config.motor_gain_scale` | `0.5` | Gain multiplier applied to each differential-ankle motor. |
| `ankle_differential_config.motor_effort_limit_scale` | `1` | Motor effort-limit multiplier. |
| `joint_availability.*` | waist yaw on; waist roll/pitch and neck/head off | Fix optional joints at asset-generation time. |
| `actuator_config` | canonical Bello map | Its `joints` mapping assigns a `JointActuatorConfig` to each joint. |

The `ankle_differential_config` fields are `motor_gain_scale` and
`motor_effort_limit_scale`. `joint_availability` has the boolean fields
`waist_yaw`, `waist_roll`, `waist_pitch`, `neck_yaw`, and `head_pitch`.
Each `actuator_config.joints` entry has a `mode` (`policy` or `hold`),
`default_pos`, optional `soft_limit`, `[kp, kd]` `gains`, and optional
`effort_limit`.

The complete asset configuration is stored in checkpoint metadata. By default,
a checkpoint runs from its exact clean Git revision, so its source descriptions
and asset-generation implementation come from that revision. Training rejects
a custom `output_root`.

Motion training supplies its `asset_config` from the
`MotionTrainingRunConfig` default factory rather than its YAML. The fixed
tracking contract is the canonical 60 kg, box-collision, sphere-hand asset with
waist yaw enabled; waist roll/pitch and neck/head joints are fixed, and all 25
remaining coordinates are policy-controlled. Motion initialization rejects an
asset that deviates from this contract.

## Locomotion environment: `env_config`

| Field | Meaning |
| --- | --- |
| `decimation` | Physics steps per policy step. The control period is `simulation_physics.timestep * decimation`. |
| `max_episode_steps` | Timeout horizon in policy steps. |
| `action_clip` | Environment action bound; must match a bounded policy's `action_limit`. |
| `root_link` | Floating-base link used by environment indexing. |
| `simulation_physics` | Runtime timestep and optional foot/terrain contact overrides. |
| `terrain` | Plane, rough heightfield, or robot-only scene parameters. |
| `model_randomization` | Per-environment joint-origin, angle-offset, and inertia variation. |
| `spawn` | Initial base pose, joint pose/velocity, root velocity, and episode staggering. |
| `contact` | Contact debounce durations and force thresholds. |
| `random_push` | Periodic base linear-velocity perturbations. |
| `numerical_safety` | Broad finite-state and solver-force bounds that terminate invalid rollouts. |
| `observation` | Deployable actor observation scaling and noise. |
| `observation_latency` | Per-episode measured actor-feedback delay range in policy steps. |
| `gait` | Fixed gait frequency and stance fraction shared by simulation, native evaluation, and hardware. |
| `critic_observation` | Critic-only observation; maintained locomotion profiles use privileged fields. |

### Simulation physics

The `simulation_physics` field `timestep` must be positive. Optional
`foot_terrain_margin`, `foot_terrain_solref`, and `foot_terrain_solimp` replace
the generated foot/terrain contact parameters. A custom `solref` time constant
must be at least twice the timestep.

### Terrain

`terrain.mode` is `none`, `plane`, or `rough`.

| Field | Meaning |
| --- | --- |
| `name`, `pos`, `size`, `rgba` | Generated terrain geom identity, pose, extent, and color. |
| `friction`, `condim` | MuJoCo sliding/torsional/rolling friction and contact dimension. |
| `seed` | Deterministic slope/heightfield seed. |
| `hfield_resolution`, `hfield_amplitude`, `hfield_base_thickness`, `hfield_frequency` | Rough-terrain heightfield dimensions and shape. |
| `slope_angle_range` | Ordered slope bounds in radians strictly inside `(-pi/2,pi/2)`. |

### Model and reset randomization

`model_randomization` is fixed per vectorized world.

| Field | Meaning |
| --- | --- |
| `enabled`, `seed` | Enable deterministic per-world randomization and select its seed. |
| `joint_origin_range` | Ordered additive joint-origin offset range. |
| `joint_angle_offset_range` | Ordered default joint-angle offset range. |
| `joint_angle_offset_scope` | Apply angle offsets to `all` active coordinates or only `policy` coordinates. |
| `joint_inertia_scale_range` | Ordered positive inertia multiplier range. |

`spawn` independently samples episode-level initial state.

| Field | Meaning |
| --- | --- |
| `clearance` | Vertical clearance above the resolved terrain surface. |
| `root_xy_range`, `root_yaw_range` | Uniform base-position and yaw ranges. |
| `joint_pos_noise`, `joint_pos_scale_range` | Additive and multiplicative joint-position variation. |
| `joint_vel_range`, `root_vel_range` | Initial joint and floating-base velocity ranges. |
| `stagger_initial_step_count` | Spread initial timeout progress across a batch. |

Contact and disturbance fields are:

| Section | Fields |
| --- | --- |
| `contact` | `foot_contact_on_seconds`, `foot_contact_off_seconds`, `foot_contact_force_threshold`, `undesired_contact_force_threshold`, `self_collision_force_threshold` |
| `random_push` | `enabled`, `interval_seconds`, `lin_vel_x`, `lin_vel_y` |
| `numerical_safety` | `max_abs_position`, `max_abs_velocity`, `max_abs_acceleration`, `max_abs_actuator_force`, `max_abs_constraint_force`, `max_abs_reward_term` |

Contact on/off durations debounce raw force-threshold contact. Random pushes add
base planar velocity at the configured interval. Numerical safety values are
broad fail-fast bounds, not training objectives.

### Actor and critic observations

Both `observation` and `critic_observation` contain `privileged`, `scale`, and
`noise`. The `noise` section also has an `enabled` switch.

Scale and noise fields are `base_lin_vel`, `base_lin_acc`, `base_ang_vel`,
`projected_gravity`, `base_height`, `command`, `gait_phase`, `joint_pos`,
`joint_vel`, `actuator_force`, and `last_action`. Scales multiply observations;
noise values are ordered uniform ranges applied before stacking.

With commands enabled, the actor and critic both receive a three-value gait
component: sine phase, cosine phase, and a gait-active bit. The default 0.9 Hz
per-leg clock has a 60% stance fraction and a half-cycle offset between the two
feet. Standing resets its phase and makes the complete component exactly zero.
`frequency_hz` is the complete cycle frequency of each leg, so the default
repeats each leg about every 1.11 seconds and offsets nominal left/right
footfalls by about 0.56 seconds: 1.8 total footfalls per second, or 108 steps
per minute. `stance_fraction` must be in `[0.5, 1.0)`.

Actor observations are the hardware policy input. Hardware locomotion rejects
a checkpoint whose actor observation is privileged. Critic-only fields do not
enter the deployed policy.

The `observation_latency` section contains `enabled` and `delay_steps`, an
inclusive integer lower/upper range. The maintained locomotion and
hardware-locomotion profiles sample a fixed delay of 0, 1, or 2 policy steps
per episode, corresponding to 0--40 ms at 50 Hz. The delayed terms are IMU
linear acceleration, angular velocity, projected gravity, and joint
position/velocity. Command and previous action remain current, and the
privileged critic remains undelayed. Native evaluation and physical deployment
do not inject synthetic delay. Older locomotion checkpoints load with latency
disabled; resuming their rollout continuation still requires the original
source revision because its environment state predates the feedback history.

## Locomotion commands: `command_config`

| Field | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Disable to train only on zero command. |
| `lin_vel_x` | `[-0.5,0.5]` | Lateral/right velocity sampling range. |
| `lin_vel_y` | `[-0.5,0.5]` | Forward velocity sampling range. |
| `ang_vel_yaw` | `[-0.5,0.5]` | Yaw-rate sampling range. |
| `standing_probability` | `0.1` | Fraction of batched training environments assigned an exact all-zero command cohort; scalar sampling uses this as a probability. |
| `linear_deadband` | `0.1` | Actor/reward standing threshold for XY command. |
| `yaw_deadband` | `0.1` | Standing threshold for yaw command. |
| `minimum_moving_linear_speed` | `0.2` | A selected moving linear axis is sampled outside this magnitude. |
| `minimum_moving_yaw_speed` | `0.2` | A selected moving yaw axis is sampled outside this magnitude. |
| `resample_seconds` | `10` | Interval between command samples. |

Moving samples are conditioned so at least one available axis lies outside its
minimum moving speed. The minimums must exceed their respective deadbands.
Training assigns the standing cohort once at batch reset and preserves each
environment's standing or moving mode across failures and periodic command
resampling. This prevents short standing episodes from reducing standing's
share of rollout frames. Training telemetry reports mode-specific mean reward,
completed-episode return and length, and completed-episode count under the
`command/standing_*`, `command/moving_*`, and `episode/{standing,moving}_*`
names.

## Locomotion reward: `reward_config`

Every term is computed independently and multiplied by its configured weight.
The `weights` section is listed below; a zero weight disables metadata and
computation where possible.

| Weight | Baseline | Effect |
| --- | ---: | --- |
| `track_lin_vel_xy` | `1.0` | Gaussian command tracking for planar velocity. |
| `track_ang_vel_z` | `1.0` | Gaussian yaw-rate command tracking. |
| `healthy_survival` | `1.5` | Reward each non-terminal control step so early negative shaping costs cannot make immediate failure preferable to continued control. |
| `lin_vel_z` | `-2.0` | Penalize vertical base velocity. |
| `ang_vel_xy` | `-0.15` | Penalize roll/pitch angular velocity. |
| `flat_orientation` | `-2.0` | Penalize gravity-vector tilt. |
| `base_height` | `-10.0` | Penalize error from `limits_pose.base_height_target`. |
| `action_rate` | `-0.01` | Penalize change from the previous action. |
| `action_limit` | `-1.0` | Penalize requested action beyond the soft action ratio. |
| `torque_l1` | `-1.2` | Penalize absolute requested torque normalized by each joint's peak torque. |
| `torque_peak` | `-12.0` | Quadratically penalize normalized torque above peak. |
| `joint_force_peak` | `-4.0` | During commanded gait, quadratically penalize the two largest hip, knee, or ankle reaction-force excesses above `0.8` body weight. It is disabled for standing because adjacent joints in a support chain can report the same static load. This uses MuJoCo's parent-child linear interaction force, not actuator torque. |
| `touchdown_impact` | `-1.0` | At the first raw foot-contact sample, penalize both measured normal contact impulse above one body-weight control impulse and pre-impact downward foot speed. Each component is bounded before weighting so isolated solver spikes cannot dominate PPO. |
| `joint_pose` | `-0.1` | Penalize weighted deviation from nominal joint pose. |
| `stand_still` | `1.0` | At standing command, reward healthy double support in proportion to bounded foot stillness. Terminal steps receive no standing bonus. |
| `stand_foot_motion` | `-1.0` | At standing command, penalize bounded foot linear and angular motion. |
| `stand_contact` | `-0.5` | At standing command, penalize each foot that is not in filtered terrain contact. Together with `stand_still`, this favors quiet double support without making continued recovery intrinsically negative. |
| `hip_deviation` | `-0.25` | Penalize hip deviation, gated by lateral command. |
| `knee_deviation` | `-0.1` | Penalize knee deviation from nominal. |
| `joint_limit` | `-1.0` | Penalize proximity beyond the joint soft-limit factor. |
| `invalid_action` | `0.0` | Optional penalty for invalid/nonfinite requested action. |
| `gait_contact` | `-1.0` | Penalize disagreement between measured foot contact and the phase-conditioned stance schedule. |
| `gait_stance_slip` | `-0.25` | Penalize planar foot motion during the phase-conditioned stance interval. |
| `gait_swing_clearance` | `-0.5` | Penalize swing-foot clearance below the configured target. |
| `undesired_contact` | `0.0` | Optional non-foot terrain-contact penalty. |
| `self_collision` | `0.0` | Optional self-collision penalty. |
| `termination` | `-100.0` | Penalize non-timeout terminal failure. |

The `links` section selects the measured link for `track_lin_vel_xy`,
`track_ang_vel_z`, `lin_vel_z`, `ang_vel_xy`, `flat_orientation`, and
`base_height`; `track_lin_vel_xy_frame` optionally selects a separate velocity
frame. The maintained locomotion profile uses pelvis velocity/yaw and height,
with torso angular velocity and orientation.

Supporting fields are:

| Section | Fields |
| --- | --- |
| `tracking` | `lin_vel_xy_moving_std`, `lin_vel_xy_still_std`, `ang_vel_z_moving_std`, `ang_vel_z_still_std` |
| `standing` | `foot_linear_speed_scale`, `foot_angular_speed_scale`, `foot_angular_weight` |
| `gait` | `stance_slip_angular_speed_scale`, `swing_clearance_target`, `joint_force_soft_limit_body_weights`, `joint_force_top_k`, `joint_force_apply_while_standing`, `touchdown_force_soft_limit_body_weights`, `touchdown_vertical_speed_scale` |
| `limits_pose` | `base_height_target`, `action_soft_limit_ratio`, `joint_soft_limit_factor`, `hip_lateral_command_index`, `hip_lateral_command_threshold` |

Rollout telemetry reports the mean, p95, p99, and worst instantaneous normalized
joint and foot-contact force maxima. The tail metrics are named
`gait/{joint_force,contact_force}_{p95,p99,worst}_body_weight`; unlike the mean
reward terms, they expose short impacts that would otherwise be diluted across
joints and rollout frames. `command/standing_positive_reward_fraction` reports
the fraction of standing transitions whose total reward is positive; it should
rise toward one as quiet standing matures.

## Locomotion termination: `done_config`

| Section | Meaning |
| --- | --- |
| `base_height` | `enabled`, `min_height`, and optional `link` for terminal minimum height. |
| `bad_orientation` | `enabled`, `limit_angle`, and optional `link` for terminal projected-gravity tilt. |
| `timeout` | `enabled`; mark `max_episode_steps` as truncation rather than physical failure. |
| `nonfinite` | `enabled`; terminate finite-state or solver-reward violations. |

The maintained locomotion profile uses a 0.55 m pelvis minimum and a 0.80 rad
torso orientation limit.

## Motion selection and environment

The motion selector fields `directory` and `sequence` resolve to
`<directory>/<sequence>.npz`. The public `motion_path` override replaces both.
The archive's schema, source identity, frame count, joint/body names, and
content hash are stored in checkpoints.

### `env_config`

| Field | Meaning |
| --- | --- |
| `decimation` | Physics steps per policy step. |
| `action_clip` | Applied normalized-action bound. |
| `formulation` | `reference_residual`, `beyondmimic`, or `beyondmimic_joint_limits`. The maintained profile uses bounded joint-limit actions. |
| `anchor_body_name` | Body used for reference alignment and root-relative observations. |
| `terrain` | Native/MJX terrain contract. Maintained motion uses a plane. |
| `reset` | Start-time, XY/yaw, joint, root pose/velocity, and default-offset randomization. |
| `observation_noise` | Independent magnitudes for anchor pose, base velocity, and joint observation noise. |
| `observation_latency` | Per-episode actor-feedback delay range in policy steps. |
| `adaptive_sampling` | Failure-EMA binning, smoothing kernel, and uniform-sampling mixture. |
| `push` | Random interval and root linear/angular velocity perturbation magnitudes. |
| `reward` | Tracking term weights, Gaussian sigmas, action/joint/torque penalties, and undesired-contact threshold. |
| `done` | Root-height, root-orientation, and end-effector tracking-error limits. |

Nested motion fields are:

| Section | Fields |
| --- | --- |
| `reset` | `random_start`, `initial_xy_range`, `initial_yaw_range`, `joint_pos_noise`, `joint_vel_noise`, `root_position_noise`, `root_rotation_noise`, `root_linear_velocity_noise`, `root_angular_velocity_noise`, `joint_default_offset_noise` |
| `observation_noise` | `enabled`, `anchor_position`, `anchor_orientation`, `base_linear_velocity`, `base_angular_velocity`, `joint_position`, `joint_velocity` |
| `observation_latency` | `enabled`, `delay_steps` (inclusive integer lower/upper bounds) |
| `adaptive_sampling` | `enabled`, `bin_seconds`, `ema_alpha`, `kernel_size`, `kernel_decay`, `uniform_ratio` |
| `push` | `enabled`, `interval_seconds`, `linear_velocity`, `angular_velocity` |
| `done` | `root_height_error`, `root_orientation_error`, `end_effector_position_error` |

Observation latency applies only to measured actor feedback. Reference terms
and the previous action use the current policy step, while the privileged
critic remains undelayed. The maintained range `[0, 2]` corresponds to 0, 20,
or 40 ms at 50 Hz. The sampled delay remains fixed for an episode. Synthetic
delay is disabled in native evaluation and hardware deployment.

### Motion reward

Positive Gaussian terms cover root position/orientation, joint
position/velocity, tracked-body position/orientation, combined body velocity,
and separate body linear/angular velocity. Negative terms cover action rate,
joint limits, normalized actuator torque, torque above each actuator's peak
rating, and undesired contact. Torque terms are multiplied by the control
timestep. Each Gaussian term has a matching sigma; zero weight disables the
term.

The maintained `motion` profile emphasizes root and tracked-body pose plus body
linear/angular velocity, disables direct joint tracking, and uses strong
joint-limit and action-rate penalties. It does not enforce simulated policy
actuator effort limits: normalized L1 torque has weight `-1.2`, while normalized
torque above peak is penalized quadratically with weight `-12.0`.

The exact reward fields are:

| Kind | Fields |
| --- | --- |
| Positive weights | `root_position_weight`, `root_orientation_weight`, `joint_position_weight`, `joint_velocity_weight`, `body_position_weight`, `body_orientation_weight`, `body_velocity_weight`, `body_linear_velocity_weight`, `body_angular_velocity_weight` |
| Penalty weights | `action_rate_weight`, `joint_limit_weight`, `torque_l1_weight`, `torque_peak_weight`, `undesired_contact_weight` |
| Gaussian scales | `root_position_sigma`, `root_orientation_sigma`, `joint_position_sigma`, `joint_velocity_sigma`, `body_position_sigma`, `body_orientation_sigma`, `body_velocity_sigma`, `body_linear_velocity_sigma`, `body_angular_velocity_sigma` |
| Contact threshold | `undesired_contact_force_threshold` |

## Checkpoints: `checkpoint_config`

| Field | Meaning |
| --- | --- |
| `enabled` | Write periodic and final checkpoints. |
| `directory` | Parent output directory. It must be outside the source repository or covered by a Git ignore rule so saving cannot dirty the exact-source checkout. |
| `name` | Stable, single-component run label. Locomotion final checkpoints are timestamped roots; motion also maintains a `final` child. |
| `every_updates` | Periodic save interval, or `null` for final-only behavior. |
| `resume_path` | Checkpoint to continue, normally from its exact clean Git revision. |
| `allow_git_commit_mismatch` | Permit resume from a different clean commit. It requires `resume_path` and bypasses only commit equality. |
| `upload_wandb_artifact` | Upload saved checkpoints when a W&B run exists. |
| `wandb_artifact_name` | Explicit artifact name or profile-derived default. |
| `wandb_artifact_type` | W&B artifact type, normally `checkpoint`. |

Evaluation requires a clean worktree and, by default, the checkpoint's recorded
Git commit. `allow_git_commit_mismatch=true` bypasses only that equality check.
Checkpoint document integrity, metadata and state layout hashes, normalization
state, saved update, task, and actor/critic observation and action runtime
sizes/layouts are still validated.
Resume additionally validates and restores environment continuation,
observation stacks, RNG, current learning rate, sampler state, environment
layout, and the full resume contract. A changed `total_timesteps` is allowed
only when the resulting update budget still covers the saved update.

## W&B: `wandb_config`

| Field | Meaning |
| --- | --- |
| `enabled` | Create a W&B run. |
| `project`, `entity`, `group`, `job_type`, `run_name` | Standard run identity and grouping. |
| `mode` | `online`, `offline`, or `disabled`. |
| `dir` | Local W&B state directory. |
| `tags` | Immutable run tags. |

Metrics use environment transitions as the W&B step. Checkpoint artifacts and
native evaluation videos are attached only when both their local feature and
the W&B run are enabled.

## Evaluation videos: `video_config`

Locomotion video fields are `enabled`, `every_updates`, `render_initial`,
`render_final`, `output_dir`, `filename_prefix`, `command`, `steps`, `width`,
`height`, `fps`, `stop_on_done`, `stochastic_actions`, `action_seed`,
`log_wandb`, `wandb_name`, and `camera_config`. The camera contains `mode`,
`distance`, `azimuth`, `elevation`, and `lookat_offset`.

Motion video fields are `enabled`, `every_updates`, `render_initial`,
`render_final`, `output_dir`, `filename_prefix`, `steps`, `width`, `height`,
`fps`, `log_wandb`, and `wandb_name`. Motion video always renders deterministic
tracking and does not accept a locomotion command or camera override.

Training video generation uses native MuJoCo outside the JAX rollout hot path.
Locomotion renders in the host training process; motion uses a spawned CPU
evaluation worker.

## Warp capacity: `warp`

| Field | Meaning |
| --- | --- |
| `contact_budget_per_env` | Default active-contact capacity multiplied by batch size when `naconmax` is not explicit. |
| `naconmax` | Explicit active-contact capacity or `null` for the derived value. |
| `naccdmax` | Optional collision-detection candidate capacity. |
| `njmax` | Constraint/Jacobian capacity. |

Capacity that is too small can fail or truncate contact processing; excessive
capacity increases device memory. The maintained locomotion and motion
profiles raise these values for their larger batches and contact workloads.

## Maintained profile intent

| Profile | Intent |
| --- | --- |
| `smoke` | One real two-environment PPO update with checkpointing, W&B, and video disabled. |
| `locomotion` | 2,048-environment gait training with slope, model/reset randomization, observation noise and latency, pushes, a larger policy, and periodic checkpoint/video evaluation every 512 updates. |
| `hardware_locomotion` | Locomotion plus longer `gamma`/GAE credit assignment and hardware-specific naming. |
| `motion` | 3,072-environment bounded motion tracking with observation and value normalization, adaptive failure sampling, pushes, and periodic checkpoint/video evaluation every 500 updates. |

Headline resolved PPO values are:

| Profile | Timesteps | Envs × steps | Epochs × minibatches | Learning rate | Gamma / GAE | Policy |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `smoke` | 4 | 2 × 2 | 1 × 1 | 0.001 | 0.99 / 0.95 | 256-256 tanh |
| `locomotion` | 150,000,000 | 2,048 × 20 | 4 × 32 | 0.0001 adaptive | 0.97 / 0.95 | 512-256-128 swish |
| `hardware_locomotion` | 8,000,000,000 | 2,048 × 20 | 4 × 32 | 0.0001 adaptive | 0.995 / 0.98 | 512-256-128 swish |
| `motion` | 2,949,120,000 | 3,072 × 24 | 5 × 4 | 0.001 adaptive | 0.99 / 0.95 | 512-256-128 ELU |

`bello-train --cfg job` prints the outer public command document; profile
resolution occurs afterward inside the command and is therefore not included
in that Hydra output. Keep maintained numerical changes in the corresponding
YAML layer and schema/default changes in the owning dataclass.
