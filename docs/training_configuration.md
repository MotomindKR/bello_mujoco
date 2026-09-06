# Training configuration

`bello-train` deliberately exposes a small command surface. Maintained YAML
profiles and immutable Python dataclass defaults jointly define the complete
training contract; YAML may omit fields that intentionally inherit their
dataclass defaults. The resolved mapping is decoded and validated before model
construction.

This document describes the nested configuration. The checked-in YAML remains
the source of maintained experiment overrides:

- [Locomotion base defaults](../configs/train/default.yaml)
- [Shared production locomotion recipe](../configs/train/locomotion.yaml)
- [Motion defaults](../configs/train_motion/default.yaml)
- [TWIST defaults](../configs/train_twist/default.yaml)
- [TWIST2 defaults](../configs/train_twist2/default.yaml)
- [Smoke profile](../configs/profiles/train/smoke.yaml)
- [Bello locomotion profile](../configs/profiles/train/bello_locomotion.yaml)
- [Bello Mini locomotion profile](../configs/profiles/train/bello_mini_locomotion.yaml)
- [Motion profile](../configs/profiles/train/motion.yaml)
- [TWIST teacher profile](../configs/profiles/train/twist_teacher.yaml)
- [TWIST student profile](../configs/profiles/train/twist_student.yaml)
- [TWIST2 profile](../configs/profiles/train/twist2.yaml)

## Resolution and ownership

Profile layers are merged from task defaults to experiment-specific overrides.
Hydra resolves interpolations, then strict dataclass decoding rejects unknown
keys. The narrow fields documented under
[`bello-train`](commands.md#bello-train) are applied last.

Do not add a second recipe in a script or command wrapper. Change:

- `configs/train/default.yaml` for locomotion base contracts;
- `configs/train/locomotion.yaml` for the production recipe shared by both
  locomotion robots;
- `configs/train_motion/default.yaml` for shared motion contracts;
- `configs/train_twist/default.yaml` for shared TWIST contracts;
- `configs/train_twist2/default.yaml` for the one-stage TWIST2 contract;
- `configs/profiles/train/*.yaml` for robot selection, dimensioned locomotion
  settings, and other maintained experiment numerics;
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
| `normalize_observations` | `false` | Track and apply actor observation moments; critic behavior is controlled separately. |
| `normalize_critic_observations` | `true` | Apply and update a separate critic observation normalizer when observation normalization is enabled. |
| `observation_norm_update_mode` | `online` | `online` applies newly accumulated moments immediately; `rollout` holds them fixed until the next rollout. |
| `observation_norm_update_updates` | `null` | Optional positive update count after which actor observation moments freeze. |
| `observation_norm_epsilon_mode` | `variance_floor` | `variance_floor` uses `sqrt(max(var, eps))`; `std_offset` uses `sqrt(max(var, 1e-8)) + eps`. |
| `observation_norm_eps` | `1e-5` | Positive epsilon interpreted by `observation_norm_epsilon_mode`. |
| `observation_norm_clip` | `10` | Positive absolute clip after observation normalization. |
| `rollout_observation_max_abs` | `10,000` | Fail an update when rollout observations exceed this finite bound. |
| `rollout_reward_max_abs` | `1,000` | Fail an update when rewards exceed this bound. |
| `rollout_value_max_abs` | `10,000` | Fail an update when critic outputs exceed this bound. |
| `target_max_abs` | `10,000` | Fail an update when value targets exceed this bound. |
| `gradient_norm_max` | `10,000` | Independent nonfinite/explosion guard beyond optimizer clipping. |
| `symmetry_loss_coef` | `0` | Nonnegative actor left/right consistency coefficient. The maintained locomotion profile leaves it disabled. |

The effective update count is
`total_timesteps // (num_envs * num_steps)`. Extra transitions below one full
rollout batch are not sampled.

### PPO update diagnostics

`train/policy_clip_fraction` reports the fraction of policy-training samples
whose probability ratio is outside `[1 - clip_eps, 1 + clip_eps]`. This is the
standard PPO `clipfrac`; it is distinct from `train/action_clip_fraction`, which
reports actions changed by the environment's physical action clamp.

There is no universal target because the useful range depends on batch size,
minibatch count, update epochs, and the policy distribution. As a practical
diagnostic, a sustained value around 0.05-0.20 usually indicates meaningful but
controlled updates. Values below 0.05 are conservative only when KL is also low
and returns have stalled. Sustained values around 0.20-0.30 deserve attention,
and values above 0.30 usually mean that too much of the batch lies outside the
PPO trust region. Interpret isolated early-training spikes with
`train/mean_approx_kl`, `train/max_approx_kl`, `train/ppo_update_fraction`, and
episode returns rather than treating clip fraction as a standalone target.

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
| `asset` | `bello` | Selects a registered robot contract. `bello` and `bello_mini` have maintained locomotion profiles; Mini motion, GMR, emulator, and hardware remain calibration-gated. |
| `output_root` | `null` | Optional prepared-asset destination for standalone preparation. Training requires `null` so an exact source revision regenerates through the canonical content-addressed cache. |
| `collision_mode` | `box` | `box` or simplified `mesh` collisions. Maintained policies use boxes. |
| `mesh_face_budget` | `1000` | Per-mesh collision simplification budget. |
| `visual_mesh_face_budget` | `200000` | Per-mesh viewer simplification budget. |
| `target_total_mass_kg` | `null` | Optional experimental uniform rescaling. Maintained locomotion uses each asset's canonical inertials without rescaling: about 55.07 kg for Bello and 43.50 kg for Bello Mini. |
| `end_effector_payload_mass_kg` | `0` | Equal payload mass attached at each sphere-hand center. This is a per-hand mass; it is added after any robot-mass rescaling and becomes part of the asset/checkpoint fingerprint. |
| `neighbor_exclusion_depth` | `2` | Exclude nearby kinematic neighbors from generated self-collision pairs. |
| `enforce_policy_effort_limits` | `true` | Enforce the same physical actuator-force saturation used by hardware. Locomotion separately reconstructs the unclipped servo demand for the over-peak reward. |
| `imu_config` | disabled | Optional generated IMU site/sensors. Its `enabled` field controls generation and `body_name` selects the instrumented body. Training observations derive simulated IMU features without requiring this native sensor block. |
| `ankle_differential_config.motor_gain_scale` | `0.5` | Gain multiplier applied to each differential-ankle motor. |
| `ankle_differential_config.motor_effort_limit_scale` | `1` | Motor effort-limit multiplier. |
| `joint_availability.disabled_joint_names` | `[]` | Explicit configured actuator joints to fix at asset-generation time. Unknown and duplicate names are rejected. Bello's waist roll/pitch and neck/head joints are already fixed in its URDF; Bello Mini's waist roll remains movable and policy-controlled. |
| `actuator_mode` | `null` | Optional mode applied to every robot actuator before explicit actuator validation. Motion/TWIST profiles set `policy`; locomotion uses the robot profile's policy/hold partition. |
| `actuator_config` | selected robot map | Its `joints` mapping assigns a `JointActuatorConfig` to each joint. Bello Mini locomotion exposes waist yaw, waist roll, and its 12 leg joints to the policy. |

The `ankle_differential_config` fields are `motor_gain_scale` and
`motor_effort_limit_scale`. `joint_availability.disabled_joint_names` is decoded
from the YAML list into a canonical sorted tuple. Every name must belong to the
selected robot's configured actuator set.
When `actuator_mode` is set together with `actuator_config`, the explicit joint
map must exactly match the derived robot map. Each `actuator_config.joints` entry
has a `mode` (`policy` or `hold`),
`default_pos`, optional `soft_limit`, `[kp, kd]` `gains`, and optional
`effort_limit`.

The complete asset configuration is stored in checkpoint metadata. By default,
a checkpoint runs from its exact clean Git revision, so its source descriptions
and asset-generation implementation come from that revision. Training rejects
a custom `output_root`.

Every public maintained experiment resolves `asset_config.asset` explicitly.
Motion, TWIST, and TWIST2 also set `actuator_mode: policy`; the selected robot
profile still supplies joint order, neutral positions, gains, limits, and effort
limits. Those maintained profiles currently select Bello. Their fixed tracking
contract is the canonical box-collision asset with no actuator joints disabled;
Bello's waist roll/pitch and neck/head joints are fixed in its source URDF, and
all 25 remaining coordinates are policy-controlled. A symmetric sphere-hand
payload may be added without changing the joint or observation contract.
Initialization rejects other deviations from this contract.

## Locomotion environment: `env_config`

| Field | Meaning |
| --- | --- |
| `decimation` | Physics steps per policy step. The control period is `simulation_physics.timestep * decimation`. |
| `max_episode_steps` | Timeout horizon in policy steps. |
| `action_clip` | Environment action bound; must match a bounded policy's `action_limit`. |
| `root_link` | Body used by environment indexing. `null` resolves to the selected robot contract's generated root body; maintained locomotion profiles explicitly use `pelvis_link`. |
| `simulation_physics` | Runtime timestep and optional foot/terrain contact overrides. |
| `terrain` | Plane, rough heightfield, or robot-only scene parameters. |
| `model_randomization` | Per-environment joint, torso inertial, and other physical variation. |
| `spawn` | Initial base pose, joint pose/velocity, root velocity, and episode staggering. |
| `contact` | Contact debounce durations and force thresholds. |
| `random_push` | Periodic base linear-velocity perturbations. |
| `numerical_safety` | Broad finite-state and solver-force bounds that terminate invalid rollouts. |
| `observation` | Deployable actor observation scaling and noise. |
| `observation_low_pass_filter` | Training-side low-pass filter for hardware-derived actor feedback. |
| `torso_com_curriculum` | Optional policy-step ramp for fixed per-world torso COM offsets. |
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

`model_randomization` samples are fixed per vectorized world. When the torso COM
curriculum is enabled, its sampled offset remains fixed while its applied scale
increases during training.

| Field | Meaning |
| --- | --- |
| `enabled`, `seed` | Enable deterministic per-world randomization and select its seed. |
| `joint_origin_range` | Ordered additive joint-origin offset range. |
| `joint_angle_offset_range` | Ordered default joint-angle offset range. |
| `joint_angle_offset_scope` | Apply angle offsets to `all` active coordinates or only `policy` coordinates. |
| `joint_inertia_scale_range` | Ordered positive inertia multiplier range. |
| `torso_mass_scale_range` | Ordered positive multiplier for `torso_link` mass and inertia. |
| `torso_com_offset_range` | Ordered per-axis offset range for the `torso_link` center of mass, in meters. |
| `added_base_mass_range` | Ordered additive mass range distributed over the pelvis, waist yaw, and torso, in kilograms. |
| `base_com_offset_range` | Ordered per-axis offset range for the pelvis center of mass, in meters. |
| `friction_range` | Ordered positive sliding-friction range. |
| `actuator_strength_range` | Ordered positive multiplier range sampled independently for proportional and derivative servo gains. |
| `gravity_fraction_range` | Ordered fraction of nominal gravity added as a model-level perturbation. |

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
component: sine phase, cosine phase, and a gait-active bit. The maintained
per-leg frequency is 0.90 Hz for both Bello and the rounded Bello Mini setting;
both use a 60% stance fraction and a half-cycle offset between the feet. Moving
training resets uniformly initialize the clock across the full gait cycle. This
keeps parallel environments phase-diverse; native evaluation and hardware runs
initialize it at phase zero. During active gait, a zero input is only a standing
request. The effective command and gait component remain moving until the next
scheduled half-cycle boundary, including when the request arrives during double
support. The boundary step still receives gait rewards; the following step uses
the zero effective command, zero gait component, and standing rewards. An
environment initialized in standing uses zero immediately. `frequency_hz` is
the complete cycle frequency of each leg, and `stance_fraction` must be in
`[0.5, 1.0)`. The configured frequency and control timestep must advance the
clock by less than half a cycle per policy step so no gait boundary can be
skipped.

Actor observations are the hardware policy input. Hardware locomotion rejects
a checkpoint whose actor observation is privileged. Critic-only fields do not
enter the deployed policy.

The `observation_low_pass_filter` section contains `enabled`, `cutoff_hz`,
`noise_before_filter`, and `residual_noise_scale`.
The maintained locomotion profiles use the same 10 Hz first-order response as
hardwared for IMU acceleration, angular velocity, orientation-derived projected
gravity, and joint velocity. Joint position remains unfiltered. Training updates
the filter at the 50 Hz policy boundary using the exact elapsed-time response;
four constant-input 200 Hz updates have the same 20 ms response. The maintained
profile injects configured IMU and joint noise before the filter, then adds 10%
of that range as residual post-filter noise; noisy joint position is delayed but
not filtered or duplicated afterward. Older checkpoints load with this
training-side filter and pre-filter noise path disabled.

The `torso_com_curriculum` section contains `enabled` and positive integer
`ramp_steps`. A value of 3,000,000 corresponds to 150,000 PPO updates for the
maintained 20-step rollout. It starts from the nominal torso COM and linearly
reaches each world's complete sampled offset, retaining the full configured
plus-or-minus 10 cm per-axis envelope for the final 45,312 updates.

The `observation_latency` section contains `enabled` and `delay_steps`, an
inclusive integer lower/upper range. The maintained Bello and Bello Mini
locomotion profiles sample a fixed delay of 0, 1, or 2 policy steps
per episode, corresponding to 0--40 ms at 50 Hz. The delayed terms are IMU
linear acceleration, angular velocity, projected gravity, and joint
position/velocity. Command and previous action remain current, and the
privileged critic remains undelayed. Native evaluation and physical deployment
do not inject synthetic delay. Older locomotion checkpoints load with latency
disabled; resuming their rollout continuation still requires the original
source revision because its environment state predates the feedback history.

`last_action` is the clipped policy action sent to the position servos. Action
rate and acceleration rewards teach smooth commands without adding a stateful
filter to training, native evaluation, or the hardware runner.

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
| `resample_seconds` | `10` | Control-time interval between command samples; episode resets do not restart it. |
| `standing_transitions_enabled` | `false` | Reassign the standing cohort during periodic resampling. During gait, latch a zero request until the next scheduled half-cycle boundary before changing effective mode. |

Moving samples are conditioned so at least one available axis lies outside its
minimum moving speed. The minimums must exceed their respective deadbands.
The maintained locomotion profiles override `standing_probability` to `0.25`.
Training assigns an exact standing cohort at batch reset. Failure resets
preserve the exact command request. A separate command-resampling clock survives
episode resets, so short episodes cannot trap an environment in one mode. Its
initial ages are randomly assigned from an evenly spaced batch distribution;
the maintained 2,048-environment profile therefore resamples four or five worlds
per control step instead of all worlds every ten seconds. Subsequent samples use
`standing_probability` for each expiring world's new request, while batch
initialization retains its exact standing count. When
`standing_transitions_enabled` is true, this exposes
both standing-to-moving and moving-to-standing transitions without a periodic
batch-wide mode pulse. A zero request during active gait is pending until the
next scheduled half-cycle boundary, even when it arrives during double support.
The effective moving command, gait phase, and moving rewards remain active
through the boundary step; zero becomes effective and standing rewards begin on
the following step. Training telemetry reports the resampled fraction,
mode-specific mean reward and planar tracking reward, completed-episode return
and length, and completed-episode count under the `command/resampled_fraction`,
`command/standing_*`, `command/moving_*`, and `episode/{standing,moving}_*`
names.

The yaw-rate tracking reward uses a first-order low-pass-filtered measurement.
The maintained recipe uses a 5 Hz cutoff at the 50 Hz policy rate; the raw IMU
observation is unchanged. Planar and yaw tracking each have weight `2.0`, so
their maximum contribution is `0.04` per control step. Moving-command tracking
uses standard deviations of `0.30 m/s` for planar velocity and `0.25 rad/s` for
yaw velocity; standing-command tolerances remain `0.25 m/s` and `0.20 rad/s`.

## Locomotion reward: `reward_config`

Every term is computed independently and multiplied by its configured weight.
The `weights` section is listed below; a zero weight disables metadata and
computation where possible. In this table, "standing command" means the
effective standing mode. A raw zero request made during gait does not activate
standing rewards until the policy step after the next scheduled half-cycle
boundary.

| Weight | Baseline | Effect |
| --- | ---: | --- |
| `track_lin_vel_xy` | `2.0` | Gaussian command tracking for planar velocity. At a standing command it is awarded only in double support. |
| `moving_lin_vel_xy_error` | `0.0` | Moving-only smooth-L1 error between signed planar velocity and command. Its constant far-field gradient remains corrective after the Gaussian tracker saturates. Bello Mini sets this to `-1.0`; Bello leaves it disabled. |
| `track_ang_vel_z` | `2.0` | Gaussian yaw-rate command tracking. At a standing command it is awarded only in double support. |
| `healthy_survival` | `1.5` | Reward each non-terminal control step so early negative shaping costs cannot make immediate failure preferable to continued control. Bello Mini sets this to `2.5`, or `+0.05` per healthy 50 Hz step, to offset its additional dense moving penalties. |
| `lin_vel_z` | `-2.0` | Penalize vertical base velocity. |
| `ang_vel_xy` | `-0.15` | Penalize roll/pitch angular velocity. |
| `flat_orientation` | `-2.0` | Penalize gravity-vector tilt. |
| `base_height` | `-10.0` | Penalize error from `limits_pose.base_height_target`. |
| `action_rate` | `-0.01` | Penalize change from the previous applied action, with extra ankle and waist weighting. |
| `action_acceleration` | `-0.005` | Penalize change in applied action delta, using the same joint weighting. |
| `action_limit` | `-1.0` | Penalize requested action beyond the soft action ratio. |
| `torque_l1` | `-1.2` | Penalize physically applied torque normalized by each motor's peak torque. |
| `torque_peak` | `-12.0` | Quadratically penalize reconstructed unclipped servo demand above peak even though dynamics enforce the physical limit. |
| `joint_force_peak` | `-4.0` | During commanded gait, quadratically penalize the two largest hip, knee, or ankle reaction-force excesses above `0.8` body weight. It is disabled for standing because adjacent joints in a support chain can report the same static load. This uses MuJoCo's parent-child linear interaction force, not actuator torque. |
| `touchdown_impact` | `-4.0` | At the first raw contact across all physics substeps, penalize the larger of integrated normal load and peak normal force above `0.6` body weight plus actual pre-impact downward foot speed. Bounded squared `tanh` shaping limits isolated impacts from dominating an update. |
| `joint_pose` | `-0.1` | Penalize weighted deviation from nominal joint pose. |
| `stand_still` | `1.0` | At standing command, reward healthy double support in proportion to bounded foot stillness. Terminal steps receive no standing bonus. |
| `stand_foot_motion` | `-1.0` | At standing command, penalize bounded foot linear and angular motion. |
| `stand_foot_heading` | `-0.5` | During healthy standing double support, penalize bounded left/right foot heading error outside the configured deadband relative to the robot command-frame forward axis. The weighted cost is capped below `0.01` per control step and is inactive during recovery, termination, and moving commands. |
| `stand_contact` | `-0.5` | At standing command, penalize each foot that is not in filtered terrain contact. Together with `stand_still`, this favors quiet double support without making continued recovery intrinsically negative. |
| `stand_upright` | `-8.0` | At standing command, apply an additional gravity-vector tilt penalty so a deeply leaned pose cannot retain most of the standing reward. |
| `hip_deviation` | `-0.25` | Penalize hip deviation during standing or when any configured command axis is active. |
| `knee_deviation` | `-0.1` | Penalize knee deviation during effective standing; a profile can explicitly retain it while moving. A pending zero request does not activate standing-only shaping until the gait scheduler completes the current step. |
| `joint_limit` | `-1.0` | Penalize proximity beyond the joint soft-limit factor. |
| `invalid_action` | `0.0` | Optional penalty for invalid/nonfinite requested action. |
| `gait_contact` | `-1.0` | Penalize disagreement between measured foot contact and the phase-conditioned stance schedule. |
| `gait_stance_slip` | `-0.25` | Penalize planar foot motion during the phase-conditioned stance interval. |
| `gait_swing_clearance` | `-0.5` | Penalize swing-foot clearance below the configured target. |
| `undesired_contact` | `0.0` | Optional non-foot terrain-contact penalty. |
| `self_collision` | `0.0` | Optional self-collision penalty. |
| `termination` | `-100.0` | Penalize non-timeout terminal failure. |

The maintained locomotion profiles override `touchdown_impact` to `-8.0` while
retaining their `1.0` body-weight touchdown-force soft limit. This restores the
impact-to-tracking balance after the profile's tracking rewards were doubled,
without changing the bounded touchdown formulation. Bello Mini inherits the same
reward configuration except for a `-1.0` moving planar smooth-L1 error penalty
and midpoint gait shaping: `gait_contact: -2.5`, `gait_swing_clearance: -1.0`,
and a `0.075` m swing-clearance target. The error penalty uses a `0.25 m/s`
Huber delta and preserves a signed corrective gradient when the Mini moves in
the opposite direction, while the gait terms more strongly discourage
persistent double support.

Both profiles apply moving hip-deviation shaping on their pelvis-frame forward
axis: command index 1 for Bello (`+Y`) and index 0 for Mini (`+X`). Bello enables
`knee_deviation_apply_while_moving`, preserving the successful profile's
moving-pose gate; Mini leaves the switch disabled so its knees remain free to
form gait while the term still regularizes standing.

The `links` section selects the measured link for `track_lin_vel_xy`,
`track_ang_vel_z`, `lin_vel_z`, `ang_vel_xy`, `flat_orientation`, and
`base_height`; `track_lin_vel_xy_frame` optionally selects a separate velocity
frame. When that override is `null`, planar tracking uses the selected robot
contract's canonical locomotion command frame. Both maintained profiles use
pelvis velocity/yaw and height, with torso angular velocity and orientation.
Both maintained profiles resolve planar velocity in `pelvis_link`. Their
robot-specific forward axes differ: Bello uses pelvis `+Y`, while Mini uses
pelvis `+X`. Mini also configures the simulated actor IMU source as
`pelvis_link`, aligning actor feedback, critic velocity, command, and reward
axes for training.

Supporting fields are:

| Section | Fields |
| --- | --- |
| `tracking` | `lin_vel_xy_moving_std`, `lin_vel_xy_still_std`, `lin_vel_xy_error_delta`, `ang_vel_z_moving_std`, `ang_vel_z_still_std`, `ang_vel_z_lpf_cutoff_hz` |
| `standing` | `foot_linear_speed_scale`, `foot_angular_speed_scale`, `foot_angular_weight`, `foot_heading_deadband`, `foot_heading_scale` |
| `gait` | `stance_slip_angular_speed_scale`, `swing_clearance_target`, `joint_force_soft_limit_body_weights`, `joint_force_top_k`, `joint_force_apply_while_standing`, `touchdown_force_soft_limit_body_weights`, `touchdown_vertical_speed_scale` |
| `smoothness` | `ankle_action_weight`, `waist_action_weight` |
| `limits_pose` | `base_height_target`, `action_soft_limit_ratio`, `joint_soft_limit_factor`, `hip_deviation_command_indices`, `hip_deviation_command_threshold`, `knee_deviation_apply_while_moving` |

Rollout telemetry reports the mean, p95, p99, and worst normalized joint-force,
foot-contact peak, and control-interval impulse maxima. The tail metrics are named
`gait/{joint_force,contact_force,contact_impulse}_{p95,p99,worst}_body_weight`; unlike the mean
reward terms, they expose short impacts that would otherwise be diluted across
joints and rollout frames. `command/standing_positive_reward_fraction` reports
the fraction of standing transitions whose total reward is positive; it should
rise toward one as quiet standing matures.

`track_ang_vel_z` uses a first-order low-pass-filtered measured yaw rate. The
filter is initialized from the measurement on reset and
`tracking.ang_vel_z_lpf_cutoff_hz` sets its cutoff; it does not filter policy
actions or the commanded yaw rate.

## Locomotion termination: `done_config`

| Section | Meaning |
| --- | --- |
| `base_height` | `enabled`, `min_height`, and optional `link` for terminal minimum height. |
| `bad_orientation` | `enabled`, `limit_angle`, and optional `link` for terminal projected-gravity tilt. |
| `timeout` | `enabled`; mark `max_episode_steps` as truncation rather than physical failure. |
| `nonfinite` | `enabled`; terminate finite-state or solver-reward violations. |

The maintained Bello profile uses a 0.55 m pelvis minimum; Bello Mini uses the
rounded 0.56 m minimum. Both use a 0.80 rad torso orientation limit. Their base
height reward targets are respectively 0.86 m and 0.87 m.

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

## TWIST: `configs/train_twist`

TWIST uses the shared PPO scalar fields but owns an explicit temporal policy
architecture and packed reference bank. Its maintained scalar values are 4,096
environments, 24 steps, five epochs, four minibatches, learning rate `2e-4`,
desired KL `0.008`, entropy `0.005`, gamma `0.99`, GAE lambda `0.95`, and clip
ratio `0.2`. TWIST alone sets Adam epsilon to `1e-8` and uses an unhalved
value-loss scale of `1.0`; other training profiles retain their established
optimizer and loss defaults.

`stage` is exactly `teacher` or `student`; the corresponding policy
architecture must be `twist_teacher` or `twist_student`. Both require
`hidden_dims=[512,512,256,128]`, `activation=swish`,
`obs_stack_size=1`, and final-hidden LayerNorm. Student history is part of the
environment observation, not the generic PPO stack.

`dataset_index` selects a schema-v1 packed index. `teacher_checkpoint_path` is
required only for the student. `simulation` configures only the dedicated
TWIST terrain; the runtime motion is always materialized from the fingerprinted
packed index. `twist_env` owns the TWIST
reward, reset, disturbances, actuator mapping, domain randomization,
termination, sensor noise, 500-step episode limit, released 0–1 frame action
delay, and ordered nine-key-body contract. It has no dependency on the
BeyondMimic or legacy motion-training environment. These values, observation
layouts, dataset/robot fingerprints, and ordering are checkpointed and
resume-checked.

The teacher receives the 20 future offsets and the released privileged suffix.
The student actor receives 11 chronological current-reference/proprioception
frames and no future or privileged state; its critic receives the teacher
reference plus privileged physical and randomization state. Student PPO adds
the frozen-teacher `KL(student || teacher)` schedule defined in the [TWIST
guide](twist.md).

## TWIST2: `configs/train_twist2`

TWIST2 is a separate one-stage PPO task, not a teacher/student mode and not a
fallback to either TWIST environment. Its policy architecture is exactly
`twist2`, it accepts no teacher checkpoint, and its typed checkpoints cannot be
resumed or evaluated through another task. The maintained profile follows the
pinned public release for 4,096 environments, 24 rollout steps, five epochs,
four minibatches, learning rate `2e-4`, desired KL `0.008`, entropy `0.005`,
gamma `0.99`, GAE lambda `0.95`, clip ratio `0.2`, 30,001 updates, a Gaussian
policy, actor observation moments collected between rollouts for the first
3,000 updates, and an explicitly raw privileged critic input.

`twist2_env` owns its 500 Hz simulation, 50 Hz policy, ten-second episodes,
current-command/history actor observation, future-reference privileged critic,
reward, termination, noise, action delay, four-second root-velocity pushes,
and fixed per-world model randomization. The public release declares but does
not execute gravity randomization, so the maintained environment leaves it
inactive. `sampling` uses unit manifest weights plus the released
per-clip difficulty curriculum. `require_released_dataset=false` accepts another
compatible packed corpus while recording its composition audit in W&B and the
checkpoint; setting it to `true` requires the training split to contain the
exact released 19,426-clip source composition with unit weights. See the
[TWIST2 guide](twist2.md) for the observation, data-split, and parity contracts.

## Checkpoints: `checkpoint_config`

| Field | Meaning |
| --- | --- |
| `enabled` | Write periodic and final checkpoints. |
| `directory` | Parent output directory. It must be outside the source repository or covered by a Git ignore rule so saving cannot dirty the exact-source checkout. |
| `name` | Stable, single-component run label. Locomotion final checkpoints are timestamped roots; motion also maintains a `final` child. |
| `every_updates` | Periodic save interval, or `null` for final-only behavior. |
| `resume_path` | Checkpoint to continue, normally from its exact clean Git revision. |
| `warm_start_path` | TWIST2-only transfer source. It restores policy and normalization but resets optimizer and runtime state; it is mutually exclusive with `resume_path`. |
| `allow_git_commit_mismatch` | Permit resume or warm start from a different clean commit. It requires one of those paths and bypasses checkout/source-revision identity only. |
| `upload_wandb_artifact` | Upload saved checkpoints when a W&B run exists. |
| `wandb_artifact_name` | Explicit artifact name or profile-derived default. |
| `wandb_artifact_type` | W&B artifact type, normally `checkpoint`. |

Evaluation requires a clean worktree and, by default, the checkpoint's recorded
Git commit. `allow_git_commit_mismatch=true` bypasses checkout and recorded
source-revision identity only.
Checkpoint document integrity, metadata and state layout hashes, normalization
state, saved update, task, and actor/critic observation and action runtime
sizes/layouts are still validated.
Checkpoints created before the PPO symmetry-loss field was introduced load it
as zero. This preserves their original policy and affects neither inference nor
the saved parameter layout. Legacy locomotion checkpoints similarly load newer
reward weights as zero and missing joint-specific smoothness multipliers as one,
preserving their original reward calculation if resumed.
Resume additionally validates and restores environment continuation,
observation stacks, RNG, current learning rate, sampler state, environment
layout, and the full resume contract. A changed `total_timesteps` is allowed
only when the resulting update budget still covers the saved update.
TWIST2 warm start instead allows a new dataset fingerprint and optimization
budget while requiring the same task, robot, GMR revision, policy architecture,
environment semantics, and observation/action layouts.

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
evaluation worker. A locomotion video failure emits an explicit warning and
disables later videos for that run without terminating PPO. Vast jobs use the
software OSMesa backend and validate rendering plus MP4 encoding before the
training preflight; MJX-Warp training remains on the selected GPU.

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
TWIST uses 32 contacts per environment and logs
`max_physics/contact_capacity_fraction` plus
`mean_physics/contact_capacity_overflow` so capacity pressure is visible.

## Maintained profile intent

| Profile | Intent |
| --- | --- |
| `smoke` | One real two-environment PPO update with checkpointing, W&B, and video disabled. |
| `bello_locomotion` | Production Bello gait training with 2,048 environments, full planar/yaw commands, graceful standing transitions, slope, reset randomization, pushes, 10 Hz pre-noise observation filtering, plus-or-minus 20% torso mass, a torso COM curriculum reaching plus-or-minus 10 cm, mean-KL adaptive learning with a `0.00003` floor, and periodic checkpoint/video evaluation every 512 updates. Symmetry and action-acceleration losses remain disabled. |
| `bello_mini_locomotion` | The production Bello PPO, command, reward, randomization, and observation recipe adapted to Bello Mini's 14-action locomotion contract, with pelvis-frame `+X` forward command/reward/actor feedback, flat-sole `-0.15/0.45/-0.30` rad leg crouch, `[0.0, 1.2]` rad knee policy range, outward held shoulder-roll pose, rounded 0.87 m height target, 0.56 m termination floor, 0.90 Hz gait, a `2.5` healthy-survival weight that offsets Mini's additional dense moving penalties, and output names. The profile disables moving knee-pose shaping and independent multiplicative joint-position reset scaling so gait remains free while the coordinated hip/knee/ankle pose stays flat-footed; the remaining randomization is active. |
| `motion` | 3,072-environment bounded motion tracking with observation and value normalization, adaptive failure sampling, pushes, and periodic checkpoint/video evaluation every 500 updates. |
| `twist_teacher` | 4,096-environment, multi-clip whole-body PPO with released future-reference encoding. |
| `twist_student` | The same dynamics and critic contract with current-reference/history actor input and frozen-teacher distillation. |
| `twist2` | Pinned one-stage TWIST2 PPO with current command, ten-frame history, privileged future-reference critic, and released clip curriculum. |

Headline resolved PPO values are:

| Profile | Timesteps | Envs × steps | Epochs × minibatches | Learning rate | Gamma / GAE | Policy |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `smoke` | 4 | 2 × 2 | 1 × 1 | 0.001 | 0.99 / 0.95 | 256-256 tanh |
| `bello_locomotion` | 8,000,000,000 | 2,048 × 20 | 4 × 32 | 0.0001 adaptive, 0.00003 floor | 0.995 / 0.98 | 512-256-128 swish |
| `bello_mini_locomotion` | 8,000,000,000 | 2,048 × 20 | 4 × 32 | 0.0001 adaptive, 0.00003 floor | 0.995 / 0.98 | 512-256-128 swish |
| `motion` | 2,949,120,000 | 3,072 × 24 | 5 × 4 | 0.001 adaptive | 0.99 / 0.95 | 512-256-128 ELU |
| `twist_teacher` | 2,949,316,608 | 4,096 × 24 | 5 × 4 | 0.0002 adaptive | 0.99 / 0.95 | 512-512-256-128 SiLU + LN |
| `twist_student` | 2,949,316,608 | 4,096 × 24 | 5 × 4 | 0.0002 adaptive | 0.99 / 0.95 | 512-512-256-128 SiLU + LN |
| `twist2` | 2,949,218,304 | 4,096 × 24 | 5 × 4 | 0.0002 adaptive | 0.99 / 0.95 | three released actor encoders + 512-512-256-128 SiLU + LN |

`bello-train --cfg job` prints the outer public command document; profile
resolution occurs afterward inside the command and is therefore not included
in that Hydra output. Keep maintained numerical changes in the corresponding
YAML layer and schema/default changes in the owning dataclass.
