# Configuration

Values belong in YAML, not duplicated documentation tables.

SONIC's `sampling_config` controls its temporal-bin, exposure-normalized
curriculum. `env_config.adaptive_sampling` is the BeyondMimic EMA recipe and
must stay at inactive defaults for SONIC. See [the sampler contract](sonic_transfer.md#adaptive-motion-sampling).

`config.timeout_bootstrap` selects next-state timeout values by default.
SONIC explicitly selects `current_state` to reproduce Any2Any TRL's
`gamma * V(s_t)` reward correction followed by done-masked GAE. This does not
change the timeout convention for locomotion or BeyondMimic.

Use `asset_config.joint_soft_limits` for task-specific target restrictions. The
mapping contains logical joint names and `[lower, upper]` radians. Unspecified
Mini joints use canonical URDF ranges. Resolved actuator limits are saved with
the checkpoint for evaluation and deployment; changing a training profile does
not change an already trained policy.

Use `asset_config.joint_gains` for task-specific `[kp, kd]` actuator gains keyed
by logical joint name. Unspecified entries retain the robot defaults. For coupled
ankles, both joint entries must match and specify each differential motor's gains,
not the effective joint impedance. Gains must be finite and nonnegative. Resolved
gains are saved with the checkpoint and used by evaluation and policy export.
Mini motion overrides gains in its profile; locomotion retains its existing gains.

Target restrictions and reward margins are separate. `asset_config.joint_soft_limits`
caps commanded targets; it does not change the URDF's physical stops. Locomotion's
`reward_config.limits_pose.joint_soft_limit_factor` and motion's
`env_config.reward.joint_limit_soft_factor` shrink the configured range only for
the joint-limit penalty. A factor of `0.9` penalizes the outer 5% at each end,
without making that margin a physical stop or a target clamp.

| Location | What to edit |
| --- | --- |
| `configs/commands/*.yaml` | Public CLI fields |
| `configs/profiles/train/*.yaml` | Experiment selection and overrides |
| `configs/train/default.yaml` | Locomotion PPO, environment, rewards, logging |
| `configs/train/locomotion.yaml` | Shared production locomotion recipe |
| `configs/train_motion/default.yaml` | BeyondMimic training |
| `src/bello_mujoco/robots/*.yaml` | Topology, nominal pose, motor gains, tracked bodies |
| `deps/bello-stack/descriptions/` | Canonical geometry and hardware configuration |

Maintained experiments: `smoke`, `bello_locomotion`, `bello_mini_locomotion`,
`bello_mini_standing`, `motion`, `bello_mini_motion`,
`bello_mini_sonic_transfer`, `bello_mini_supermimic`,
`bello_mini_supermimic_smoke`, `bello_mini_supermimic_pilot`, and
`bello_mini_supermimic_amass`.
The CLI accepts its public top-level overrides;
edit the selected recipe for nested reward/physics changes. SONIC's corpus,
immutable artifact identity, held-out clip names, pretrained checkpoint, and
immutable source revision are explicit inputs; its fixed LoRA/PPO recipe lives
in `configs/profiles/train/bello_mini_sonic_transfer.yaml`.
The SONIC profile opts into trajectory minibatches, per-minibatch adaptive
learning rate, sample-standard-deviation advantages and the reference critic's
normalizer. Shared PPO defaults for other profiles are unchanged. Its corpus
array-size guard and separate actor/critic optimization controls are documented
in [SONIC transfer](sonic_transfer.md).

## SuperMimic configuration

The `bello_mini_supermimic_lafan` recipe uses whole LAFAN sequences. Its
`config.max_episode_steps` caps collection episodes, while reference arrays and
native full-clip evaluation retain the complete recording. A null cap preserves
the reference-end timeout behavior. `networks.matmul_precision` explicitly
selects `float32`, `tensorfloat32`, or `bfloat16` matrix multiplication for the
networks. LAFAN uses BF16 network operands/activations with FP32 accumulation,
master parameters, normalization statistics, and optimizer state. Physical state,
kinematics, rewards, and returns stay FP32. ONNX export preserves the network's
BF16 rounding using CPU-supported casts around FP32 matrix multiplication.
The LAFAN comparison uses the full 4,096-environment, ten-by-2,048 dynamics,
and 4,096-imagination batches. Its transition target covers exactly 14,000
update cycles after collection warmup with no wall-time cutoff. It disables
command delay and feedback filtering/latency for reference comparison, inherited
from the common SuperMimic recipe. Hardware filtering/delay commissioning uses
separate explicitly configured settings. The return sampler
uses alpha 0.2 and a 0.001 historical-difficulty EMA.
The common SuperMimic recipe sets `asset_config.self_collisions: false`, matching
the unofficial reproduction. Robot-ground contacts remain enabled. Other tasks
retain the model's default self-collision setting.
SuperMimic allocates a shared contact pool of 128 contacts per environment and
1,024 constraint rows per environment. Collection reports usage and rejects an
update if either physics capacity overflows.
See [LAFAN preparation and training](supermimic.md) for corpus constraints and
the separate training/held-out evaluation reports.

The [full profile](../configs/profiles/train/bello_mini_supermimic.yaml) owns
`config` (learning/replay), `environment` (control, feedback, and randomization),
`observations`, `tracking`, `networks`, `sampling`, `dataset`, `warp`,
`checkpoint_config`, and `wandb_config`. These decode into
`SuperMimicTrainingRunConfig`; there is no PPO config in this task. The profile
inherits only the Mini motion asset configuration, preserving its tuned gains
and joint target restrictions. The
[smoke profile](../configs/profiles/train/bello_mini_supermimic_smoke.yaml)
overrides model widths and learning sizes for a compiled-loop check.
The [pilot profile](../configs/profiles/train/bello_mini_supermimic_pilot.yaml)
retains the full networks and 32-step imagination, while reducing environment,
replay, and optimizer batch sizes for initial single-GPU learning measurements.
The [AMASS profile](../configs/profiles/train/bello_mini_supermimic_amass.yaml)
uses 1,024 environments with the full networks. Its `max_training_seconds` stops
training after 75 minutes, including setup and compilation, and saves the full
final state even when the transition target is unfinished. The deadline is
checked between complete optimization iterations. This operational budget may
change on resume; the learning contract remains fixed. An unset budget runs to
`config.total_timesteps`.

Control timing has one owner, `environment.control`: physics timestep,
decimation, command delay, and action clipping. Observation delay is sampled in
control steps; command delay must be a whole physics substep within one control
period. Supported physical randomization is friction, calibration, and torso
COM, all explicitly supplied to the world model and privileged critic. Unknown
dynamics randomization and unobserved pushes fail setup.

`config.replay_steps` counts vector steps, so storage scales with `num_envs`.
`config.max_replay_bytes` checks the actual transition footprint before replay
allocation; `dataset.max_array_bytes` separately limits uncompressed reference
arrays. Neither estimates all training VRAM. Warmup must hold a model window,
and the training budget must permit actor updates after dynamics warmup.
The complete observation, reward, normalization, sampling, resume, and export
semantics are documented in [SuperMimic](supermimic.md).

```bash
uv run bello-train --cfg job
uv run --group training bello-train experiment=bello_mini_locomotion seed=1 num_envs=4096
```

Hydra documents decode into frozen dataclasses, validating fields, dimensions,
ranges, and incompatible combinations before simulation. Add settings to their
owning dataclass and YAML together, without aliases or parallel CLI defaults.

For motion actor inputs, select `env_config.observation_mode` in the motion
profile: `proprioceptive` (default) or `full_state`. This changes the actor input
layout and requires retraining; the critic and simulator terminations retain
full state. See [motion observation modes](motion_tracking.md#observation-modes).

## Locomotion components

- `torso.py`: level-ground torso lean/twist targets and independent slew.
- `gait.py`: cadence, directional start, command slew, delayed stop, standing blend.
- `actions.py`: motor targets and calibration offsets.
- `observations.py`: actor/critic assembly, noise, filtering, feedback history,
  and shared delay sampling.
- `contacts.py`: contact-force classification and contact debouncing.
- `foot_geometry.py`: sole geometry, support points, and shared terrain normals.
- `randomization.py`: spawn and pushes.
- `rewards.py`: tracking, alignment, crossing, clearance, contact, force, smoothness.
- `locomotion_state.py` and `locomotion_config.py`: rollout state and typed settings.

The motor-coordinate action convention requires fresh training. Keep training,
native evaluation, and hardware observation/filter conventions aligned.

Mini uses shared joint-posture, torso-orientation, foot-heading, and contact-slip
terms across walking and standing. Standing alignment uses deadband Huber
costs, including during single support; speed and lean do not relax leg posture,
alignment, or thermal penalties. Explicit per-joint posture weights live in the
Mini training profile. Full-size Bello retains its active standing-specific terms.

### Torso orientation commands

`command_config.torso` owns roll/pitch/yaw ranges (radians), per-axis slew rates
(radians/second), the neutral-command probability, and the tracked body name.
Mini's locomotion/standing recipes enable these commands. They are sampled
independently of standing/walking, on the velocity-command resampling clock;
episode startup begins at neutral and ramps toward the sampled target.

Roll and pitch describe lean relative to level ground. Yaw describes torso twist
relative to the pelvis heading, not an absolute compass heading or a turn rate.
Axes are anatomical forward/left/up, accounting for the robot's CAD frame;
rotations use `Rz(yaw) Ry(pitch) Rx(roll)`. The nominal CAD torso orientation
defines zero. The existing velocity/yaw-rate command and gait scheduler remain
independent; a torso-only command does not initiate stepping.

The actor and critic receive a separate three-angle `torso_command` observation.
`track_torso_orientation` rewards `exp(-(SO(3) angle error / std)^2)` using
`reward_config.tracking.torso_orientation_std`. Fixed upright penalties must be
disabled when nonzero torso targets are sampled. Mini also removes the nominal
waist-angle penalty so it does not oppose twist; leg posture, foot placement,
torque penalties, joint/motor limits and absolute tilt termination remain active.
This observation change requires fresh training and export: changing a previous
bundle's commit metadata cannot add this capability.

`video_config.torso_command` selects a fixed target for training videos. Native
evaluation and deployment use the checkpoint's same torso slew rates and bounds.

`foot_crossing` penalizes signed lateral clearance between the current sole
collision primitives, accounting for yaw, roll, pitch, and thickness. It remains
active during both swing and stance. `gait.sole_clearance_margin` sets the
desired edge buffer; `gait.crossing_scale` sets the Huber penalty scale. This is
a conservative lateral projection, not the closest 3D distance: separating feet
fore/aft or vertically does not exempt crossed or overlapping lateral paths.
`gait/sole_lateral_clearance` reports this signed edge gap; the existing center
separation metric remains available. Standing width still targets the nominal
stance rather than a wider walking stance. These reward changes require
training; they do not alter an already exported policy.

Mini's standing motor-torque cost covers every physical actuator, including held
arms and each differential-ankle motor, above half its model's rated output
torque. Ratings live in `robots/motor_torque_ratings.yaml`; motor identities and
coordinate assignments come from the pinned hardware description. Missing
coverage or non-unit actuator gears fail at initialization. This replaces the
hip-roll-only penalty; it does not change servo limits or model a thermal sensor.

`env_config.random_push.curriculum_steps` ramps disturbance strength from
`initial_scale` to full strength using the persistent per-world training clock,
shared with the torso COM curriculum. Episode resets do not restart this clock;
checkpoint resume restores it. A zero duration disables the push curriculum.

`env_config.random_push.interval_seconds` is a [minimum, maximum] range.
Training independently samples a control-step countdown per environment, both
for the first push after reset and after each push. Countdown state resumes with
the checkpoint and is independent of the episode/gait clocks. Fixed evaluation
push scenarios remain deterministic for repeatable comparisons.

Runtime initialization validates the selected robot, including default-pose
collisions. Unit tests exercise algorithms and invalid-input handling with
synthetic inputs. Do not test a robot's YAML values, joint count, nominal angles,
or geometry inventory as fixed expectations.
