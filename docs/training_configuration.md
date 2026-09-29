# Configuration

Values belong in YAML, not duplicated documentation tables.

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

Use `asset_config.joint_default_positions` for sparse nominal joint-angle
overrides in radians, including held-arm targets. Unspecified joints retain their
defaults; modes, gains and limits do not change. The actual compiled home pose
is checked for joint-limit violations and self-collisions at initialization.

`asset_config.end_effector_payload_mass_kg` adds that mass **per attachment**.
`end_effector_payload_offset_m` moves its center from the canonical attachment
center, expressed in each attachment's local link frame. This changes inertia,
not visuals or collision geometry, and requires positive payload mass when
nonzero. These are independent spherical hand loads, not a shared rigid object.
`target_total_mass_kg` scales the robot before those payloads are added.

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
`bello_mini_standing`, `bello_mini_loaded_standing`, `bello_mini_payload_standing`, `motion`, `bello_mini_motion`,
and `bello_mini_barbell_motion`.
The CLI accepts its public top-level overrides;
edit the selected recipe for nested reward/physics changes.

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

## Rigid barbell profiles

`asset_config.barbell` describes one rigid free body and two shaft-axis hinge grips.
Each hand's local +Y stays aligned with the shaft; only rotation around that
axis is free. Two point constraints spaced 0.10 m along the axis implement
each hinge, with no extra mass or free-body coordinates. All four constraints
disconnect in unloaded worlds. The approved hand-local -Z offset is unchanged.
It is incompatible with `end_effector_payload_mass_kg`. `target_total_mass_kg`
scales the robot before adding the bar. Normal walking profiles leave it null.

`env_config.model_randomization.payload_distribution` defaults to `fixed`.
`standing` requires enabled model randomization and a rigid bar, and samples
25% unloaded, 25% 30 kg, and 50% uniform 5--30 kg per world. It is never used by
`bello_mini_barbell_motion`, which now selects a fixed 15 kg reference and scene.
Python and Rust require the screened clip mass to match the scene mass; changing
the configured load requires rebuilding and screening the reference.
Generic mass/inertia randomization applies only to robot bodies.

`env_config.arm_pose_path` enables the standing arm trajectory bank and extended
observations. `arm_pose_sha256` is filled by the training entrypoint and checked
on restore; the bank is bundled with the checkpoint. Missing or failed banks
are errors. Feedback measurements cover all joints, the action vector remains
leg/waist only, and commanded arm joint targets are appended after last action.
No exact payload mass is given to the actor. The training profile preserves
existing filtering, latency, gains and torque clamps.

See [barbell preparation and validation](barbell_feasibility.md) for generation,
reference review, the current feasibility blocker and pending runtime gates.

For exploratory barbell simulation, `env_config.allow_unscreened_reference`
(default `false`) permits a failed exact-reference torque screen without changing
the archive result or relaxing mass, torque, grip, or collision constraints.
The fixed-15 kg profile enables this setting; saved simulation checkpoints retain
it. Rust deployment continues to require a passed screen.


### EEF-load standing pretraining

`bello_mini_payload_standing` trains the existing 14 leg/waist actions with the
49.5 kg robot and a nominal 15 kg total load, split equally across fixed hand
payload bodies. It shares the loaded-standing actor's measured arm position,
velocity and commanded-target observations. Exact payload mass is not an actor
input. Walking profiles retain their original actions and observations.

`env_config.model_randomization.payload_mass_range_kg` controls the positive
standing load interval. With `payload_distribution: standing`, 25% of worlds
receive zero load, 25% receive the upper endpoint, and 50% sample uniformly in
the interval. Each world's mass stays fixed throughout training. The proxy uses
[5, 15] kg; the existing rigid-bar default remains [5, 30] kg. Each EEF child
receives half the total, including zero mass and inertia in unloaded worlds.
Native derived inertial constants are rebuilt per world before GPU rollout.
Generic robot randomization does not alter these payload bodies.

Prepare the bank with `uv run python -m bello_mujoco.evaluation.payload_standing`.
This writes `outputs/payload-standing-15kg-v2/standing.npz` and a JSON report.
Symmetric arm excursions use four-second quintic transitions and endpoint holds.
Collision and target-limit checks cover every 50 Hz frame; inverse dynamics with
foot friction cones screens 5 Hz samples and the endpoint against peak torque.
Only passing trajectories enter the bank. This screen does not validate thermal
endurance or prove feasibility under every randomized robot model.

The bank is hashed and copied into checkpoints. Gains, torque limits, feedback
filters and latency stay unchanged. Small pushes and quiet-foot rewards train
load compensation. Periodic training videos use the nominal fixed 15 kg proxy;
final evaluation must separately cover 0, 5, 10 and 15 kg and robot/sensor seeds.

The proxy has no bar collision geometry or two-hand mechanical coupling. It is
balance pretraining, with physical arm resets at the reviewed handover posture, not
pickup/release or motion-endpoint training. Validate the resulting policy on the
continuous rigid-bar cycle and fine-tune there if necessary. Preserve live
outputs from both controllers during handoff; hardware remains separately gated.


The corrected payload-standing profile distinguishes physical reset joint angles
(`env_config.arm_reset_joint_positions`, named held joints in radians) from the
position commands. Fixed nominal-15 kg gravity preload is included in the arm
command defaults. This is the same command bank for all positive payloads; no
exact mass input or online gravity compensation was added to the actor/runtime.
All held joints must be listed exactly once; overrides require an EEF arm bank
and are rejected for rigid-bar scenes. Native and GPU resets share this pose.

`stand_body_still` rewards double-support standing with low linear and angular
body velocity, and is zero during robot self-contact. Its exponential scales are
`standing.body_linear_speed_scale` and `standing.body_angular_speed_scale`.
`stand_self_contact` penalizes forceful robot/robot contact counts using the
existing contact classifier. Both weights default to zero for walking recipes.
The payload profile uses weights +2 and -4, tighter velocity/torso tracking and
stronger torso angular-velocity cost. Contact is penalized; no grip-error
termination has been added. Push curriculum now ramps over 500,000 policy steps.

Reference preparation now integrates the actual arm PD response at 0/5/10/15 kg,
with pitch and actuator-strength variations, before inverse-dynamics screening.
A fixed root/leg rig isolates arm response; it is only a screening fixture, never
the training simulator. The screen rejects any forceful self-contact during
startup or the trajectory, joint-limit violations, arm saturation, and nominal
whole-body inverse-dynamics demand above 98% of peak. The preparation output is
`outputs/payload-standing-15kg-v2`. Full floating-body policy evaluation is still
required. Periodic fixed-camera quiet and four-direction push videos now record
settling, drift, self-contact and torque diagnostics in JSONL files.
