# Mini shared locomotion rewards

The previous recovery recipe made standing alignment conditional on both feet
contacting. A synthetic probe of the actual reward function found that an
18 cm stagger earned +0.120 per step with one foot down but -0.242 with both
feet down. A roughly 17.4-degree tilt could disable the alignment cost entirely.
Saved run xetme3xj improved push survival while double support fell to 7%.
These observations motivated structural corrections, not a blanket increase
in standing weights.

The subsequent v5 run (`ybbh0npu`, source `7e28d11`) used bounded alignment
and a 25% geometry floor but still failed to settle. At update 81,920, the
right-push trial already had about 20 cm stagger and 17 degrees torso tilt
before the kick; its first five seconds matched quiet standing exactly.
None of the ten saved quiet-standing trials met the settling criterion.

## Current recipe

Standing is a zero-velocity command with shared posture and geometry costs.
Mini disables recovery relaxation, without a replacement timer. It removes
`stand_still`, `stand_contact`, `stand_foot_motion`, `stand_foot_heading`,
`stand_upright`, `stand_waist_deviation`, `hip_deviation`, and `knee_deviation`.
Their legacy implementations remain for saved checkpoints and the unchanged
Bello profile, but are inactive in this Mini recipe.

| Shared term | Mini weight | Definition |
| --- | ---: | --- |
| `flat_orientation` | -5 | Squared torso projected-gravity XY magnitude. |
| `joint_pose` | -1 | Sum of per-joint weighted squared nominal-pose errors. |
| `foot_heading` | -1 | Mean squared foot heading error relative to command-frame forward, in radians. |
| `foot_slip` | -0.5 | Sum of squared planar foot speeds in actual filtered terrain contact. |

The single pose regularizer gives waist yaw/roll weights 2, hip roll/yaw 1,
ankles 0.1, and hip pitch/knees 0.05. These are soft costs, not joint locks.
`limits_pose.joint_pose_weights` names every policy joint exactly once and is
validated at environment initialization. Held upper-body joints are excluded.
An empty mapping preserves historical weighting for existing checkpoints.
Mini disables scheduled `gait_stance_slip` to avoid duplicating slip shaping.
The standing-only overlay inherits shared slip and heading costs.

Two Bello-specific standing alignment costs remain. In meters, let
`D(e) = Huber(max(abs(e) - 0.02, 0) / 0.05)`. Fore-aft and width costs are
each `-0.25 * D(error)`, using `standing.contact_independent_alignment: true`.
They include single support and retain the existing 0.2-second standing blend
after the extra stopping step. They do not use `h/(1+h)` or recovery relaxation.
A 20 cm stagger costs 0.0155 per 20 ms step; a 10 cm width error costs 0.0055.
Together, 0.021 is 26.25% of the maximum planar plus yaw tracking reward (0.08),
before other costs. Larger errors retain a linear cost increase.
Shared heading, pose, orientation and slip are not standing-blend gated.

Signed crossing, stumble/impact shaping, hard effort limits and unclipped
torque-demand penalties remain. The former hip-roll-only objective is replaced
by the all-motor standing torque penalty below. General torque costs remain
active in every mode. No blanket self-collision penalty is enabled without
identifying contact pairs.

## All-motor standing torque penalty

`stand_motor_continuous_torque_excess: -8.0` replaces
`stand_hip_roll_continuous_torque_excess` (now zero in Mini). Both terms cannot
be enabled together. For the 28 physical motors, including held upper-body
motors, the new term is:

```
threshold_i = 0.5 * rated_output_torque_i
excess_i = max(abs(applied_motor_torque_i) / threshold_i - 1, 0)
reward = -8 * control_dt * standing_blend * is_standing * mean(excess_i)
```

It is a soft instantaneous cost, not a hard limit or an accumulated temperature
model. It has no tilt/speed recovery gate. It is inactive for moving commands
and follows the existing standing transition blend. Averaging now covers 28
motors rather than two hip rolls: one motor at twice its threshold costs
`0.16 / 28` per 20 ms step; all motors at twice threshold cost `0.16`.
Peak constraints and moving-mode torque penalties are unchanged.

| Model | Rated output torque (Nm) | Penalty threshold (Nm) | Mini motors |
| --- | ---: | ---: | ---: |
| [RS00](https://robstride.com/products/robStride00) | 5 | 2.5 | 6 wrists |
| [RS02](https://robstride.com/products/robStride02) | 6 | 3 | 4 ankle motors |
| [RS03](https://robstride.com/products/robStride03) | 20 | 10 | 2 waist, 4 hip roll/yaw, 4 shoulder pitch/roll |
| [RS04](https://www.robstride.com/products/robStride04) | 40 | 20 | 2 hip pitch, 2 knees |
| [RS06](https://www.robstride.com/products/robStride06) | 11 | 5.5 | 2 shoulder yaw, 2 elbows |

These manufacturer ratings were checked on 2026-09-14. They serve as the
continuous-torque reference, not a guarantee of indefinite static holding under
arbitrary cooling/ambient conditions. The pinned protocol catalog has no rated
torque field, so verified ratings are kept in `robstride/ratings.py`; motor
identity/model assignments still come from the canonical pinned hardware
description, not duplicated per-joint configuration. Unknown ratings, missing
motor coverage, duplicate targets and non-unit actuator gears fail at load.

Each model actuator is bound by its actual joint/tendon target. For differential
ankles, `actuator_force` already represents each motor's tendon force/torque;
the penalty does not apply 3 Nm separately to ankle pitch/roll joint torques.
For example, two motor torques of -3/+3 Nm produce 6 Nm pitch and zero roll
through the current differential while both motors remain at threshold.
Motor direction signs do not change an absolute-torque penalty. The mapping is
resolved at initialization; rollout math uses a constant rating vector.

The 25% standing share, +/-15 cm torso COM range/curriculum, command ranges,
tracking tolerances, extra stop step, gait cadence, lateral starts, IMU frames,
feedback filtering/latency and actuator effort limits are unchanged. This is
a fresh-training recipe; old checkpoints explicitly retain their old reward
mode when metadata is loaded.

Checkpoint frequency is 1024 updates and video frequency 8192 updates. Each
video event evaluates the same policy in walking, 30-second quiet standing,
and four separate directional push cases. Settling is evaluated independently
of reward and survival, with physical traces and overlays. See
[training configuration](training_configuration.md#evaluation-videos-video_config).
The periodic suite is nominal seed 0, not a complete randomized or emulator
qualification. Multi-seed, transition-phase and native/emulator comparisons
remain required before deployment; no trained-policy success is implied by
unit tests or GPU smoke checks.

## Push scheduling (unchanged)

Training uses instantaneous additive world-XY root-velocity kicks, not sustained
forces. Each episode has its own `step_count`. At a 20 ms control step, a kick
occurs before integration when the existing count is a positive multiple of
250: 5, 10, 15 seconds, and so on if the episode continues. There is no kick at
reset. Failure/timeout resets restart the interval; command changes do not.
Worlds initially reset together share a schedule until episode resets diverge.
The interval is fixed, not sampled or globally timed. There is no settling,
contact, gait-phase or standing/walking condition. The standard 20-second
episode normally resets before a 20-second kick.

Each event independently samples X and Y uniformly from [-0.5, 0.5] m/s at full
strength and adds this delta to existing velocity. Maximum diagonal delta is
about 0.707 m/s. No Z or angular kick is added. The persistent training
curriculum scales amplitude from 0.2 to 1 over 488,282 per-world control steps,
approximately 1B transitions with 2,048 worlds. Initial per-axis bounds are
therefore +/-0.1 m/s. Episode resets do not reset curriculum progress; the
5-second interval does not change with the curriculum.

Native videos do not run this repeated training-push scheduler. Quiet standing
has no kick for 30 seconds. Each directional trial has one fixed +/-0.5 m/s
world-X or world-Y kick at 5 seconds, then up to 15 seconds of recovery. It is
not curriculum-scaled and is applied even if the robot has not settled.
`push_precondition_settled` and `push_recovery_settled` distinguish that failure
from successful recovery. Right/left denote world -Y/+Y, not instantaneous
body axes after the robot turns.

## Validation boundary

Tests cover shared reward activation, actual-contact slip, linear alignment
tails, unchanged motor torque costs under tilt/speed, all-motor/differential
coverage, explicit pose mappings,
reward-input recovery sequences, push interval boundaries and legacy metadata.
Synthetic sequences are not dynamically feasible recovery demonstrations.
A native no-policy check holding nominal servo targets tipped backward after
1.6 seconds, without torque saturation or self-contact. Fixed servo targets are
not a balance controller; this does not prove the pose is infeasible, but the
nominal reference must not be treated as a demonstrated stable equilibrium.
The nominal whole-robot COM projects inside the sole support footprint, about
4.18 cm behind the sole centers (6.82 cm ahead of the rear bounding edge).
This geometric check alone does not establish dynamic or torque feasibility.
Fresh policy training and multi-seed evaluation remain necessary. Changing
this profile does not modify any running job or old checkpoint.

## Reference basis

Reference review informed the structure, not a wholesale coefficient copy:

- [Booster T1](https://github.com/BoosterRobotics/booster_gym/blob/da396a06d6eed99e2de72d7749c48ee8748950f9/envs/t1.py)
  keeps foot geometry costs independent of double-support gating.
- [Playground T1](https://github.com/google-deepmind/mujoco_playground/blob/4057c147714b6ac09b395377f1a1724bbeacc4d3/mujoco_playground/_src/locomotion/t1/joystick.py)
  freezes zero-command phase while retaining foot-height/pose incentives.
- [Unitree G1](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py)
  retains selected hip/waist posture costs; its 2% explicit standing fraction
  is not our target. We adapt the structure, not its coefficients or command
  distribution. Fore-aft alignment and the half-rated motor torque objective
  are Bello-specific.
