# Whole-body locomotion ablations

These experimental recipes inherit `bello_mini_locomotion`. They use the same
PPO settings, seed 0, 2,048 environments, command distribution, randomization,
and 1,000,038,400 transitions (24,415 updates). They train from scratch. The
baseline restores the deployed update-106496 posture objective: fixed upright
penalty -5 and waist joint-posture weights 2. Torso commands are disabled, omitted
from actor/critic observations, and have no tracking reward. The contact decoder
remains corrected. These are fresh runs, not continuations of deployed weights.

| Experiment suffix (`bello_mini_ablation_...`) | Cumulative change |
| --- | --- |
| `baseline` | Velocity-only commands, deployed posture objective, leg/waist action space |
| `tracking` | Narrower tracking kernels, stronger linear/yaw rewards and Huber error penalties, preserve small nonzero command components |
| `gait` | Two-sided swing-height target, command-aware transverse swing-speed cost, non-saturating impact costs, continuous substep peak-load cost, torso damping and roll/pitch centroidal momentum cost |
| `whole_body` | All actuators policy-controlled, with corresponding joint observations and separate arm posture weights |

For example:

```bash
uv run --group training bello-train experiment=bello_mini_ablation_whole_body
uv run bello-vast-train experiment=bello_mini_ablation_whole_body branch=experiment/whole-body-locomotion-ablations max_price=0.65 duration_hours=6
```

The action space is derived from `asset_config.actuator_mode: policy`. Motor
targets, differential coupling, limits, feedback history, actor/critic layouts,
checkpoint metadata and export retain their existing dimension-driven paths.
All joints, including arms and wrists, are observed and actuated. This does not
use a reference motion or require a different PPO architecture.
The Vast preflight checks a PPO update, checkpoint restore, native rollout/video
and deployment export before starting the full locomotion run.

New reward configuration:

- `tracking.respect_command_deadband: false` uses each commanded velocity
  component without rounding small values to zero. Command sampling and gait
  transition deadbands retain their existing meanings.
- `gait.swing_clearance_overshoot_weight` penalizes sole-corner height above the
  phase-shaped target; zero retains the existing lower-bound objective.
- `gait.unbounded_touchdown` replaces the two saturating squared-tanh impact
  costs with positive Huber costs of normalized force excess and downward speed.
- `weights.foot_contact_load` applies Huber excess-load cost throughout contact,
  including force peaks after touchdown. Loads use physics-substep maxima and
  impulse divided by body weight and control timestep, as in the existing
  touchdown feature collection.
- `weights.swing_lateral_velocity` penalizes swing foot velocity transverse to
  the local commanded tangent, including commanded yaw about the pelvis. It
  fades near zero local command speed and at swing endpoints, and is disabled
  in actual contact. `gait.swing_lateral_speed_scale` provides its velocity scale.
- `weights.centroidal_momentum_xy` penalizes world roll/pitch angular momentum
  about the robot COM, divided by current mass times target height squared.
  It uses each world's spatial inertias, including domain randomization. Yaw
  momentum is unpenalized to retain turning authority. The implementation is
  checked against native MuJoCo `mj_subtreeVel`, including randomized inertias.

Existing `ang_vel_xy` already measures torso angular velocity in these recipes;
the gait variant increases its damping and adds a small action-acceleration
cost. Arm posture weights allow shoulder pitch/elbow motion while keeping
wrists closer to nominal. These weights are experimental, not validated gains.

## Comparing runs

Compare equal update checkpoints, rather than raw total reward across recipes.
Track `tracking/vx_abs_error`, `tracking/vy_abs_error`,
`tracking/yaw_abs_error`, existing squared velocity errors, episode lengths,
torso orientation error, contact-force maxima, foot slip and crossing. Signed
`tracking/vx_error` and `tracking/vy_error` expose persistent underspeed/drift.

Repeat the nine-command trajectory suite on each selected checkpoint: stand,
forward 0.25/0.5 m/s, backward 0.3 m/s, lateral +/-0.3 m/s, turns +/-0.5 rad/s,
and forward 0.3 m/s plus yaw 0.3 rad/s. Check stance world-frame slip and
swing-relative paths separately. Retain pushes, command transitions and multiple
evaluation seeds; smoother feet do not count as improvement if tracking or
survival worsens. One seed and a 1B-transition pilot cannot establish robustness
or convergence; follow with matched training seeds before selecting a policy.

These four runs retain the gait clock. Removing it needs a separate experiment
with phase-free foot rewards and matching deployment observations.

## Contact-force correction and bounded pilot

The original `629bfd8` ablations are invalid for assessing contact shaping.
They read the first constraint row as the normal force. With pyramidal friction,
normal force is the sum of all pyramid edges (`2 * (condim - 1)` rows), or one
row for frictionless contacts. The old result depends on friction direction and
can be zero under load. This affects contact classification, slip, impact,
non-foot contact and self-collision costs. Native evaluation already used
`mj_contactForce`. The decoder now agrees with that API for pyramidal and
elliptic cones; motion contact classification uses the same correction.

The velocity-only 500M run `kd985miw` survived its final forward evaluation and
alternated support, but failed tracking and slip gates: planar velocity RMSE
0.203 m/s, yaw-rate RMSE 0.319 rad/s, and contact-point slip 0.341 m/s.
Its generic slip reward measured foot-body origin velocity, and phase-specific
stance-slip shaping was disabled.

`bello_mini_contact_pilot` now uses loaded contact-point slip for both slip costs.
The phase-specific cost applies a Huber loss with a 0.10 m/s speed scale during
desired stance, with weight -1.0. Generic squared slip has weight -2.0; contact
timing is -8.0 and swing clearance is -2.0. Clearance overshoot and lateral swing
motion are also penalized. The pilot retains the velocity-only tracking
variant's 14-action policy, gait clock, strict tracking objective, seed and
velocity command distribution; it has no torso commands. These changes target
sliding and weak support timing and require fresh behavioral evaluation.
The contact-slip run `cyjap1pn` reduced slip to 0.049 m/s at 252M transitions,
but retained a backward bias and fell at 7.34 seconds under a +0.5 m/s command.
Native replay of forward, backward, standing and lateral commands found no
velocity-frame sign discrepancy (maximum reward/evaluation difference below
5e-7 m/s). The next pilot triples `moving_lin_vel_xy_error` to -12 and doubles
`moving_ang_vel_z_error` to -4, retaining the narrow tracking kernels and contact
costs. This tests whether a stronger error cost overcomes the backward bias;
it is not a verified gait fix. It starts fresh because rewards changed.
It trains from scratch for 500,039,680 transitions and records forward
walking every 2,048 updates and at completion, without the expensive push suite.
Checkpoints are saved every 512 updates (about 21M transitions). Vast's advance
stop request also saves a final checkpoint before budget teardown.
The previous torso-conditioned 100M pilot and its continuations are superseded;
their 600-input actor checkpoints cannot resume this 570-input actor task.
This is a diagnostic run, not a deployment recipe. A passing
forward rollout still requires standing, lateral, turning, transition and
multiple-seed evaluation before selecting a controller.

The subsequent direction-tracking run `1a3aqjdw` completed 500M transitions.
Its final +0.5 m/s rollout survived 10 seconds, with mean forward speed 0.315 m/s,
velocity RMSE 0.241 m/s and contact slip 0.045 m/s. After the 2-second warmup,
torso lateral tilt was 15.8 degrees RMS and 47.9 degrees peak-to-peak; waist
roll moved through 44.5 degrees. The original acceptance gate did not measure
this sway.

The current pilot (`mini_torso_stability_seed0`) strengthens existing soft
posture costs: waist-roll joint-pose weight 2 -> 10, torso `flat_orientation`
-5 -> -10, and torso `ang_vel_xy` -0.15 -> -0.5. Waist yaw and leg posture
weights retain their previous values. Linear velocity and yaw-rate tracking
measure the pelvis, with planar velocity expressed in the pelvis yaw frame;
uprightness and roll/pitch angular-velocity costs measure the torso. Foot-heading
costs measure the feet. The strict velocity objective, gait conditioning and
disabled torso commands remain in place. These are trial gains for fresh
training, not a demonstrated improvement or hard joint limits.

```bash
uv run --group training bello-train experiment=bello_mini_contact_pilot
```

The frozen `video_config.walking_acceptance` config enables reward-independent
checks. Defaults discard the first 2 seconds and require at least 6 measured
seconds, no termination or invalid values, planar velocity RMSE <= 0.10 m/s,
yaw-rate RMSE <= 0.20 rad/s, and mean stance slip <= 0.05 m/s. Slip is the
normal-load-weighted tangential velocity at actual foot-ground contact points,
not the moving sole center. Debounced single support must occupy >= 20% of the
window, each leg must have >= 5%, and there must be >= 4 support alternations.
Contact debounce is 0.04 seconds. These are explicit screening thresholds, not
proof of hardware safety or general locomotion quality.

The gate also requires lateral tilt RMS <= 10 degrees and peak absolute tilt
<= 20 degrees after warmup. `lateral_tilt_rms_max` and `lateral_tilt_peak_max`
are configured in radians. Tilt uses the uprightness reward's configured body
(torso in these recipes), projected onto the physical lateral axis of the
velocity-command yaw frame. This respects each robot's forward-axis convention
and permits moderate balancing motion. It reports RMS, peak absolute tilt and
peak-to-peak excursion as `walking/lateral_tilt_*`, all in radians.

Acceptance requires a translating command of at least 0.2 m/s and first-failure
stopping. `walking/accepted` and individual `walking/pass_*` checks are logged
with the video phase. The adjacent `.walking.jsonl` is uploaded to W&B and stores
commands, measured velocities, body XY/yaw, sole XYZ, support, forces and slip.
It also records `orientation_lateral_tilt` in radians.
The training logger also preserves the `tracking/*` velocity-error diagnostics;
previous ablations computed these but accidentally dropped them during reduction.

Validation on 2026-10-12: a static 1 kg GPU contact reports 9.81013 N with the
corrected decoder versus 9.81010 N in native MuJoCo; the old pyramidal decoder
reports 2.45253 N. A Mini checkpoint spot check found equal viewer/training
physical parameters and matching native trajectories, with CPU/GPU one-step
maximum qpos error 2.49e-7, qvel error 4.97e-5, and same-state actor feedback
error 2.77e-5. These checks identify and repair the contact measurement mismatch;
they do not establish long-horizon CPU/GPU policy equivalence.
