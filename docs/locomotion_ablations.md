# Whole-body locomotion ablations

These experimental recipes inherit `bello_mini_locomotion`. They use the same
PPO settings, seed 0, 2,048 environments, command distribution, randomization,
and 1,000,038,400 transitions (24,415 updates). They train from scratch. The
baseline is the current recipe, not the historical update-106496 checkpoint.

| Experiment suffix (`bello_mini_ablation_...`) | Cumulative change |
| --- | --- |
| `baseline` | Current rewards and leg/waist action space |
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
