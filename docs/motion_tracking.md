# Whole-Body Motion Tracking

The motion-tracking task trains Bello to reproduce GMR reference motion in
MJX-Warp. It is separate from the velocity-command locomotion environment and
uses the existing PureJaxRL-style PPO implementation.

## Robot Contract

The task always prepares `full_body` Bello with:

- `waist_roll_joint` rewritten as fixed
- `waist_pitch_joint` rewritten as fixed
- `waist_yaw_joint` retained and actuated
- both wrist subtrees replaced by end-effector spheres
- the source URDF's five primitive foot collision shapes per foot retained

The result has `nq=34`, `nv=33`, and `nu=27`. Fixed links remain named bodies,
so `waist_roll_link` and `torso_link` are still available to FK and rewards.
The robot is anatomically forward along local `+Y`; the importer does not rotate
the reference into an X-forward convention.

## Import A GMR Clip

GMR pickle input is a trusted setup-time format. Convert it once:

```bash
bello-import-gmr-motion \
  /home/gmmyung/GMR/motion_data/bello_validation/walk1_subject1_bello_from_smplx.pkl \
  motions/local/walk1_subject1_fixed_waist.npz
```

`motions/local/` is ignored by Git. Training depends only on the resulting
versioned `.npz`; it does not import GMR or read GMR robot assets.

The legacy pickle stores `dof_pos` in the source MuJoCo `qpos[7:]` order. That
order differs from the generated model's actuator order. The importer applies
an explicit joint-name permutation, converts root quaternions from XYZW to
WXYZ, enforces quaternion continuity, checks hard joint limits, derives `qvel`
with `mj_differentiatePos`, and evaluates tracked-body FK and world velocities.

The canonical clip contains:

```text
format_version, fps, source_sha256
joint_names, body_names
qpos[T, 34], qvel[T, 33]
body_pos[T, 10, 3], body_quat_wxyz[T, 10, 4]
body_lin_vel[T, 10, 3], body_ang_vel[T, 10, 3]
foot_contacts[T, 2]
```

Runtime loading rejects missing fields, nonfinite values, incorrect shapes,
unnormalized quaternions, or a joint order that differs from the loaded model.
Sampling uses linear interpolation for translations and velocities and
shortest-arc SLERP for orientations.

Render the kinematic reference before training:

```bash
bello-render-motion-reference \
  motions/local/walk1_subject1_fixed_waist.npz \
  outputs/motion_tracking/reference_walk.mp4
```

## Environment Contract

`BelloMotionTrackingEnv` runs at the repository defaults of 200 Hz physics and
50 Hz control. Reset samples a reference phase, places that phase at a random
world XY/yaw transform, initializes `qpos` and `qvel` from the aligned
reference, and applies configured joint noise. Batched `step_batch` uses one
MJX-Warp world batch and auto-resets completed environments.

The actor observes the next control-step reference and current proprioception:

```text
reference_joint_pos[27]
reference_joint_vel[27]
current_joint_pos[27]
current_joint_vel[27]
last_action[27]
root_position_error_heading[3]
root_orientation_error[3]
root_linear_velocity_error_heading[3]
root_angular_velocity_error_heading[3]
reference_phase_sin_cos[2]
```

Actor observation size is 149. The asymmetric critic adds tracked-body relative
position and orientation errors for a total size of 209. Observation stacking
is handled by PPO; the default motion config uses four frames.

Actions are normalized residual position targets around the configured default
pose. The default motion config uses `default_action_scale_ratio=0.5`, clips
actions to `[-1, 1]`, clips resulting position targets to actuator control
ranges, and retains per-joint force limits.

Positive exponential rewards track root position/orientation, joint
position/velocity, body-relative position/orientation, and body velocity.
Action-rate and joint-limit costs are negative terms. Root height, root
orientation, end-effector divergence, and nonfinite state are terminal failures.
Reaching the end of the reference is a nonterminal timeout boundary so PPO may
bootstrap the terminal reference state.

## Train And Evaluate

Validate the full setup without optimization:

```bash
bello-train-motion-ppo
```

Start the default 4096-environment training run:

```bash
bello-train-motion-ppo dry_run=false
```

The default config is `configs/train_motion/default.yaml`. It enables final
Orbax checkpointing under `checkpoints/`; W&B remains disabled until explicitly
enabled:

```bash
bello-train-motion-ppo \
  dry_run=false \
  wandb_config.enabled=true \
  wandb_config.run_name=bello-walk-fixed-waist
```

Render a deterministic policy from a completed checkpoint:

```bash
bello-eval-motion-policy \
  checkpoints/bello_wholebody_<timestamp> \
  outputs/motion_tracking/policy.mp4
```

The checkpoint records the task name, canonical motion path, asset config,
motion environment config, PPO config, policy/critic observation sizes, action
size, optimizer state, and value-normalization state. Evaluation rejects a
locomotion checkpoint instead of applying it to the wrong environment.
