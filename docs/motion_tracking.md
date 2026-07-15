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
- non-foot collision meshes replaced by fitted boxes
- the source URDF's five primitive foot collision shapes per foot retained

The result has `nq=34`, `nv=33`, and `nu=27`. Fixed links remain named bodies,
so `waist_roll_link` and `torso_link` are still available to FK and rewards.
The robot is anatomically forward along local `+Y`; the importer does not rotate
the reference into an X-forward convention.

## Import GMR Motions

GMR pickle input is a trusted setup-time format. Convert it once:

```bash
bello-import-gmr-motion \
  /home/gmmyung/GMR/motion_data/bello_validation/walk1_subject1_bello_box_from_smplx.pkl \
  motions/local/walk1_subject1_fixed_waist.npz
```

`motions/local/` is ignored by Git. Training depends only on the resulting
versioned `.npz`; it does not import GMR or read GMR robot assets.

The complete local LAFAN1 archive can be extracted and retargeted in GMR with:

```bash
cd /home/gmmyung/GMR
nix develop -c python scripts/prepare_lafan1_bello_dataset.py --jobs 8
```

Import every resulting Bello pickle into the canonical 14-body training format:

```bash
cd /home/gmmyung/bello_mujoco_wholebody
nix develop -c uv run bello-import-gmr-motion-dataset \
  /home/gmmyung/GMR/motion_data/lafan1_bello \
  motions/local/lafan1 \
  --jobs 8
```

Both stages are resumable: existing outputs are reused unless `--overwrite` is
supplied. Each output directory contains a manifest with frame counts,
durations, hashes, and any failures. Dataset files remain local because
LAFAN1-derived artifacts must follow the source dataset's license.

Single-file imports reject any primitive-box self-intersection. The dataset
importer retains every finite, joint-limit-valid sequence and records collision
diagnostics in both the archive and manifest. This is intentional: sparse box
overlap in a kinematic reference is useful quality metadata, while the dynamic
training model still enforces contacts and penalizes undesired self-contact.
Sequences with persistent or deep overlap should be repaired before committing
compute to a long PPO run.

The legacy pickle stores `dof_pos` in the source MuJoCo `qpos[7:]` order. That
order differs from the generated model's actuator order. The importer applies
an explicit joint-name permutation, converts root quaternions from XYZW to
WXYZ, enforces quaternion continuity, checks hard joint limits, derives `qvel`
with `mj_differentiatePos`, evaluates primitive-box self-collisions, and
computes tracked-body FK and world velocities. A single-file import rejects
self-intersections; a dataset import records them as described above.

The canonical clip contains:

```text
format_version, fps, source_sha256
joint_names, body_names
qpos[T, 34], qvel[T, 33]
body_pos[T, B, 3], body_quat_wxyz[T, B, 4]
body_lin_vel[T, B, 3], body_ang_vel[T, B, 3]
foot_contacts[T, 2]
self_collision_frames, max_self_collision_penetration_m
deepest_self_collision_pair[2]
```

`B=10` for the baseline tracked-body profile and `B=14` for BeyondMimic.

Upstream GMR pickle exports may omit `foot_contacts`. The importer fills that
optional legacy field with zeros; the current motion-tracking environment
preserves the array in its canonical schema but does not use it for observations
or rewards.

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

Check whether the reference can be followed by the configured PD actuators
without policy feedback:

```bash
bello-check-motion-feasibility \
  motions/local/walk1_subject1_fixed_waist.npz \
  --output outputs/motion_tracking/walk_feasibility.json
```

This CPU MuJoCo replay uses the same 200 Hz physics, 50 Hz control, actuator
gains, effort limits, box collisions, and primitive feet as training. It reports
the first environment-equivalent terminal condition, root and joint tracking
error, force saturation, contact coverage, and stance-foot slip. Failure in this
open-loop replay means balance feedback is required; nonfinite state or broad
actuator saturation indicates a modeling or reference problem that should be
fixed before PPO.

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

Actions are normalized residual position targets around the next reference
pose. A zero action therefore supplies the known kinematic motion as PD
feedforward, while PPO learns balance and tracking corrections. The default
motion config uses `default_action_scale_ratio=0.25`, clips actions to `[-1, 1]`,
clips resulting position targets to actuator control ranges, and retains
per-joint force limits.

Positive exponential rewards track root position/orientation, joint
position/velocity, body-relative position/orientation, and body velocity.
Action-rate and bounded joint-limit costs are negative terms, keeping the total
per-step reward finite and bounded even during a large joint excursion. Root
height, root orientation, end-effector divergence, and nonfinite state are
terminal failures.
Reaching the end of the reference is a nonterminal timeout boundary so PPO may
bootstrap the terminal reference state.

## Train And Evaluate

### Paper-aligned BeyondMimic profile

The baseline above remains available for its existing checkpoints. The separate
`beyondmimic` profile changes the MDP rather than treating the paper as a PPO
hyperparameter preset. Generate its 14-body motion archive with:

```bash
bello-import-gmr-motion \
  /home/gmmyung/GMR/motion_data/bello_validation/walk1_subject1_bello_box_from_smplx.pkl \
  motions/local/walk1_subject1_fixed_waist_beyondmimic.npz \
  --tracked-body-profile beyondmimic
```

The additional targets are the left/right hip-roll and shoulder-roll links.
The profile uses `torso_link` as its anchor and implements the paper's online
hybrid transform: the global torso reference is retained, while non-anchor
targets preserve reference height and roll/pitch but follow the robot anchor's
XY position and yaw drift.

Its single-timestep actor observation contains reference joint position and
velocity as phase, torso pose error with a 6D rotation representation, actual
base twist in the base frame, joint position/velocity, and previous action.
Actor observations are corrupted during training and actor/critic observations
are empirically normalized. The critic additionally receives clean current
body poses relative to the torso. Running normalization statistics are saved in
checkpoints and restored for training and evaluation.

Actions are nominal-centered position setpoints with per-joint scale
`0.25 * effort_limit / stiffness`. They are not centered on the reference and
are not clipped to kinematic joint ranges. The generated MuJoCo force limits
remain active. Bello's existing hardware-oriented gains remain in use because
the source model does not provide motor reflected-inertia values from which the
paper's 10 Hz impedance can be derived defensibly.

Rewards contain only global anchor pose, relative body pose, global body linear
and angular velocity, action rate, soft joint limit, and undesired-contact
terms. Termination uses torso height, torso pitch/roll through projected
gravity, and ankle/wrist height errors. Adaptive one-second failure bins,
backward failure kernels, uniform sampling coverage, reset pose/velocity
perturbations, joint calibration offsets, observation noise, and interval root
velocity pushes are enabled.

Start the complete 30,000-update, 4096-environment run with:

```bash
bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name beyondmimic
```

Select any imported sequence by name, without its `.npz` suffix:

```bash
bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name beyondmimic \
  motion.sequence=fallAndGetUp2_subject2
```

The dataset root is configured by `motion.directory`. Checkpoint directories,
W&B run names, and evaluation video prefixes include `motion.sequence`, so
independent sequence runs do not overwrite one another.

The complete config is
`experiments/train_motion/beyondmimic.yaml`. W&B logging, checkpointing, and
evaluation videos are enabled and run every 500 updates.

MJX-Warp uses one shared model for all worlds in the current vectorized
backend. Consequently, per-environment ground-friction and torso-COM model
randomization from the Isaac Lab implementation are not active; reset-state,
sensor, calibration, and push randomizations are per environment. Adding the
two shared-model parameters requires batched models or multiple independently
compiled training lanes, not mutation of `mjx.Data`.

### Baseline profile

Validate the full setup without optimization:

```bash
bello-train-motion-ppo
```

Start the default 1024-environment training run:

```bash
bello-train-motion-ppo dry_run=false
```

The reproducible single-walk overfit baseline is:

```bash
bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name walk_overfit \
  wandb_config.enabled=true
```

It runs 1,024 updates with random reference phase, fixed world alignment, mild
joint noise, a constant `1e-4` learning rate, and checkpoints plus evaluation
videos every 100 updates.

The default config is `configs/train_motion/default.yaml`. It writes periodic
Orbax checkpoints every 100 updates plus a final leaf under `checkpoints/`, and
renders an evaluation video every 100 updates plus the final update. W&B
remains disabled until explicitly enabled:

```bash
bello-train-motion-ppo \
  dry_run=false \
  wandb_config.enabled=true \
  wandb_config.run_name=bello-walk-fixed-waist
```

When W&B is enabled, each scheduled evaluation MP4 is uploaded under
`eval/motion_policy` at the corresponding environment-step index. Configure the
schedule and destination with:

```bash
bello-train-motion-ppo \
  dry_run=false \
  wandb_config.enabled=true \
  video_config.every_updates=50 \
  video_config.wandb_name=eval/bello_walk
```

Videos are also retained locally under
`outputs/motion_tracking/eval_videos/eval_<update>.mp4`. Set
`video_config.enabled=false` to disable rendering, or
`video_config.log_wandb=false` to keep local videos without uploading them.

Motion evaluation runs in a dedicated process started before W&B or JAX creates
threads. That process restores the scheduled checkpoint, uses on-demand GPU
allocation for its batch-one rollout, and encodes MP4s with PyAV. A native
renderer failure or Python exception is reported as a warning and cannot
terminate PPO; the checkpoint is written before its video is requested.

The default Warp constraint capacity is `njmax=2048`. Do not reduce it to the
locomotion default of 256: randomized whole-body reference phases can create
more simultaneous constraints, and a smaller buffer reports `nefc overflow`
before destabilizing the rollout. The default policy uses bounded
`tanh_gaussian` actions, learning rate `3e-4`, and initial log standard
deviation `-2.0` for a conservative first sustained run.

Periodic checkpoints are stored below one timestamped run directory:

```text
checkpoints/bello_wholebody_<timestamp>/
  update_000100/
  update_000200/
  final/
```

Each leaf contains policy parameters, optimizer state, update count, adaptive
learning rate, value and observation-normalization statistics, RNG state, and
the source motion hash. Resume from a leaf by increasing
`config.total_timesteps` if necessary:

```bash
bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name walk_overfit \
  checkpoint_config.resume_path=/absolute/path/to/update_000500
```

Resume rejects a different robot asset, motion export, environment, policy
shape, or PPO configuration. Only `total_timesteps` may differ. Environments
are freshly reset after restore rather than checkpointing the full batched MJX
simulation state.

Render a deterministic policy from a completed checkpoint:

```bash
bello-eval-motion-policy \
  checkpoints/bello_wholebody_<timestamp>/final \
  outputs/motion_tracking/policy.mp4
```

The checkpoint records the task name, canonical motion path, asset config,
motion environment config, PPO config, policy/critic observation sizes, action
size, optimizer state, recovery state, and source motion hash. Evaluation rejects a
locomotion checkpoint instead of applying it to the wrong environment.
