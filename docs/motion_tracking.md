# Whole-Body Motion Tracking

The motion-tracking task trains Bello to reproduce GMR reference motion in
MJX-Warp. It is separate from the velocity-command locomotion environment and
uses the existing PureJaxRL-style PPO implementation.

The supported motion command surface is:

```text
bello-import-gmr-motion
bello-import-gmr-motion-dataset
bello-render-motion-reference
bello-check-motion-feasibility
bello-train-motion-ppo
bello-eval-motion-policy
bello-run-motion-hardware
```

Use `uv run <command>` outside an activated development shell.

Physical deployment is intentionally separate from native MuJoCo evaluation.
See [hardware_motion.md](hardware_motion.md) for strict checkpoint validation,
read-only preflight, guarded arming, and the current short-horizon IMU odometry
constraint.

## Robot Contract

The task always prepares `full_body` Bello with:

- `waist_roll_joint` rewritten as fixed
- `waist_pitch_joint` rewritten as fixed
- `waist_yaw_joint` retained and actuated
- `neck_yaw_joint` rewritten as fixed
- `head_pitch_joint` rewritten as fixed
- both wrist subtrees replaced by end-effector spheres
- non-foot collision meshes replaced by fitted boxes
- the source URDF's five primitive foot collision shapes per foot retained

The result has `nq=32`, `nv=31`, and `nu=25`. Fixed links remain named bodies,
so `waist_roll_link` and `torso_link` are still available to FK and rewards.
The robot is anatomically forward along local `+Y`; the importer does not rotate
the reference into an X-forward convention.

Motion archives and policies expose 25 logical one-DoF joints. The physical
differential ankle tendons, four ankle motors, force limits, and motor-coordinate
control limits remain active in the generated model. Reference poses and both
policy action formulations use the model's `actuator_coordinate_transform` to
map logical ankle pitch/roll targets into physical motor coordinates.

The canonical `joint_names` and policy action order is:

```text
left_hip_pitch_joint
left_hip_roll_joint
left_knee_joint
left_ankle_pitch_joint
left_ankle_roll_joint
right_hip_pitch_joint
right_hip_roll_joint
right_hip_yaw_joint
right_knee_joint
right_ankle_pitch_joint
right_ankle_roll_joint
left_hip_yaw_joint
waist_yaw_joint
right_shoulder_pitch_joint
right_shoulder_roll_joint
right_shoulder_yaw_joint
right_elbow_pitch_joint
right_elbow_yaw_joint
right_wrist_pitch_joint
left_shoulder_pitch_joint
left_shoulder_roll_joint
left_shoulder_yaw_joint
left_elbow_pitch_joint
left_elbow_yaw_joint
left_wrist_pitch_joint
```

## Import GMR Motions

Only trusted local GMR pickle input is supported. Set `GMR_REPO` to a compatible
GMR checkout, then convert a single motion from this repository's root:

```bash
export GMR_REPO=/path/to/GMR

uv run bello-import-gmr-motion \
  "$GMR_REPO/motion_data/bello_validation/walk1_subject1_bello_box_from_smplx.pkl" \
  motions/local/walk1_subject1_fixed_waist_head.npz
```

`motions/local/` is ignored by Git. Training depends only on the resulting
versioned `.npz`; it does not import GMR or read GMR robot assets.

The complete LAFAN1 archive can be extracted and retargeted in GMR, then
imported here in the canonical 14-body format:

```bash
(
  cd "$GMR_REPO"
  nix develop -c python scripts/prepare_lafan1_bello_dataset.py --jobs 2
)

uv run bello-import-gmr-motion-dataset \
  "$GMR_REPO/motion_data/lafan1_bello_fixed_head" \
  motions/local/lafan1_fixed_head \
  --jobs 2
```

The ASAP TairanTestbed source motions can be converted and retargeted in GMR:

```bash
(
  cd "$GMR_REPO"
  nix develop -c python scripts/prepare_asap_bello_dataset.py --jobs 2
)

uv run bello-import-gmr-motion-dataset \
  "$GMR_REPO/motion_data/asap_bello_fixed_head" \
  motions/local/asap_fixed_head \
  --jobs 2
```

Both pipelines are resumable: existing outputs are reused unless `--overwrite`
is supplied. Each output directory contains a manifest with frame counts,
durations, upstream GMR hashes, collision diagnostics, and any failures. The
maintained local datasets are 51 ASAP clips and 77 fixed-head LAFAN1 clips; all
use the exact 25-joint `qpos[T, 32]` and `qvel[T, 31]` contract.

The tracked single-foot balance examples under `motions/asap_fixed_head/` come
from [ASAP](https://github.com/LeCAR-Lab/ASAP) revision
`df5320cc47dd8cad97961bdfabfe402dd62ad999`, distributed under its
[MIT license](https://github.com/LeCAR-Lab/ASAP/blob/df5320cc47dd8cad97961bdfabfe402dd62ad999/LICENSE).
Their hashes and regeneration provenance are recorded in that directory's
README.

LAFAN1 is the
[Ubisoft La Forge Animation Dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset),
licensed under
[CC BY-NC-ND 4.0](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt).
The retargeted Bello binaries are not distributed by this repository. Regenerate
them locally for noncommercial use, retain the attribution and license notice,
and keep them below the ignored `motions/local/lafan1_fixed_head/` directory.
They must not be committed or silently uploaded by cloud tooling.

The older `motions/local/lafan1/` export has 27 joints, including movable neck
yaw and head pitch, with `qpos[T, 34]` and `qvel[T, 33]`. It is stale and
unsupported: do not copy it, select it in a preset, or upload it. Motion training
supports only the fixed-head 25-joint morphology.

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
qpos[T, 32], qvel[T, 31]
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
uv run bello-render-motion-reference \
  motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz \
  outputs/motion_tracking/reference_balance.mp4
```

Check whether the reference can be followed by the configured PD actuators
without policy feedback:

```bash
uv run bello-check-motion-feasibility \
  motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz \
  --output outputs/motion_tracking/balance_feasibility.json
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
reference_joint_pos[25]
reference_joint_vel[25]
current_joint_pos[25]
current_joint_vel[25]
last_action[25]
root_position_error_heading[3]
root_orientation_error[3]
root_linear_velocity_error_heading[3]
root_angular_velocity_error_heading[3]
reference_phase_sin_cos[2]
```

Actor observation size is 139. The asymmetric critic adds six values per
tracked body, so its size is `139 + 6 * B`: 199 for the 10-body baseline profile
and 223 for the maintained 14-body datasets. Observation stacking is handled by
PPO; the default motion config uses four frames.

Actions are normalized residual position targets around the next reference
pose. A zero action therefore supplies the known kinematic motion as PD
feedforward, while PPO learns balance and tracking corrections. The default
motion config uses one quarter of each logical joint range as its residual
scale, clips actions to `[-1, 1]`, clips resulting logical position targets to
their allowed ranges, maps the targets through the physical ankle transmission,
and retains motor-coordinate control and per-motor force limits.

Positive exponential rewards track root position/orientation, joint
position/velocity, body-relative position/orientation, and body velocity.
Action-rate and bounded joint-limit costs are negative terms, keeping the total
per-step reward finite and bounded even during a large joint excursion. Root
height, root orientation, end-effector divergence, and nonfinite state are
terminal failures.
Reaching the end of the reference is a nonterminal timeout boundary so PPO may
bootstrap the terminal reference state. Reward terms and their configured
weights are per-step values and are not multiplied by the control timestep.

## Train And Evaluate

### Paper-aligned BeyondMimic profile

The baseline above remains available for its existing checkpoints. The separate
`beyondmimic` profile changes the MDP rather than treating the paper as a PPO
hyperparameter preset. Generate its 14-body motion archive with:

```bash
uv run bello-import-gmr-motion \
  "$GMR_REPO/motion_data/bello_validation/walk1_subject1_bello_box_from_smplx.pkl" \
  motions/local/walk1_subject1_fixed_waist_head_beyondmimic.npz \
  --tracked-body-profile beyondmimic
```

The additional targets are the left/right hip-roll and shoulder-roll links.
The profile uses `torso_link` as its anchor and implements the paper's online
hybrid transform: the global torso reference is retained, while non-anchor
targets preserve reference height and roll/pitch but follow the robot anchor's
XY position and yaw drift.

Its single-timestep actor layout is:

```text
reference_joint_pos[25]
reference_joint_vel[25]
torso_position_error[3]
torso_orientation_error_6d[6]
base_linear_velocity_base_frame[3]
base_angular_velocity_base_frame[3]
current_joint_pos[25]
current_joint_vel[25]
last_action[25]
```

The actor size is 140. The critic adds clean torso-relative body positions and
6D orientations, nine values per body, for a 266-value observation with the
14-body profile.
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

The adaptive sampler follows the paper and official reference implementation.
Terminal counts update a shared per-bin EMA as
`f_bar = 0.999 * f_bar + 0.001 * f`. Before sampling, every bin receives the
absolute floor `0.1 / S`; a replicated-boundary, non-causal convolution uses
the normalized kernel `[1, 0.8, 0.8^2]` so failures increase the probability of
the preceding two bins. The convolved scores are normalized, a bin is drawn
multinomially, and the initial phase is uniform within that bin. W&B reports
`sampling/entropy`, `sampling/top_probability`, normalized
`sampling/top_bin`, and the mean/max failure EMA.

Start the complete 30,000-update, 4096-environment run with:

```bash
uv run bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name beyondmimic
```

Select any imported sequence by name, without its `.npz` suffix:

```bash
uv run bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name beyondmimic \
  motion.sequence=fallAndGetUp2_subject2
```

Select an ASAP skill by overriding both the dataset and sequence:

```bash
uv run bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name beyondmimic \
  motion.directory=motions/local/asap_fixed_head \
  motion.sequence=video_CR7_level1_filter_amass
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
uv run bello-train-motion-ppo
```

Start the default 1024-environment training run:

```bash
uv run bello-train-motion-ppo dry_run=false
```

The reproducible single-walk overfit baseline is:

```bash
uv run bello-train-motion-ppo \
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
uv run bello-train-motion-ppo \
  dry_run=false \
  wandb_config.enabled=true \
  wandb_config.run_name=bello-walk-fixed-waist-head
```

When W&B is enabled, each scheduled evaluation MP4 is uploaded under
`eval/motion_policy` at the corresponding environment-step index. Configure the
schedule and destination with:

```bash
uv run bello-train-motion-ppo \
  dry_run=false \
  wandb_config.enabled=true \
  video_config.every_updates=50 \
  video_config.wandb_name=eval/bello_walk
```

Videos are also retained locally under
`outputs/motion_tracking/eval_videos/eval_<update>.mp4`. Set
`video_config.enabled=false` to disable rendering, or
`video_config.log_wandb=false` to keep local videos without uploading them.

Each motion evaluation runs in a fresh spawn-isolated CPU process. That process
restores the scheduled checkpoint, performs a native MuJoCo rollout, and sends
raw frames to FFmpeg through the repository's failure-safe video writer. A
native renderer failure or Python exception is reported as a warning and cannot
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
learning rate, value and observation-normalization statistics, adaptive-sampler
failure EMA, RNG state, and the source motion hash. Metadata also names the
formulation, actor/critic/action layouts, environment contract, motion schema
and content hash, and runtime fingerprints. Resume from a leaf by increasing
`config.total_timesteps` if necessary:

```bash
uv run bello-train-motion-ppo \
  --config-dir "$PWD/experiments/train_motion" \
  --config-name walk_overfit \
  checkpoint_config.resume_path=/absolute/path/to/update_000500
```

Resume rejects a different robot asset, motion export, environment, policy
shape, PPO configuration, or locomotion/motion task mismatch. A motion file may
move only when its schema and content hash still match the checkpoint. Only
`total_timesteps` may differ. Unsupported legacy checkpoints are rejected.
The full batched MJX environment state, observation stacks, optimizer, adaptive
learning rate, normalization statistics, adaptive sampler, and RNG are restored
exactly, so the next update is a true continuation rather than a fresh reset.

Render a deterministic policy from a completed checkpoint:

```bash
uv run bello-eval-motion-policy \
  checkpoints/bello_wholebody_<timestamp>/final \
  outputs/motion_tracking/policy.mp4
```

If the recorded clip has moved, provide its new path explicitly; evaluation
accepts it only after the complete schema and file-content hash match:

```bash
uv run bello-eval-motion-policy \
  checkpoints/bello_wholebody_<timestamp>/final \
  outputs/motion_tracking/policy.mp4 \
  --motion /new/path/to/the_same_clip.npz
```

The checkpoint records the task name, canonical motion path, asset config,
motion environment config, PPO config, policy/critic observation sizes, action
size, optimizer state, recovery state, and source motion hash. Evaluation rejects a
locomotion checkpoint instead of applying it to the wrong environment.
