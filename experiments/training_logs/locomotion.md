# Locomotion Reward and Environment Tuning Notes

Date: 2026-07-09

These notes track the walking recipe at `experiments/train/locomotion.yaml`.
They are experiment notes, not package defaults.

## Current Recipe

Run:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion
```

The locomotion recipe inherits `configs/train/default.yaml` directly. It does
not inherit `experiments/train/standing.yaml`, but it explicitly bootstraps from
the stabilized standing setup and then aligns the supported PPO knobs with the
Unitree G1/RSL-RL recipe:

- full-body asset with sphere end effectors
- lower body and waist yaw controlled by policy
- waist roll/pitch, head, arms, and fingers held
- `env_config.root_link: torso_link`
- MuJoCo timestep `0.005s` with `env_config.decimation: 4`
- `env_config.contact.foot_contact_on_seconds: 0.0`
- `env_config.contact.foot_contact_off_seconds: 0.0`
- `env_config.contact.foot_contact_force_threshold: 1.0`
- `env_config.contact.undesired_contact_force_threshold: 1.0`
- `num_envs: 4096`
- `num_steps: 24`, equivalent to `24 * 0.02s = 0.48s`
- `total_timesteps: 4_915_200_000`, equivalent to 50,000 PPO updates
- `max_episode_steps: 1000`, equivalent to `20s`
- inherited PPO entropy coefficient `ent_coef: 0.01`, matching Unitree/RSL-RL
- `policy.hidden_dims: [512, 256, 128]`
- `policy.activation: elu`
- `policy.obs_stack_size: 5`, matching Unitree's policy observation history

The policy observation group follows Unitree more closely by omitting
`base_lin_vel`, `base_height`, and `gait_phase`. The critic uses a separate
clean privileged observation group that includes `base_lin_vel`.

The locomotion command distribution now starts with a forward-only range above
the gait command gate and expands by curriculum. Bello is anatomically
Y-forward, so this applies the forward range to `lin_vel_y` while keeping
`lin_vel_x` lateral:

```yaml
command_config:
  enabled: true
  lin_vel_x: [-0.1, 0.1]
  lin_vel_y: [0.15, 0.35]
  ang_vel_yaw: [-0.1, 0.1]
  standing_probability: 0.02
  resample_seconds: 10.0
  curriculum:
    enabled: true
    limit_lin_vel_x: [-0.3, 0.3]
    limit_lin_vel_y: [-0.5, 1.0]
    limit_ang_vel_yaw: [-0.2, 0.2]
    expand_ang_vel_yaw: false
    increment: 0.1
    success_threshold: 0.8
```

The curriculum uses completed-episode `track_lin_vel_xy` reward sums divided by
full episode duration, mirroring Unitree's
`episode_sums[track_lin_vel_xy] / max_episode_length_s` criterion. Linear X/Y
ranges expand when the normalized score exceeds `0.8 *
track_lin_vel_xy.weight` at an episode-length boundary. Yaw expansion is
disabled for now because the referenced Unitree G1 config only registers the
linear command curriculum.

Unitree-style fixed perturbations are also enabled, separate from curriculum:

```yaml
env_config:
  random_push:
    enabled: true
    interval_seconds: 5.0
    lin_vel_x: [-0.5, 0.5]
    lin_vel_y: [-0.5, 0.5]
  observation:
    scale:
      base_ang_vel: 0.2
      joint_vel: 0.05
    noise:
      enabled: true
      base_ang_vel: [-0.2, 0.2]
      projected_gravity: [-0.05, 0.05]
      joint_pos: [-0.01, 0.01]
      joint_vel: [-1.5, 1.5]
```

Noise is additive uniform noise before scaling, following IsaacLab's observation
manager. Random pushes set the floating-root XY velocity every `5s` per
environment after the first step of an episode and are logged under
`train/mean_random_push/*`.

## Unitree-Style Foot Rewards

The Unitree G1 29-DoF locomotion task in `unitree_rl_lab` uses humanoid-specific
foot shaping rather than the quadruped-style `feet_air_time` recipe:

- `feet_gait`: positive reward for each foot whose contact state matches the
  scheduled stance/swing phase, command-gated at nonzero command speed.
- `foot_clearance_reward`: positive exponential reward that is highest when
  moving feet are near the target clearance height.
- no humanoid `feet_air_time` term in the G1/H1 velocity configs.

The local reward implementation was changed to match that shape:

```yaml
reward_config:
  foot_clearance:
    enabled: true
    weight: 1.0
    target: 0.10
    std: 0.05
    tanh_mult: 2.0
  gait_contact:
    enabled: true
    weight: 0.5
    command_speed_threshold: 0.10
  feet_air_time:
    enabled: false
```

`gait_contact` is no longer a negative mismatch penalty. It sums matched feet
and multiplies by the positive weight and control timestep. With two feet,
perfect phase/contact agreement contributes:

```text
2 * 0.5 * 0.02 = +0.02 per control step
```

Near-standing commands receive zero gait reward, so the standing part of the
command distribution does not pressure the policy into an open-loop walking
phase.

`foot_clearance` is no longer a low-swing penalty. It was initially aligned to
Unitree's ungated whole-foot exponential score:

```text
exp(-sum((foot_height - target) ** 2 * tanh(tanh_mult * lateral_speed)) / std)
```

This was later changed to use contact gating because local videos showed
planted feet exploiting the ungated shape.

## Unitree G1 Weight Alignment

The locomotion recipe now follows the Unitree G1 reward weights where the local
pipeline has an equivalent term. The local reward implementation also follows
the Unitree raw-value reductions for these terms:

- yaw-frame linear velocity tracking
- summed joint velocity, joint acceleration, action-rate, and energy penalties
- summed absolute joint-limit violation
- summed contact-gated lateral plus scaled angular foot speed for `foot_slip`

```yaml
reward_config:
  track_lin_vel_xy.weight: 1.0
  track_ang_vel_z.weight: 0.5
  alive.weight: 0.15
  lin_vel_z.weight: -2.0
  ang_vel_xy.weight: -0.05
  flat_orientation.weight: -5.0
  base_height.weight: -10.0
  joint_vel.weight: -0.001
  joint_acc.weight: -0.00000025
  action_rate.weight: -0.05
  energy.weight: -0.00002
  joint_limit.weight: -5.0
  foot_slip.weight: -0.2
  foot_clearance.weight: 1.0
  gait_contact.weight: 0.5
  undesired_contact.weight: -1.0
```

`torque` is disabled because Unitree G1 uses `energy` rather than a direct
torque-L2 term. `termination` is also disabled to match Unitree's reward setup:
early termination is handled through done/reset and loss of future alive/tracking
reward rather than an explicit terminal penalty.

Remaining mismatches:

- Unitree G1 uses `base_height.target_height: 0.78`; Bello keeps an explicit
  `torso_link` target of `1.03`.
- Unitree uses IsaacLab/PhysX articulation state and contact sensors while
  Bello uses MJX/Warp state and MuJoCo contact data.
- Unitree uses separate deviation penalties for arms, waist, and hip roll/yaw.
  The local pipeline currently has broad held-policy deviation groups, so the
  recipe enables `hold_joint_deviation.weight: -0.1` as the closest upper-body
  approximation and leaves `policy_joint_deviation` disabled rather than
  penalizing every policy-controlled leg joint.
- Unitree reads articulation-provided `joint_acc`; Bello uses a finite
  difference of actuator joint velocity across control steps.

## 2026-07-09 - PPO/env alignment pass

Aligned the locomotion training scale and policy config with the Unitree G1
RSL-RL runner where the local PPO supports an equivalent knob:

- changed `config.num_envs: 1024 -> 4096`
- kept `config.num_steps: 24`
- changed `config.total_timesteps: 98_304_000 -> 4_915_200_000`, matching
  `4096 envs * 24 steps * 50,000 updates`
- added variable-width policy MLP support and set
  `policy.hidden_dims: [512, 256, 128]`
- set `policy.activation: elu`
- set `policy.obs_stack_size: 5`
- disabled policy observation terms that Unitree does not expose to the actor:
  `base_lin_vel`, `base_height`, and `gait_phase`

This intentionally keeps the local default config small; the Unitree-scale
sample budget lives in `experiments/train/locomotion.yaml`.

## 2026-07-09 - Unitree substep layout

Changed the global MuJoCo simulation timestep from `0.002s` to `0.005s` and the
default training decimation from `10` to `4`. This matches Unitree's
`dt=0.005, decimation=4` substep layout while preserving the same `0.02s`
policy control timestep, so existing reward dt-scaling and 24-step PPO rollouts
still represent `0.48s` of simulated control time.

## 2026-07-09 - KL spike diagnosis

Latest run inspected: `qy115pyo`, `locomotion_unitree_aligned_seed0`.

Observed failure pattern:

- `approx_kl` intermittently jumped from normal values near `0.03` to very large
  values, including `> 1e6`.
- `learning_rate` had already reached the adaptive floor of `1e-5`, so this was
  not recoverable by the next-update LR scheduler alone.
- `action_mean_abs_max` grew into the hundreds while `action_abs_max` stayed at
  the action clip of `10`.
- `ctrl_saturation_fraction` stayed around `0.30`, meaning a large fraction of
  controls were running at actuator limits.

Interpretation:

The actor mean escaped far outside the clipped action range. PPO was also
storing the clipped action in the transition and computing Gaussian log-prob
ratios against that clipped boundary action. Once actor means became huge,
small changes in mean caused extreme log-ratio/KL values around saturated
actions.

Applied local fixes:

- PPO now stores the raw Gaussian sampled action and log-prob for PPO ratios.
- The environment receives a clipped copy of that action.
- Control saturation diagnostics are computed from the clipped env action, while
  raw action magnitude remains visible through `action_abs_max`.
- Added `env_action_abs_max` to show the clipped action magnitude.
- Added same-update `target_kl` stopping with default `0.02`. When a minibatch
  has `approx_kl > target_kl`, remaining minibatches in that PPO update are
  skipped. New metrics:
  - `target_kl_exceeded`
  - `ppo_update_fraction`

Recommendation: stop old run `qy115pyo`; it is already in the escaped-mean
regime and should not be used as evidence for the patched PPO path.

## 2026-07-09 - Done threshold alignment

Compared Bello locomotion termination against Unitree RL Lab G1 29-dof
locomotion:

- Unitree G1 uses `episode_length_s: 20.0`; Bello already matches with
  `1000` control steps at `0.02s`.
- Unitree G1 uses `root_height_below_minimum(minimum_height=0.2)`.
- Unitree G1 uses `bad_orientation(limit_angle=0.8)`.
- Unitree G1 does not define a separate locomotion termination reward term.

Changed `experiments/train/locomotion.yaml`:

- `done_config.base_height.min_height: 0.55 -> 0.20`
- `done_config.base_height.link: bello_root`
- replaced the old independent `roll_pitch` done term with
  `bad_orientation.limit_angle: 0.80`
- `done_config.bad_orientation.link: bello_root`

Rationale: the live run was mostly ending through `done/base_height`, while the
height threshold was far stricter than Unitree G1 and the orientation threshold
was looser. The new setting should terminate large tilts earlier but avoid
prematurely ending recoverable low-height states.

Implementation fix:

- Unitree G1 imports `bad_orientation` from IsaacLab. The actual implementation
  terminates on the tilt angle from projected gravity:
  `acos(-projected_gravity_b.z) > limit_angle`.
- Bello now computes the same projected-gravity tilt angle for
  `done/bad_orientation` instead of checking independent roll and pitch
  thresholds.
- Unitree `root_height_below_minimum` checks the articulation root world
  position z. Bello locomotion now sets both done links to the synthetic
  `bello_root` freejoint body, so the root-height and root-orientation done
  conditions do not inherit `env_config.root_link: torso_link`.

## 2026-07-09 - Pipeline parity fixes from deep audit

Resolved high/medium implementation discrepancies found during the Unitree/RSL
audit:

- Generated MuJoCo position actuators now get explicit `forcelimited=true` and
  `forcerange`. By default the range is copied from the compiled URDF joint
  `actuatorfrcrange`; per-joint `effort_limit` can override it when hardware
  runtime limits differ from URDF. This keeps `data.actuator_force`, effort
  observations, and effort rewards consistent with applied joint actuation.
- PPO now normalizes advantages once across the full rollout batch before
  shuffling/minibatching, matching the rsl_rl default.
- Checkpoint metadata now stores value-normalization mean/variance/count so
  normalized-value training state is not silently lost.
- Single-env eval/video rollouts now reset the env and observation stack at done
  boundaries. `stop_on_done=false` continues recording into the next episode
  instead of continuing from a terminal state.
- Pure MuJoCo video observations now apply `ObservationConfig.scale` and rotate
  synthetic-root angular velocity into the root body frame, matching
  `BelloLocomotionEnv.observation`.
- Reset randomization is explicit under `env_config.spawn`. Current locomotion
  keeps randomized root XY/yaw, but uses zero joint position/velocity reset
  noise and no initial step-count staggering to match the Unitree G1 reset
  semantics more closely.
- `alive` reward is zero on terminal failure transitions while timeouts still
  receive alive reward. `termination` remains terminal-only.
- `joint_limit` now uses actuator control ranges, so configured soft limits are
  actually the penalized limits.
- `torque` reward now sums actuator-force squared terms rather than averaging.
- `undesired_contact` now applies a normal-force threshold and counts unique
  non-foot bodies instead of raw contact slots.

Locomotion config changes from this audit:

- `env_config.spawn.root_xy_range: [-0.5, 0.5]`
- `env_config.spawn.root_yaw_range: [-3.14, 3.14]`
- `env_config.spawn.joint_pos_noise: [0.0, 0.0]`
- `env_config.spawn.joint_vel_range: [0.0, 0.0]`
- `env_config.spawn.stagger_initial_step_count: false`
- `env_config.contact.foot_contact_on_seconds: 0.0`
- `env_config.contact.foot_contact_off_seconds: 0.0`
- `env_config.contact.foot_contact_force_threshold: 1.0`
- `env_config.contact.undesired_contact_force_threshold: 1.0`

## 2026-07-10 - Foot-contact reward signal alignment

Foot-contact-dependent rewards were reviewed as a likely locomotion blocker.
The main issue was not the reward weights alone; the contact labels feeding
`foot_slip`, `gait_contact`, and future air-time terms could be stale or too
sensitive:

- foot contact was accepted from any active terrain contact slot, including tiny
  solver/grazing contacts
- locomotion then applied a 40 ms on/off debounce, delaying contact flips by two
  control steps
- reset and done memory assumed both feet were in contact, even when the actual
  thresholded contact state was different

Aligned the implementation closer to Unitree/IsaacLab contact semantics:

- added `env_config.contact.foot_contact_force_threshold`, default `1.0`
- foot contact now requires terrain contact normal force above that threshold
- locomotion and standing recipes set foot-contact debounce to `0.0`
- reset initializes `last_foot_contact` as no contact instead of hard-coding
  both feet in contact; the first physics step then fills contact memory from
  thresholded terrain contact
- done memory keeps the current thresholded contact instead of resetting to
  synthetic both-feet contact

## 2026-07-09 - Locomotion run `u7zvr5v1` diagnosis

Latest active W&B run inspected:

- Run: `u7zvr5v1`, `locomotion_unitree_aligned_seed0`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/u7zvr5v1`
- Config: `num_envs: 4096`, `num_steps: 24`, `max_episode_steps: 1000`
- Command curriculum was still at the initial range `[-0.1, 0.1]` for x/y/yaw.

Observed from sampled history:

- Mean episode length improved early, peaking around `394` steps in the sampled
  rows, then fell back toward `~280` steps.
- Failures were almost entirely `done/bad_orientation`; `done/base_height` was
  effectively zero after the Unitree-style height threshold change.
- `action_mean_rms` grew from near zero to `>5`, and
  `ctrl_saturation_fraction` rose to roughly `0.36-0.38`, so the policy was
  driving many controls into actuator/soft-limit saturation.
- The default video command was `[0.25, 0.0, 0.0]`, outside the still-unexpanded
  training command range, so early videos were evaluating an out-of-distribution
  command.

Root-cause fixes applied:

- `reset()` now calls `mjx.kinematics` after setting qpos/qvel/ctrl and reset
  randomization. Before this, named-body position/orientation terms such as
  `torso_link` height were stale at reset; with `root_link: torso_link`, the
  initial observation stack after every reset contained invalid body-frame
  values until the first step. This intentionally avoids full `mjx.forward`,
  which would allocate contact/solver buffers on every batched auto-reset.
- `reward_config.base_height` was corrected to track `torso_link` with
  `target: 1.03`. The active configs use `root_link: torso_link`, and after the
  reset-forward fix the nominal torso reset height is valid immediately. The
  earlier `0.86m` target was a root/pelvis height, so using it with torso
  tracking rewarded lowering the torso by roughly `17cm`.
- The default locomotion training video command was set to `[0.1, 0.0, 0.0]`,
  which was inside the then-current initial curriculum range. This was later
  changed again during the command-pressure pass below.

Verification:

- Dry-runs for `standing` and `locomotion` resolve after the config changes.
- A reset probe now reports `torso_link` height around `1.03m` immediately after
  reset, and torso base-height reward is near zero at the nominal pose.

## 2026-07-09 - NaN after reset-kinematics fix

Run `bjgt2nlu` produced real numerical NaNs, not just placeholder NaNs from
`max_episode_return`. The first bad row was at env step `294912`, the third PPO
update with `4096 * 24` samples per update.

Observed at the first bad row:

- `train/mean_done/nonfinite` was approximately `1 / 98304`, so exactly one
  rollout sample hit the nonfinite done condition.
- `train/mean_reward/base_height`, `train/mean_reward/joint_acc`, and
  `train/mean_reward/energy` became NaN on that transition.
- `train/mean_reward`, `train/loss`, `train/approx_kl`, and value-normalization
  state became NaN afterward.

Diagnosis:

- The environment detected nonfinite qpos/qvel and marked the transition
  terminal, but still computed reward terms from the nonfinite `next_state`.
- That single NaN reward poisoned GAE targets and value normalization. Later
  simulator states could be finite again, but the learner state was already bad.

Fix:

- Nonfinite terminal transitions now keep only the finite termination reward and
  zero all other reward components before episode-return accounting.
- `done/nonfinite` now checks qpos, qvel, named-body position/orientation,
  body-frame velocities, and actuator forces because reward terms read all of
  those fields.
- PPO now uses the reset observation, not the terminal pre-reset observation,
  for terminal bootstrap policy calls. Timeout bootstraps still use the
  pre-reset timeout observation.
- Added a regression test that injects a NaN into one vectorized env and checks
  `done/nonfinite`, auto-reset, finite reward components, and the expected
  termination reward.

## 2026-07-09 - Remove hard target-KL stop from locomotion

Locomotion now sets `config.target_kl: null` while keeping
`config.schedule: adaptive` and `config.desired_kl: 0.01` from the default PPO
config.

Rationale:

- The Unitree/rsl_rl-style config uses adaptive KL learning-rate control through
  `desired_kl`, not a same-update hard early stop.
- Bello's `target_kl` guard skips remaining minibatch updates when a minibatch
  exceeds the threshold. That is useful for instability debugging, but can
  throttle humanoid locomotion learning when exploration is high.
- After this change, watch `train/approx_kl` and `train/learning_rate` directly.
  `train/ppo_update_fraction` should stay at `1.0` because the hard early-stop
  path is disabled.

## 2026-07-09 - G1 gait phase alignment

Correction to earlier speculation: Unitree G1 29-dof velocity training does use
a phase-based `feet_gait` reward while leaving `gait_phase` commented out in
both policy and critic observations. Therefore Bello should not disable
`gait_contact` just because `env_config.observation.gait_phase=false`.

Actual mismatch found:

- Unitree computes gait phase from `episode_length_buf * step_dt`, with ordinary
  reset episode length behavior.
- Bello locomotion had `spawn.stagger_initial_step_count=true`, which randomized
  the hidden gait phase at reset. That extra randomization was not in the
  Unitree G1 recipe and made the hidden gait schedule less consistent.

Aligned locomotion reset settings with Unitree G1:

- `spawn.stagger_initial_step_count: false`
- `spawn.joint_pos_noise: [0.0, 0.0]`, matching Unitree
  `reset_joints_by_scale(position_range=(1.0, 1.0))`
- `spawn.joint_vel_range: [0.0, 0.0]`. Unitree G1 passes
  `reset_joints_by_scale(velocity_range=(-1.0, 1.0))`, but its default joint
  velocity is zero, so scaling still resets joint velocity to zero.

Kept:

- `reward_config.gait_contact.enabled: true`
- `env_config.observation.gait_phase: false`

## 2026-07-09 - Disable reset joint-velocity randomization

Turned off locomotion reset joint-velocity randomization by setting:

```yaml
env_config.spawn.joint_vel_range: [0.0, 0.0]
```

Rationale: Bello's reset code samples this range as an absolute joint velocity.
Unitree's `reset_joints_by_scale(velocity_range=(-1.0, 1.0))` scales default
joint velocity, and the G1 default is zero, so Unitree effectively resets joint
velocity to zero.

## 2026-07-09 - Unitree-style privileged critic observation

Implemented separate actor and critic observation streams in the config system.
If `env_config.critic_observation` is absent, PPO falls back to the actor
observation for backward compatibility. Locomotion now configures the critic to
match Unitree G1 29-dof:

- actor/policy observation: noisy `base_ang_vel`, `projected_gravity`,
  velocity command, relative joint position, scaled relative joint velocity,
  and last action; history length `5`
- critic observation: clean `base_lin_vel`, scaled `base_ang_vel`,
  `projected_gravity`, velocity command, relative joint position, scaled
  relative joint velocity, and last action; history length `5`
- no critic noise/corruption
- no gait phase in actor or critic observation

Dry-run after the change:

- actor raw observation size: `48`
- actor stacked observation size: `240`
- critic raw observation size: `51`
- critic stacked observation size: `255`

Checkpoint metadata now stores `critic_obs_size` so evaluation restores the
critic network with the correct input width and validates runtime compatibility.

## 2026-07-10 - Forward command pressure pass

Changed only the command distribution and diagnostic video command; reward
weights and reward implementations were left unchanged.

Superseded note: this pass still assumed X-forward locomotion. The later
"Bello locomotion forward axis" note corrects the forward command to `lin_vel_y`.

Config changes:

```yaml
command_config.lin_vel_x: [0.15, 0.35]
video_config.command: [0.3, 0.0, 0.0]
checkpoint_config.name: locomotion_command_pressure_seed0
wandb_config.run_name: locomotion_command_pressure_seed0
```

Rationale:

- The previous initial X command range `[-0.1, 0.1]` made standing or near-zero
  velocity a high-scoring behavior under the exponential tracking reward.
- `gait_contact.command_speed_threshold` is `0.10`, and the previous video
  command was exactly `[0.1, 0.0, 0.0]`, so strict `>` gating did not activate
  gait-contact pressure in the diagnostic video.
- Starting X commands at `0.15` ensures nearly all non-standing sampled commands
  activate gait contact while keeping the curriculum, Unitree-style foot
  rewards, PPO settings, observation layout, and random pushes unchanged.

## 2026-07-10 - Step-pressure reward pass

The forward-command pressure run did not visibly force stepping, so this pass
raises the existing velocity-tracking and gait-contact signals without changing
the reward implementations:

```yaml
reward_config.track_lin_vel_xy.weight: 3.0
reward_config.gait_contact.weight: 2.0
checkpoint_config.name: locomotion_step_pressure_seed0
wandb_config.run_name: locomotion_step_pressure_seed0
```

Rationale:

- At the previous weights, planted double support still received one correct
  gait-contact foot per phase, so the difference between double support and a
  real alternating contact pattern was only about `0.01` reward per control
  step.
- With `gait_contact.weight: 2.0`, that gap is about `0.04` per control step,
  making persistent double support much less competitive.
- `track_lin_vel_xy.weight: 3.0` makes failing to move under the
  `[0.15, 0.35]` forward command range more expensive while keeping the reward
  shape unchanged.
- `foot_clearance.weight` was intentionally left at `1.0`: the Unitree-style
  clearance term gives stationary feet the maximal raw score, so increasing it
  would strengthen the planted-foot loophole rather than force stepping.
- `feet_air_time` remains disabled for now. Enabling it is the next stronger
  intervention if gait pressure alone still does not produce swing/landing
  behavior, but it is more likely to create hopping artifacts.

## 2026-07-10 - Exploration-pressure pass

The step-pressure run still did not visibly produce enough stepping behavior, so
this pass increases PPO exploration while keeping the command and reward changes
from the previous passes:

```yaml
config.ent_coef: 0.03
config.policy.init_log_std: 0.5
config.policy.max_log_std: 1.5
checkpoint_config.name: locomotion_step_entropy_seed0
wandb_config.run_name: locomotion_step_entropy_seed0
```

Rationale:

- `ent_coef: 0.03` is 3x the Unitree/RSL-style `0.01`, encouraging the policy
  to keep broader action distributions while searching for stepping behavior.
- `init_log_std: 0.5` starts action standard deviation at about `1.65` instead
  of `1.0`.
- `max_log_std: 1.5` raises the cap to about `4.48`, so the entropy term is not
  immediately clipped by the old `max_log_std: 1.0` ceiling.

Watch these metrics closely:

- `train/action_std_mean`
- `train/ctrl_saturation_fraction`
- `train/approx_kl`
- `train/mean_done/bad_orientation`
- `train/mean_foot_contact/raw`

If contact remains near constant double support, this exploration change did
not reach the stepping behavior. If saturation and bad-orientation failures
spike, the entropy pressure is too high and is producing thrashing rather than
useful stepping exploration.

## 2026-07-10 - Back off unstable entropy/std

Run `34jcuwvt` (`locomotion_step_entropy_seed0`) failed at PPO update `199` with
`nonfinite_update`. The last finite W&B summary row was update `198`.

Relevant last finite metrics:

- `train/action_std_mean: 2.57`
- `train/action_std_max: 3.59`
- `train/action_abs_max: 19.5`
- `train/env_action_abs_max: 10.0`
- `train/ctrl_saturation_fraction: 0.385`
- `train/mean_done/bad_orientation: 0.033`
- `train/mean_reward/joint_vel: -7700`
- `train/mean_reward/joint_acc: -4813`
- `train/value_loss_raw: 7.8e17`

Diagnosis: the `init_log_std: 0.5`, `max_log_std: 1.5`, `ent_coef: 0.03`
combination produced broad, saturated exploration. The robot was mostly
thrashing into velocity/acceleration penalties and bad-orientation resets rather
than discovering controlled steps.

Backed off to a safer setting:

```yaml
config.ent_coef: 0.02
config.policy.init_log_std: 0.0
config.policy.max_log_std: 1.0
checkpoint_config.name: locomotion_step_entropy_safe_seed0
wandb_config.run_name: locomotion_step_entropy_safe_seed0
```

Keep the command-pressure and gait-pressure changes, but do not use high
Gaussian std as the main stepping mechanism. If this still does not step, the
next intervention should be a directed stepping term such as carefully gated
air-time/swing reward rather than more undirected entropy.

## 2026-07-10 - Count contacted-foot rotation as slip

Extended the existing `foot_slip` reward to penalize contacted-foot angular
motion as slip, not just lateral translation:

```yaml
reward_config.foot_slip.angular_speed_scale: 0.05
```

The raw slip speed is now:

```text
lateral_speed + angular_speed_scale * angular_speed
```

and it is still multiplied by `foot_contact` and the configured negative
`foot_slip.weight`. The `0.05` scale treats foot rotation as an equivalent
tangential slip speed using a 5 cm lever arm. This keeps the term in roughly the
same units as lateral foot speed while making twisting/rolling a contacted foot
undesirable.

## 2026-07-10 - Contact-gated foot clearance

Changed `foot_clearance` so contacted feet no longer receive the free maximum
clearance reward. The term now computes a moving non-contact gate:

```text
swing_gate = (1 - foot_contact) * tanh(tanh_mult * lateral_speed)
```

and returns the weighted average of per-foot target-height scores only across
active swing feet:

```text
exp(-(foot_height - target) ** 2 / std)
```

If all feet are in contact, or no non-contact foot is moving, `foot_clearance`
is zero. This intentionally diverges from Unitree's ungated clearance formula to
remove the planted-foot loophole observed in Bello locomotion runs.

## 2026-07-10 - KL spike and action-std runaway in run 3qpx6oll

Run `gyungmin/bello-mujoco/3qpx6oll` failed at update 594. Local W&B history
shows the first major break at update 476 (`46,792,704` env steps), not exactly
45M:

- `train/approx_kl` spiked to `0.171`
- `train/value_loss_raw` jumped to `6.37e13`
- `train/mean_reward` dropped to about `-1.8e4`
- the drop was dominated by `joint_vel` and `joint_acc`

After that, the adaptive LR reduced from `4.44e-4` to `2.96e-4`, but the bad
update had already been committed because this recipe had `target_kl: null`.
The value normalizer was also contaminated by a huge finite target. Later
outliers at updates 560 and 573 pushed `value_norm_mean` to about `-1e8` and
`value_norm_std` to about `9e10`; the run then failed with a nonfinite PPO
update at 594.

Two concrete implementation/config problems were found:

- PPO rollout action handling was unstable around clipping. An earlier
  mitigation made stored actions, old log-probs, and env actions all agree at
  the clipped boundary, but later locomotion-hard runs showed that this lets a
  runaway Gaussian mean produce huge clipped-boundary log-ratio/KL values.
  Current PPO rollout stores the raw Gaussian action/log-prob for policy ratios
  and sends a clipped copy to the env.
- `locomotion.yaml` overrode every actuator entry without `effort_limit`, so
  W&B showed `effort_limit: null` and generated MJCF omitted `forcerange`.
  Effort limits are now explicit in the recipe; generated XML again includes
  Robstride peak-torque force ranges such as hip pitch/knee
  `forcerange="-120 120"`, hip roll/yaw `forcerange="-60 60"`, ankle
  `forcerange="-36 36"`, and waist yaw `forcerange="-120 120"`.

Stability config changes:

```yaml
config.target_kl: 0.02
config.ent_coef: 0.01
warp.njmax: 512
```

The `njmax` increase addresses the observed `nefc overflow` messages requesting
up to `321`. The entropy change returns to the Unitree-style coefficient; with
the separated raw/env action path, `env_action_abs_max` should stay bounded by
the env action clip, while raw `action_abs_max` and `action_mean_abs_max` remain
useful warnings for action-distribution drift.

## 2026-07-10 - Easy fixed-forward locomotion diagnostic

Added `experiments/train/locomotion_easy.yaml` as a lower-variance walking
debug recipe:

```bash
uv run bello-train-ppo --config-dir "$PWD/experiments/train" --config-name locomotion_easy
```

Purpose: test whether the current PPO/env/reward pipeline can learn one simple
forward command before reintroducing the harder Unitree-aligned ingredients.

Main differences from `locomotion.yaml`:

- fixed command: `lin_vel_x: [0.0, 0.0]`, `lin_vel_y: [0.50, 0.50]`,
  `ang_vel_yaw: [0.0, 0.0]`
- no command curriculum
- no random pushes
- no observation noise
- no randomized spawn position/yaw/joint state
- no `gait_contact`, so there is no hidden periodic contact schedule
- forward velocity tracking is stronger, while uprightness/control penalties
  are lighter
- small `feet_air_time` and contact-gated `foot_clearance` rewards are enabled
  to give a direct stepping incentive without phase scheduling
- actuator limits are inherited from `locomotion.yaml`, including lower-body
  Robstride peak-torque limits and waist `120 Nm` limits

This is not intended as the final locomotion recipe. If it cannot learn a
fixed `0.5 m/s` forward command, the remaining issue is likely in the pipeline
or robot dynamics rather than curriculum design.

## 2026-07-10 - Bello locomotion forward axis

The fixed-forward diagnostic showed the torso moving coherently but visually
sideways. The asset convention explains this: Bello is anatomically Y-forward,
while the earlier recipes put forward pressure on `lin_vel_x`. Because
`track_lin_vel_xy` compares commands to yaw-frame body XY velocity, `lin_vel_x`
was rewarding local lateral motion.

Changed `locomotion.yaml` and `locomotion_easy.yaml` so forward walking pressure
is on `lin_vel_y`; `lin_vel_x` remains the lateral axis for this asset. The
training/evaluation command vectors were changed accordingly, for example the
easy diagnostic video command is now `[0.0, 0.5, 0.0]`.

## 2026-07-10 - CPU videos can sample stochastic policy actions

Run `gyungmin/bello-mujoco/hoc8i3bu` reported timeout-only training episodes:

```text
train/mean_episode_length: 1000
train/mean_done/timeout: 0.0009969
train/mean_done/terminal: 0
train/mean_done/base_height: 0
train/mean_done/bad_orientation: 0
```

But the W&B rollout videos showed falls around the `2s` mark. Those videos were
CPU MuJoCo rollouts using deterministic actor-mean actions:

```yaml
video.simulation_backend: mujoco
video.stochastic_actions: false
video.stop_on_done: false
```

Training rollouts use stochastic actions in MJX/Warp. We intentionally keep
training videos on CPU MuJoCo because the GPU is already occupied by training,
but deterministic actor-mean CPU rollouts can fail even when stochastic training
rollouts are surviving.

Added `video_config.stochastic_actions` and `video_config.action_seed`. When
enabled, CPU video rollouts sample from the actor Gaussian using the learned
`log_std`, clip the sampled action to the env action range, and then step CPU
MuJoCo. The locomotion recipes now keep `simulation_backend: mujoco` and
`stop_on_done: false`, but set `stochastic_actions: true` so videos better
match the action distribution used during PPO rollout collection.

Follow-up: the stochastic CPU videos can still fall around `2-3s` while the
batched MJX training metrics report timeout-only episodes. This is still a
valid backend/rollout-distribution discrepancy. Video captions include the
single-render rollout's `first_done_step`, `first_done_seconds`, completed
episode count, and done-cause counts; scalar `train/mean_*` metrics still
describe the batched PPO training rollout.

## 2026-07-10 - Command curriculum diagnostics

Removed scalar W&B logging for video rollout diagnostics so single-render stats
do not look like batched PPO metrics. Added `train/command_curriculum/*`
diagnostics for curriculum state: enabled flag, returned episode count,
episode and normalized tracking reward, success threshold, period boundary, and
range expansion decisions.

## 2026-07-10 - Reintroduce modest gait-contact reward

`gait_contact` rewards each foot whose contact state matches the deterministic
two-foot schedule. The term is command-speed gated and dt-scaled, but its raw
value is a match count in `[0, 2]`, so large weights can dominate velocity
tracking. Earlier locomotion experiments also had a hidden-phase issue:
`gait_contact.enabled: true` while actor and critic observations had
`gait_phase: false`.

Changed `locomotion.yaml` and `locomotion_easy.yaml` to expose gait phase to
both actor and critic observations whenever gait contact is active. The
introduced gait-contact weight is intentionally modest:

```yaml
env_config.observation.gait_phase: true
env_config.critic_observation.gait_phase: true
reward_config.gait_contact:
  enabled: true
  weight: 0.5
  command_speed_threshold: 0.10
```

At the default `0.02s` control timestep, a perfect two-foot schedule match
contributes at most `0.5 * 2 * 0.02 = 0.02` reward per step. This gives the
policy some stepping-phase pressure without overpowering the fixed-forward
velocity reward.

## 2026-07-10 - Track locomotion velocity at the pelvis

The sideways-walking failure was better framed as a measurement-body issue than
as a waist posture reward. The torso can face forward while the lower body is
effectively moving sideways, so constraining `waist_yaw_joint` would add another
posture term without fixing what the velocity objective measures.

Changed `track_lin_vel_xy.link` and `track_ang_vel_z.link` to `bello_root` in
`locomotion.yaml` and `locomotion_easy.yaml`. In the generated MJCF, the URDF
`pelvis_link` is folded into the floating root body named `bello_root`, while
`pelvis_link` remains a mesh/geom name. Torso height/uprightness remains on
`torso_link`, but commanded robot velocity is now evaluated from the pelvis/root
body.

## 2026-07-10 - Add command-conditioned locomotion hard recipe

Added `experiments/train/locomotion_hard.yaml` as a next-step recipe after the
fixed-command diagnostic. It inherits the stable `locomotion_easy` structure,
including box collision and policy history, but reintroduces command
conditioning:

```yaml
command_config:
  lin_vel_x: [-0.05, 0.05]
  lin_vel_y: [0.25, 0.55]
  ang_vel_yaw: [-0.05, 0.05]
  standing_probability: 0.03
  resample_seconds: 6.0
  curriculum:
    enabled: true
    limit_lin_vel_x: [-0.25, 0.25]
    limit_lin_vel_y: [0.10, 0.85]
    limit_ang_vel_yaw: [-0.25, 0.25]
    expand_ang_vel_yaw: true
    increment: 0.05
    success_threshold: 0.75
```

The curriculum expands X/Y/yaw command ranges only after completed episodes
clear the existing velocity-tracking success test. Push events are enabled but
kept mild at `8s` intervals with root XY velocity sampled from `[-0.20, 0.20]`.
The recipe also enables modest reset randomization, modest actor observation
noise, a small torque penalty, `termination=-1.0`, and `target_kl=0.02` so PPO
does not accept large same-update KL jumps unchecked.

## 2026-07-10 - Stand-still command samples and self-collision penalty

Command-conditioned locomotion needs explicit zero-command samples, not only
small nonzero velocities from the uniform command ranges. Updated default,
`locomotion.yaml`, and `locomotion_hard.yaml` command sampling to use:

```yaml
command_config.standing_probability: 0.10
```

This makes roughly 10% of command resamples exact `[0, 0, 0]` stand-still
commands. The fixed-forward `locomotion_easy.yaml` diagnostic intentionally
keeps `standing_probability: 0.0` because it is not a mixed-command recipe.

Added `foot_clearance.command_speed_threshold: 0.05` to active locomotion
recipes. Below that command speed, foot clearance contributes zero reward, so a
standing command does not ask the policy to lift or swing its feet.

Added `reward/self_collision`, backed by
`env_config.contact.self_collision_force_threshold: 1.0`. The term counts
active robot-robot contact slots above the force threshold and applies a small
timestep-scaled penalty:

```yaml
reward_config.self_collision:
  enabled: true
  weight: -1.0
```

Use this as a broad guardrail against leg/arm/body interpenetration during
aggressive exploration. It is separate from `undesired_contact`, which only
counts non-foot robot contacts against terrain.

## 2026-07-10 - Add locomotion termination penalty

Enabled a meaningful event-driven termination penalty in the locomotion recipes:

```yaml
reward_config.termination:
  enabled: true
  weight: -2.0
```

The term is not timestep-scaled and is only applied on terminal failures such as
bad height/orientation/nonfinite termination. Episode-length timeouts remain
truncations and do not receive this penalty. The `-2.0` value follows the
earlier Unitree-style scaling rationale: a reference `termination=-100` under
dt-scaled rewards at a `0.02s` control step is effectively about `-2`.

## 2026-07-10 - Command-speed gait frequency

Changed the deterministic gait schedule from a fixed `0.80s` cycle to a
command-speed-dependent frequency:

```text
fstep(v) = clip(1.25 + 0.55 v, 1.2, 2.2)
```

where `v = norm(command[:2])`, the commanded linear speed. The config
representation keeps `cycle_time: 0.80` as the zero-speed baseline and adds:

```yaml
env_config.gait:
  command_speed_frequency_gain: 0.55
  min_frequency: 1.20
  max_frequency: 2.20
```

This affects both the gait-contact reward mask and the `gait_phase` observation.
The CPU MuJoCo render/eval observation path uses the same formula so rollout
videos do not feed a different phase signal than MJX training.

## 2026-07-10 - Remove target-limit action penalty

Removed the `target_limit` reward term. It penalized the requested residual
position target before MuJoCo clipped it to actuator `ctrlrange`, which made the
policy pay for asking past the target range even when the physical joint had not
violated its range.

What remains:

- `joint_limit` still penalizes actual joint-position violation of actuator
  control ranges.
- MuJoCo position actuators still have generated `ctrlrange` values for action
  scaling, diagnostics, and soft-limit rewards.
- Actuators with effort limits still have `forcelimited="true"` and generated
  force ranges.

So removing `target_limit` removes the action-target loss, but not the physical
force bounds.

## 2026-07-10 - Disable actuator control target clipping

Changed generated MuJoCo position actuators from `ctrllimited="true"` to
`ctrllimited="false"` and removed the Python-side clipping from
`_action_to_ctrl()` / `mujoco_action_to_ctrl()`. The policy can now request
position targets beyond the generated actuator `ctrlrange`.

The generated `ctrlrange` is still retained as soft-limit metadata for action
scale fallback, diagnostics, and `joint_limit` reward references. Motor force
caps remain active through generated `forcelimited="true"` / `forcerange`
attributes, so oversized targets can drive actuators to their torque limits but
not produce unbounded torque.

Also bumped the prepared-asset cache version so existing cached MJCFs with
`ctrllimited="true"` are not reused silently.

## 2026-07-10 - Increase actual joint-limit penalty after unclipping targets

After allowing actuator targets to exceed generated `ctrlrange`, increased the
actual joint-position limit penalty by 10x in locomotion recipes:

```yaml
locomotion.yaml:
  reward_config.joint_limit.weight: -50.0

locomotion_easy.yaml / locomotion_hard.yaml:
  reward_config.joint_limit.weight: -20.0
```

This keeps the action-target penalty removed, but makes physical soft-limit
violations much more expensive once the joint itself crosses the configured
range.

## 2026-07-10 - Replace locomotion joint-limit pressure with invalid-action reward

Checkpoint rollout diagnostics showed `joint_limit` had become the dominant
penalty after actual joints were already outside the soft actuator ranges. The
locomotion recipes now disable the actual joint-position `joint_limit` term and
enable `invalid_action` instead:

```yaml
reward_config:
  joint_limit:
    enabled: false
  invalid_action:
    enabled: true
    weight: -100.0
```

`invalid_action` penalizes the requested policy position target outside the
generated actuator `ctrlrange` before physics turns that request into a joint
position violation. It is still timestep-scaled like other continuous reward
terms.

## 2026-07-10 - Remove low-side knee soft-limit margin

Set both locomotion knee lower soft limits from `0.10472` rad to `0.0` rad, so
the soft range no longer adds a 6 degree extension-side buffer above the hard
knee limit. The high-side soft limit remains `1.98968` rad, keeping the 6
degree flexion-side buffer.

## 2026-07-11 - Make invalid-action pressure normalized and quadratic

Changed `invalid_action` from a linear radian-space violation to a normalized
squared violation. This originally used the symmetric action scale:

```text
target = default_ctrl + policy_action_scale * action
violation = max(soft_low - target, 0) + max(target - soft_high, 0)
invalid_action = sum((violation / policy_action_scale)^2)
```

This makes the penalty depend on how far the requested target is outside the
soft range in action units, so small-scale joints such as knees and ankles are
not under-penalized relative to hips. Because the raw magnitude is now larger
for severe out-of-range targets, the active locomotion recipe weights were reset
from `-100.0` to `-1.0`. The later asymmetric action mapper was removed, so
`invalid_action` again uses this same linear action-scale target formula.

## 2026-07-11 - Separate PPO raw actions from clipped env actions

Run `y999e9iv` showed the policy was best around update `7308`, then collapsed
after the actor mean drifted into hundreds of action units. The env only saw
clipped actions, but PPO had been storing/logging clipped actions and computing
Gaussian log-prob ratios at the clipped boundary. That made KL unstable once the
mean escaped.

Rollout collection now uses two action tensors:

```text
raw_action = mean + std * noise
log_prob = gaussian_log_prob(raw_action)
env_action = clip(raw_action, -env.action_clip, env.action_clip)
```

`Transition.action` is the raw Gaussian action used by PPO loss. The new
`Transition.env_action` is the clipped action passed to `step_batch` and used
for control saturation diagnostics. `action_abs_max` now reports the raw action
magnitude, while `env_action_abs_max` reports the clipped environment action.

## 2026-07-13 - Normalize Bello policy action range

Changed Bello `ACTION_CLIP` from `10.0` to `2.0` and updated standing /
locomotion policy joint `action_scale` values so a two-unit action can cover the
configured soft-limit range:

```text
clipped_action in [-2, 2]
negative_scale = min(action_scale, (default_ctrl - soft_low) / 2)
positive_scale = min(action_scale, (soft_high - default_ctrl) / 2)
target = default_ctrl + (positive_scale if action >= 0 else negative_scale) * action
```

This preserves `action=0` as the configured default pose while keeping clipped
policy targets inside the configured soft range even when the default pose is
off-center.

The first hard run after this change, `bozou38g`, failed during update 1 with
`nefc overflow` and nonfinite PPO metrics. That was expected in hindsight:
`init_log_std: 0.0` sampled unit-std normalized actions in a much larger
physical target space. Reduced locomotion startup exploration to
`init_log_std: -1.5`, capped `max_log_std` at `-0.5`, lowered locomotion
learning rate to `1e-4`, and lowered hard locomotion entropy pressure to
`ent_coef: 0.003`. A one-update GPU smoke run with these settings completed
without nonfinite metrics, with `approx_kl ~= 0.006` and
`target_kl_exceeded = 0`.

## 2026-07-11 - Make orientation, yaw tracking, and torque more relevant in hard locomotion

Checkpoint `locomotion_hard_seed0` at update `12207` fulfilled fixed forward
walking well, but the reward breakdown showed several stabilizing terms were
too small to materially shape the policy:

```text
fixed GPU eval, command (0, 0.5, 0):
  track_lin_vel_xy:  99.6% of max
  track_ang_vel_z:   86.2% of max
  gait_contact:      95.3% of max
  foot_clearance:    96.1% of max
  flat_orientation: -0.00031 / step
  torque:           -0.00057 / step
```

For `locomotion_hard.yaml`, increased the terms that should matter for command
tracking and smooth upright motion without making them dominate forward
tracking:

```yaml
reward_config:
  track_ang_vel_z.weight: 1.0      # from 0.5
  ang_vel_xy.weight: -0.2          # from -0.05
  flat_orientation.weight: -10.0   # from -2.0
  torque.weight: -0.00002          # from -0.000005
```

At the same rollout state, this would roughly make yaw tracking worth up to
`+0.02/step`, torso roll/pitch angular-velocity penalty about 4x larger,
flat-orientation penalty about 5x larger, and torque penalty about 4x larger.

Follow-up: torque was increased another 2x in `locomotion_hard.yaml`:

```yaml
reward_config:
  torque.weight: -0.00004
```

## 2026-07-12 - Shorten hard locomotion run budget

Reduced `experiments/train/locomotion_hard.yaml` from `2,400,000,000` to
`600,000,000` requested env steps. With `num_envs: 2048` and `num_steps: 24`,
this composes to `12,207` PPO updates, or `599,998,464` actual env steps after
integer update rounding.

## 2026-07-12 - Add standing foot penalties and broaden hard commands

Added two near-zero-command standing penalties:

```yaml
reward_config:
  standing_foot_vel:
    enabled: true
    weight: -2.0
    command_speed_threshold: 0.08
  standing_foot_air_time:
    enabled: true
    weight: -3.0
    command_speed_threshold: 0.08
```

`standing_foot_vel` penalizes horizontal foot speed while standing.
`standing_foot_air_time` penalizes accumulated foot air time while standing.
Both are timestep-scaled continuous penalties and are gated off for non-standing
commands.

To reduce forward-only specialization in `locomotion_hard.yaml`, widened the
initial command distribution and made the curriculum less dependent on a long
run:

```yaml
command_config:
  lin_vel_x: [-0.15, 0.15]
  lin_vel_y: [0.15, 0.60]
  ang_vel_yaw: [-0.20, 0.20]
  standing_probability: 0.20
  curriculum:
    limit_lin_vel_x: [-0.50, 0.50]
    limit_lin_vel_y: [0.00, 1.50]
    limit_ang_vel_yaw: [-1.00, 1.00]
    increment: 0.025
    success_threshold: 0.85
```

Also increased `track_ang_vel_z.weight` from `1.0` to `2.0` so yaw commands are
not dominated by forward velocity tracking.

## 2026-07-13 - Bound normalized actions to soft control ranges

Run `fjkrj9hp` survived the first update with the lower exploration settings but
failed at update 156. The last finite summary was already physically unstable:
`train/mean_reward/ang_vel_xy ~= -4348`, `joint_vel ~= -1579`,
`joint_acc ~= -790`, `mean_episode_length ~= 49`, and `value_norm_std ~= 2e10`.
`env_action_abs_max` was clipped at `2.0`, but `ctrl_margin_min ~= -1.74`,
meaning a clipped normalized action could still request targets well outside
the configured soft control range because default poses are off-center.

Changed Bello action mapping so `0` still means the configured default pose,
`-ACTION_CLIP` maps to the soft lower target, and `+ACTION_CLIP` maps to the
soft upper target using asymmetric per-joint side scales. This keeps policy
targets inside soft ranges even though the generated MJCF keeps
`ctrllimited="false"`.

Verification smoke:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion_hard \
  config.total_timesteps=9830400 \
  wandb_config.enabled=false \
  checkpoint_config.enabled=false \
  video_config.enabled=false
```

Completed 200 PPO updates without `nefc overflow` or nonfinite PPO metrics.
Final diagnostics included `ctrl_margin_min ~= 0.033`,
`ctrl_saturation_fraction = 0`, `nonfinite_update = 0`,
`mean_reward ~= 0.034`, and `value_norm_std ~= 2.22`.

## 2026-07-13 - Restore linear action mapping with actuator ctrl limits

The asymmetric side-scale mapper above kept controls inside soft ranges, but it
made action scale kink at zero because positive and negative action directions
could use different slopes. A no-env-clip variant was tested:

```text
raw_action = mean + std * noise
ctrl_target = default_ctrl + policy_action_scale * raw_action
```

It failed a 200-update disabled-W&B smoke at update 59 with `nefc overflow` and
a nonfinite PPO update, so raw unbounded actions are still not acceptable even
with actuator control limits. A linear mapper with `action_clip=2.0` and
`ctrllimited="true"` still failed at update 83, suggesting repeated hard
requests against actuator limits remain destabilizing.

Kept the simpler linear mapper, set Bello `action_clip=1.0`, and changed
generated position actuators back to `ctrllimited="true"`. PPO still stores raw
Gaussian actions/log-probs, but the environment receives a clipped action copy;
MuJoCo/MJX then clamps requested actuator targets to `ctrlrange`. Force limits
remain active through `forcelimited="true"`. The generated asset cache version
was bumped so old `ctrllimited="false"` MJCFs are not reused silently.

## 2026-07-13 - Align Bello action semantics with reference residual targets

Inspection of Unitree RL Lab / Isaac Lab, legged-gym, and MuJoCo Playground
showed the common locomotion convention is a bounded normalized action mapped as
a residual joint-position target:

```text
bounded_action = clip(raw_action, -1, 1)
ctrl_target = default_ctrl + action_scale * bounded_action
```

Unitree/IsaacLab uses `JointPositionActionCfg(... scale=0.25,
use_default_offset=True)`. legged-gym clips actions in the env, applies
`default_dof_pos + action_scale * action`, then clips torques. MuJoCo
Playground's JAX PPO uses tanh-bounded actions and humanoid envs use
`default_pose + action * action_scale`.

Removed the Bello asymmetric side-scale compatibility path:

```text
negative_scale = min(action_scale, (default_ctrl - soft_low) / action_clip)
positive_scale = min(action_scale, (soft_high - default_ctrl) / action_clip)
```

The env, MuJoCo renderer, diagnostics, and `invalid_action` reward now use the
single clipped affine target formula. Generated MuJoCo position actuators stay
`ctrllimited="false"` and `forcelimited="true"`; target saturation is no longer
used as a safety layer. Lower-body action scales were reduced from full-soft
range spans to conservative residual deltas:

```text
hip pitch/roll/yaw: 0.35 rad
knee:               0.45 rad
ankle pitch:        0.20 rad
ankle roll:         0.06 rad
waist yaw:          0.15 rad
```

## 2026-07-13 - Use Unitree/rsl_rl-style action scale as gain

The previous `action_clip=1.0` setting made the reduced `action_scale` values a
hard target-range cap. That is closer to tanh-normal MuJoCo Playground semantics
than Unitree/IsaacLab/rsl_rl semantics. Unitree's `JointPositionActionCfg`
interprets `scale=0.25` as a gain on the raw Gaussian action; it does not imply
that only `default +/- 0.25` is reachable.

Changed Bello `action_clip` from `1.0` to `5.0` while keeping the single affine
mapper:

```text
env_action = clip(raw_action, -5, 5)
ctrl_target = default_ctrl + policy_action_scale * env_action
```

This keeps a finite safety bound for PPO/MJX stability but lets policy actions
use the configured action scales as gains, not as the full reachable joint
target range.

## 2026-07-13 - Derive default action scales from actuator ranges

Removed the explicit lower-body policy `action_scale` values from the Python
defaults and standing/locomotion recipes. Bello now uses the existing default
rule for those joints:

```text
policy_action_scale = 0.25 * (actuator_ctrl_high - actuator_ctrl_low)
```

The generated actuator `ctrlrange` uses the configured soft limit when present,
so action scales now follow the actual configured target range unless a joint
sets `action_scale` explicitly.

## 2026-07-13 - Replace foot mesh collision with URDF primitives

Moved foot collision from STL-derived meshes/boxes into the source URDFs. Each
`*_ankle_roll_link` originally had four collision primitives: two boxes and two
cylinders. The primitive origins were converted from the supplied coordinates
relative to the previous STL collision origin into link-frame URDF origins. The
left foot mirrors the STL-frame x offsets so the primitives land under the left
ankle instead of inheriting the left STL frame's 180-degree yaw displacement.

The visual STL feet remain unchanged. Simulation contact now uses the primitive
foot shapes consistently across `mesh`, `foot-mesh`, and `box` collision modes.

## 2026-07-13 - Correct foot primitive tilt axis

The supplied foot primitive tilt was initially composed as pitch because I
assumed the foot's forward direction was local `+x`. The foot forward direction
is local `+y`, so the `-15deg` sole-box tilt and `-3deg` toe/cylinder tilt are
roll rotations in the STL-foot frame. Recomputed the URDF primitive `rpy` values
for both full-body and lower-body sources while keeping the primitive centers
and sizes unchanged.

## 2026-07-13 - Add mid-foot primitive box

Added a third foot collision box per side, bringing each ankle-roll foot to five
collision primitives: three boxes and two cylinders. The new box uses the
supplied STL-frame center `(133.825, 24.7345, -866.018) mm`, size
`(106, 41.95, 50) mm`, and the same `-3deg` local roll convention as the toe
box/cylinders.

## 2026-07-14 - Strengthen hard-locomotion gait pressure

Repeated seed-0 launches of the unchanged hard-locomotion recipe produced
different first-rollout contact counts under MJX-Warp, and earlier full runs
split between alternating gait and near-constant double support. Three
matched 520-update baseline repetitions all reached gait, but retained more
foot contact than desired. Tested a single-variable ablation:

```yaml
reward_config:
  gait_contact:
    weight: 2.0  # from 0.5
```

At update 500, the three-run means changed as follows:

```text
                                  baseline       gait_contact=2.0
mean_foot_contact/filtered:       0.6956         0.5938
mean_reward/feet_air_time:        0.001157       0.002031
mean_reward/track_lin_vel_xy:     0.07236        0.07063
```

All three stronger-gait repetitions entered the gait branch by update 100,
with contact in `[0.6425, 0.6683]` and air-time reward in
`[0.000548, 0.000726]`. The corresponding baseline ranges were
`[0.7087, 0.7307]` and `[0.000206, 0.000350]`.

Deterministic zero-command MJX-Warp rollouts from these early checkpoints still
terminated before 600 steps, but did not regress against the matched baseline:
baseline first-done steps were `98`, `218`, and `113`; stronger-gait first-done
steps were `278`, `108`, and `117`. These checkpoints are too early to establish
final stand-still quality, so long-run zero-command evaluation remains required.

The weight remains command-gated at speed `> 0.10`, preserving the explicit
standing-foot penalties for stand-still samples. No command distribution,
entropy, curriculum, contact threshold, or PPO setting changed in this
ablation.

## 2026-07-14 - Add stand-still observation countermeasures

The full stronger-gait run walked reliably at a forward command but still lost
orientation at zero command. Inspection found two observation-level causes:

- the contact schedule correctly requested double support below the standing
  threshold, while the observed gait phase continued cycling at its 1.25 Hz
  baseline frequency;
- the hard-recipe actor did not observe base linear velocity, even though it was
  expected to suppress zero-command drift (the critic already observed it).

The gait phase now freezes at zero whenever the command norm is at or below
`gait.standing_command_threshold`, matching the existing double-support mask.
The hard recipe now also enables actor `base_lin_vel`. A 520-update smoke
checkpoint with these two observation changes walked for the full 1,000-step
forward-command episode but still lost orientation after 109 zero-command
steps, within the spread of the earlier short runs.

A follow-up short run tried a command-gated default-pose penalty, but it did not
improve the zero-command rollout (94 steps). A zero-action control showed why:
the configured nominal pose itself lost orientation after 76 steps in MJX-Warp,
so forcing that pose would encode an unstable target. The term was discarded.

Instead, the hard recipe raises `standing_probability` from `0.20` to `0.35` so
the policy receives 75% more exact-zero-command experience while learning its
own dynamically stable pose. The final countermeasures are therefore the frozen
standing gait phase, actor base-linear-velocity feedback, and increased standing
sampling. The actor observation change requires training a new checkpoint.

## 2026-07-14 - Track command velocity at the pelvis

Changed the easy/hard locomotion velocity targets from the floating
`bello_root` wrapper to `pelvis_link`. `track_lin_vel_xy.link` and its
`frame_link` now both use the pelvis, and `track_ang_vel_z.link` uses the pelvis
as well. Torso-linked height, orientation, vertical-velocity, and roll/pitch
angular-velocity terms are unchanged.
MuJoCo's URDF importer previously fused the root pelvis into `bello_root`, so
asset preparation now disables static-body fusion and keeps `pelvis_link` as an
actual model body. The asset cache version was incremented to invalidate models
generated with the fused hierarchy.

## 2026-07-14 - Add scheduled swing-foot velocity tracking

Added `swing_foot_velocity`, a command-gated reward for matching the scheduled
swing foot's XY velocity relative to the pelvis in the pelvis yaw frame. Its
target is derived from gait duty factor as
`duty_factor / (1 - duty_factor) * command[:2]`, then capped at `2.0 m/s`.
The term uses a bounded exponential score with `std: 0.25`, is disabled below
linear command speed `0.10`, and is enabled in the easy/hard locomotion recipe
at weight `0.5`. Directional vector error prevents sideways or backward foot
motion from receiving the same reward as commanded swing motion.

## 2026-07-14 - Correct body-frame velocity measurements

Audited the environment velocity helpers against MuJoCo's canonical
`mj_objectVelocity(..., mjOBJ_XBODY, ...)` result. The previous implementation
treated `mjData.cvel` as body-local velocity and rotated it into the world frame.
MuJoCo instead stores `cvel` in the globally oriented, subtree-COM-centered
c-frame. This corrupted named-body linear and angular observations, pelvis
command tracking, foot slip and clearance measurements, and scheduled swing-foot
velocity tracking. The free-root angular-velocity path also rotated an already
local free-joint angular velocity a second time.

The MJX path now reproduces MuJoCo's spatial motion transform: it shifts linear
velocity from the root subtree COM to the requested body-frame origin, then
rotates world-oriented motion into the body frame only when a local result is
requested. CPU evaluation uses `mj_objectVelocity` directly. A regression test
checks root, pelvis, torso, and both feet against MuJoCo under a nontrivial base
orientation and generalized velocity.

## 2026-07-14 - Clean baseline converged to standing still

Inspected W&B run `nzjdd3h7`, trained from commit `589ad22` with the original
clean config. By update 2,048 it survived the full 1,000-step episode, but its
filtered foot-contact fraction reached `0.9815`. The deterministic rollout
remained in near-constant double support and did not travel forward.

The reward explains the behavior. With commands sampled uniformly from
`0.15-0.35 m/s`, 10% exact-zero commands, and tracking `sigma: 0.25`, a policy
with zero velocity has expected raw tracking score `0.79626`. After the `0.02s`
control-step scale, that is `0.015925` reward per step. The run reported
`0.015920`, effectively the exact standing-still prediction. Perfect tracking
could improve the episode return by only about `4.1`, while one fall cost
`100`. Avoiding all stepping was therefore the rational optimum rather than an
exploration failure.

The clean forward-discovery control now uses spherical hands, a fixed
`0.3 m/s` command with no standing samples, linear tracking weight `2.0` and
`sigma: 0.05`, and termination weight `-5.0`. At zero velocity the linear term
now scores only `exp(-0.3^2 / 0.05) = 0.1653`, making forward motion worth
roughly `33.4` additional return per full episode. Gait, contact, clearance,
air-time, and swing-foot rewards remain disabled so the next run still tests
whether locomotion emerges without gait shaping.

## 2026-07-14 - Clean rerun overflow and policy escape

Run `k7sbun7q` from the original clean commit failed at update 487. Immediately
before failure, MJX-Warp reported `nefc overflow` requests up to `1627` while
the config allowed only `njmax: 512`. The last finite row also showed that this
was not an isolated capacity warning: `action_mean_abs_max` had reached `14.49`,
the clipped environment action reached `5.0`, `ctrl_margin_min` was `-2.45`,
`value_loss_raw` was about `4e23`, and `value_norm_std` was about `7.6e12`.
The next update was nonfinite and correctly skipped by PPO.

The clean config now uses `njmax: 2048`, enough for the largest reported
per-world constraint count. It also adopts a constant learning rate of `1e-4`,
`init_log_std: -1.5`, and `max_log_std: -0.5`, plus explicit conservative
residual action scales for the 13 policy joints. The rate is constant because a
20-update smoke run showed the adaptive schedule increasing `1e-4` to
`3.375e-4`, undoing the intended safety limit. These changes address both the
undersized Warp solve and the policy/control escape that created the
pathological contacts.

A subsequent 512-update, 2,048-environment local run completed without a
constraint overflow or nonfinite update. At update 512, however, its action mean
had reached `3.05` and `ctrl_margin_min` was `-0.69`. The environment action
bound is now an explicit `BelloEnvConfig.action_clip` setting. The clean recipe
uses `1.0`, while the default stays `5.0` to avoid silently changing existing
recipes. With the clean recipe's residual scales, the tighter bound prevents
the policy from requesting the extreme joint targets seen in the failed run.

## 2026-07-14 - Replace touchdown air time with positive-biped support

Replaced the old landing-event `feet_air_time` bonus with the continuous
positive-biped formulation used by Isaac-style humanoid locomotion. The reward
is active only during single support and uses the shorter of the stance foot's
current contact duration and the swing foot's current air duration, capped at
`0.40s`. Double support, flight, and planar commands at or below `0.10 m/s`
receive zero. The reward is now timestep-scaled like other continuous terms.

The clean control still has both foot terms disabled. Three configs isolate the
follow-up experiments: `locomotion_clean_slip` enables only foot slip at
`-0.10`, `locomotion_clean_air_time` enables only air time at `0.25`, and
`locomotion_clean_feet` combines them for use only if both single-term
ablations improve the three-seed median.

## 2026-07-14 - Air-time run learned a high-frequency shuffle and diverged

W&B run `n3c9maha` used the air-time-only config and failed with a nonfinite PPO
update at update 1,294. The update-1,293 policy survived for a mean `982.5` of
1,000 steps and tracked forward velocity well (`track_lin_vel_xy = 0.03167` per
step), but the rollout video showed very short alternating steps. Mean filtered
foot contact was `0.6475`, and air-time reward was only `0.0001003`, about 0.3%
of linear tracking reward. Action rate cost was only `-0.0005906`, and joint
acceleration was disabled.

Policy stability deteriorated at the same time. Target-KL truncation first
appeared at update 959 and became recurrent after update 1,009. At update 1,293,
approximate KL was `0.0268` against target `0.02`, only 10% of PPO minibatches
were applied, raw action-mean magnitude reached `5.47` despite the environment
bound of `1.0`, and 5.65% of applied controls were saturated. The existing
`invalid_action` metric remained exactly zero because PPO passed only the
clipped action to the environment, hiding the out-of-range request.

The rollout path now passes the raw action request to the environment. Bello
uses the raw request for `invalid_action` and the bounded action for physics,
observations, and action-rate cost. `action_clip_fraction` reports how often the
bound changes a sampled action. The `locomotion_clean_air_time_smooth` follow-up
raises air-time weight to `0.50`, action-rate cost to `-0.05`, enables joint
acceleration at `-1e-6`, and uses a `5e-5` linearly decaying learning rate,
`0.5` gradient norm, and `0.015` target KL.

A 128-update local MJX-Warp screen held the learning rate at `5e-5` to
approximate the early portion of the full schedule. It completed with no
nonfinite or KL-truncated update: final KL was `0.00338`, all PPO minibatches
were applied, action clipping affected `0.85%` of sampled dimensions, and the
pre-clip invalid-action cost was nonzero. Maximum raw action mean was `1.23`,
down slightly from `1.25` in the failed run at a similar update, while the new
penalty now supplies a restoring gradient rather than a flat clipped-reward
region. Gait quality still requires the 512-update video gate; this short run
only validates early optimization stability.

## 2026-07-14 - Replace occupancy-only gait ablations with event timing

The 1,536-update, three-seed follow-up found that stance-width weight `-2.0`
was the only treatment that preserved robust tracking and survival. The
1.20-second clock, stronger invalid-action pressure, and tanh policy all traded
lower measured cadence for worse survival. Across successful runs, PPO remained
finite, applied every minibatch, and never exceeded target KL. Reward design,
not optimizer stability, remained the limiting factor.

The clocked seed-0 policy produced about `5.16` combined footfalls per second
against a scheduled target near `1.76`. `gait_contact` scores contact occupancy,
while contact debounce was zero, so short repeated transitions could retain
substantial reward. The next pipeline keeps the stance recipe as
`locomotion_gait_base` and adds an explicit `contact_timing` event penalty. A
touchdown is penalized only when the preceding air interval is below its
minimum; liftoff uses the preceding contact interval. Accepted contacts are
debounced before duration memory and rewards consume them.

`locomotion_gait` is the full candidate: 40 ms on/off debounce, 1.0-second
clock, gait-contact `0.35`, timing penalty `-0.10`, swing velocity `0.25`, and
foot slip `-0.05`. The previous `locomotion_clean_*` ablation YAML chain was
removed. The executable matrix in `scripts/run_locomotion_ablation.sh` now
isolates debounce, timing, slip, clock, swing velocity, and full-stack
interactions without adding another config file per treatment.

## 2026-07-15 - Separate physical joint and policy action limits

The Berkeley policy saturated a large fraction of its bounded action means even
though actual joint positions usually remained inside the configured actuator
target ranges. The old `joint_limit` path could not distinguish those states:
it compared physical joint positions against actuator `ctrlrange`, mixing
physical safety with action-scale metadata.

`joint_limit` now derives a symmetric soft interval directly from each MuJoCo
hard joint range. `soft_limit_factor: 0.9` matches the common Isaac-style 90%
range and restores a knee-extension margin instead of treating the knee hard
lower bound as soft. Its cost remains the standard summed linear violation.

The independent `action_limit` term starts at `soft_limit_ratio: 0.8` of the
environment action bound and increases quadratically to one at the boundary,
averaged over policy dimensions. The Berkeley candidate enables it at weight
`-1.0`. Four scripted conditions isolate the hard-bound control, joint margin,
action margin, and their combination.

## 2026-07-15 - Restore adaptive KL control for the Berkeley recipe

The four joint/action-limit ablations all approached the hard `0.03` KL guard
under the fixed `3e-4` learning rate. Late in the runs, PPO applied only about
4--14% of its 128 planned minibatch updates. The two conditions without the
action-limit reward also pinned `max_log_std: 1.0` (standard deviation `e`) and
saturated roughly one third of deterministic action means. The action-limit
conditions suppressed saturation but still lost most PPO updates, isolating
the constant learning rate as a shared optimizer problem rather than a reward
ablation effect.

The Berkeley recipe now uses the existing adaptive schedule with
`desired_kl: 0.01`, `min_learning_rate: 1e-5`, `max_learning_rate: 3e-4`, and
`adaptive_lr_factor: 1.5`, while retaining `target_kl: 0.03` as the hard
same-update stop. Capping `max_log_std` at `0.0` limits policy standard
deviation to `1.0`. A single combined 90% joint-margin plus 80% action-margin
run replaces the stopped four-way matrix for the initial stability check.
