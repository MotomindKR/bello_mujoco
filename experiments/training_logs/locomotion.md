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

- PPO rollout stored and evaluated the unclipped sampled action while the env
  executed a clipped action. At update 573, raw `action_abs_max` was `7.6e7`
  while `env_action_abs_max` was still `10`. Rollout collection now passes
  `env.action_clip` into `sample_action_with_mean`, so stored actions, old
  log-probs, and env actions agree.
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
the clipped-action fix, `action_abs_max` should stay bounded by the env action
clip, while `action_mean_abs_max` remains useful as a separate warning for mean
drift.

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
squared violation:

```text
target = default_ctrl + policy_action_scale * action
violation = max(soft_low - target, 0) + max(target - soft_high, 0)
invalid_action = sum((violation / policy_action_scale)^2)
```

This makes the penalty depend on how far the requested target is outside the
soft range in action units, so small-scale joints such as knees and ankles are
not under-penalized relative to hips. Because the raw magnitude is now larger
for severe out-of-range targets, the active locomotion recipe weights were reset
from `-100.0` to `-1.0`.
