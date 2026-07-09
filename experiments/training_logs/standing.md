# Standing Reward and Environment Tuning Notes

Date: 2026-07-08

These notes capture practical findings from tuning the standing-only Bello task.
They are local experiment notes, not packaged defaults.

## Current Recipe

Use:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name standing
```

Config:

- `experiments/train/standing.yaml`
- zero command task via `command_config.enabled: false`
- command, gait phase, and actuator-force observations removed from the policy
  observation; base linear velocity and base height included for balance
- `env_config.root_link: torso_link`, so base observations and null-link
  reward/done terms use the torso instead of the generated free-root wrapper
- lower body and waist yaw controlled by policy; waist roll/pitch, head, arms,
  and fingers held
- `obs_stack_size: 10`
- default PPO settings inherited from `configs/train/default.yaml`, including
  adaptive KL, value normalization, `ent_coef: 0.003`, `init_log_std: 0.0`,
  `max_log_std: 1.0`, and `num_steps: 24`
- native additive standing reward with `alive`, zero-velocity tracking,
  base-height/uprightness shaping, small joint-velocity/torque/action-rate
  penalties, and terminal-failure penalty
- `max_episode_steps: 256`, base-height done threshold `0.55`, roll/pitch
  done threshold `1.35 rad`

## Diagnostics

Before standing-specific reward fixes:

- Zero action terminated at 6 control steps.
- With `control_timestep = 0.02s`, this is `0.12s`.
- Termination reason was `done/base_height`.
- Initial stochastic policy also terminated around 6 steps.

After reducing exploration noise and relaxing base-height termination:

- Zero action terminated at 134 control steps, about `2.68s`.
- Initial stochastic policy survived at least 100 control steps, about `2.0s`.
- This made PPO rollouts long enough to provide a useful learning signal.

At this historical checkpoint, the recipe still used a DM-Control-style reward
with loose base-height and roll/pitch done checks. The active recipe no longer
has `dm_control_stand`; it uses the native additive reward summarized above.

W&B run observed:

- Run: `standing_seed0`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/872gaypg`
- Around update 196:
  - `mean_episode_return`: about `214`
  - `max_episode_return`: about `612`
  - `mean_reward`: about `1.66`
  - `mean_done/base_height`: about `0.0077`
  - `mean_done/timeout`: `0`
  - average episode length estimate: `1 / 0.0077 ~= 129` steps, about `2.6s`

Visual rollout at `outputs/train_videos/standing_000175.mp4`:

- policy stands briefly
- starts leaning by roughly frame 100
- falls by roughly frame 150
- stays down because `video_config.stop_on_done: false`

Foot-distance-only diagnostic:

- Config: `experiments/train/foot_distance.yaml`
- Reward: only `reward/foot_distance`, the horizontal distance between the two
  foot bodies
- Initial observation: the policy visibly spreads the legs before falling
- User observation: this diagnostic showed better standing time than the
  standing reward recipe
- Interpretation: a wider support polygon is helping stability, and the
  standing reward recipe is probably under-rewarding stance geometry or
  over-emphasizing small-control/default behavior. This makes a pure PPO
  implementation bug less likely, because the same rollout/update path can
  discover a simple stabilizing behavior when the reward has an easy geometric
  gradient.
- Follow-up: keep fall done checks disabled for this diagnostic if testing
  post-fall behavior. If the policy still does not spread its legs while lying
  down, the problem is not fall reset truncation.

W&B run observed after disabling fall done checks:

- Run: `foot_distance_seed0`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/07g1n298`
- Around update 127:
  - `mean_reward`: about `0.0116`
  - `mean_reward/foot_distance`: about `0.0116`
  - all other reward terms: `0`
  - average horizontal foot distance estimate: `0.0116 / 0.02 ~= 0.58m`
  - `mean_episode_length`: `256`
  - `mean_done/timeout`: `1 / 256 = 0.00390625`
  - `mean_done/base_height`: `0`
  - `mean_done/roll_pitch`: `0`
  - `ctrl_saturation_fraction`: about `0.044`
  - `action_rms`: about `0.78`
  - `action_mean_rms`: about `0.49`
  - `approx_kl`: about `0.0027`
  - `value_loss`: about `0.012`

Interpretation:

- The no-fall-done diagnostic is learning monotonically. `mean_reward` increased
  from about `0.0058` at update 1 to about `0.0116` at update 127, which means
  average foot distance roughly doubled from `0.29m` to `0.58m`.
- It learns slower than the previous fall-reset foot-distance diagnostic at the
  same wall-clock stage, because fallen states stay in the rollout instead of
  being reset to easy standing starts.
- PPO statistics look healthy: KL is small, entropy remains high, value loss is
  controlled, and control saturation is present but not dominating yet.

High-exploration follow-up:

- Config changed in `experiments/train/foot_distance.yaml`.
- `config.ent_coef`: `0.003 -> 0.03`
- `config.policy.init_log_std`: `-0.5 -> 1.8`
- `config.policy.max_log_std`: explicitly set to `1.8`, so the diagnostic is
  not clipped by the standing/default `max_log_std: 1.0`
- This changes the initial action std from `exp(-0.5) ~= 0.61` to
  `exp(1.8) ~= 6.05`, about `10x`.
- Run/checkpoint/video names now use `foot_distance_high_exploration_seed0` /
  `foot_distance_high_exploration`.
- One-update smoke check stayed finite, but produced very high exploration:
  `action_rms ~= 6.6`, `action_abs_max = 10`, and
  `ctrl_saturation_fraction ~= 0.58`. This is intentionally aggressive and
  should be compared against the previous foot-distance run for learning speed
  rather than treated as a stable final recipe.

W&B run observed for high-exploration foot-distance diagnostic:

- Run: `foot_distance_high_exploration_seed0`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/7myi0i7s`
- Around update 34:
  - `mean_reward`: about `0.0243`
  - previous normal foot-distance run reached about `0.0209` only at update 256
  - `action_rms`: about `6.38`
  - `action_mean_rms`: about `4.20`
  - `ctrl_saturation_fraction`: about `0.65`
  - `approx_kl`: about `0.023`
  - `value_loss`: about `0.081`
- Interpretation: exploration scale was a primary bottleneck for the
  foot-distance diagnostic. The setting is intentionally violent and saturation
  heavy, but it proves the policy/update path can discover the simple
  geometric behavior quickly.

Applied standing follow-up:

- Config changed in `experiments/train/standing.yaml`.
- `config.ent_coef`: `0.003 -> 0.03`
- `config.policy.init_log_std`: `-0.5 -> 1.8`
- Run/checkpoint/video names now use `standing_high_exploration_seed0` /
  `standing_high_exploration`.
- Reward shape is otherwise unchanged so this run isolates exploration scale on
  the standing objective.

Survival-only standing follow-up:

- Config changed in `experiments/train/standing.yaml`.
- Enabled reward terms for this run were only:
  - `alive: enabled: true, weight: 1.0`
  - `termination: enabled: true, weight: -1.0`
- `dm_control_stand` and all pose, velocity, action, contact, and gait terms are
  disabled.
- With `control_timestep = 0.02`, alive reward contributes `0.02` per step, so a
  256-step timeout is worth about `5.12` before any terminal penalty. A fall
  still receives the event-driven `-1.0` termination penalty.
- Run/checkpoint/video names now use
  `standing_survival_high_exploration_seed0` /
  `standing_survival_high_exploration`.
- Purpose: isolate whether high exploration plus a pure survival objective is
  enough for PPO to discover a stable balance behavior.

W&B run observed for survival-only high-exploration standing:

- Run: `standing_survival_high_exploration_seed0`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/aroemmim`
- Around update 162:
  - `mean_episode_length`: about `53`
  - `max_episode_length`: about `129`
  - `mean_reward/alive`: about `0.020`
  - `mean_reward/termination`: about `-0.019`
  - `mean_reward`: about `0.0009`
  - `ctrl_saturation_fraction`: about `0.71`
  - `action_std_mean`: about `7.59`
- Interpretation: survival-only is too sparse under the high-exploration PPO
  setting. The positive alive reward and terminal penalty mostly cancel, while
  the policy keeps high-entropy saturated actions and falls quickly. This
  behaved similarly to the high-exploration `dm_control_stand + termination`
  run, which also stayed dominated by fall rate.

Applied next standing follow-up:

- Config changed in `experiments/train/standing.yaml`.
- `config.ent_coef`: `0.03 -> 0.01`
- `config.policy.init_log_std`: `1.8 -> 0.65`
- Reward terms for this run became:
  - `alive: enabled: true, weight: 1.0`
  - `foot_distance: enabled: true, weight: 1.0`
  - `termination: enabled: true, weight: -1.0`
- `dm_control_stand` remained disabled.
- Run/checkpoint/video names now use
  `standing_survival_stance_exploration_seed0` /
  `standing_survival_stance_exploration`.
- One-update smoke check stayed finite with `action_std_mean ~= 1.92`,
  `ctrl_saturation_fraction ~= 0.33`, `mean_reward/alive = 0.02`, and
  `mean_reward/foot_distance ~= 0.0056`.
- This stance-width cue was later removed from the standing recipe after
  deciding to keep the Unitree-style standing experiment survival-only.

Applied Unitree/rsl_rl-style PPO follow-up:

- Config changed in `experiments/train/standing.yaml`.
- Standing reward terms for this run were survival-only:
  - `alive: enabled: true, weight: 1.0`
  - `termination: enabled: true, weight: -1.0`
  - `foot_distance: enabled: false`
  - `dm_control_stand: enabled: false`
- Matched supported PPO parameters from the Unitree/rsl_rl config:
  - `num_steps: 24`
  - `total_timesteps: 1_228_800_000`, equivalent to 50,000 updates with 1,024
    envs and 24 steps per env
  - `update_epochs: 5`
  - `num_minibatches: 4`
  - `learning_rate: 0.001`
  - `gamma: 0.99`
  - `gae_lambda: 0.95`
  - `clip_eps: 0.2`
  - `ent_coef: 0.01`
  - `max_grad_norm: 1.0`
  - `policy.init_log_std: 0.0`, matching `init_noise_std: 1.0`
- Did not change the MLP shape or activation. The local policy config did not
  support separate actor/critic layer lists or ELU yet.
- Did not implement `schedule: adaptive` / `desired_kl: 0.01` at this point;
  the local PPO then supported fixed LR or linear annealing, not rsl_rl-style
  adaptive KL scheduling.
- Run/checkpoint/video names now use
  `standing_unitree_ppo_survival_seed0` /
  `standing_unitree_ppo_survival`.
- Dry-run confirmed `num_updates: 50000` with `num_steps: 24`.
- One-update smoke check with one env and 24 steps should show only
  `mean_reward/alive` and `mean_reward/termination` as nonzero reward terms.
  Tiny smoke batches could have high `approx_kl` because adaptive KL scheduling
  had not been added yet.
- Corrected smoke check after disabling `foot_distance` showed
  `mean_reward/alive = 0.02`, `mean_reward/foot_distance = 0`,
  `mean_reward/dm_control_stand = 0`, `mean_reward/termination = 0` on the
  first non-terminal rollout.

## Reward Tuning Lessons

Semantic simplicity does not mean the RL objective is simple.
Standing still can be a hard exploration problem if the default controller
collapses quickly.

Zero action means:

```text
ctrl = default_ctrl + action_scale * 0 = default_ctrl
```

For standing, zero action should ideally be a reasonable baseline controller.
If zero action falls almost immediately, PPO is not refining a stable behavior;
it is discovering active balance from short terminal failures.

Avoid making slow collapse profitable.

If `track_lin_vel_xy` is large, a robot that falls slowly can receive high
reward because its XY velocity is near zero. With commands disabled,
`track_lin_vel_xy` tracks zero velocity, which is appropriate for standing, but
its weight should stay modest.

Negative per-step penalties can encourage early termination.

If the agent receives large negative rewards every step while struggling, dying
early can avoid future penalties. Keep early bootstrapping rewards shaped so
survival/uprightness has clear positive value, not just many large penalties.

Useful standing terms for the additive velocity-tracking recipe:

- `alive`: positive survival pressure
- `foot_contact`: positive planted-foot pressure
- `base_height`: moderate penalty around target height
- `flat_orientation`: moderate uprightness penalty
- `lin_vel_z`: discourages vertical collapse
- `ang_vel_xy`: discourages pitching/rolling motion
- `joint_vel`: discourages thrashing
- `action_rate`: discourages jittery controls
- `policy_joint_deviation`: nudges policy joints toward the default stance
- `termination`: optional terminal event penalty; not scaled by
  `control_timestep`

Useful standing term for the now-removed DM Control-style recipe:

- `dm_control_stand`: bounded positive product of standing height, uprightness,
  near-zero horizontal velocity, and small action

Less useful for pure standing:

- `gait_contact`: locomotion schedule, disable
- `feet_air_time`: locomotion foot-swing reward, disable
- `foot_clearance`: locomotion swing clearance, disable
- `gait_phase` observation: disable when not using a gait schedule
- command observation: disable when command is always zero

## MuJoCo Playground Comparison

MuJoCo Playground humanoid examples use explicit pose/default-stance
regularization on policy-controlled joints.

- G1 has `pose`, `joint_deviation_hip`, `joint_deviation_knee`, and
  `stand_still` terms around the default pose.
- Berkeley Humanoid has a `pose` term with weight `-1.0`.
- H1 has `stand_still` around the default pose when commands are near standing.
- The DM Control humanoid stand task uses bounded standing/upright/small-control
  factors instead of large unbounded negative terms.

Reward scale matters. The local reward path now follows the Playground
locomotion convention for continuous terms by multiplying them by
`control_timestep`. At the default `0.02s` control timestep, a continuous term
with `weight: 1.0` and raw value `1.0` contributes `0.02` per step. Event terms
such as `feet_air_time` and `termination` are not timestep-scaled; their weights
are event magnitudes.

## Historical DM-Control Reward Shape

This section is superseded. The package no longer defines `dm_control_stand`,
and the active `standing.yaml` uses the native additive reward summarized in
the current recipe section.

The previous DM-Control-style standing recipe used:

```yaml
reward_config:
  dm_control_stand:
    enabled: true
    weight: 1.0
    height_link: head_link
    upright_link: torso_link
    move_link: bello_root
    stand_height: 1.4
    stand_margin: 0.35
    upright_threshold: 0.9
    upright_margin: 1.9
    move_margin: 2.0
    action_margin: 1.0
  track_lin_vel_xy:
    enabled: false
  track_ang_vel_z:
    enabled: false
  alive:
    enabled: false
  lin_vel_z:
    enabled: true
    weight: -2.0
  ang_vel_xy:
    enabled: true
    weight: -0.5
  flat_orientation:
    enabled: true
    weight: -5.0
  base_height:
    enabled: true
    weight: -10.0
    target: 0.86
  joint_vel:
    enabled: false
  action_rate:
    enabled: false
  policy_joint_deviation:
    enabled: false
  foot_contact:
    enabled: false
  foot_clearance:
    enabled: false
  gait_contact:
    enabled: false
  feet_air_time:
    enabled: false
  termination:
    enabled: false
```

Rationale:

- match DM Control Humanoid Stand's product-style reward as closely as the
  env abstraction allowed at that time
- make reward bounded and mostly nonnegative
- scale reward by control timestep so returns are roughly in seconds of good
  standing, not raw control-step count
- add dt-scaled early-warning penalties for vertical velocity, roll/pitch
  angular velocity, orientation drift, and base-height drift
- avoid terminal penalties that can distort the standing objective
- use `head_link` height and `torso_link` uprightness, matching DM Control's
  head-height and torso-upright factors

## Environment Tuning Lessons

The base-height termination threshold is a curriculum knob for additive
locomotion-style recipes.

Current standing done thresholds:

```yaml
done_config:
  base_height:
    enabled: true
    min_height: 0.55
  roll_pitch:
    enabled: true
    max_abs_roll: 1.35
    max_abs_pitch: 1.35
```

This resets clearly falling samples earlier than the initial bootstrap threshold
while still allowing some recoverable leaning states.

Initial action noise matters for balance.

The current package default `init_log_std: 0.0` gives action std `1.0`, while
the standing recipe caps learned std at `exp(1.0) ~= 2.72`. Earlier lower-noise
diagnostics used `init_log_std: -2.0` for std about `0.135`; the foot-distance
pipeline diagnostic intentionally uses `init_log_std: 1.8` and `max_log_std:
1.8` for aggressive exploration.

Keep rollout videos with `stop_on_done: false` while debugging.

This shows the post-termination behavior instead of hiding falls at the first
done frame. It makes collapse modes easier to diagnose.

## What To Watch In W&B

Primary progress signals:

- `train/mean_done/base_height` and `train/mean_done/roll_pitch` should be
  nonzero early if the policy is falling.
- `train/mean_done/timeout` should become more common only as standing improves.
- `train/mean_episode_return` should increase.
- `train/max_episode_return` should rise as episodes approach the 256-step
  standing horizon.

Reward component sanity checks:

- `train/mean_reward/alive`, `train/mean_reward/track_lin_vel_xy`, and
  `train/mean_reward/track_ang_vel_z` should stay positive for near-standing
  rollouts.
- `train/mean_reward/base_height`, `train/mean_reward/flat_orientation`,
  `train/mean_reward/lin_vel_z`, `train/mean_reward/ang_vel_xy`,
  `train/mean_reward/joint_vel`, `train/mean_reward/torque`, and
  `train/mean_reward/action_rate` should move toward zero as balance improves.
- Disabled locomotion/contact terms such as `gait_contact`, `feet_air_time`,
  and `foot_clearance` should stay at zero.

If `mean_episode_return` improves but rollout videos still show collapse, the
policy is probably learning to delay falling rather than solving standing.

If `mean_reward` is positive while videos show collapse, inspect the component
terms. A collapsed policy may still earn some alive or zero-velocity reward
unless base-height/uprightness penalties and terminal failures are visible in
the breakdown.

## Next Likely Steps

If the native additive standing recipe still plateaus around 2-4 seconds:

1. Check whether `base_height`, `flat_orientation`, velocity penalties, or
   terminal-failure rate is the limiting factor.
2. Increase ankle/hip damping or gains if zero action continues to sink.
3. Keep waist held until lower-body balance is reliable.
4. If learning is too sparse after falls, relax `base_height.min_height` or
   `roll_pitch` thresholds further, but make sure `reward/termination` only
   applies to true terminal failures, not timeouts.
5. Add policy-pose regularization only if the native reward solves height and
   uprightness but finds unnatural joint configurations.

## Actuator Gain Notes

Corrected runtime PD gains for lower-body standing experiments:

```text
hip_pitch: kp=200, kd=5
hip_roll:  kp=150, kd=5
hip_yaw:   kp=150, kd=5
knee_mix:  kp=220, kd=6
ankle:     kp=22,  kd=1
waist_yaw: kp=180, kd=12
```

`bello_mujoco` position actuator controls are joint-position targets, so these
gains determine the effective corrective torque generated from target-position
error. The ankle gain correction is lower than the previous local value
(`40/2`), so if standing remains ankle-limited, inspect action scale and
actuator modeling before simply increasing these gains again.

## 2026-07-08 Deep Audit: Falling Despite High Stand Reward

Latest corrected-gain W&B run inspected: `7py6rnfk`.

Observed pattern:

- `train/mean_done/base_height` plateaued around `0.009`, roughly a
  100-110 control-step episode, or about 2.0-2.2 seconds at the 20 ms control
  timestep.
- `train/mean_done/roll_pitch` and `train/mean_done/timeout` stayed near zero,
  so episodes were ending because the root sank below the configured
  base-height threshold, not because the robot first toppled by roll/pitch.
- `train/mean_reward/dm_control_stand` stayed near `0.019` out of the
  per-step maximum `0.02` while the policy was still failing. That means the
  main standing reward was almost saturated at the state where the
  `base_height` done condition reset the rollout.
- Rollout videos showed an upright start, then crouch/sink, then fall. With
  `stop_on_done: false`, the video continues after the first done frame, which
  makes the collapse visible.

Code audit:

- PPO rollout and GAE handling looked conventional: terminal transitions do
  not bootstrap, and batched training resets done envs before feeding the next
  stacked observation.
- Evaluation/viewer stacking already uses `stop_on_done` to decide whether the
  observation stack resets, so continuing past done no longer pretends the env
  reset.
- Single-env `env.step` does not reset on done; batched training does. That is
  expected for viewer/eval usage.
- A stale MJX/Warp contact-cache detail exists after `tree_where` reset
  selection, but stepping from the selected reset state matched a clean reset
  in qpos/qvel/xpos/xquat/cvel/reward/info, so it does not explain this
  training failure.
- Pure MuJoCo and MJX zero-action rollouts agree: with the corrected gains the
  default pose still sinks below the old base-height threshold after about
  1.1 seconds, then fully collapses later. This points to task/design signal,
  not a backend mismatch.

Most likely root issue:

- The policy did not directly observe root height or root linear velocity, even
  though standing rewards/done conditions depended on height, horizontal
  movement, and vertical movement.
- The standing experiment terminated at `base_height < 0.78` while the
  DM-Control-style head-height reward was still near saturated. This cut off
  the lower-reward crouch/fall samples that would have made the objective more
  identifiable.

Change made after audit:

```yaml
env_config:
  observation:
    base_lin_vel: true
    base_height: true

done_config:
  base_height:
    enabled: false
```

The new observation knobs are opt-in package features, so default observation
sizes stay backward compatible. `standing.yaml` opts into them. Disabling
base-height termination for standing keeps height as a continuous reward signal
and lets roll/pitch or timeout end episodes instead of resetting during a
high-reward crouch.

## 2026-07-08 Training Length Adjustment

Changed `standing.yaml` from `104,857,600` to `524,288,000` total timesteps.
With `num_envs: 2048` and `num_steps: 256`, this keeps the per-env rollout
horizon at 5.12 seconds and raises PPO training from 200 updates to 1000
updates.

Rationale:

- `num_steps: 256` is not unusually short for PPO humanoid training.
- Increasing `num_steps` while holding total timesteps fixed would reduce the
  number of policy updates.
- The previous standing budget may have sounded large in environment steps but
  was only 200 optimizer/update cycles, which is low for this robot.

## 2026-07-08 Live Run Diagnosis: Not A Duration Problem

Live run inspected: `lsm64twd`.

After about 10 minutes:

```text
env steps: 57,671,680
runtime: 601 s
mean_episode_return: -0.74
max_episode_return: 0.96
mean_reward: -0.0058
done/base_height: 0
done/roll_pitch: 0.0078
reward/dm_control_stand: 0.0171
reward/base_height: -0.0024
reward/flat_orientation: -0.0083
reward/lin_vel_z: -0.0011
reward/ang_vel_xy: -0.0111
```

This is enough data to reject "just train longer" as the primary explanation.
The positive stand reward was present, but the auxiliary negative penalties
dominated it, so the training objective was no longer the simple nonnegative
DM-Control-style standing objective.

Changed `standing.yaml` again:

```yaml
config:
  total_timesteps: 67108864
  num_envs: 1024
  num_steps: 128
  num_minibatches: 4

reward_config:
  dm_control_stand:
    enabled: true
  lin_vel_z:
    enabled: false
  ang_vel_xy:
    enabled: false
  flat_orientation:
    enabled: false
  base_height:
    enabled: false
```

Rationale:

- Standing should first be tested as a positive bounded task reward where
  falling gives near-zero reward, not as a mix of a positive task reward and
  several larger negative penalties.
- `1024 x 128` gives 131,072 samples per update, so a 10-minute run produces
  many more optimizer updates than the throughput-oriented `2048 x 256` shape.
- The rollout horizon is still 2.56 seconds, which is longer than the
  zero-action fall window, so the first failure mode remains visible to PPO.

## 2026-07-08 Live Run Diagnosis: Roll/Pitch Done Still Cuts Off The Signal

Live run inspected: `v2tpwgsp`.

The simplified positive-only reward is now moving in the right direction, but
it is still far from solved:

```text
env steps: 18,743,296
runtime: 331 s
mean_episode_return: 1.93
max_episode_return: 3.39
mean_reward: 0.0165
done/roll_pitch: 0.0084
done/timeout: 0
```

The trend improved from early training, where mean return was around `0.83` and
`done/roll_pitch` was about `0.0157`, but rollout videos at updates 25-125
still fall after roughly 2 seconds.

Next issue:

- `done/roll_pitch` still resets the env while the policy is in the fall
  window.
- That prevents the reward from acting like DM Control Humanoid Stand, where
  fallen states continue and receive near-zero reward instead of resetting
  immediately.

Changed next-run standing config:

```yaml
env_config:
  max_episode_steps: 300

done_config:
  roll_pitch:
    enabled: false
```

This keeps no pose-based fall termination for the standing experiment and uses
a 6 second timeout. The goal is to show the policy the low-reward fallen state
without wasting 20 seconds of rollout after every fall.

## 2026-07-08 Live Run Diagnosis: Janky Mean Reward

Live run inspected: `6bg52tpl`.

Run config:

```text
num_envs: 1024
num_steps: 128
max_episode_steps: 300
base_height done: disabled
roll_pitch done: disabled
```

The janky `train/mean_reward` curve is mostly a timeout/reset synchronization
artifact, not W&B noise.

Because pose-based fall termination is disabled, every env only resets on
timeout. All vectorized envs start with the same `step_count`, so their timeouts
stay synchronized. The metric history alternates between rollout chunks that
contain an upright reset window and chunks where most envs are already fallen.

Evidence:

```text
done/timeout = 0.0078125 = 1 / num_steps
```

When this appears, it means each env timed out once during that 128-step PPO
rollout. Those updates include fresh reset/upright samples and have higher
`mean_reward`. Neighboring updates often contain mostly fallen near-zero-reward
states and drop close to zero.

Secondary issue:

- The first update had a very high `approx_kl` around `0.77`, and later updates
  stayed around `0.03-0.06`. PPO updates may still be aggressive for this tiny
  bounded reward scale, but the dominant graph jank matches timeout phase
  synchronization.

Likely next curriculum tweak:

- For the first standing sanity task, set `max_episode_steps` equal to
  `num_steps` (`128`) so every update sees the same reset-to-fall window.
- After the policy reliably stands for that horizon, increase timeout to
  `300`, then later to `1000`.
- A more general fix is asynchronous timeout randomization across vectorized
  envs, but that should be a deliberate env feature rather than hidden behavior.

Applied next-run curriculum:

```yaml
env_config:
  max_episode_steps: 128
```

This intentionally makes the first standing curriculum a 2.56 second task,
matching the PPO rollout horizon. It should remove timeout phase sawtoothing
from `train/mean_reward` and make early W&B curves easier to interpret.

Updated curriculum after discussion:

```yaml
config:
  num_steps: 256
env_config:
  max_episode_steps: 256
```

This keeps timeout synchronized with the PPO rollout to avoid phase jank, but
uses a 5.12 second standing horizon. With `num_envs: 1024`, the batch size is
262,144 samples/update and the 67,108,864-step budget gives 256 PPO updates.

## 2026-07-08 Midpoint Inspection: Run `6bg52tpl`

Active run inspected around step `43,253,760`, about 64% through its
`67,108,864` step budget.

Important config mismatch:

```text
run config:
  num_steps: 128
  max_episode_steps: 300

repo standing.yaml at inspection time:
  num_steps: 256
  max_episode_steps: 256
```

The active run was started before the later `256/256` curriculum patch, so it
is still the older unsynchronized-horizon experiment.

Metrics:

```text
best mean_episode_return: 1.99
best max_episode_return: 4.69
latest mean_reward: 0.00046
latest mean_episode_return: 0
done/timeout on timeout updates: 0.0078125
done/base_height: 0
done/roll_pitch: 0
```

Interpretation:

- It is not solving the task by midpoint.
- It improved early from near-zero returns to roughly 2 seconds of good return,
  but then plateaued.
- `done/timeout = 0.0078125 = 1 / 128`, confirming the reward graph still has
  rollout-phase jank from `max_episode_steps=300` with `num_steps=128`.
- Latest videos through update 300 still fall. The best sampled video around
  update 275 stays upright until about frame 150, then falls by about frame
  175. Other nearby videos fall earlier, around frame 100-125.

Decision:

- Do not use `6bg52tpl` as evidence for the `256/256` curriculum; it is running
  the previous config.
- Stop/restart before judging the then-current recipe.

## 2026-07-08 PPO Conservatism Adjustment

The standing runs showed suspiciously large PPO policy movement:

```text
v2tpwgsp first update approx_kl: ~28.7
6bg52tpl first update approx_kl: ~0.77
later updates often: ~0.03-0.06
```

For PPO this is too aggressive for a tiny bounded reward scale. A more stable
range to aim for is roughly `0.005-0.02`.

Changed next-run standing config:

```yaml
config:
  learning_rate: 0.0001
  update_epochs: 2
  clip_eps: 0.1
  ent_coef: 0.001
```

Rationale:

- Lower `learning_rate` and fewer `update_epochs` reduce policy movement per
  rollout.
- Lower `clip_eps` tightens PPO's trust-region-style clipping.
- A small `ent_coef` discourages premature collapse to a deterministic falling
  strategy.
- `policy.init_log_std` stays at `-2.0` for this pass so the main variable is
  optimizer conservatism, not larger reset-time action noise.

## 2026-07-08 Late Inspection: Run `qxpx6577`

Active run inspected around step `59,768,832`, about 89% through its
`67,108,864` step budget.

Config confirmed:

```text
num_envs: 1024
num_steps: 256
max_episode_steps: 256
update_epochs: 2
learning_rate: 0.0001
clip_eps: 0.1
ent_coef: 0.001
```

Metrics:

```text
latest mean_episode_return: 1.83
best mean_episode_return: 1.95
latest max_episode_return: 3.69
best max_episode_return: 4.91
latest mean_reward: 0.0071
best mean_reward: 0.0076
latest approx_kl: 0.00078
latest entropy: -8.79
done/timeout: 0.00390625 = 1 / 256
done/base_height: 0
done/roll_pitch: 0
```

Interpretation:

- The conservative PPO settings fixed the previous excessive KL issue.
- They likely overshot: after the first few updates, `approx_kl` stayed around
  `0.001`, so the policy is changing very slowly.
- The policy improved from the initial return around `1.23` to about `1.8-1.95`,
  but it plateaued well below the 5.12 maximum for a 256-step standing episode.
- Some stochastic rollouts reach high returns near `4.9`, so the system can
  occasionally remain standing for most of the horizon. The deterministic video
  rollout still falls around frame 100-125, so the mean policy is not capturing
  those successful samples reliably.

Decision:

- Do not call `qxpx6577` solved.
- The next experiment should be less conservative than this run but still much
  safer than the original aggressive PPO settings, for example
  `learning_rate: 0.0002`, `update_epochs: 2`, `clip_eps: 0.1`.
- Add action/ctrl diagnostics before further guessing: action RMS/max, control
  range saturation fraction, and maybe mean/std of per-action policy means.

Applied next-run changes:

```yaml
config:
  learning_rate: 0.0002
  update_epochs: 2
  clip_eps: 0.1
  ent_coef: 0.001
```

Added rollout diagnostics to PPO metrics:

```text
action_rms
action_abs_max
action_mean_rms
action_mean_abs_max
action_std_mean
action_std_min
action_std_max
ctrl_saturation_fraction
ctrl_margin_min
```

Watch `approx_kl` together with these diagnostics. A useful next run should
move `approx_kl` back toward roughly `0.005-0.02` without driving
`ctrl_saturation_fraction` high or pushing action means to large constant
offsets.

## 2026-07-08 Inspection: Run `zpenj7vh`

Run `zpenj7vh` used the intended next config:

```text
total_timesteps: 67,108,864
num_envs: 1024
num_steps: 256
max_episode_steps: 256
learning_rate: 0.0002
update_epochs: 2
clip_eps: 0.1
ent_coef: 0.001
policy.init_log_std: -2.0
```

Final W&B summary:

```text
mean_episode_return: 1.857
max_episode_return: 4.056
mean_reward/dm_control_stand: 0.00725
done/timeout: 0.00390625 = 1 / 256
done/base_height: 0
done/roll_pitch: 0
action_mean_rms: 0.261
action_mean_abs_max: 1.238
action_std_mean: 0.136
ctrl_saturation_fraction: 0.0286
approx_kl: 0.000027
```

Sampled history showed improvement until roughly `35M` env steps, then a
plateau/drift:

```text
~3.7M steps:  mean return 1.23, approx_kl 0.010
~31.5M steps: mean return 1.84, approx_kl 0.0046
~35.4M steps: mean return 1.97, approx_kl 0.0049
~58.0M steps: max return 4.53, approx_kl 0.0013
~64.7M steps: mean return 1.90, approx_kl 0.0005
final:        mean return 1.86, approx_kl 0.000027
```

Direct deterministic checkpoint rollout:

```text
zero action:     return 1.25, falls around step 75
trained policy:  return 1.73, falls around step 100
perfect horizon: return 5.12
```

Interpretation:

- PPO is learning something: the mean policy delays the fall versus zero action.
- It is not learning a stable recovery/balance controller.
- With only timeout termination, each env stays down after falling, producing
  many zero-reward lying-on-ground samples until step 256.
- The final very low KL is mostly consistent with LR annealing reaching the end
  of the schedule, not necessarily a standalone PPO bug.
- One policy DOF is structurally suspect for this task:
  `waist_pitch_joint` has `ctrlrange=[0.0, 0.8]`, `default=0.0`, so the default
  control is already at a hard lower limit. During the fall the policy commands
  negative waist pitch, which is clipped to the same lower-limit target.
- MuJoCo Playground's Berkeley humanoid locomotion tasks terminate on a fall
  instead of spending the rest of the rollout down. This supports re-enabling
  fall resets for standing, without bringing back a large event penalty.

Applied next-run changes:

```yaml
asset_config:
  actuator_config:
    joints:
      waist_yaw_joint:
        mode: hold
      waist_roll_joint:
        mode: hold
      waist_pitch_joint:
        mode: hold

config:
  anneal_lr: false

done_config:
  base_height:
    enabled: true
    min_height: 0.55
  roll_pitch:
    enabled: true
    max_abs_roll: 1.35
    max_abs_pitch: 1.35
```

Expected signal in the next run:

- `done/base_height` or `done/roll_pitch` should be nonzero once policies fall.
- `done/timeout` should rise only when policies survive the whole 256-step
  horizon.
- Action size should drop from `15` to `12`.
- Zero-action waist saturation should disappear because waist joints are held
  outside the policy action vector.

## 2026-07-08 Inspection: Run `lxjtrwr0`

Run `lxjtrwr0` used the fall-reset, held-waist standing config:

```text
action_size: 12
anneal_lr: false
done/base_height enabled: true
done/roll_pitch enabled: true
termination reward: disabled
```

Final W&B summary:

```text
mean_episode_return: 1.822
max_episode_return: 4.434
mean_reward/dm_control_stand: 0.01713
done/any: 0.00948
done/base_height: 0.00948
done/roll_pitch: 0
done/timeout: 0
approx_kl: 0.0254
action_mean_rms: 0.297
ctrl_saturation_fraction: 0.0011
```

Sampled history:

```text
~3.7M steps:  mean return 1.22, done/base_height 0.0132
~18.9M steps: mean return 1.79, max return 4.23
~25.7M steps: mean return 1.97, max return 4.30
~48.8M steps: mean return 1.84, max return 4.74
~64.7M steps: mean return 1.82, max return 4.75
final:        mean return 1.82, done/timeout 0
```

Direct deterministic checkpoint rollout:

```text
zero action:    first done at step 74, return 1.25
policy action:  first done at step 104, return 1.77
perfect horizon return: 5.12
```

Interpretation:

- The fall reset mechanics are now working: all completed episodes ended by
  `base_height`, and no policies reached timeout in the final rollout.
- Holding waist DOFs removed the zero-action saturation problem, but did not
  materially improve the deterministic policy.
- The policy still delays the fall relative to zero action, but converges to a
  local behavior that falls after about two seconds.
- With fall resets enabled and `termination` disabled, there is no one-shot cost
  for ending an episode. The rollout stream repeatedly returns to the easy
  initial upright state, so falling is only penalized indirectly by missing
  future within-episode reward. For this task that signal appears too weak.

Applied next-run change:

```yaml
reward_config:
  termination:
    enabled: true
    weight: -1.0
```

This is an event penalty, not a dense shaping term: it fires once on `done`.
Expected next-run signal:

- Early policies should have lower or even negative `mean_episode_return`
  because falls now cost one reward unit.
- `done/base_height` should decrease over training.
- `done/timeout` should become nonzero if the policy actually survives the full
  256-step horizon.

## 2026-07-08 Physics/Observation Audit

After the termination-penalty run still failed, the next inspection focused on
the plant and observation path instead of reward tuning.

Raw physics findings:

```text
default zero-action fall:       first done around step 74
best single constant DOF:       right_ankle_pitch action -2 -> step 117
best coarse sagittal constant:  knees -2, ankles -1 -> step 184
best random constant action:    step 129
```

Interpretation:

- The default reset posture is not a quiet static equilibrium. It starts with
  the feet just above the ground due to spawn clearance, contacts settle after
  the initial drop, then the COM drifts out of the foot support area.
- A hand-picked static posture can delay the fall, but did not survive the
  256-step horizon. Starting directly from that target qpos did not make a
  stable equilibrium, so this is not just "PPO failed to find one constant
  target".
- The actuator mapping itself looks sane: MuJoCo compiles position actuators to
  `kp * (ctrl - qpos) - kv * qvel`, and qpos/ctrl indices line up.

Observation-scale findings from a zero-action rollout:

```text
t=5 actuator_force maxabs: 33, rms: 8.6
t=10 actuator_force maxabs: 38, rms: 10.9
t=25 actuator_force maxabs: 22, rms: 5.9
joint_vel maxabs in the same window: roughly 0.4-2.5
base/angular velocities: roughly 0-0.7 before the actual fall
```

The policy MLP uses tanh activations and there is no observation normalization.
The policy observation included all 29 actuator forces, including held upper-body
actuators the policy cannot control. MuJoCo Playground's Berkeley humanoid policy
state includes linvel/gyro/gravity/joint angles/joint velocities/last action; it
keeps actuator forces in privileged state, not the normal actor observation.

Applied next-run observation fix:

```yaml
env_config:
  observation:
    joint_mode: policy
    actuator_force: false
```

Expected next-run signal:

- `obs_size` should drop from `109` to `46`.
- If raw force/irrelevant held-joint observations were saturating the actor, the
  policy should improve earlier than prior runs and eventually show nonzero
  `done/timeout`.
- If it still fails at the same horizon, the next suspect is the reset/default
  stance or the physical model itself, not PPO.

## 2026-07-09 Standing Failure Audit

Run `7xmd2gp7` used the reduced actor observation:

```text
obs_size: 46
policy_obs_size: 460
action_size: 12
obs_stack_size: 10
init_log_std: -2.0
final action_std_mean: 0.139
```

Final W&B summary:

```text
mean_episode_return: 1.029
max_episode_return: 3.723
mean_reward: 0.00880
mean_reward/dm_control_stand: 0.01748
mean_reward/termination: -0.00867
done/any: 0.00867
done/base_height: 0.00867
done/timeout: 0.0000038
action_mean_rms: 0.209
action_mean_abs_max: 1.266
approx_kl: 0.0109
```

Deterministic checkpoint rollout in CPU MuJoCo ended at step 103. The learned
policy did not discover the useful transient action family found by direct
probes. It mostly kept knees near the default or bent them more:

```text
step 1:
  left_knee action -0.409 -> ctrl 0.266
  right_knee action -0.284 -> ctrl 0.322
step 76:
  left_knee action +0.144 -> ctrl 0.515
  right_knee action +0.117 -> ctrl 0.503
step 103 done:
  left_knee action +0.083 -> ctrl 0.487
  right_knee action +0.185 -> ctrl 0.533
```

Direct physics probes after the failed run:

```text
collision mode comparison, full_body + sphere end-effectors:
  mesh zero action:            done 74
  mesh best transient action:  done 184
  foot-mesh zero action:       done 76
  foot-mesh best transient:    done 138
  box zero action:             done 69
  box best transient:          done 86

asset comparison:
  lower_body zero action:      done 59
  lower_body best transient:   done 91
  full_body no spheres zero:   done 73
  full_body no spheres best:   done 148
  full_body spheres zero:      done 74
  full_body spheres best:      done 184
```

Interpretation:

- `mesh` contact and sphere end-effectors are not the cause of the failure; this
  combination was the best of the tested variants.
- Lower-body-only is worse, so the issue is not simply the upper-body mass.
- Raising foot friction did not make the robot stand. The feet are not merely
  sliding away under low default friction.

Search results:

```text
constant action search:
  6000+ candidates over lower-body actions
  best: done 185
  no candidate survived the 256-step horizon

reset pose search:
  7000+ candidates with qpos == ctrl inside configured limits
  best: done 174
  no candidate survived the 256-step horizon

low-dimensional feedback-controller search:
  best: done 180
  no sampled feedback controller survived the 256-step horizon
```

The best action targets all push toward a straighter lower body, but even those
do not create a static standing equilibrium. The earlier `done 184` result is a
transient: it starts from the original reset qpos and commands straighter knee
and ankle targets. Starting directly from candidate reset poses is worse.

Additional code audit:

- PPO GAE and `done` masking still look conventional.
- Batched envs reset on `done`; training does not keep fallen states inside a
  reset episode.
- The observation stack resets when the env resets.
- MuJoCo free-joint angular velocity is already body-frame, so raw
  `base_ang_vel` is correct. `base_lin_vel` is world-frame and is correctly
  rotated into the body frame.

Applied next-run changes:

```yaml
config:
  gamma: 0.995
  ent_coef: 0.003
  policy:
    init_log_std: -0.5

reward_config:
  dm_control_stand:
    action_margin: 5.0
```

Rationale:

- The previous `init_log_std=-2.0` left final std near `0.14`; the policy stayed
  close to the bad default and did not sample the large knee/ankle actions that
  at least delay the fall.
- `gamma=0.995` makes the effective horizon less myopic for a 20 ms control
  timestep.
- `action_margin=5.0` avoids suppressing large balance actions while keeping the
  DM-Control-style small-control multiplier in place.

Expected next-run signal:

- `mean_episode_length` and `max_episode_length` are now logged explicitly.
- A useful change should push `max_episode_length` above the direct-probe
  ceiling of roughly 185 and should make `done/timeout` clearly nonzero.
- If the run still cannot exceed that ceiling, the likely next fix is not reward
  tuning but a robot model/home-stance/control-prior change.

## 2026-07-09 - Unitree-style survival-only PPO collapse

Run `standing_unitree_ppo_survival_seed0` (`pk3gu4m2`) reproduced the same
failure signature as the previous long run (`o0xh9ftv`):

```text
_step: 22,831,104
mean_reward: -0.0745
mean_episode_length: 11
max_episode_length: 11
done/base_height: 0.0945
done/roll_pitch: 0
done/timeout: 0
mean_reward/alive: 0.0200
mean_reward/termination: -0.0945
action_rms: NaN
action_mean_rms: NaN
action_std_mean: NaN
approx_kl: NaN
value_loss: NaN
```

This is not just poor reward shaping. Once action statistics, KL, and value loss
are NaN, the policy/optimizer state is numerically invalid and the rest of the
run should be ignored.

Likely cause in this specific config:

- We copied Unitree-style `learning_rate=1e-3`, `ent_coef=0.01`,
  `num_steps_per_env=24`, and `init_noise_std=1.0`, but the local PPO does not
  implement Unitree/rsl_rl's adaptive KL schedule.
- With survival-only reward, the policy receives a weak action-dependent signal.
  Entropy pressure plus the high fixed learning rate can push the Gaussian policy
  scale until sampled actions/log-probs become nonfinite.
- The configured `total_timesteps=1,228,800,000` is also too large for debugging:
  `50,000 updates * 1024 envs * 24 steps`.

Next debugging rule:

- Treat any run with NaN `action_*`, `approx_kl`, or `value_loss` as dead.
- Do not run Unitree-scale 50k iterations locally unless adaptive KL or an
  equivalent policy-scale guard is implemented.
- For short standing diagnostics, prefer a finite debug budget and either lower
  entropy/LR or implement the missing adaptive KL behavior before comparing
  reward variants.

Applied PPO stability fix:

- Added `config.schedule` with explicit `constant`, `linear`, and `adaptive`
  modes. `schedule: null` keeps the old `anneal_lr` behavior for existing
  configs.
- Added Unitree/rsl_rl-style adaptive KL controls:
  - `desired_kl: 0.01`
  - `min_learning_rate: 1e-5`
  - `max_learning_rate: 1e-2`
  - `adaptive_lr_factor: 1.5`
- Replaced hidden Optax LR scheduling with an explicit LR carried in
  `RunnerState`; W&B now logs `train/learning_rate` and
  `train/next_learning_rate`.
- Added policy `min_log_std` / `max_log_std` bounds and clamp the learned
  Gaussian log-std after each PPO minibatch update.
- Switched PPO `approx_kl` to the nonnegative ratio-based estimator and clips
  log-ratios before exponentiation to avoid overflow.
- Wrapped the optimizer in Optax finite-update protection so nonfinite gradients
  are not applied to parameters.
- Updated `experiments/train/standing.yaml` to use `schedule: adaptive` and the
  new run/checkpoint/video prefix
  `standing_unitree_ppo_survival_adaptive_seed0`.

Verified W&B result:

- Run: `standing_unitree_ppo_survival_adaptive_seed0`
- Run id: `1ccjt78y`
- URL: `https://wandb.ai/gyungmin/bello-mujoco/runs/1ccjt78y`
- At `_step = 17,178,624`:
  - `mean_episode_length = 216.1`
  - `max_episode_length = 256`
  - `mean_done/timeout = 0.00240`
  - `mean_done/base_height = 0.00106`
  - `mean_done/roll_pitch = 0.00122`
  - `mean_reward = 0.0154`
  - `mean_reward/alive = 0.0200`
  - `mean_reward/termination = -0.00456`
  - `action_std_mean = 2.28`
  - `ctrl_saturation_fraction = 0.383`
  - `approx_kl = 0.0132`
  - `value_loss = 0.0330`
  - `learning_rate = 1.98e-4`
- The immediately previous non-adaptive Unitree-style survival run
  `pk3gu4m2` collapsed to `mean_episode_length = 11` with NaN action, KL, and
  value metrics by `_step = 25,141,248`.

Conclusion: the adaptive-KL / policy-scale guard was the missing PPO stability
piece for this recipe. It did not merely improve the metric slightly; it moved
survival-only standing from numerical collapse to frequent full-horizon
episodes under the same reward family.

Applied next recipe change:

- Re-enabled `reward_config.dm_control_stand` with `weight: 1.0`.
- Kept `alive` and `termination` enabled, so the objective for that run became:
  - continuous alive reward
  - continuous DM-Control-style stand/upright/small-motion reward
  - event-driven termination penalty
- Kept `dm_control_stand.action_margin: 5.0` to avoid suppressing the larger
  balance actions that were needed for survival.
- Renamed run/checkpoint/video prefix to
  `standing_unitree_ppo_dm_control_stand_adaptive_seed0`.
- Expected signal: `mean_episode_length` should stay near the successful
  survival-only adaptive run, while `mean_reward/dm_control_stand` should become
  nonzero and increase as the policy learns a taller/upright posture.

Observed active DM-control adaptive run:

- Run: `standing_unitree_ppo_dm_control_stand_adaptive_seed0`
- Run id: `8gglk654`
- Around `_step = 10M`, the run reached its best sampled reward:
  - `mean_reward ~= 0.0327`
  - `mean_episode_length ~= 217`
  - `mean_reward/dm_control_stand ~= 0.0167`
  - `action_std_mean ~= 1.55`
  - `ctrl_saturation_fraction ~= 0.29`
- By `_step = 37M`, the run had not numerically collapsed, but reward and
  posture quality drifted down:
  - `mean_reward ~= 0.0302`
  - `mean_episode_length ~= 214`
  - `mean_reward/dm_control_stand ~= 0.0147`
  - `action_std_mean ~= 3.14`
  - `ctrl_saturation_fraction ~= 0.48`

Interpretation:

- `max_episode_length = 256` means some episodes time out; it does not mean the
  policy reliably stands.
- The policy found a useful survival behavior, then continued becoming more
  stochastic because `ent_coef = 0.01` still rewarded entropy.
- Adaptive KL prevented NaNs, but it did not prevent late high-variance actions
  and actuator saturation from degrading the DM-control posture reward.

Applied stabilization follow-up:

- Reduced `config.ent_coef: 0.01 -> 0.003`.
- Tightened `config.policy.max_log_std: 2.0 -> 1.0`, capping learned action std
  at `exp(1) ~= 2.72` instead of `exp(2) ~= 7.39`.
- Reduced `config.total_timesteps` from `1,228,800,000` to `49,152,000`, i.e.
  `2,000 updates * 1,024 envs * 24 steps`.
- Renamed run/checkpoint/video prefix to
  `standing_unitree_ppo_dm_control_stand_stable_seed0`.
- Expected signal: preserve the early standing improvement while preventing
  late entropy-driven growth in `action_std_mean` and `ctrl_saturation_fraction`.

Value-loss follow-up on the stable DM-control run:

- Active run `standing_unitree_ppo_dm_control_stand_stable_seed0` showed
  value loss rising from roughly `0.01-0.02` early to `0.10-0.12` while
  `mean_episode_return` rose from about `1.2` to `8.5+` and
  `mean_episode_length` rose from about `60` to `245`.
- This does not look like numerical divergence. It looks like critic lag after
  the actor found longer-surviving rollouts and shifted the return target
  distribution upward.
- Applied `config.vf_coef: 1.0` for the standing recipe to match the usual
  rsl_rl value-loss coefficient and give the critic more weight while fitting
  the new return scale.
- Expected signal: similar or better episode length/reward, with value loss
  flattening sooner instead of continuing to climb as returns improve.

Applied value-normalization follow-up:

- Added PPO value normalization and enabled it in `experiments/train/standing.yaml`
  with `config.normalize_value: true`.
- The critic now learns normalized value targets when enabled, while rollout and
  GAE denormalize critic predictions back into raw reward units.
- Added W&B metrics:
  - `train/value_loss`: normalized clipped value loss when value normalization is
    enabled
  - `train/value_loss_raw`: raw return-unit critic error for interpretability
  - `train/value_norm_mean`, `train/value_norm_std`, `train/value_norm_count`
- Renamed run/checkpoint/video prefix to
  `standing_unitree_ppo_dm_control_stand_valuenorm_seed0`.
- Expected signal: `value_loss` should be easier to interpret across the run,
  while `value_loss_raw` may still rise when the policy discovers longer
  survival rollouts.

Applied timeout/termination semantics fix:

- Found that `done/timeout` was included in the same `done` bit used by
  `reward/termination` and PPO GAE masking.
- This meant a successful episode-length timeout received the event-driven
  termination penalty and was treated as a zero-value terminal state.
- Added `done/terminal` for true failure conditions: base height, roll/pitch, or
  nonfinite state.
- Kept `done/any` as the reset/completed-episode flag, so timeouts still reset
  envs and are logged as completed episodes.
- Changed `reward/termination` to use `done/terminal`, so timeouts no longer get
  the termination penalty.
- Changed PPO GAE to bootstrap timeout transitions from the pre-reset timeout
  observation while still cutting lambda continuation across the reset boundary.
- Expected signal: `mean_reward/termination` should track actual falls, not
  successful horizon timeouts; timeout-heavy successful runs should have less
  value-target distortion.

Applied native additive standing reward follow-up:

- Removed the special `dm_control_stand` reward term from the package reward
  config and from the active standing recipe.
- Reformulated the standing objective using native reward-pipeline terms:
  - `alive: weight 1.0`
  - `track_lin_vel_xy: weight 0.5`, with command disabled so this tracks zero
    horizontal velocity
  - `track_ang_vel_z: weight 0.25`, tracking zero yaw velocity
  - `base_height: weight -4.0, target 0.86`
  - `flat_orientation: weight -1.0`
  - `lin_vel_z: weight -0.5`
  - `ang_vel_xy: weight -0.1`
  - `joint_vel: weight -0.0005`
  - `torque: weight -0.00001`
  - `action_rate: weight -0.01`
  - `termination: weight -1.0`, now only for true terminal failures, not timeout
- Kept the penalties intentionally modest because all continuous terms are
  multiplied by `control_timestep`; the objective should remain mostly positive
  for near-standing rollouts instead of encouraging early failure to avoid
  large per-step penalties.
- Renamed run/checkpoint/video prefix to
  `standing_unitree_ppo_native_stand_valuenorm_seed0`.
- Expected signal: no `mean_reward/dm_control_stand` metric should be emitted.
  Native terms should show `track_lin_vel_xy` and `track_ang_vel_z` positive,
  while `base_height`, `flat_orientation`, `lin_vel_z`, `ang_vel_xy`,
  `joint_vel`, `torque`, and `action_rate` move toward zero as standing
  stabilizes.

Default PPO/policy config follow-up:

- Moved the reusable PPO/policy defaults from `experiments/train/standing.yaml`
  into `configs/train/default.yaml` and the Python dataclass defaults:
  - adaptive KL schedule with `learning_rate: 0.001`, `desired_kl: 0.01`
  - `num_steps: 24`, `update_epochs: 5`, `num_minibatches: 4`
  - `ent_coef: 0.003`, `vf_coef: 1.0`, `max_grad_norm: 1.0`
  - `normalize_value: true`
  - `policy.init_log_std: 0.0`, `policy.max_log_std: 1.0`
  - `policy.obs_stack_size: 10`
- Changed `standing.yaml` to inherit `default` directly instead of
  `rtx5090_4096x256`, so those PPO/policy values are inherited rather than
  repeated locally.
- Left standing-specific settings in `standing.yaml`: 1,024 envs, 49.152M total
  steps, standing observations, disabled command, standing reward, done limits,
  actuator overrides, and logging/checkpoint names.
