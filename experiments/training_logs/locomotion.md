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
- `env_config.contact.foot_contact_on_seconds: 0.04`
- `env_config.contact.foot_contact_off_seconds: 0.04`
- `num_envs: 4096`
- `num_steps: 24`, equivalent to `24 * 0.02s = 0.48s`
- `total_timesteps: 4_915_200_000`, equivalent to 50,000 PPO updates
- `max_episode_steps: 1000`, equivalent to `20s`
- inherited PPO entropy coefficient `ent_coef: 0.01`, matching Unitree/RSL-RL
- `policy.hidden_dims: [512, 256, 128]`
- `policy.activation: elu`
- `policy.obs_stack_size: 5`, matching Unitree's policy observation history

The policy observation group now follows Unitree more closely by omitting
`base_lin_vel`, `base_height`, and `gait_phase`. Bello still uses the same
observation for actor and critic; Unitree uses a larger privileged critic
observation group, which is not implemented locally yet.

The locomotion command distribution now starts with the narrow Unitree G1 range
and expands by curriculum:

```yaml
command_config:
  enabled: true
  lin_vel_x: [-0.1, 0.1]
  lin_vel_y: [-0.1, 0.1]
  ang_vel_yaw: [-0.1, 0.1]
  standing_probability: 0.02
  resample_seconds: 10.0
  curriculum:
    enabled: true
    limit_lin_vel_x: [-0.5, 1.0]
    limit_lin_vel_y: [-0.3, 0.3]
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

`foot_clearance` is no longer a low-swing penalty. It computes:

```text
exp(-sum((foot_height - target) ** 2 * tanh(tanh_mult * lateral_speed)) / std)
```

This makes the term positive and target-seeking for moving feet, while suppressing
clearance pressure for stationary feet.

## Unitree G1 Weight Alignment

The locomotion recipe now follows the Unitree G1 reward weights where the local
pipeline has an equivalent term. The local reward implementation also follows
the Unitree raw-value reductions for these terms:

- yaw-frame linear velocity tracking
- summed joint velocity, joint acceleration, action-rate, and energy penalties
- summed absolute joint-limit violation
- summed contact-gated lateral foot speed for `foot_slip`

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

- Unitree G1 uses `base_height.target_height: 0.78`; Bello keeps `0.86`.
- Unitree uses a privileged critic observation group. Bello currently feeds the
  same stacked policy observation to both actor and critic.
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
