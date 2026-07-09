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
the stabilized standing setup:

- full-body asset with sphere end effectors
- lower body and waist yaw controlled by policy
- waist roll/pitch, head, arms, and fingers held
- `env_config.root_link: torso_link`
- `env_config.contact.foot_contact_on_seconds: 0.04`
- `env_config.contact.foot_contact_off_seconds: 0.04`
- `num_envs: 4096`
- `num_steps: 150`, equivalent to `150 * 0.02s = 3.0s`
- `max_episode_steps: 512`

Command curriculum is intentionally modest:

```yaml
command_config:
  enabled: true
  lin_vel_x: [0.0, 0.4]
  lin_vel_y: [-0.05, 0.05]
  ang_vel_yaw: [-0.2, 0.2]
  standing_probability: 0.2
  resample_seconds: 6.0
```

## Gait Reward Adjustment

Initial locomotion config enabled `gait_contact` with `weight: -0.2`. This
penalizes mismatch between detected foot contacts and the deterministic gait
schedule. The calculation is:

```text
mean((foot_contact - gait_contact_mask) ** 2) * weight * dt
```

With two feet, one off-schedule contact state gives:

```text
0.5 * -0.2 * 0.02 = -0.002 per control step
```

This is not a large absolute penalty, but it is dense and phase-specific. Early
exploration steps are unlikely to match the fixed gait phase, so this can
discourage discovering any stepping behavior before the policy has learned a
coherent gait.

Change:

```yaml
reward_config:
  gait_contact:
    enabled: false
```

Revisit `gait_contact` only after the policy reliably moves its feet and tracks
forward commands. At that point it can become a shaping/cadence term instead of
a discovery barrier.

## Air-Time Adjustment

`feet_air_time` is sparse: it pays only on first contact after a swing interval,
not every step while the foot is airborne. The previous `weight: 0.05` was too
small relative to dense penalties and velocity reward available from standing
still.

Change:

```yaml
reward_config:
  feet_air_time:
    enabled: true
    weight: 0.2
    min_air_time: 0.15
    max_air_time: 0.60
```

For a foot landing after `0.60s` of air time, the per-foot bonus is:

```text
(0.60 - 0.15) * 0.2 = 0.09
```

Because the implementation averages over two feet, a single-foot landing yields
about `0.045` event reward. This is large enough to make stepping discovery
visible without requiring the fixed gait-contact schedule.

## Foot Clearance Sign

`foot_clearance` has a negative weight because it is implemented as a penalty
for insufficient swing-foot height:

```text
swing * max(target_height - foot_height, 0) ** 2
```

The term is zero when the swing foot is at or above the clearance target and
negative when the swing foot is too low. So the negative sign does not penalize
lifting the foot; it penalizes dragging or low swing clearance.

## Termination Scale

Early locomotion run `h83zq3jp` showed frequent base-height terminal failures:

```text
mean_episode_length: about 74.5 steps
done/terminal:       about 0.0134 per step
reward/alive:        about +0.0025 per step
reward/termination:  about -0.0033 per step
```

With `termination.weight: -0.25`, the average failed episode received only
about `-0.25` event penalty. The accumulated alive reward over a 74.5-step
episode was about `+0.186`, so alive plus termination was only mildly negative
per failure. Dense velocity tracking before falling could still dominate this.

Change:

```yaml
reward_config:
  termination:
    weight: -0.5
```

Keep `alive.weight: 0.125` unchanged for this test. Increasing alive at the same
time would make short survival before falling more attractive.
