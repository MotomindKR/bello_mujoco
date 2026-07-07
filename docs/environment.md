# Environment

## Main Class

```python
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv

env = BelloLocomotionEnv.load(AssetPrepConfig(asset="full_body"))
```

Single-env API:

```python
state, obs = env.reset()
state, obs, reward, done, info = env.step(state, action)
```

Vectorized API:

```python
state, obs = env.reset_batch(batch_size=512)
state, obs, reward, done, info = env.step_batch(state, actions)
```

`step_batch` is `jax.vmap` over `step` plus vectorized auto-reset for done envs.

## State

`BelloState` contains:

```text
data                  MJX data
last_action           previous policy action
command               [lin_vel_x, lin_vel_y, yaw_vel]
step_count
episode_return
episode_length
last_episode_return
last_episode_length
```

The command field remains a fixed 3-vector even when commands are disabled. When command config is disabled, it is always zero.

## Action Space

Actions are residual position targets for policy actuators:

```text
ctrl[policy_ctrl_indices] = default_ctrl + policy_action_scale * action
```

The result is clipped to MuJoCo actuator ctrl ranges. Input actions are clipped by `ACTION_CLIP = 10.0`.

Default full-body:

```text
nu: 35
action_size: 35
```

If joints are configured as `mode="hold"`, those actuator indices are removed from the policy action vector. The env still fills held actuator controls from `default_ctrl`.

## Observation Space

Default full-body observation:

```text
base_ang_vel[3]
projected_gravity[3]
command[3]
joint_pos[35]
joint_vel[35]
actuator_force[35]
last_action[35]
```

Default full-body size:

```text
149
```

Observation components are controlled by `ObservationConfig`:

```yaml
joint_mode: all_actuated | policy
base_ang_vel: true
projected_gravity: true
command: true
joint_pos: true
joint_vel: true
actuator_force: true
last_action: true
```

Examples:

```bash
bello-train-ppo --env-config.observation.no-command
bello-train-ppo --env-config.observation.no-actuator-force
bello-train-ppo --env-config.observation.joint-mode policy
```

`joint_mode` affects `joint_pos`, `joint_vel`, and `actuator_force`:

- `all_actuated`: observe all actuators.
- `policy`: observe only policy actuator indices.

## Commands

Commands are `[lin_vel_x, lin_vel_y, yaw_vel]`.

Use command sampling:

```bash
bello-train-ppo \
  --command-config.lin-vel-x -0.2 0.6 \
  --command-config.lin-vel-y -0.2 0.2 \
  --command-config.ang-vel-yaw -0.3 0.3
```

Hide commands from the policy while still using them for reward:

```bash
bello-train-ppo --env-config.observation.no-command
```

Disable command generation entirely:

```bash
bello-train-ppo --command-config.no-enabled
```

When command generation is disabled, resets ignore explicit command input and RNG command sampling.

## Rewards

Reward terms live in `RewardConfig`. Each term has its own config object. Most terms have:

```yaml
enabled: bool
weight: float
```

Link-aware terms also have:

```yaml
link: body_name
```

Tracking/target terms add term-specific fields such as `sigma` or `target`.

Default reward terms:

```text
track_lin_vel_xy
track_ang_vel_z
alive
lin_vel_z
ang_vel_xy
flat_orientation
base_height
joint_vel
action_rate
energy
joint_limit
hold_joint_deviation
foot_contact
foot_slip
foot_clearance
undesired_contact
termination
```

Example:

```bash
bello-train-ppo \
  --reward-config.track-lin-vel-xy.weight 1.2 \
  --reward-config.track-lin-vel-xy.sigma 0.2 \
  --reward-config.base-height.target 0.75
```

Disabled reward terms are ignored by link/body validation.

## Done Conditions

Done terms live in `DoneConfig`:

```text
base_height
roll_pitch
timeout
nonfinite
```

Example:

```bash
bello-train-ppo \
  --done-config.base-height.min-height 0.75 \
  --done-config.roll-pitch.max-abs-roll 0.8 \
  --done-config.roll-pitch.max-abs-pitch 0.8
```

Disable terms:

```bash
bello-train-ppo \
  --done-config.base-height.no-enabled \
  --done-config.roll-pitch.no-enabled
```

Disabled done terms are ignored by link/body validation.

## Terrain

Terrain is added at runtime, not in source URDFs.

Modes:

```text
none
plane
rough
```

The locomotion env defaults to a flat plane. Use rough terrain:

```bash
bello-train-ppo \
  --env-config.terrain.mode rough \
  --env-config.terrain.seed 3 \
  --env-config.terrain.hfield-amplitude 0.04 \
  --env-config.terrain.hfield-resolution 65
```

The rough heightfield is fixed for a loaded model and shared across vectorized envs. Change `env_config.terrain.seed` between training runs to change the generated terrain scene.

## Split-Body Assets

`lower_body` loads with default locomotion terms because feet exist.

`upper_body` has no foot links. Disable foot rewards and link-dependent base terms or retarget them:

```bash
bello-train-ppo \
  --asset-config.asset upper_body \
  --reward-config.foot-contact.no-enabled \
  --reward-config.foot-slip.no-enabled \
  --reward-config.foot-clearance.no-enabled \
  --reward-config.track-lin-vel-xy.no-enabled \
  --reward-config.track-ang-vel-z.no-enabled \
  --reward-config.lin-vel-z.no-enabled \
  --reward-config.ang-vel-xy.no-enabled \
  --reward-config.flat-orientation.no-enabled \
  --reward-config.base-height.no-enabled \
  --done-config.base-height.no-enabled \
  --done-config.roll-pitch.no-enabled
```

## Foot Metadata

Foot bodies are detected by body name:

```text
name.endswith("ankle_roll_link")
```

Current foot links:

```text
left_ankle_roll_link
right_ankle_roll_link
```

Foot metadata drives contact, slip, and clearance rewards.
