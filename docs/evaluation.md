# Evaluation Details

`bello-eval-policy` restores a PPO checkpoint, runs one real-time CPU MuJoCo
environment, reads commands from an SDL game controller, and streams the robot
scene and diagnostics to a spawned Rerun Viewer.

## Running Evaluation

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

Rerun opens separately. Evaluation runs until Start, `Ctrl-C`,
`eval_config.steps`, or a done condition with `stop_on_done=true`.
It writes no MP4, W&B, or `.rrd` artifact; training MP4s remain unchanged.

The default `mujoco` backend steps one CPU MuJoCo environment. To visualize a
checkpoint through the MJX-Warp physics backend used by training, select the
single-environment `mjx` backend:

```bash
uv run bello-eval-policy \
  checkpoint_path=checkpoints/ppo_seed_0 \
  eval_config.simulation_backend=mjx
```

MJX runs on the active JAX device. Rerun still receives a synchronized CPU
`MjData` copy for scene transforms and diagnostics; that copy is not stepped and
does not affect the rollout.

## Gamepad Controls

SDL normalizes Xbox, DualShock/DualSense, and compatible controllers:

- left stick X: `lin_vel_x`
- inverted left stick Y: `lin_vel_y` (forward is positive)
- inverted right stick X: yaw (pushing right turns right)
- A / Cross: reset the episode
- Start / Options: exit

Input inside the deadzone is zero. Remaining input is rescaled and mapped
independently to the positive or negative command bound stored in checkpoint
metadata. If the controller is absent or disconnects, commands immediately
become zero while evaluation scans for reconnection. Connection, reset, done,
and exit events are logged in Rerun.

Disable `eval_config.gamepad.enabled` to use the fixed
`eval_config.command=[lin_vel_x,lin_vel_y,yaw_vel]` value.

## Config

```yaml
checkpoint_path: Path
eval_config:
  command: [0.0, 0.0, 0.0]
  steps: null
  stop_on_done: false
  simulation_backend: mujoco  # mujoco | mjx
  stochastic_actions: false
  action_seed: 0
  gamepad:
    enabled: true
    controller_index: 0
    deadzone: 0.1
  runtime:
    source: checkpoint  # checkpoint | override
```

`steps=null` runs until stopped. Evaluation is always paced by the environment
control timestep. Deterministic evaluation uses the actor mean;
`stochastic_actions=true` samples the Gaussian policy with `action_seed`.

`runtime.source=checkpoint` restores asset, environment, reward, command, done,
and Warp configuration from checkpoint metadata. `override` merges changed
nested values over that metadata and must preserve observation/action sizes.

## Rerun Data

The default blueprint provides:

- a Z-up 3D prepared URDF with a root-following orbit camera, terrain, and
  animated root/joints
- reward totals/components and episode diagnostics
- commanded and measured linear/yaw velocities overlaid in one chart
- one tab per policy joint overlaying measured joint position, logical joint
  target, applied normalized policy action, and associated actuator force
- aggregate actor means, applied actions, and control saturation telemetry
- joint positions/velocities and actuator forces
- done flags, controller connection state, and an event log

Data has synchronized monotonic step and simulation-time timelines. Rerun can
pause, scrub, and replay received data without pausing the live simulation.

## Runtime Behavior

The evaluator loads one visual environment. The `mujoco` backend advances a CPU
`MjData`; the `mjx` backend loads MJX-Warp with `batch_size=1` and advances an
MJX state through `BelloLocomotionEnv.step`. Each real-time control step polls
the controller, evaluates and clips the policy, advances the selected backend,
and streams transforms and telemetry. Done episodes reset automatically unless
`stop_on_done=true`; automatic and gamepad resets rebuild observation and
contact memory.

## Validation

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-eval-policy --help
uv run pytest tests/test_eval_policy.py tests/test_hydra_config.py -q
```

For manual verification, connect a supported controller and run a checkpoint.
Confirm motion and plots stay synchronized, A/Cross resets, Start exits, and a
disconnect immediately zeros all command channels.
