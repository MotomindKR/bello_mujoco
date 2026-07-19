# Bello MuJoCo reference

This repository provides Bello asset preparation, MuJoCo and MJX-Warp
simulation, PPO locomotion training, checkpoint evaluation, Vast.ai training
orchestration, and a RobStride hardware-in-the-loop emulator.

[README.md](../README.md) is the quick start. This page defines the supported
surface and routes to the workflow-specific documentation.

## Supported commands

The installed command-line interface is intentionally small:

```text
bello-prepare-assets
bello-train-ppo
bello-eval-policy
bello-vast-train
bello-robstride-runner
```

All Python commands work as `uv run <command>` outside an activated development
shell. Use `<command> --help` and `<command> --hydra-help` for the current config
tree rather than relying on copied option lists.

## Robot assets

The only robot source of truth is:

```text
assets/source/bello/
```

The only prepared/runtime asset is `full_body`. Prepare it with:

```bash
uv run bello-prepare-assets asset=full_body
```

The tracked `assets/source/bello/lower_body/` and
`assets/source/bello/upper_body/` directories are source material for the
full-body description, not standalone prepared asset choices.

The preparation pipeline validates and converts the tracked descriptions,
simplifies visual and collision meshes, creates training and viewer MJCF,
configures actuators, optionally replaces hands with spheres, and always
generates and actuates the physical differential ankle tendons. Runtime terrain
is added after asset generation; it is not embedded in the source descriptions.

Prepared assets are deterministic cache artifacts under
`/tmp/bello-mujoco-assets/<config-hash>/` by default. They are not source files
and must not be committed.

## Simulation environment

`BelloLocomotionEnv` provides single-environment and vectorized APIs:

```python
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv

env = BelloLocomotionEnv.load(AssetPrepConfig(asset="full_body"))
state, observation = env.reset()
state, observation, reward, done, info = env.step(state, action)

states, observations = env.reset_batch(batch_size=512)
states, observations, rewards, dones, info = env.step_batch(states, actions)
```

The training backend is MJX-Warp. Native CPU MuJoCo is used for deterministic
evaluation, rendering, and HIL. The implementations share prepared models,
control semantics, runtime physics overrides, and observation/reward contracts.

Policy actions are normalized logical joint-position targets. Action zero keeps
the nominal target, while `-1` and `+1` reach the lower and upper logical soft
limits respectively. The environment
clamps logical targets to configured limits, applies any transmission mapping,
and defensively clamps actuator-coordinate targets before simulation. Held
joints remain under configured impedance control without appearing in the
policy action vector. Differential ankles retain a joint-space policy interface
while the model uses coupled tendon motor coordinates.

The canonical sphere-hand training config exposes 13 policy joints and holds
the remaining 16 actuated joints. Wrist-yaw and finger coordinates are absent
from that prepared morphology. Its deployable actor frame is:

```text
root-frame linear acceleration[3]
root-frame angular velocity[3]
projected gravity[3]
left/right foot phase sin/cos[4]
command[3]
policy joint position[13]
policy joint velocity[13]
last action[13]
```

With commands enabled this is 55 values per frame. The privileged critic adds
root-frame linear velocity, root height, and policy-actuator force for 72 values
per frame. The package training default stacks ten frames, producing actor and
critic inputs of 550 and 720 values. Observation noise and scaling do not change
the layout.

Environment configuration covers:

- actor and optional privileged critic observations;
- observation scaling, noise, and temporal stacking;
- velocity commands, standing samples, and periodic command resampling;
- reset distributions, pushes, contact filtering, and episode limits;
- plane, randomized-slope, and rough terrain;
- joint-origin, target-offset, and inertia randomization;
- reward, termination, gait, and safety terms.

The rollout and reward hot path remains JAX-compatible. Asset generation,
checkpoint I/O, W&B, rendering, and CLI setup remain host-side operations.

## Training

The maintained experiment recipes are:

```text
experiments/train/smoke.yaml
experiments/train/locomotion.yaml
experiments/train/locomotion_plane_slope.yaml
```

The smoke recipe exercises the real Bello/MJX-Warp PPO path with a small
budget. The standard recipe is the maintained locomotion baseline. The
plane-slope recipe is a longer treatment with independently tilted planes and
different optimizer/reward settings. Both preserve the same physical ankle
transmission and joint-space policy contract.

Run a recipe with:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion
```

Training uses a local JAX PPO implementation with Equinox policies and Optax.
It supports separate actor and critic observations, observation stacking and
normalization, GAE, configurable learning-rate schedules, KL guardrails,
nonfinite-update rejection, Orbax checkpoints, W&B metrics, and rollout video.

See [training.md](training.md) for the maintained training contract and smoke
commands.

## Checkpoints and evaluation

Atomic checkpoints contain policy/optimizer state, exact update-boundary
continuation state, and strict metadata needed to rebuild the policy and
environment. Evaluate one with:

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

The evaluator can run CPU MuJoCo or a single MJX-Warp environment and streams a
live 3D scene and diagnostics to Rerun. It supports deterministic or stochastic
actions, a fixed command, and live SDL gamepad commands. See
[evaluation.md](evaluation.md).

## Vast.ai

`bello-vast-train` runs one complete, branch-pinned remote training lifecycle:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion.yaml \
  --max-price 1.25
```

It validates the selected commit and recipe, rents a compatible GPU instance,
sets up the locked environment, runs training with W&B and checkpoint upload,
and destroys the instance. See [vast_ai.md](vast_ai.md) for credentials,
recovery, and billing-safe teardown behavior.

## RobStride HIL

`bello-robstride-runner` exposes configured Bello actuators as RobStride
devices over SocketCAN while CPU MuJoCo supplies rigid-body dynamics. It covers
the platform-visible impedance-control and feedback protocol, coupled
transmissions, and optional IMU serial output.

```bash
./scripts/setup_hil.sh
uv run bello-robstride-runner
```

See [robstride_vcan.md](robstride_vcan.md).

## Configuration

Training, asset preparation, and evaluation use Hydra over typed dataclass
configs. Nested overrides use dotted paths:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name smoke \
  config.seed=1
```

Unknown runtime config keys are rejected. Maintained experiment recipes inherit
the package training defaults and override only their explicit treatment.
Checkpoint evaluation restores runtime configuration from metadata and rejects
observation or action layout mismatches.

## Validation

Run the repository gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

For a real simulation and optimizer smoke, use the maintained smoke recipe.
Focused tests are organized by subsystem under `tests/`, including assets,
model/control, contacts, observations/rewards, PPO, checkpoints, evaluation,
Vast lifecycle handling, RobStride CAN, and IMU behavior.

## Generated files

Do not commit generated environments, caches, prepared assets, checkpoints,
W&B runs, or videos. Common paths are:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
wandb/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/
```
