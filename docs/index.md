# Bello MuJoCo reference

This repository provides Bello asset preparation, MuJoCo and MJX-Warp
simulation, PPO locomotion and whole-body motion training, checkpoint
evaluation, Vast.ai training orchestration, and a RobStride hardware-in-the-loop
emulator.

[README.md](../README.md) is the quick start. This page defines the supported
surface and routes to the workflow-specific documentation.

## Supported commands

The installed command-line interface is intentionally small:

```text
bello-prepare-assets
bello-train-ppo
bello-eval-policy
bello-import-gmr-motion
bello-import-gmr-motion-dataset
bello-render-motion-reference
bello-check-motion-feasibility
bello-train-motion-ppo
bello-eval-motion-policy
bello-vast-train
bello-robstride-runner
bello-run-locomotion-hardware
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
pelvis-mounted IMU-frame linear acceleration[3]
pelvis-mounted IMU-frame angular velocity[3]
projected gravity[3]
left/right foot phase sin/cos[4]
command[3]
policy joint position[13]
policy joint velocity[13]
last action[13]
```

With commands enabled this is 55 values per frame. The IMU terms have the same
specific-force and gyro semantics as the configured pelvis-mounted MuJoCo
accelerometer and gyro. The privileged critic adds root-frame linear velocity,
terrain-relative root clearance, and policy-actuator force for 72 values per
frame. The package training default stacks ten frames, producing actor and
critic inputs of 550 and 720 values. Observation noise and scaling do not change
the layout.

The generated IMU site is the source of truth for its body frame. When sensor
generation is disabled, model loading carries the same body name from the asset
configuration; the environment does not expose an independent frame override.

The default actor scales linear acceleration by `0.1`, angular velocity by
`0.2`, and joint velocity by `0.05`. The critic uses the same shared-component
scales, plus `2.0` for root linear velocity and `0.01` for actuator force. These
fixed physical scales replace mutable online normalization statistics.

Environment configuration covers:

- actor and optional privileged critic observations;
- observation scaling, noise, and temporal stacking;
- velocity commands, standing samples, and periodic command resampling;
- reset distributions, pushes, contact filtering, and episode limits;
- numerical bounds for finite physics excursions and constraint forces;
- flat or seeded-slope planes and rough terrain;
- joint-origin, target-offset, and inertia randomization;
- reward, termination, gait, and safety terms.

The rollout and reward hot path remains JAX-compatible. Asset generation,
checkpoint I/O, W&B, rendering, and CLI setup remain host-side operations.

## Whole-body motion tracking

The motion task fixes waist roll/pitch and neck yaw/head pitch, replaces both
hands with spheres, uses box body collisions and primitive feet, and exposes 25
logical policy joints at 50 Hz over 200 Hz simulation. Differential ankle
targets are transformed into the always-active physical tendon motor
coordinates.

Two formulations share the hardened PPO trainer: next-reference residual
tracking and the paper-aligned BeyondMimic tracker. The latter retains its
nominal-centered Gaussian actions, asymmetric normalized observations,
hybrid-body targets, perturbations, pushes, and adaptive failure-bin sampling.

Tracked ASAP balance examples can be rendered or replayed directly:

```bash
uv run bello-render-motion-reference \
  motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz \
  outputs/motion_tracking/reference_balance.mp4

uv run bello-check-motion-feasibility \
  motions/asap_fixed_head/video_single_foot_balance_level1_filter_amass.npz \
  --output outputs/motion_tracking/balance_feasibility.json
```

The complete ASAP and fixed-head LAFAN1 datasets are regenerated locally under
the ignored `motions/local/` tree. Only the 25-joint fixed-head schema is
supported; the old 27-joint LAFAN1 export is explicitly unsupported. LAFAN1
retargets are not distributed or silently uploaded because the source is
CC BY-NC-ND 4.0. See [motion_tracking.md](motion_tracking.md) for the named
observation/action layouts, provenance, import commands, maintained presets,
checkpoint contract, and evaluation workflow.

## Training

The maintained experiment recipes are:

```text
experiments/train/smoke.yaml
experiments/train/locomotion.yaml
experiments/train/hardware_locomotion.yaml
```

The smoke recipe exercises the real Bello/MJX-Warp PPO path with a small
budget. The standard recipe is the maintained locomotion baseline and uses one
seeded sloped plane per run, viewed from randomized robot headings. It preserves
the physical ankle transmission and joint-space policy contract; longer
experiments override its training budget explicitly rather than duplicating the
recipe. The hardware recipe inherits that algorithm while fixing waist
roll/pitch and neck/head joints to match the physical sphere-hand 25-motor
inventory.

Run a recipe with:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion
```

Training uses a local JAX PPO implementation with Equinox policies and Optax.
It supports separate actor and critic observations, observation stacking and
explicit scaling, GAE, configurable learning-rate schedules, KL guardrails,
invalid-update rejection, Orbax checkpoints, W&B metrics, and rollout video.

See [training.md](training.md) for the maintained training contract and smoke
commands.

## Checkpoints and evaluation

Atomic checkpoints contain policy/optimizer state, exact update-boundary
continuation state, strict metadata, and fingerprints for source/generated
assets and their referenced files, exact runtime scenes, implementation code,
`uv.lock`, and active transitive runtime distribution versions. Evaluate one
with:

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

The evaluator can run CPU MuJoCo or a single MJX-Warp environment and streams a
live 3D scene and diagnostics to Rerun. It supports deterministic or stochastic
actions, a fixed command, and live SDL gamepad commands. See
[evaluation.md](evaluation.md).

Motion checkpoints additionally bind the formulation and motion content hash,
normalization and adaptive-sampler state, and named observation/action layouts.
`bello-eval-motion-policy` rejects locomotion checkpoints and incompatible or
tampered motion archives.

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

Locomotion is the default task. Motion runs require `--task motion-tracking`
and a compatible motion recipe. Repository-tracked clips may be used remotely;
ignored local datasets are never implicitly added to an upload.

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

## Physical locomotion

`bello-run-locomotion-hardware` deploys a compatible locomotion checkpoint
through the blocking Python client supplied by `~/bello_stack`. It validates the
complete 25-coordinate inventory, group rates and state, online motors and IMU,
checkpoint morphology, command bounds, telemetry freshness, initial pose, and
tilt before control. The default invocation is a read-only preflight and cannot
acquire a control lease.

The armed path starts from measured motor positions, ramps to the policy nominal
pose, runs deterministic inference at 50 Hz, and relies on both local limits and
the daemon's command/lease fault handling. See
[hardware_locomotion.md](hardware_locomotion.md) before using physical hardware.

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
./scripts/test.sh
```

The full runner uses a fresh CPU JAX process for each test module to bound
compiler-cache memory. Run a focused module directly with `uv run pytest`.

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
artifacts/
outputs/
motions/local/
/tmp/bello-mujoco-assets/
```
