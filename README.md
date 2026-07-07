# bello_mujoco

Minimal Nix shell plus uv-managed Python dependencies for MuJoCo/MJX robot-learning work.

## Setup

```bash
nix develop
```

Nix only provides `uv` and Git. The shell hook installs uv-managed Python 3.12, runs `uv sync`, and activates `.venv`. Python and Python packages are managed by uv from [pyproject.toml](/home/gmmyung/bello_mujoco/pyproject.toml) and [uv.lock](/home/gmmyung/bello_mujoco/uv.lock).

With direnv:

```bash
direnv allow
```

## Stack

- Simulator/env: `mujoco`, `mujoco-mjx[warp]`
- JAX: `jax[cuda13]` from Python wheels
- RL implementation style: local PureJaxRL-style code, `rlax`
- NN/model: `equinox`
- Optimizer: `optax`
- Experiment tracking: `wandb`
- Checkpointing: `orbax-checkpoint`
- Visualization/logging: `rerun-sdk`
- Typing/asserts: `jaxtyping`, `chex`
- Config/CLI: `tyro`

## Assets

Prepare generated Bello MuJoCo models:

```bash
bello-prepare-assets --asset full_body
```

`full_body` is the default. The generator also supports `lower_body` and `upper_body`; `down_body` is accepted as an alias for `lower_body`.

The generator reads the Bello URDF/STL description from the repository-local source assets in `assets/source/bello/<asset>` and writes a collision-only training MJCF to a deterministic cache directory:

```text
/tmp/bello-mujoco-assets/<config-hash>/<asset>/mjcf/bello_<asset>.xml
```

Generated files are local build artifacts and are not tracked by git. Asset preparation uses a file lock around each config-hash cache directory, so concurrent runs with the same config do not write the same directory at the same time.

Tests generate their own assets under `/tmp/bello-mujoco-test-assets-*` and remove them when the pytest session exits. For manual generation into another location, pass `--output-root`; omitting it uses the deterministic `/tmp/bello-mujoco-assets/<config-hash>` cache.

It also writes a separate viewer MJCF with visual geoms:

```text
/tmp/bello-mujoco-assets/<config-hash>/<asset>/mjcf/bello_<asset>_viewer.xml
```

Generated mesh assets are split by purpose:

- `<output-root>/<asset>/visual_meshes`: simplified visual STL meshes attached as non-colliding MuJoCo visual geoms for the viewer.
- `<output-root>/<asset>/meshes`: lower-budget simplified STL meshes used only by collision geoms.

`visual_meshes` is generated locally and ignored by git. The training model does not include these render-only geoms. The default visual mesh budget is separate from the collision budget:

```bash
bello-prepare-assets --visual-mesh-face-budget 200000 --mesh-face-budget 1000 --foot-collision-mesh-face-budget 1000
```

For `full_body`, the environment action is the full MuJoCo actuator vector (`nu=35`) in actuator order. The observation scaffold is a single-frame 114-dim vector: base angular velocity, projected gravity, command, full actuator joint positions, full actuator joint velocities, and previous full action.

The default training collision profile uses simplified mesh collision for every robot collision geom:

```bash
bello-prepare-assets --collision-mode mesh
```

The default generated full-body collision set has 40 mesh collision geoms. Body collision meshes are welded and simplified with a 1k-face target; the simplifier may stop above that target to preserve topology.

Arm end-effectors can be replaced during asset preparation with 65 mm diameter spheres attached 20 mm below `right_wrist_yaw_joint` and/or `left_wrist_yaw_joint`. The selected revolute joint and every child link/joint below it are removed before the sphere is attached to the wrist link:

```bash
bello-prepare-assets --end-effector-sphere-mode both
```

Valid modes are `none`, `both`, `right`, and `left`.

The generator also adds MuJoCo contact excludes for nearby robot bodies by default:

```bash
bello-prepare-assets --neighbor-exclusion-depth 2
```

Depth 2 excludes parent/child, grandparent/grandchild, and sibling body collision. This removes most mechanically-neighboring self-collision while preserving non-neighbor body contacts. Ground or terrain is added only by the runtime that loads the robot model.

Other collision profiles are available when needed:

```bash
bello-prepare-assets --collision-mode foot-mesh  # foot meshes, remaining robot collisions as boxes
bello-prepare-assets --collision-mode box        # all robot collision geoms as boxes
```

## Smoke Test

```bash
bello-smoke-env
pytest
```

Expected smoke-test shape:

```text
action_size: 35
observation_size: 114
```

`bello-smoke-env --step` also runs one jitted MJX-Warp step. The default smoke check verifies model load, GPU-backed MJX model creation, reset, and the action/observation contract.

Batch timing is handled by the benchmark command:

```bash
bello-benchmark-env --batch-size 512 --step-count 5
```

## Training and W&B

`bello-train-ppo` is dry-run only by default. Use it first to verify the resolved model, batch shapes, and config:

```bash
bello-train-ppo
```

Run a tiny end-to-end PPO smoke without W&B:

```bash
bello-train-ppo --no-dry-run --config.total-timesteps 4 --config.num-envs 2 --config.num-steps 2
```

Enable W&B for experiment tracking:

```bash
bello-train-ppo \
  --no-dry-run \
  --wandb-config.enabled \
  --wandb-config.project bello-mujoco
```

For local/offline logging:

```bash
bello-train-ppo \
  --no-dry-run \
  --wandb-config.enabled \
  --wandb-config.mode offline
```

The CLI logs the PPO, asset, reward, command, done, terrain, warp, and resolved environment config blocks. Scalar PPO metrics and reward components are tracked against environment steps. W&B also supports native video logging, which the evaluation workflow can use once policy checkpoint playback is wired.

Reward, command, and done configs are tyro-exposed nested dataclasses:

```bash
bello-train-ppo \
  --no-dry-run \
  --reward-config.lin-vel-tracking-weight 1.2 \
  --done-config.min-base-height 0.75 \
  --command-config.lin-vel-x -0.2 0.6
```

Terrain is also runtime-configured. The locomotion env and training CLI default to a flat MuJoCo plane without modifying generated robot assets:

```bash
bello-train-ppo --terrain-config.mode plane
```

Use a simple deterministic heightfield for rough-terrain smoke runs:

```bash
bello-train-ppo \
  --terrain-config.mode rough \
  --terrain-config.seed 3 \
  --terrain-config.hfield-amplitude 0.04 \
  --terrain-config.hfield-resolution 65
```

The rough heightfield is fixed for a loaded model and shared by vectorized envs. Change `--terrain-config.seed` between runs to generate a different cached terrain scene.

Disable terrain for robot-only model debugging:

```bash
bello-train-ppo --terrain-config.mode none
```

## Vectorized API

The environment exposes both single-env and batched methods:

```python
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig

env = BelloLocomotionEnv.load(AssetPrepConfig(asset="full_body"))
state, obs = env.reset()
state, obs, reward, done, info = env.step(state, action)

batch_state, batch_obs = env.reset_batch(batch_size=1024)
batch_state, batch_obs, reward, done, info = env.step_batch(batch_state, actions)
```

`step_batch` is a `jax.vmap` wrapper over `step`; it has the same collision-profile compile cost as the single step on first use. MJX-Warp is the only simulator backend used by this project.

## PPO Scaffold

The PPO implementation in [ppo.py](/home/gmmyung/bello_mujoco/src/bello_mujoco/train/ppo.py) follows the PureJaxRL training-loop style without depending on PureJaxRL as a package:

- Equinox actor-critic with a diagonal Gaussian continuous-action policy
- Optax optimizer with global gradient clipping and optional linear LR annealing
- `lax.scan` rollout collection
- reverse-scan GAE
- shuffled minibatch PPO updates with clipped policy and value losses
- `reset_batch` / `step_batch` environment interface

Dry-run initialization:

```bash
bello-train-ppo --dry-run
```

Actual training:

```bash
bello-train-ppo --no-dry-run
```

The environment reward is still a placeholder (`0.0`), so this is currently an algorithm scaffold rather than a meaningful locomotion training run.

PPO smoke test on a tiny continuous-action CartPole:

```bash
bello-ppo-cartpole-smoke
```

This runs a short JIT-compiled PPO training loop against [cartpole.py](/home/gmmyung/bello_mujoco/src/bello_mujoco/envs/cartpole.py), which exists only to validate the PPO hot path independently of MuJoCo mesh-collision compile time.
