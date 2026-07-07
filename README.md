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

- Simulator/env: `mujoco`, `mujoco-mjx[warp]`, `mujoco-warp`
- JAX: `jax[cuda13]` from Python wheels
- RL implementation style: local PureJaxRL-style code, `rlax`
- NN/model: `equinox`
- Optimizer: `optax`
- Experiment tracking: `aim`
- Checkpointing: `orbax-checkpoint`
- Visualization/logging: `rerun-sdk`
- Typing/asserts: `jaxtyping`, `chex`
- Config: `tyro`, `hydra-core`

## Assets

Prepare the full-body Bello MuJoCo model:

```bash
bello-prepare-assets
```

The generator reads the Bello URDF/STL description from `motomind_platform`, copies source descriptions into `assets/source/bello`, converts ASCII STL files to simplified binary STL files, and writes the training MJCF to:

```text
assets/generated/bello/full_body/mjcf/bello_full_body.xml
```

The generated MJCF is full-body. The environment action is the full MuJoCo actuator vector (`nu=35`) in actuator order. The observation scaffold is a single-frame 114-dim vector: base angular velocity, projected gravity, command, full actuator joint positions, full actuator joint velocities, and previous full action.

The default training collision profile keeps the left/right ankle-roll foot geoms as simplified mesh collision and converts the remaining robot mesh collision geoms to boxes derived from each mesh bound:

```bash
bello-prepare-assets --collision-mode foot-mesh
```

The generator also adds MuJoCo contact excludes for nearby robot bodies by default:

```bash
bello-prepare-assets --neighbor-exclusion-depth 2
```

Depth 2 excludes parent/child, grandparent/grandchild, and sibling body collision. This removes most mechanically-neighboring self-collision while preserving non-neighbor body and ground contacts.

Other collision profiles are available when needed:

```bash
bello-prepare-assets --collision-mode mesh  # all simplified mesh collision; slow MJX compile
bello-prepare-assets --collision-mode box   # all robot collision geoms as boxes
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

`bello-smoke-env --step` also times two jitted MJX steps by default and reports `step_seconds`. The default simulator backend is MJX-Warp on GPU; pass `--mjx-impl jax` to compare against MJX-JAX. The first entries include JIT/MJX lowering and GPU execution setup; with the default foot-mesh profile this is currently practical, while the all-mesh profile can compile very slowly. Use `--step-count N` to inspect later steady-state steps. The default smoke check verifies model load, GPU-backed MJX model creation, reset, and the action/observation contract.

Batch timing:

```bash
bello-smoke-env --mjx-impl warp --batch-size 512 --step --step-count 5
```

Standalone MJWarp timing and JAX/Warp zero-copy check:

```bash
bello-mjwarp-smoke --nworld 512 --step-count 20
```

## Vectorized API

The environment exposes both single-env and batched methods:

```python
env = BelloLocomotionEnv.load()
state, obs = env.reset()
state, obs, reward, done, info = env.step(state, action)

batch_state, batch_obs = env.reset_batch(batch_size=1024)
batch_state, batch_obs, reward, done, info = env.step_batch(batch_state, actions)
```

`step_batch` is a `jax.vmap` wrapper over `step`; it has the same collision-profile compile cost as the single step on first use. MJX-Warp is the default backend because it handles contact-heavy batched simulation much better than MJX-JAX on NVIDIA GPUs.

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
