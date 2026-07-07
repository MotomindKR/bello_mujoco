# bello_mujoco

MuJoCo/MJX-Warp robot-learning environment and PureJaxRL-style PPO training scaffold for Bello.

This repo is intentionally Nix-light and Python-managed by uv. Nix provides `uv` and `git`; uv owns Python, the virtualenv, and all Python dependencies.

## Quick Start

```bash
nix develop
```

With direnv:

```bash
direnv allow
```

The shell hook:

- installs uv-managed Python 3.12 into `.uv-python`
- syncs dependencies into `.venv`
- activates `.venv`
- keeps uv and Warp caches repo-local via `.uv-cache` and `.warp-cache`

Run a basic environment check:

```bash
bello-smoke-env --step
```

Expected default full-body contract:

```text
action_size: 35
observation_size: 149
mjx_impl: warp
terrain: plane
```

## Common Commands

Prepare assets:

```bash
bello-prepare-assets --asset full_body
```

Dry-run training setup:

```bash
bello-train-ppo
```

Run a tiny PPO smoke:

```bash
bello-train-ppo \
  --no-dry-run \
  --config.total-timesteps 4 \
  --config.num-envs 2 \
  --config.num-steps 2 \
  --config.num-minibatches 1 \
  --config.update-epochs 1
```

Benchmark vectorized stepping:

```bash
bello-benchmark-env --batch-size 512 --step-count 5
```

Run the test and lint gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

## Documentation

- [Repository spec](docs/spec.md)
- [Setup and local caches](docs/setup.md)
- [Asset pipeline](docs/assets.md)
- [Environment API, actions, observations, rewards, and dones](docs/environment.md)
- [Configuration tree](docs/configuration.md)
- [Training, checkpointing, W&B, and evaluation](docs/training.md)
- [Testing and audit checklist](docs/testing.md)
- [Contribution guide](CONTRIBUTION.md)
- [Agent instructions](AGENTS.md)

## Current Stack

- Simulator: `mujoco`, `mujoco-mjx[warp]`
- JAX: `jax[cuda13]` from Python wheels
- RL utilities: `rlax`
- Policy/model: `equinox`
- Optimizer: `optax`
- PPO implementation: local PureJaxRL-style code
- Tracking: W&B
- Checkpointing: Orbax
- Visualization/video: MuJoCo renderer/viewer, `imageio[ffmpeg]`
- Installed but not currently wired into runtime flows: `rerun-sdk`
- Config CLI: `tyro`

## Generated Files

Generated assets, Python environments, and runtime caches are local artifacts and should not be committed:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
/tmp/bello-mujoco-assets/<config-hash>/
```

Source robot assets under `assets/source/bello/` are tracked and are the only robot descriptions used by this repo.
