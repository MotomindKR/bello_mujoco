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
- exports `MUJOCO_GL=egl` for MuJoCo rendering in the Nix shell

Run a basic environment check:

```bash
bello-smoke-env step=true
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
bello-prepare-assets asset=full_body
```

Dry-run training setup:

```bash
bello-train-ppo
```

Use the local RTX 5090 baseline config:

```bash
bello-train-ppo --config-name rtx5090_4096x256
```

Run a tiny PPO smoke:

```bash
bello-train-ppo \
  dry_run=false \
  config.total_timesteps=4 \
  config.num_envs=2 \
  config.num_steps=2 \
  config.num_minibatches=1 \
  config.update_epochs=1
```

Benchmark vectorized stepping:

```bash
bello-benchmark-env batch_size=512 step_count=5
```

Run the test and lint gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

## Documentation

- [Repository reference](docs/index.md)
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
- Config CLI: `hydra-core`

## Generated Files

Generated assets, Python environments, and runtime caches are local artifacts and should not be committed:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/<config-hash>/
```

Source robot assets under `assets/source/bello/` are tracked and are the only robot descriptions used by this repo.
