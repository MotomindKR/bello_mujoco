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
- exports `MUJOCO_GL=egl` on Linux and `MUJOCO_GL=glfw` on Apple Silicon macOS

Editable checkouts use `configs/` and `assets/source/` from the repository.
Built wheels include those directories as package data so installed console
scripts can resolve Hydra configs and Bello source assets without a repo root.

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

Prepare and dry-run fixed-waist whole-body tracking:

```bash
bello-import-gmr-motion \
  /home/gmmyung/GMR/motion_data/bello_validation/walk1_subject1_bello_from_smplx.pkl \
  motions/local/walk1_subject1_fixed_waist.npz
bello-train-motion-ppo
```

See [whole-body motion tracking](docs/motion_tracking.md) for the robot,
canonical motion, environment, reward, evaluation, and periodic W&B video
contracts.

Launch a branch-pinned Vast.ai training run:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion_hard.yaml \
  --max-price 1.25
```

See [the Vast.ai workflow](docs/vast_ai.md) before renting an instance.

Benchmark vectorized stepping:

```bash
bello-benchmark-env batch_size=512 step_count=5
```

Run a live policy viewer:

```bash
bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

This spawns a Rerun Viewer and reads live velocity commands from an
SDL-compatible game controller.

Run the test and lint gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

## Documentation

- [Repository reference](docs/index.md)
- [Whole-body motion tracking](docs/motion_tracking.md)
- [Vast.ai training workflow](docs/vast_ai.md)
- [Contribution guide](CONTRIBUTION.md)
- [Agent instructions](AGENTS.md)

## Current Stack

- Simulator: `mujoco`, `mujoco-mjx[warp]`
- JAX: `jax[cuda13]` on Linux, CPU `jax` on Apple Silicon macOS
- RL utilities: `rlax`
- Policy/model: `equinox`
- Optimizer: `optax`
- PPO implementation: local PureJaxRL-style code
- Tracking: W&B
- Checkpointing: Orbax
- Evaluation visualization: Rerun
- Training video: MuJoCo renderer, `imageio[ffmpeg]`
- Config CLI: `hydra-core`

## Generated Files

Generated assets, Python environments, and runtime caches are local artifacts and should not be committed:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
wandb/
.vast/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/<config-hash>/
```

Source robot assets under `assets/source/bello/` are tracked and are the only robot descriptions used by this repo.
