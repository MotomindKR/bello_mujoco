# Contribution Guide

## Scope

This repo is a robot-learning scaffold. Changes should preserve the ability to:

- generate assets from tracked source URDF/STL files
- load MuJoCo/MJX-Warp models
- run vectorized env steps
- initialize and run PPO
- checkpoint and evaluate policies

Keep changes focused. Avoid unrelated refactors during behavior changes.

## Development Environment

Use the Nix/uv environment:

```bash
nix develop
```

or:

```bash
direnv allow
```

Python dependencies are managed by uv. Do not install project dependencies through system Python.

## Generated Artifacts

Do not commit generated artifacts:

```text
.venv/
.uv-cache/
.uv-python/
.warp-cache/
.wandb/
checkpoints/
outputs/
/tmp/bello-mujoco-assets/
/tmp/bello-mujoco-test-assets-*
```

Generated robot assets should remain reproducible from:

```text
assets/source/bello/
src/bello_mujoco/assets_pipeline/
src/bello_mujoco/actuators.py
```

## Asset Naming

Valid asset names are:

```text
full_body
lower_body
upper_body
```
Do not add extra asset aliases.

## Code Style

Run:

```bash
uv run ruff format src tests
uv run ruff check src tests
```

Prefer:

- dataclass configs for CLI-exposed options
- explicit per-term reward/done configs
- structured XML parsing for URDF/MJCF edits
- focused tests for changed behavior

Avoid:

- generated asset churn in git
- hidden dependency on external robot asset paths
- ad hoc string manipulation of XML when an XML parser is reasonable
- broad refactors mixed into training behavior changes

## Test Gate

Minimum gate:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

For env/training changes, also run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env --step
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  --no-dry-run \
  --config.total-timesteps 4 \
  --config.num-envs 2 \
  --config.num-steps 2 \
  --config.num-minibatches 1 \
  --config.update-epochs 1
```

## Commit Checklist

Before committing:

```bash
git status --short
git diff --stat
git diff --check
```

Then run the relevant tests and smoke commands. Commit only intended source/docs/test changes.

## Documentation

When adding or changing user-facing behavior, update docs in `docs/` and keep `README.md` as a concise entry point.

Use `docs/configuration.md` for config changes, `docs/environment.md` for env/action/observation/reward/done changes, and `docs/assets.md` for asset generation changes.
