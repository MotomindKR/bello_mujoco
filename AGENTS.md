# Agent Instructions

This file is for automated coding agents working in this repo.

## First Principles

- Read the existing code before editing.
- Keep generated assets out of git.
- Keep changes scoped to the user request.
- Prefer existing dataclass config patterns and tyro CLI exposure.
- Do not reintroduce external robot asset paths.
- Do not add asset aliases.

## Environment

Use the repo environment:

```bash
nix develop
```

or rely on direnv:

```bash
direnv allow
```

Use uv for Python commands:

```bash
uv run ...
```

## Editing

- Use structured parsers for URDF/MJCF/XML changes.
- Use `apply_patch` for manual edits.
- Do not commit generated outputs.
- Keep ASCII unless editing a file that already requires non-ASCII.
- Update docs when CLI/config/user-visible behavior changes.

## Validation

Standard checks:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

For simulation/training changes:

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

For CLI/config docs, verify help output:

```bash
uv run bello-train-ppo --help
uv run bello-prepare-assets --help
```

## Current Contracts

Default full-body env:

```text
asset: full_body
terrain: plane
mjx_impl: warp
action_size: 35
observation_size: 149
```

Valid assets:

```text
full_body
lower_body
upper_body
```

Default observation components:

```text
base_ang_vel
projected_gravity
command
joint_pos
joint_vel
actuator_force
last_action
```

## Git

Before committing:

```bash
git status --short
git diff --stat
git diff --check
```

Do not revert user changes unless explicitly asked. If a worktree is dirty, distinguish your edits from pre-existing edits before committing.
