# Contribution Guide

## Scope

Keep changes focused and intentional. A contribution should preserve asset
generation, MJX-Warp environment stepping, PPO initialization/training,
checkpointing, and evaluation unless it is explicitly changing one of those
contracts.

Avoid mixing unrelated refactors with behavior changes.

## Branch Naming

Use short lowercase branch names:

```text
<type>/<short-description>
```

Allowed types:

```text
feat      new user-facing behavior
fix       bug fix
docs      documentation only
test      tests only
refactor  internal cleanup without behavior change
chore     tooling, metadata, dependency, or repo maintenance
experiment temporary exploration that is not expected to merge as-is
```

Examples:

```text
feat/rough-terrain
fix/contact-worldid
docs/single-index
refactor/reward-configs
chore/update-lockfile
```

## Pull Request Shape

- Explain what changed and why.
- Call out behavior, config, asset, reward, training, or checkpoint changes.
- Include the validation commands you ran.
- Keep generated files out of the diff.
- Update [docs/index.md](docs/index.md) and any relevant topic doc, especially
  [docs/training_and_evaluation.md](docs/training_and_evaluation.md) or
  [docs/hardware.md](docs/hardware.md), when user-facing behavior changes.

## Validation

Run the smallest relevant checks during iteration, then run the full gate before
merging:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

The full runner isolates test modules in fresh CPU JAX processes. During
iteration, run focused files directly with `uv run pytest`.

For simulation or training changes, also run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train experiment=smoke
```

The public commands are backed by `configs/commands/`; maintained training
numerics live in `configs/profiles/train/`. See
[docs/index.md](docs/index.md) for supported workflows, the training smoke, and
links to topic-specific docs.
