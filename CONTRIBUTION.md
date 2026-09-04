# Contribution Guide

## Changes

- Keep each change focused.
- Read the affected implementation, tests, and documentation before editing.
- Do not mix unrelated cleanup with behavior changes.
- Do not commit generated assets, checkpoints, caches, videos, or W&B output.
- Initialize dependencies with `git submodule update --init --recursive`.
- Change canonical robot or hardware descriptions in `bello-stack`; update the
  `deps/bello-stack` gitlink here only after that stack commit is available.
- Change the locomotion pose producer or protobuf contract in
  `RTMW-SMPLX-Teleop`; update the `deps/RTMW-SMPLX-Teleop` gitlink and Bello
  consumer validation together only after that producer commit is available.

Use branch names in the form `<type>/<short-description>`. Supported types are
`feat`, `fix`, `docs`, `test`, `refactor`, `chore`, and `experiment`.

## Before opening a pull request

- Explain what changed and why.
- Add or update tests for changed behavior.
- Update the relevant user documentation.
- Report the validation commands you ran.
- Check that the diff contains no generated or unrelated files.

Start documentation changes from [docs/index.md](docs/index.md). Public command
fields belong in [docs/commands.md](docs/commands.md); training settings belong
in [docs/training_configuration.md](docs/training_configuration.md).

The private pose and remote-command Python modules are generated from the
authoritative schemas under `deps/bello-stack/components/bello-*-api/proto`.
Treat the submodule as read-only here and regenerate the bindings after
advancing the pinned revision to a version that changes either protocol:

```bash
uv run bash scripts/protobuf.sh generate
```

## Validation

Run focused tests while iterating:

```bash
uv run pytest -q tests/<relevant_test>.py
```

Run the full gate before merging:

```bash
uv lock --check
uv run bash scripts/protobuf.sh --check
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

The compiled MJX-Warp integration cases require a GPU backend. Run them on a
CUDA development host with:

```bash
JAX_PLATFORMS=cuda ./scripts/test.sh tests/test_motion_tracking.py
```
