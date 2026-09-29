# Contribution Guide

## Changes

- Keep each change focused.
- Read the affected implementation, tests, and documentation before editing.
- Do not mix unrelated cleanup with behavior changes.
- Do not commit generated assets, checkpoints, caches, videos, or W&B output.
- Initialize dependencies with `git submodule update --init --recursive`.
- Change canonical robot or hardware descriptions in `bello-stack`; update the
  `deps/bello-stack` gitlink here only after that stack commit is available.
- Change the live XRoboToolkit pose producer or protobuf contract in
  `pico-smplx-bridge`; update the `deps/pico-smplx-bridge` gitlink and Bello
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

The private pose Python module comes from the pinned stack pose API. Operator
bindings come from `rust/bello-policy/proto/operator.proto`. Regenerate bindings
when either authoritative schema changes:

```bash
uv run bash scripts/protobuf.sh generate
```

The same script generates the private PICO tracking bindings from
`deps/pico-smplx-bridge/pose_stream.proto`. This submodule supplies the wire
contract and tracking server. The systemd tracking installer provisions its
standalone environment and native SDK; Bello consumers only import generated
protobufs.
CI checks out the private bridge using the `PICO_SMPLX_DEPLOY_KEY` repository
secret, paired with a read-only deploy key on `pico-smplx-bridge`.

## Validation

Implement the explicit `training/ppo_types.py:PPOEnvironment` contract for each
task, including initialization, task-state updates, and metrics. Task-neutral
normalization and metric reducers live outside PPO optimization. Keep scalar and
batched locomotion transitions on their shared post-physics path; test changes
with synthetic JIT/vmap fixtures before GPU rollouts.

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
JAX_PLATFORMS=cuda uv run --group training pytest -m integration tests/test_motion_tracking.py
```

The default suite excludes `integration` and sets a CPU backend unless explicitly
overridden. Full robot dynamics and rollout tests are opt-in; run each module in
a separate process to bound compiled-state memory. Pure algorithm and validator
tests belong in the default suite. Do not encode robot YAML values, joint counts,
or nominal-pose collision expectations in tests: validate the selected model at
runtime and test that validator with synthetic geometry.

To check the exported bundle against the Rust CPU consumer, run
`cargo build --locked -p bello-policy` from this repository, set `BELLO_POLICY_BIN` to the executable and
`ORT_DYLIB_PATH` to the ONNX Runtime library. Build inside
`nix develop .#rust-runtime` on Linux for the matching MuJoCo library, then run:

```bash
uv run pytest -m integration tests/test_deployment_runtime.py tests/test_native_locomotion.py tests/test_native_motion.py
```

This uses synthetic policies and temporary local Git fixtures, without a robot,
GPU, hardware service, or external repository writes.
