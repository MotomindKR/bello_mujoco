# bello_mujoco

Simulation, reinforcement learning, evaluation, hardware emulation, and robot
deployment for the Bello humanoid. The repository supports locomotion and
motion-tracking policies built from the robot descriptions pinned through
`deps/bello-stack`.

## Setup

Use the reproducible development environment:

```bash
git submodule update --init --recursive
nix develop
```

Direnv may enter it automatically. Outside that environment, run commands
through `uv run` after `uv sync --frozen`. The uv workspace installs the pinned
`bello-hardware` Python binding from the submodule; no sibling checkout is
required.

## Commands

Invoke commands as `uv run <command> field=value`.

| Command | Purpose |
| --- | --- |
| `bello-prepare-assets` | Generate the Bello URDF and MuJoCo models. |
| `bello-train` | Train smoke, locomotion, hardware-locomotion, motion, TWIST, TWIST2, or SONIC-transfer policies. |
| `bello-eval` | Evaluate locomotion, render motion tracking, or score and render TWIST/TWIST2/SONIC policies. |
| `bello-motion` | Import, inspect, validate, and render motion data. |
| `bello-twist-data` | Validate licensed manifests and build packed TWIST datasets. |
| `bello-sonic-data` | Generate synchronized G1/SMPL sidecars and build Bello SONIC datasets. |
| `bello-vast-train` | Run a bounded Vast.ai training job. |
| `bello-run-emulator` | Run the RobStride hardware-interface emulator. |
| `bello-run-hardware` | Run checkpoint preflight or an armed robot policy. |

Training profiles live under `configs/profiles/train/`; shared task settings
live under `configs/train/`, `configs/train_motion/`,
`configs/train_twist/`, `configs/train_twist2/`, or `configs/train_sonic/`. Read the
[TWIST guide](docs/twist.md) and [TWIST2 guide](docs/twist2.md) before supplying
licensed datasets. The [SONIC transfer guide](docs/sonic.md) covers the
synchronized AMASS/GMR data boundary and release-weight export.

## Documentation

- [Documentation index](docs/index.md)
- [Command and argument reference](docs/commands.md)
- [Training configuration](docs/training_configuration.md)
- [Training and evaluation](docs/training_and_evaluation.md)
- [Motion tracking](docs/motion_tracking.md)
- [TWIST teacher and student](docs/twist.md)
- [One-stage TWIST2](docs/twist2.md)
- [SONIC transfer learning](docs/sonic.md)
- [Hardware runner](docs/hardware.md)
- [Emulator](docs/emulator.md)
- [Vast.ai training](docs/vast_ai.md)

Contributor workflow is documented in
[CONTRIBUTION.md](CONTRIBUTION.md). Repository-specific agent guidance is in
[AGENTS.md](AGENTS.md).

## Validation

```bash
uv lock --check
uv run bash scripts/protobuf.sh --check
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```
