# bello_mujoco

Simulation, reinforcement learning, evaluation, hardware emulation, and robot
deployment for the Bello humanoid. The repository supports locomotion and
motion-tracking policies built from the robot descriptions pinned through
`deps/bello-stack`. Registered static assets are `bello` and `bello_mini`;
both have maintained locomotion-training profiles. Bello Mini also has
calibrated offline and live-upper-body SMPL-X GMR profiles for repository
tooling and locomotion deployment.
The reviewed Mini hardware description supports both the vCAN emulator and
physical locomotion deployment; maintained Mini motion/TWIST training remains
task-tuning-gated. A separate RTMW-driven command controls Linker O6/L6 hands
over classic CAN without routing hand targets through `bello-hardwared`.

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
| `bello-train` | Train maintained Bello or Bello Mini locomotion, or Bello motion, TWIST, and one-stage TWIST2 policies. |
| `bello-eval` | Evaluate maintained Bello or Bello Mini locomotion, or Bello motion and TWIST/TWIST2 policies. |
| `bello-motion` | Import, inspect, validate, and render motion data. |
| `bello-twist-data` | Validate licensed manifests and build packed TWIST datasets. |
| `bello-vast-train` | Run a bounded Vast.ai training job. |
| `bello-run-emulator` | Run the RobStride hardware-interface emulator. |
| `bello-run-hardware` | Run checkpoint preflight or an armed robot policy. |
| `bello-run-hands` | Validate RTMW hand tracking or directly control Linker O6/L6 hands over classic CAN. |

Training profiles live under `configs/profiles/train/`; shared task settings
live under `configs/train/`, `configs/train_motion/`, and
`configs/train_twist/` or `configs/train_twist2/`. Read the
[TWIST guide](docs/twist.md) and [TWIST2 guide](docs/twist2.md) before supplying
licensed datasets.

## Documentation

- [Documentation index](docs/index.md)
- [Command and argument reference](docs/commands.md)
- [Training configuration](docs/training_configuration.md)
- [Training and evaluation](docs/training_and_evaluation.md)
- [Motion tracking](docs/motion_tracking.md)
- [TWIST teacher and student](docs/twist.md)
- [One-stage TWIST2](docs/twist2.md)
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
