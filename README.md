# bello_mujoco

Simulation, reinforcement learning, evaluation, hardware emulation, and robot
deployment for the Bello humanoid. The repository supports locomotion and
motion-tracking policies built from the robot sources in `assets/source/bello/`.

## Setup

Use the reproducible development environment:

```bash
nix develop
```

Direnv may enter it automatically. Outside that environment, run commands
through `uv run`.

## Commands

Invoke commands as `uv run <command> field=value`.

| Command | Purpose |
| --- | --- |
| `bello-prepare-assets` | Generate the Bello URDF and MuJoCo models. |
| `bello-train` | Train smoke, locomotion, hardware-locomotion, or motion policies. |
| `bello-eval` | Evaluate locomotion in Rerun or render motion tracking. |
| `bello-motion` | Import, inspect, validate, and render motion data. |
| `bello-vast-train` | Run a bounded Vast.ai training job. |
| `bello-run-emulator` | Run the RobStride hardware-interface emulator. |
| `bello-run-hardware` | Run checkpoint preflight or an armed robot policy. |

Training profiles live under `configs/profiles/train/`; shared task settings
live under `configs/train/` and `configs/train_motion/`.

## Documentation

- [Documentation index](docs/index.md)
- [Command and argument reference](docs/commands.md)
- [Training configuration](docs/training_configuration.md)
- [Training and evaluation](docs/training_and_evaluation.md)
- [Motion tracking](docs/motion_tracking.md)
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
