# bello_mujoco

MJX-Warp PPO training, native MuJoCo evaluation, and hardware integration for
Bello and Bello Mini. Single-motion tracking uses BeyondMimic. Canonical URDFs,
meshes, motor descriptions, and hardware clients come from pinned `deps/bello-stack`.

## Setup and run

```bash
git submodule update --init --recursive
nix develop
uv run bello-train experiment=bello_mini_locomotion
uv run bello-eval task=locomotion checkpoint_path=checkpoints/RUN gamepad_enabled=false steps=1500
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
```

Training requires CUDA. Evaluation uses CPU inference, native CPU simulation,
and Mesa llvmpipe software rasterization on Linux; use Lima on macOS.

Maintained experiments are `smoke`, `bello_locomotion`, `bello_mini_locomotion`,
`bello_mini_standing`, and `motion`. Values live in YAML, not documentation tables.
Checkpoints require their exact clean root Git commit and pinned dependencies.
Historical policies run from historical checkouts, without compatibility fallbacks.

## Guides

- [Commands](docs/commands.md) and [configuration locations](docs/training_configuration.md)
- [Training, evaluation, and checkpoints](docs/training_and_evaluation.md)
- [Robot descriptions](docs/robot_assets.md) and [motion tracking](docs/motion_tracking.md)
- [Hardware](docs/hardware.md), [emulator](docs/emulator.md), and [Lima](docs/macos_vcan.md)
- [Vast.ai](docs/vast_ai.md)
- [Contributing and validation](CONTRIBUTION.md)

## Validate

```bash
uv lock --check
uv run ruff check src tests
uv run ruff format --check src tests
./scripts/test.sh
```

The default suite is CPU-only; full robot dynamics and compiled rollouts are
opt-in. Runtime initialization validates the actual description and nominal
pose instead of tests hard-coding robot configuration values.
