# bello_mujoco

MJX-Warp PPO and model-based training, native MuJoCo evaluation, and hardware integration for
Bello and Bello Mini. Single-motion tracking uses BeyondMimic. Canonical URDFs,
meshes, motor descriptions, and hardware clients come from pinned `deps/bello-stack`.

## Setup and run

```bash
git submodule update --init --recursive
nix develop
uv run bello-eval task=locomotion checkpoint_path=checkpoints/RUN gamepad_enabled=false steps=1500
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
```

The default environment installs CPU dependencies. For GPU training on Linux x86:

```bash
nix develop .#training
uv run --group training bello-train experiment=bello_mini_locomotion
```

The training shell uses a separate `.venv-training` with CUDA dependencies.
Vast.ai enables the same training group automatically. Evaluation uses CPU
inference, native CPU simulation, and Mesa llvmpipe software rasterization on
Linux; use Lima on macOS.

Maintained experiments are `smoke`, `bello_locomotion`, `bello_mini_locomotion`,
`bello_mini_standing`, `motion`, `bello_mini_motion`,
`bello_mini_sonic_transfer`, `bello_mini_supermimic`,
`bello_mini_supermimic_smoke`, `bello_mini_supermimic_pilot`, and
`bello_mini_supermimic_amass`, and `bello_mini_supermimic_lafan`.
SuperMimic trains Mini from scratch with learned
dynamics and imagined policy gradients; see [the workflow](docs/supermimic.md).
SONIC transfer is an explicit pretrained-only
workflow; see [SONIC transfer](docs/sonic_transfer.md). Values live in YAML, not documentation tables.
Checkpoints require their exact clean root Git commit and pinned dependencies.
Historical policies run from historical checkouts, without compatibility fallbacks.

Hardware policy control uses `rust/bello-policy`; there is no Python hardware policy runner or
Python hardware-client dependency. Live retargeting dependencies remain in the
default `retargeting` dependency group.
Physical deployment remains gated on [trained-policy commissioning](docs/hardware.md).

Live PICO body following uses direct XRoboToolkit poses through the updated
tracking bridge: `bello-run-reference asset=bello_mini xrobot.endpoint=HOST:50052`.
Body following and the hand server's tracking source use `xrobot.*` and typed
native tracking data. Application hand targets use the [leased hand API](docs/hand_control.md)
independently of tracking. Neither tracking consumer needs a neutral SMPL-X model
or bridge T-pose calibration. Install the
receiver and bridge with `scripts/install_systemd.py tracking`; the viewer is
not installed. Update the bridge before its consumers; see [PICO startup](docs/pico_startup.md).

## Guides

- [Commands](docs/commands.md) and [configuration locations](docs/training_configuration.md)
- [Training, evaluation, and checkpoints](docs/training_and_evaluation.md)
- [SuperMimic for Bello Mini](docs/supermimic.md)
- [Robot descriptions](docs/robot_assets.md) and [motion tracking](docs/motion_tracking.md)
- [Bello Mini policy console](docs/policy_console.md)
- [Barbell mode](docs/barbell_mode.md)
- [Hardware](docs/hardware.md), [emulator](docs/emulator.md), and [Lima](docs/macos_vcan.md)
- [Systemd daemon startup and operations](docs/systemd.md)
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
