# bello_mujoco

Training, native MuJoCo evaluation, motion tracking, RobStride HIL, and
physical-hardware deployment for the Bello humanoid.

The repository owns its robot descriptions under `assets/source/bello/`.
Generated assets, environments, checkpoints, W&B runs, videos, and caches are
not source files.

## Setup

```bash
nix develop
```

Outside the development shell, prefix commands with `uv run`.

## Supported commands

The public interface consists of seven Hydra commands:

```text
bello-prepare-assets
bello-train
bello-eval
bello-motion
bello-vast-train
bello-run-robstride
bello-run-hardware
```

Use `<command> --help`, `<command> --hydra-help`, or `<command> --cfg job`.
Unknown configuration keys fail instead of falling back to aliases.

Prepare the canonical 60 kg, sphere-hand physical morphology:

```bash
uv run bello-prepare-assets
```

Prepare its native visualizer assets:

```bash
uv run bello-prepare-assets profile=visualizer
```

Validate the MJX-Warp PPO path:

```bash
uv run bello-train experiment=smoke
```

Train the physical locomotion policy:

```bash
uv run bello-train \
  experiment=hardware_locomotion \
  total_timesteps=2000000000 \
  run_name=hardware-locomotion
```

Train bounded BeyondMimic motion tracking:

```bash
uv run bello-train \
  experiment=motion \
  motion_path=motions/asap_fixed_head/video_single_foot_balance_level4_filter_amass.npz
```

Evaluate locomotion in native MuJoCo and Rerun:

```bash
uv run bello-eval \
  task=locomotion \
  checkpoint_path=checkpoints/hardware-locomotion/final \
  command='[0.5,0.0,0.0]'
```

Render a motion checkpoint:

```bash
uv run bello-eval \
  task=motion \
  checkpoint_path=checkpoints/motion/final \
  output_path=outputs/evaluation/motion.mp4
```

Import, inspect, or render a motion through `bello-motion`:

```bash
uv run bello-motion operation=import source=legacy.pkl output=motions/local/walk.npz
uv run bello-motion operation=inspect source=motions/local/walk.npz
uv run bello-motion operation=render source=motions/local/walk.npz output=outputs/walk.mp4
```

Start the RobStride HIL emulator after creating its VCAN interfaces:

```bash
./scripts/setup_hil.sh
uv run bello-run-robstride
```

The runner enforces the canonical motor torque limits. A direct nominal-pose
simulation is available with `mode=prescribed_gains`.

Run a read-only physical-hardware preflight:

```bash
uv run --no-cache \
  --with ../bello-stack/components/bello-hardware-python \
  bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/hardware-locomotion/final
```

Arming requires both `arm=true` and the explicit operator confirmation. Set
`duration_seconds=null` for an unbounded run. Motion deployment additionally
requires `bello-state-estimatord` PoseService at `127.0.0.1:50052`.

Launch one billing-safe Vast.ai job:

```bash
uv run bello-vast-train \
  experiment=hardware_locomotion \
  branch=main \
  max_price=1.25 \
  duration_hours=6 \
  run_name=hardware-locomotion
```

## Canonical contracts

- Training uses MJX-Warp; evaluation, rendering, HIL, and deployment use native
  MuJoCo.
- The physical morphology has sphere hands, 60 kg total mass, 25 physical
  coordinates, fixed waist roll/pitch and neck/head, and active waist yaw.
- Locomotion controls 13 lower-body/waist coordinates and holds the arms.
- Motion tracking controls all 25 coordinates with bounded asymmetric
  joint-limit actions.
- New checkpoints use semantic schema 7. Older checkpoints require the archived
  source revision that created them.

See [docs/index.md](docs/index.md) for the focused workflow guides.

## Validate changes

```bash
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

Repository rules and contribution workflow are in [AGENTS.md](AGENTS.md) and
[CONTRIBUTION.md](CONTRIBUTION.md).
