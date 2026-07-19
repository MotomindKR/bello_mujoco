# bello_mujoco

MuJoCo/MJX-Warp simulation, reinforcement-learning training, policy evaluation,
and hardware-in-the-loop tooling for the Bello humanoid.

The repository uses repo-owned Bello descriptions under `assets/source/bello/`.
Nix supplies `uv` and Git; `uv` manages Python 3.12 and the project dependencies.

## Setup

```bash
nix develop
```

With direnv:

```bash
direnv allow
```

Outside the activated shell, prefix commands with `uv run`.

## Prepare the robot assets

```bash
uv run bello-prepare-assets asset=full_body
```

Asset preparation converts the tracked Bello descriptions into deterministic
training and viewer MJCF, with configurable collision geometry, mesh
simplification, end-effector spheres, joint availability, IMU metadata, and
motor gain/effort scaling. The physical differential ankle tendons and their
actuators are always generated. Generated assets are cached outside the source
tree.

## Train

Validate the complete training path with the lightweight recipe:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name smoke
```

Run the maintained locomotion recipe:

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name locomotion
```

The standard recipe uses one seeded sloped plane per run and the physical ankle
transmission. Root-yaw randomization exposes the policy to that incline from
different headings. Change its budget with an explicit Hydra override instead
of selecting a duplicate recipe.
Training is dry-run by default only in the package configuration; maintained
experiment recipes explicitly select their intended execution mode.

See [training.md](docs/training.md) for the runtime contract, checkpoints,
tracking, and validation workflow.

## Evaluate

```bash
uv run bello-eval-policy checkpoint_path=checkpoints/ppo_seed_0
```

Evaluation restores the environment and policy contract from checkpoint
metadata, runs either CPU MuJoCo or single-environment MJX-Warp, and streams the
robot and diagnostics to Rerun. An SDL-compatible gamepad can provide live
velocity commands.

See [evaluation.md](docs/evaluation.md).

## Train on Vast.ai

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion.yaml \
  --max-price 1.25
```

The launcher pins a remote commit, provisions one compatible instance, performs
the remote setup and training, uploads configured checkpoint snapshots through
W&B, and destroys the instance. See [vast_ai.md](docs/vast_ai.md) before
renting.

## Run the RobStride HIL emulator

```bash
./scripts/setup_hil.sh
uv run bello-robstride-runner
```

The runner simulates Bello in CPU MuJoCo and exposes its actuators as RobStride
devices on Linux SocketCAN. See [robstride_vcan.md](docs/robstride_vcan.md).

## Validate changes

```bash
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

The full test runner starts each test module in a fresh CPU JAX process so
compiled executables cannot accumulate until the host runs out of memory. Use
`uv run pytest tests/test_<area>.py` for a focused iteration.

The supported installed commands are:

- `bello-prepare-assets`
- `bello-train-ppo`
- `bello-eval-policy`
- `bello-vast-train`
- `bello-robstride-runner`

## Repository layout

```text
assets/source/bello/       tracked robot descriptions
configs/                   installed Hydra and HIL configuration
experiments/train/         maintained training recipes
src/bello_mujoco/          simulation, training, evaluation, and HIL code
tests/                     automated validation
docs/                      user-facing reference
```

Do not commit generated environments, caches, W&B runs, checkpoints, videos, or
prepared assets. Common generated paths include `.venv/`, `.uv-cache/`,
`.warp-cache/`, `.wandb/`, `checkpoints/`, `artifacts/`, `outputs/`, and
`/tmp/bello-mujoco-assets/`.

For contribution workflow and repository rules, see [CONTRIBUTION.md](CONTRIBUTION.md)
and [docs/index.md](docs/index.md).
