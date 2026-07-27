# bello_mujoco

`bello_mujoco` is the simulation, reinforcement-learning, evaluation, HIL, and
physical-deployment repository for the Bello humanoid. It owns the robot model
used by every runtime and keeps training, native MuJoCo evaluation, RobStride
emulation, and hardware policy execution on one explicit contract.

The repository is intentionally narrower than a general robotics framework. It
supports the Bello morphology, its locomotion and motion-tracking policies, and
the operational paths needed to train and deploy those policies reproducibly.

## System scope

The main artifact flow is:

```text
assets/source/bello
        |
        v
asset preparation -----> generated URDF, runtime MJCF, viewer MJCF
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
               MJX-Warp environments                       RobStride HIL
                         |
                         v
                        PPO
                         |
                         v
                schema-7 checkpoint
                         |
                         +-------------------+
                         |                   |
                         v                   v
                 native evaluation   physical hardware
```

One source robot description feeds every downstream runtime. Prepared assets
record their source and configuration fingerprints. Checkpoints bind the
policy, observation/action layouts, environment contract, robot assets, and
normalization state needed to reproduce evaluation or safely reject an
incompatible runtime.

## Execution paths

| Path | Backend | Purpose |
| --- | --- | --- |
| Asset preparation | MuJoCo compiler and structured XML processing | Build deterministic collision, visual, URDF, and MJCF artifacts from repository-owned sources. |
| Locomotion training | JAX, MJX-Warp, PPO | Learn command-conditioned walking and standing with batched simulation. |
| Motion tracking | JAX, MJX-Warp, PPO | Track canonical motion clips with bounded joint-limit actions and adaptive sampling. |
| Evaluation | Native MuJoCo | Run the deployment dynamics path, stream locomotion to Rerun, or render motion tracking to video. |
| RobStride HIL | Native MuJoCo, SocketCAN, serial, gRPC | Emulate the physical motor, IMU, and pose interfaces against simulated dynamics. |
| Hardware deployment | `bello-hardware` client and optional pose estimator | Validate and execute locomotion or motion policies on the robot with fail-closed controls. |
| Remote training | Vast.ai, SSH, detached process, W&B | Provision one bounded training job, monitor it, retain checkpoints, and verify rental teardown. |

Host-side Python handles configuration, assets, orchestration, logging,
checkpoint I/O, rendering, and hardware services. Rollout and reward math stays
JAX-compatible on the training path.

## Canonical robot and policy contracts

- Robot descriptions under `assets/source/bello/` are the only asset source of
  truth. External robot paths and asset aliases are unsupported.
- The physical profile is a 60 kg robot with box collisions, sphere hands,
  active waist yaw, and fixed waist roll/pitch, neck yaw, and head pitch.
- The physical layout has 25 controllable coordinates after the fixed-joint
  profile is applied. Differential ankles map joint coordinates to their paired
  motor coordinates.
- Locomotion policies act on 13 lower-body and waist coordinates while the
  hardware runner holds the arm coordinates.
- Motion policies act on all 25 coordinates using asymmetric bounds from the
  nominal pose to each joint's soft limit.
- Training uses MJX-Warp. Evaluation and HIL step native MuJoCo dynamics.
  Physical deployment uses a native model to reconstruct the policy,
  observation, and transmission contract, while the robot supplies the actual
  dynamics.
- Current checkpoints use semantic schema 7. Older schemas require the source
  revision that created them; there is no silent compatibility path.

## Configuration model

The supported interface is seven Hydra commands with strict key sets backed by
`configs/commands/`. Training commands select maintained profiles from
`configs/profiles/train/`, which are layered over the task defaults in
`configs/train/` or `configs/train_motion/`.

Unknown keys are rejected. The narrow public overrides documented for each
command take precedence over its selected profile. Reward, environment, and
other maintained training numerics are changed in profile YAML rather than
through a one-off wrapper or duplicate recipe.

| Command | Responsibility |
| --- | --- |
| `bello-prepare-assets` | Prepare the canonical physical or viewer asset set. |
| `bello-train` | Run smoke, locomotion, hardware-locomotion, or motion PPO training. |
| `bello-eval` | Evaluate a locomotion or motion checkpoint through native MuJoCo. |
| `bello-motion` | Import, inspect, batch-import, or render canonical motion data. |
| `bello-vast-train` | Launch one branch-pinned, billing-bounded Vast.ai training lifecycle. |
| `bello-run-robstride` | Run RobStride HIL or prescribed-gain simulation. |
| `bello-run-hardware` | Perform hardware preflight or explicitly armed policy execution. |

The [command reference](docs/commands.md) documents every public field, default,
supported domain, and mode-specific behavior. The
[training configuration reference](docs/training_configuration.md) explains the
nested PPO, environment, reward, asset, checkpoint, logging, video, and Warp
sections.

## Safety and reproducibility boundaries

Physical execution is read-only unless both `arm=true` and the exact operator
confirmation are supplied. Hardware runners validate checkpoint task and
morphology, motor inventory, telemetry freshness, IMU bounds, command timing,
and control-group state before and during execution. Safety failures are
terminal and unwind the acquired control sessions; explicit completion and
operator-interrupt paths stop and disable the groups.

Vast.ai orchestration records durable rental state under `.vast/` and attempts
teardown after success, failure, timeout, or interruption. W&B is the supported
remote checkpoint transport.

Generated assets, caches, W&B state, checkpoints, videos, and remote lifecycle
state are runtime artifacts, not source. They must not be committed.

## Repository map

| Path | Ownership |
| --- | --- |
| `assets/source/bello/` | Canonical robot descriptions and meshes. |
| `configs/commands/` | Public Hydra command documents. |
| `configs/profiles/train/` | Maintained experiment numerics. |
| `configs/robstride/` | Motor inventory, bus assignment, and motor-kind limits. |
| `motions/asap_fixed_head/` | Repository-tracked canonical motion clips. |
| `src/bello_mujoco/assets_pipeline/` | Asset paths, mesh rewriting, MJCF compilation, and caching. |
| `src/bello_mujoco/envs/` | Locomotion, motion tracking, and reward formulation. |
| `src/bello_mujoco/training/` | PPO, checkpointing, tracking, training loops, and Vast orchestration. |
| `src/bello_mujoco/evaluation/` | Native rollout reconstruction, Rerun, and video rendering. |
| `src/bello_mujoco/hardware/` | Hardware policy runners, EBIMU, and pose service. |
| `src/bello_mujoco/robstride/` | RobStride protocol, motor simulation, and HIL runner. |
| `src/bello_mujoco/motion/` | Motion schema, import, feasibility, and reference tools. |
| `tests/` | Contract, numerical, safety, and workflow tests. |

## Environment

Enter the reproducible development environment with:

```bash
nix develop
```

Direnv may enter it automatically. Outside that environment, run project
commands through `uv run`. Physical deployment additionally supplies the local
`bello-hardware` component through uv; the exact requirement is documented in
the command reference.

## Documentation

- [Command reference](docs/commands.md)
- [Training configuration](docs/training_configuration.md)
- [Training and evaluation contracts](docs/training_and_evaluation.md)
- [Motion data and tracking](docs/motion_tracking.md)
- [Physical hardware](docs/hardware.md)
- [RobStride HIL](docs/robstride_vcan.md)
- [Vast.ai lifecycle](docs/vast_ai.md)
- [Repository reference](docs/index.md)

Contribution rules are in [AGENTS.md](AGENTS.md) and
[CONTRIBUTION.md](CONTRIBUTION.md).

## Validation

The full repository gate is:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
./scripts/test.sh
```

`scripts/test.sh` runs each test module in a fresh CPU JAX process to avoid
cross-module accelerator state and memory contamination.
