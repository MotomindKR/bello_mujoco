# Bello MuJoCo reference

The [README](../README.md) is the quick start. These guides define the supported
contracts:

- [Training and evaluation](training_and_evaluation.md)
- [Motion data and tracking](motion_tracking.md)
- [Physical hardware](hardware.md)
- [RobStride HIL](robstride_vcan.md)
- [Vast.ai](vast_ai.md)

## Repository shape

```text
assets/source/bello/       robot source of truth
configs/commands/          seven public Hydra command documents
configs/profiles/train/    maintained training profiles
configs/robstride/         canonical motor inventory and limits
motions/asap_fixed_head/   tracked reference clips
src/bello_mujoco/
  assets_pipeline/         asset preparation and project data paths
  envs/                    locomotion, motion tracking, and rewards
  evaluation/              native rendering and policy evaluation
  hardware/                physical runners, IMU, and pose service
  motion/                  motion clips, import, and feasibility tools
  robstride/               CAN protocol, simulation, and HIL runner
  training/                PPO, checkpoints, tracking, and Vast orchestration
  cli.py                   seven public Hydra command adapters
  configuration.py         config decoding and maintained profiles
tests/                     contract and integration tests
```

The supported prepared asset name is `full_body`. There are no asset aliases or
external robot-description fallbacks.

## Runtime boundaries

MJX-Warp is the training backend. Native MuJoCo is the evaluation, rendering,
HIL, and physical-deployment backend. Host-side Python handles setup, assets,
logging, checkpoints, and services; rollout and reward math remains JAX
compatible.

The public configuration is profile-oriented. Internal topology, link names,
coordinate partitions, numerical guardrails, and safety limits are validated
constants rather than user-selectable Hydra branches.

Generated paths include `.venv/`, `.uv-cache/`, `.uv-python/`, `.warp-cache/`,
`.wandb/`, `assets/generated/`, `artifacts/`, `outputs/`, and `checkpoints/`.
They must not be committed.
