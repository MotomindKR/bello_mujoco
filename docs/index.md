# Bello MuJoCo reference

The [README](../README.md) defines the project scope and architecture. These
guides define the supported interfaces and contracts:

- [Command reference](commands.md)
- [Training configuration](training_configuration.md)
- [Training and evaluation contracts](training_and_evaluation.md)
- [Motion data and tracking](motion_tracking.md)
- [Physical hardware](hardware.md)
- [RobStride emulator](emulator.md)
- [Vast.ai lifecycle](vast_ai.md)

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
  robstride/               CAN protocol, transmission, and hardware emulator
  training/                PPO, checkpoints, tracking, and Vast orchestration
  configuration.py         config decoding and maintained profiles
tests/                     contract and integration tests
```

The supported prepared asset is `full_body`. Its canonical URDF is
`assets/source/bello/full_body/urdf/bello_full_body.urdf`, which directly
references meshes under the lower- and upper-body directories. The component
URDFs in those directories are retained as engineering references; asset
preparation does not merge or read them.

## Runtime boundaries

MJX-Warp is the training backend. Native MuJoCo is the evaluation, rendering,
and hardware-emulation backend; physical deployment uses the native model for
policy-contract reconstruction without stepping simulated dynamics. Host-side
Python handles setup, assets, logging, checkpoints, and services; rollout and
reward math remains JAX compatible.

The public configuration is profile-oriented. Internal topology, link names,
coordinate partitions, numerical guardrails, and safety limits are validated
constants rather than user-selectable Hydra branches.

Generated paths include `.venv/`, `.uv-python/`, `.warp-cache/`, `.wandb/`,
`.vast/`, `assets/generated/`, `artifacts/`, `outputs/`, and `checkpoints/`.
They must not be committed. The uv download cache lives under
`$XDG_CACHE_HOME/uv`, outside the repository.

## Distribution boundary

Training, Vast orchestration, and exact-source checkpoints operate from a Git
checkout. The tracked reference motion clips are intentionally repository data,
not installed wheel data.

This repository currently records neither a license grant nor a provenance
manifest for the robot descriptions and meshes. Establish both before
distributing the project or its assets outside their current environment.
