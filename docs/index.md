# Bello MuJoCo reference

The [README](../README.md) defines the project scope and architecture. These
guides define the supported interfaces and contracts:

- [Command reference](commands.md)
- [Training configuration](training_configuration.md)
- [Training and evaluation contracts](training_and_evaluation.md)
- [Motion data and tracking](motion_tracking.md)
- [TWIST teacher, student, and dataset pipeline](twist.md)
- [One-stage TWIST2](twist2.md)
- [Physical hardware](hardware.md)
- [RobStride emulator](emulator.md)
- [Vast.ai lifecycle](vast_ai.md)

## Repository shape

```text
configs/commands/          public Hydra command documents
configs/profiles/train/    maintained training profiles
deps/bello-stack/          pinned robot, hardware, and client dependency
motions/asap_fixed_head/   tracked reference clips
src/bello_mujoco/
  assets_pipeline/         asset preparation and project data paths
  envs/                    locomotion, motion tracking, TWIST/TWIST2, and rewards
  evaluation/              native rendering and policy evaluation
  hardware/                physical runners, IMU, and pose service
  motion/                  motion clips, import, and feasibility tools
  robstride/               CAN protocol, transmission, and hardware emulator
  training/                PPO, checkpoints, tracking, and Vast orchestration
  configuration.py         config decoding and maintained profiles
tests/                     contract and integration tests
```

The supported prepared asset is `full_body`. Its canonical URDF is
`deps/bello-stack/descriptions/bello/full_body/urdf/bello_full_body.urdf`,
which directly references the lower- and upper-body meshes in that submodule.
The same gitlink pins the hardware topology, RobStride model catalog, and
Python hardware client used by source-checkout workflows.

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
checkout with initialized submodules. Exact-source checks require both the root
worktree and recursive submodules to be clean and at their pinned commits. The
wheel embeds only the pinned URDF and meshes needed to prepare simulation
assets; the hardware topology, RobStride catalog, Python hardware binding, and
tracked reference motion clips remain checkout-only. Checkpoint training,
evaluation, and hardware deployment also remain exact-source checkout
workflows; the wheel supports asset preparation and non-checkpoint simulation
imports.

The package and embedded robot assets are proprietary and internal-only; see
the repository `LICENSE`. The wheel smoke test resolves every URDF mesh and
prepares both simulation and viewer MJCFs from the installed package. External
redistribution requires a separate written agreement with the applicable
rights holder.
