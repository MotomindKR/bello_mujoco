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
- [Robot asset contracts](robot_assets.md)
- [URDF inertial overhaul](urdf_inertial_overhaul.md)
- [RobStride emulator](emulator.md)
- [Vast.ai lifecycle](vast_ai.md)

## Repository shape

```text
configs/commands/          public Hydra command documents
configs/profiles/train/    maintained training profiles
deps/bello-stack/          pinned robot, hardware, and client dependency
deps/GMR/                  pinned offline and live motion retargeter
deps/Fast-SAM-3D-Body/     pinned SMPL-X skeletal gRPC contract
deps/RTMW-SMPLX-Teleop/    pinned RTMW gRPC producer and protobuf contract
motions/asap_fixed_head/   tracked reference clips
src/bello_mujoco/
  assets_pipeline/         asset preparation and project data paths
  envs/                    locomotion, motion tracking, TWIST/TWIST2, and rewards
  evaluation/              native rendering and policy evaluation
  hardware/                physical runners, IMU/pose services, and live GMR
  motion/                  motion clips, import, and feasibility tools
  robots/                  packaged, validated robot topology contracts
  robstride/               CAN protocol, transmission, and hardware emulator
  training/                PPO, checkpoints, tracking, and Vast orchestration
  configuration.py         config decoding and maintained profiles
  robot.py                 robot-contract schema and strict loader
tests/                     contract and integration tests
```

The supported prepared assets are `bello` and `bello_mini`. Their canonical
URDFs are respectively
`deps/bello-stack/descriptions/bello/urdf/bello.urdf` and
`deps/bello-stack/descriptions/bello_mini/urdf/bello_mini.urdf`; both reference
meshes stored in the same pinned submodule. Both assets have maintained
locomotion-training profiles. The `bello` asset also retains its motion and
hardware profiles; Bello Mini GMR/motion, emulator, and deployable-hardware
profiles remain deliberately absent until tuning and bench calibration.
The same gitlink pins the RobStride model catalog and Python hardware client
used by source-checkout workflows.

Each asset identifier resolves one packaged contract under
`src/bello_mujoco/robots/`. The contract declares source, generated, and hardware
paths, floating-base and IMU bodies, locomotion command frame, sagittal action
mirror, ordered feet, explicit support collision geometries and forward axes,
the actuator joint partition and defaults, payload attachment links,
randomization bodies, and coupled actuator coordinates. These files are strictly
decoded internal topology; they are not Hydra override surfaces.

Motion-capable assets add a strict `<asset>_motion.yaml` profile in the same
directory. It is the source of import joint schemas, tracked bodies, end effectors,
and TWIST key-body order; training configs select the asset explicitly and preserve
the resulting observation order in checkpoints.

## Runtime boundaries

MJX-Warp is the training backend. Native MuJoCo is the evaluation, rendering,
and hardware-emulation backend; physical deployment uses the native model for
policy-contract reconstruction without stepping simulated dynamics. Host-side
Python handles setup, assets, logging, checkpoints, and services; rollout and
reward math remains JAX compatible.

The public configuration is profile-oriented. Locomotion is selected with the
robot-specific `bello_locomotion` or `bello_mini_locomotion` experiment; runtime
evaluation and deployment continue to use the generic `task=locomotion`
interface and recover the robot contract from checkpoint metadata. Internal
topology, link names, coordinate partitions, numerical guardrails, and safety
limits are validated contracts rather than user-selectable Hydra branches.

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
