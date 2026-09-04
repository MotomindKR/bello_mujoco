# Robot asset contracts

Each supported robot asset has one packaged YAML contract under
`src/bello_mujoco/robots/`. This is the static morphology boundary shared by asset
generation, native MuJoCo, and MJX-Warp setup. It is strictly decoded into the
frozen dataclasses in `robot.py`; unknown fields, missing fields, invalid dimensions,
and inconsistent coupled actuators are errors.

The robot contract owns:

- source URDF and mesh directories, plus generated output paths;
- the platform hardware description and motor-model catalog, when available;
- generated floating-base body and joint names and the default root pose;
- the default IMU body, ordered foot bodies, and foot-forward axes;
- the complete actuator joint order, policy/hold partition, neutral positions,
  soft limits, gains, and effort limits;
- base, torso, and central bodies used by model randomization, plus optional
  payload attachment points; and
- coupled actuator coordinates such as Bello's ankle differentials.

These values are not Hydra options. Runtime experiments may deliberately override
actuator modes or gains with `ActuatorConfig`, but the robot file remains the
canonical default and defines which joints belong to the morphology.
`ActuatorConfig.for_asset(...)` is the explicit way to materialize those defaults;
standalone actuator configs do not silently select Bello.

Training selects the contract through `asset_config.asset`. Every maintained base
training document makes that selection explicitly; experiment files under
`configs/profiles/train/` only override task-specific values. New robot identifiers
should be globally descriptive, even though the existing `full_body` identifier is
retained as the public Bello asset name.

Motion-capable robots also have a strict `<asset>_motion.yaml` contract beside the
static robot file. It owns legacy import joint schemas, fixed joints, tracked-body
sets, end effectors, the reference root, ordered TWIST key bodies, and the robot's
GMR boundary. The GMR section selects its source and target types, pinned IK config,
task profile, source landmarks and feet, output bodies, standing joints, and
knee-pole diagnostics. Motion import, live/offline retargeting, tracking termination,
and TWIST training all resolve these semantics from the same selected asset. Hydra
still records the key-body order because it is part of the checkpoint observation
contract, and training validates it against the robot's motion profile.

## Registered assets

- `full_body` is the production Bello morphology with maintained motion,
  differential-ankle, emulator, and physical-hardware contracts.
- `bello_mini` is the reviewed 43.503025 kg Bello Mini morphology. Its all-zero
  joint pose uses a root height of 0.8731232537850352 m so both sole surfaces
  are exactly at `z=0`. The static contract exposes all 28 movable joints and
  keeps waist roll and both seven-DoF arms in hold mode by default. Its logical
  ankle pitch/roll joints are simulated directly because the physical
  differential coefficients have not been calibrated. Its massless fixed
  `imu_link` is 50 mm below the midpoint between the reviewed left/right
  hip-pitch joint origins (`[-0.01, 0, -0.2300000011]` m in pelvis coordinates).
  It is yawed `-pi/2` from the Mini pelvis so its +X-right, +Y-forward, +Z-up
  axes reproduce original Bello's pelvis/IMU-axis convention.

  Joint travel ranges come from the designer's left-side nominal mechanical
  ranges. User visual review reversed the Mini URDF signs for hip pitch, hip
  roll, shoulder roll, and elbow pitch; the other left-side signs retain the
  designer mapping. The right side is derived geometrically across `y=0`:
  pitch keeps its sign while roll/yaw reverse. Encoder-zero correspondence and
  physical stops still require confirmation. IDs 35/45 are named `wrist_yaw`,
  matching the designer's seven-DoF arm terminology.

The Bello Mini source directory includes `motor_topology.yaml` as a
calibration manifest, not as a `bello-hardwared` configuration. CAN IDs increase
along each kinematic chain: left/right arms use `31..37` / `41..47` on `can1`,
waist yaw/roll use `51,52` on `can1`, and left/right legs use `61..66` /
`71..76` on `can0`. The recorded CAD-axis signs are geometric estimates only;
live RobStride command directions, encoder zeros, and ankle coefficient matrices
remain null until bench calibration. Consequently, the robot contract has
`hardware_description: null`; hardware and emulator entrypoints fail explicitly
instead of borrowing the production Bello topology. A Bello Mini motion/GMR
profile is also intentionally absent until it is tuned.

## Adding a humanoid

1. Add or update the canonical URDF and meshes in `bello-stack`, then deliberately
   advance this repository's pinned submodule gitlink.
2. Add a robot YAML beside the existing robot files and register its asset
   identifier in `robot.py`. Do not add aliases or fallback paths.
3. If the robot supports motion tracking, add its strict `<asset>_motion.yaml`
   profile and a compatible IK config in pinned GMR. A morphology without that file
   remains valid for other tasks, but motion workflows fail explicitly.
4. Point the robot YAML at its hardware description and motor-model catalog, or set
   those fields to `null` when physical/emulated hardware is unsupported.
5. Extend the wheel's explicit description includes in `pyproject.toml` when the
   new source directories must be available outside a source checkout.
6. Prepare the asset. Preparation validates configured links and joints against the
   source URDF before mesh conversion, then validates the actuator set again after
   disabled joints have been fixed.
7. Add contract, asset-generation, installed-wheel, native-model, and MJX tests for
   the new identifier.

A robot profile makes asset generation and core model loading morphology-aware. It
does not make every task portable automatically. Locomotion rewards and termination,
motion-tracking body sets, retargeting maps, observation layouts, and hardware motor
topology are task or platform contracts. A new humanoid needs explicit compatible
profiles for the workflows it supports.

## Remaining cleanup boundary

The robot and motion profile boundaries now cover asset generation, live and offline
GMR selection, tracked bodies, TWIST key bodies, and emulator topology selection.
New robots still need an upstream GMR IK mapping and explicit compatibility with any
physical hardware adapter. The main remaining multi-robot work is task-specific
reward and termination assumptions plus public model/environment APIs that still
carry Bello-specific class names. Renaming those APIs is intentionally separate
because it would be a breaking public change rather than a configuration cleanup.
