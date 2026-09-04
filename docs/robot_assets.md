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
- the default IMU body, locomotion command frame, explicit sagittal action
  mirror, ordered foot bodies, and foot-forward axes;
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

Training selects the contract through `asset_config.asset`. The maintained
`bello_locomotion` and `bello_mini_locomotion` experiment profiles make that
selection explicitly after inheriting the shared locomotion recipe. Robot
identifiers must be globally descriptive; the production robot is `bello`,
paired with `bello_mini` for the smaller morphology.

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

- `bello` is the production Bello morphology with maintained motion,
  differential-ankle, emulator, and physical-hardware contracts.
- `bello_mini` is the reviewed 43.50 kg Bello Mini morphology. Its all-zero
  joint pose uses a root height of about 0.873 m so both sole surfaces are at
  `z=0`. The static contract exposes all 28 movable joints: waist yaw, waist
  roll, and the 12 leg joints are policy-controlled, while both seven-DoF arms
  are held. Its logical
  ankle pitch/roll joints are simulated directly because the physical
  differential coefficients have not been calibrated. Its massless fixed
  `imu_link` is 50 mm below the midpoint between the reviewed left/right
  hip-pitch joint origins (`[-0.01, 0, -0.23]` m in pelvis coordinates).
  Bello is physically forward along pelvis `+Y`; Bello Mini is physically
  forward along pelvis `+X`. The Mini locomotion profile therefore measures
  planar tracking in the yaw-aligned `imu_link` frame, preserving the public
  command convention of `+X` right and `+Y` forward for both robots. Each
  robot's configured action mirror and foot-forward axes likewise encode its
  physical geometry; in particular, Mini hip pitch keeps sign across sides
  while Mini hip roll changes sign.

  A reviewed nonzero locomotion pose and narrower policy soft limits are not yet
  available from the source description. Mini therefore still starts from its
  all-zero pose; its experiment explicitly disables the otherwise-inert
  multiplicative joint-position reset range, and the knees begin at their hard
  lower limit. Other model, observation, velocity, terrain, and push
  randomization remains active. This is a tuning limitation for the first
  locomotion run, not a hidden fallback.

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
profile is also intentionally absent until it is tuned. These gates do not
prevent simulation locomotion training with `bello_mini_locomotion`.

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
