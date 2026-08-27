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
- bodies used by model randomization and optional payload attachment points; and
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
sets, end effectors, the reference root, and ordered TWIST key bodies. Motion import,
tracking termination, and TWIST training all resolve these semantics from the same
selected asset. Hydra still records the key-body order because it is part of the
checkpoint observation contract, and training validates it against the robot's
motion profile.

## Adding a humanoid

1. Add or update the canonical URDF and meshes in `bello-stack`, then deliberately
   advance this repository's pinned submodule gitlink.
2. Add a robot YAML beside `full_body.yaml` and register its asset identifier in
   `robot.py`. Do not add aliases or fallback paths.
3. If the robot supports motion tracking, add its strict `<asset>_motion.yaml`
   profile. A morphology without that file remains valid for other tasks, but motion
   workflows fail explicitly.
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

The robot and motion profile boundaries now cover asset generation, GMR import,
tracked bodies, TWIST key bodies, and emulator topology selection. The largest
remaining multi-robot work is the live SMPL-X/GMR retarget map, which is still a
pinned Bello solver contract, followed by task reward assumptions and public
model/environment APIs that still carry Bello-specific class names. Renaming those
APIs is intentionally separate because it would be a breaking public change rather
than a configuration cleanup.
