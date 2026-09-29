# Robot descriptions

Supported names are `bello` and `bello_mini`, without aliases. Canonical URDFs:

- `deps/bello-stack/descriptions/bello/urdf/bello.urdf`
- `deps/bello-stack/descriptions/bello_mini/urdf/bello_mini.urdf`

Change descriptions in `bello-stack`, commit there, then advance this repository's
gitlink. Visual meshes and collision primitives are packaged with the URDFs.
Runtime compilation does not simplify meshes, substitute collision shapes,
disable joints, copy descriptions, or modify URDF mesh paths.

Mini uses the measured-assembly, left/right-symmetric inertials in the pinned
description, including hand masses and a separate fixed head proxy. Allocation
and geometry assumptions are recorded in its
[asset provenance](../deps/bello-stack/descriptions/bello_mini/ASSET_PROVENANCE.md).
Both training tasks consume those inertials directly; do not add the already
included battery, hands, or head again as training payloads. Older trained
policies are not automatically adapted to a changed mass distribution.

## Compilation

`compile_robot(ModelConfig(...))` uses MuJoCo's URDF importer and adds the free
base, actuators, native fixed ankle tendons, contact exclusions, sensors, and
home keyframe. Physics and visual MJCF files are process-local. Physics omits
non-colliding visuals; both models share inertials and dynamics. There is no
preparation command, persistent cache, manifest, lease, or embedded-wheel fallback.

Initialization checks required links/joints, primitive collision geometry,
actuator coverage, limits, and nominal-pose self-collisions. Collision errors name the shapes and penetration;
correct the nominal actuator pose or canonical collision geometry. The actual
selected configuration is checked at runtime, not prescribed by snapshot tests.

## Controls

`src/bello_mujoco/robots/*.yaml` owns nominal joint positions, policy/hold
ownership, and motor gains. Transmission coefficients come only from the
canonical URDF's differential reductions. Policy targets use motor
coordinates directly. Tendons map motor forces into joint forces; observed
kinematics and joint-limit rewards remain in joint coordinates. Gains are explicit
motor gains, with no hidden coupled-motor multiplier.

Physical joint ranges come from the URDF. Experiment-specific restrictions belong
in `asset_config.joint_soft_limits` in the training profile, keyed by logical joint
name. They must contain the nominal pose and stay within the physical range. They
limit policy targets, not simulated physical joint travel. Mini's shared defaults
use the URDF ranges; locomotion and standing inherit their narrower knee targets
from the locomotion profile. Motion training does not inherit that restriction.

Unused normal-Bello waist/head articulations are removed from the description,
with geometry/inertia consolidated into surviving links. Useful fixed head,
camera, and IMU frames remain. No active joint is disabled during compilation.
Optional mass/payload experiments live in `asset_config`; payload inertia is
applied at the declared attachment position.

## Mini visual formats

The canonical Mini URDF references 28 GLBs with embedded materials. Their
millimetre vertices retain the URDF's 0.001 scale. Head and hand proxy visuals
are removed; all physical frames, inertials and collisions remain unchanged.
Electron and Kiss3d render the GLBs. MuJoCo's Python and Rust importers use an
explicit temporary URDF referencing retained companion STLs. The newly detailed
head and hands are absent from that visual path; their physical links and
collisions remain. Missing companions fail at import.
See the stack's Mini asset provenance for source and geometry details.
