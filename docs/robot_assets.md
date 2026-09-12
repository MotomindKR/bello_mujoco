# Robot descriptions

Supported names are `bello` and `bello_mini`, without aliases. Canonical URDFs:

- `deps/bello-stack/descriptions/bello/urdf/bello.urdf`
- `deps/bello-stack/descriptions/bello_mini/urdf/bello_mini.urdf`

Change descriptions in `bello-stack`, commit there, then advance this repository's
gitlink. Visual meshes and collision primitives are packaged with the URDFs.
Runtime compilation does not simplify meshes, substitute collision shapes,
disable joints, copy descriptions, or modify URDF mesh paths.

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

Unused normal-Bello waist/head articulations are removed from the description,
with geometry/inertia consolidated into surviving links. Useful fixed head,
camera, and IMU frames remain. No active joint is disabled during compilation.
Optional mass/payload experiments live in `asset_config`; payload inertia is
applied at the declared attachment position.
