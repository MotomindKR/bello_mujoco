# ONNX policy runtime

Task-specific CPU runtime owned by `bello_mujoco`. It consumes the pinned
`bello-stack` hardware client; the stack does not depend on policy tasks.
The deployment artifact contains
normalization, motor-target mapping, and source/inference metadata. Rust checks
the exact clean source commit and live dimensions; no checkpoint migrations or
model fingerprint cache are involved.

```bash
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
cargo run -p bello-policy -- --policy /path/to/policy.zip --source /path/to/bello_mujoco --check
cargo run -p bello-policy -- --policy /path/to/policy.zip --source /path/to/bello_mujoco
```

The second command accepts one stacked observation JSON array per stdin line
and emits action/motor-target JSON. It does not connect to hardware or enable
motors. Bundles contain `policy.onnx`, self-contained `physics.xml`, and, for
motion tracking, `motion.npz`. The loader exposes XML bytes for the native
MuJoCo adapter. Raw-observation inference does not initialize a physics model;
motion telemetry uses headless MuJoCo 3.12.0 for forward kinematics only.
Build from the repository root with `cargo build --locked -p bello-policy`.
For motion bundles, `--check` also loads the clip and initializes the physics
model without connecting to hardware. Motion observations resolve joints by
name: policy/actuator order need not match MuJoCo's internal qpos/qvel order.
On Linux, `nix develop .#rust-runtime` supplies the native libraries.
Outside that shell, set `MUJOCO_DYNAMIC_LINK_DIR` to the directory containing
the MuJoCo 3.12.0 shared library for the build, and make that library available
to the platform loader at runtime (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`).

`--telemetry` instead reads JSON objects containing `motor_position`,
`motor_velocity`, `orientation_xyzw`, `angular_velocity`, `linear_acceleration`,
`command`, and `torso_command` (three requested roll/pitch/yaw radians). Motor arrays
follow the bundle's coordinate order. Acceleration
is hardwared's gravity-removed local acceleration. Command is the requested
robot-frame velocity, not normalized joystick axes. Output includes the raw
observation, action, motor target, and gait state. The checkpoint owns gait
timing, directional starts, stopping buffers, command slew, and observation
history. Hardwared owns sensor filtering; this process does not filter twice.

For a motion bundle, `--telemetry` accepts `motor_position`, `motor_velocity`,
`position` (root, world metres), `orientation_xyzw`, `linear_velocity_world`,
`angular_velocity_local`, `reference_time` (seconds), and `initialize`.
Set `initialize: true` on the first sample or when restarting the clip. This
aligns the bundled reference root XY/yaw to the measured pose and resets history.
The runner samples the reference one control step ahead, computes anchor-link
kinematics, and checks the checkpoint's height/orientation/end-effector limits
for `full_state` bundles. For `proprioceptive` bundles, `position` and
`linear_velocity_world` may be omitted and are not consumed. Root orientation
and angular velocity are still required; kinematics use an arbitrary origin,
never a fabricated height estimate. Only orientation termination is available
in this telemetry mode. Simulation training/evaluation retains all terminations.
Output contains the stacked observation, raw/clipped action, and motor target.
No action low-pass filter is applied. Quaternion/vector operations use `glam`;
`npyz` reads the bundled NumPy arrays without a Python process.

SONIC bundles contain only the clips selected at export. Prerecorded playback
requires one to be chosen before history initialization:

```bash
cargo run --locked -p bello-policy -- --policy /path/to/sonic.zip \
  --source /path/to/bello_mujoco --clip walk --check
cargo run --locked -p bello-policy -- --policy /path/to/sonic.zip \
  --source /path/to/bello_mujoco --clip walk --telemetry
cargo run --locked -p bello-policy -- --policy /path/to/sonic.zip \
  --source /path/to/bello_mujoco --clip walk \
  --hardware-config rust/bello-policy/config/hardware.yaml
```

SONIC telemetry accepts `motor_position`, `motor_velocity`,
`orientation_xyzw`, `angular_velocity_local`, `reference_time`, and
`initialize`. A primary SONIC bundle cannot be armed. Buffered live HIL uses a
locomotion primary bundle and the optional `tracking` task in hardware YAML;
see [Tracking configuration and controller behavior](../../docs/policy_console.md#buffered-tracking-hil).
It shares the named-motion transition lifecycle and keeps physical commissioning
as a separate validation step.

Read-only hardware preflight:

```bash
cargo run --locked -p bello-policy -- --policy /path/to/policy.zip \
  --source /path/to/bello_mujoco --hardware-config rust/bello-policy/config/hardware.yaml
```

Preflight checks inventory, coordinate coverage, nominal targets/gains, IMU
frames, telemetry timestamps and freshness, group state, and IMU plausibility.
It performs no lease acquisition or motor writes.

For isolated emulator validation, add `--arm`. The runner hosts the operator
stream on `control.operator_listen` (port 50055). A neutral controller sample and
a two-second Back+Start chord enter Hold. The default Crane startup ramps to
nominal Hold; a one-second A press enters Stand. Optional `control.lying_start`
adds Back-down startup, with free ankles, balance confirmation and left-stick
straightening. See [assisted startup](../../docs/policy_console.md#assisted-back-down-startup)
for configuration and controller phases.
From Stand, D-pad left/right selects Locomotion, Motion, or configured
Tracking/Sit. Stand is an intermediate state, not a selectable mode. Within
Motion, D-pad up/down selects a preloaded named motion from the visible list;
holding A for one second starts it. Once fault cleanup is ready, a fresh
one-second A hold can clear the fault with neutral inputs.
See the [console guide](../../docs/policy_console.md) for transitions, connection
loss, packaging and direct service ownership. Stop cleanup automatically damps and
releases leases. No terminal confirmation is required.

Each locomotion/motion sample and the `startup_target` event includes
`motor_rails`, keyed by power-board name. Each reading contains `voltage_v`,
`current_a` (signed, including regeneration), `sample_age_ms`, `stale`, `sequence`,
and `connection_generation`. Age includes the daemon-reported sample age and
time since local receipt. Staleness uses the board's configured timeout or the
daemon's stale flag. Unknown, offline, malformed, or disconnected readings are
`null`, never a fabricated zero; a robot without power boards logs `{}`.
The logger reads a background telemetry cache and reconnects automatically.
Power telemetry does not add an RPC to the control loop or gate motor control.

Synthetic lifecycle checks cover rejection, stale feedback, operator abort,
signals, and ambiguous start timeouts. A synthetic hold actor has completed the
native policy/hardwared/MuJoCo lifecycle on isolated Lima vCAN. Trained-policy
dynamics commissioning remains pending. This is the only policy runner;
physical deployment stays gated on the trained-policy checks in
[the deployment guide](../../docs/hardware.md). Synthetic success is not
physical-deployment approval.
Synthetic motion telemetry checks compare observations/actions/targets against
JAX through root alignment, coupled coordinates, history reset, quaternion sign
changes, clip-end lookahead, and both BeyondMimic action mappings.
ONNX Runtime must provide the API supported by the pinned `ort` crate.

## Locomotion torso commands

Operator control requires `control.torso_command: [0, 0, 0]`. Optional live
following supplies torso commands only during Locomotion. Stand uses neutral
commands. Raw telemetry inference still accepts explicit torso commands.

## Live motion control

Pass the locomotion standing bundle to `--policy`. Configure all motion bundles
in the hardware YAML; they are preloaded before control acquisition:

```yaml
motions:
  - name: Wave
    policy: /absolute/path/to/wave.zip
    settings:
      start_time_seconds: 0.0
      transition_seconds: 0.2
      max_tilt_error_radians: 0.5
      max_raw_action_abs: 5.0
      pose:
        endpoint: http://127.0.0.1:50053
        reference_frame: odom
        tracked_frame: pelvis_link
        timeout_seconds: 2.0
        max_age_ms: 100.0
        max_linear_velocity: 5.0
        max_angular_velocity: 20.0
        max_imu_orientation_error: 0.25
```

The example is for a `full_state` bundle. For a newly trained `proprioceptive`
bundle, omit the entire `pose` mapping (or set `pose: null`); providing it is an
error, not an ignored endpoint. The runner does not connect to a pose service.
It converts hardwared IMU orientation/gyro into the floating-base frame using
the bundled mounting rotation, then computes the articulated anchor orientation
from measured joints. Estimator position, linear velocity, and absolute-height
guards are not used. The bundle, not a deployment override, selects the mode.
See [training modes](../../docs/motion_tracking.md#observation-modes).

For full-state motion, use the actual pose producer endpoint and floating-base frame. The stream must
provide world-frame linear velocity and root-local angular velocity, not just
position/orientation. Preflight checks both bundles' motor mapping and validates
each bundle's gains against hardware limits,
pose freshness, and pose/IMU agreement using the model's fixed IMU mounting
rotation. Both bundles must match the checked-out source commit.
A differently named pose frame is accepted only if the bundled model proves
it is rigidly fixed to and coincident with the floating base, including rotation.
Offset, rotated, or articulated frames are rejected rather than reinterpreted.

Crane Hold ramps to the standing bundle's nominal pose. Confirm Stand, select a named
motion from the visible list, then hold A for one second to blend into it.
Completion or the first
Ctrl-C blends back to Stand; another Ctrl-C, SIGTERM or holding L1 + R1 for one second stops control.
Malformed/stale feedback and rejected commands stop control. Standing and motion
retain their own gains, periods and histories. Targets and KP/KD blend over
`motions[].settings.transition_seconds`, including interrupted blends. Hold uses
standing gains; `control.gain_scale` multiplies all gains. Logs record unscaled
`motor_gains`. Velocity axes are used only in Locomotion. Named motions and
upper-body following may be configured together; following is disabled outside
Locomotion and never overrides a named motion.

The top-level `max_tilt_radians` remains the absolute upright limit for
preflight, HOLD, standing, and locomotion. During motion, the runner instead
compares the measured and reference gravity directions in the logical IMU
frame, accounting for its fixed mounting rotation. `motions[].settings.max_tilt_error_radians`
limits this angular deviation, not absolute lean; opposite-direction leans
are distinguishable, while world yaw is ignored. The example uses 0.5 rad
(about 29 degrees); commission this tolerance in the emulator before hardware.
This replaces the removed `motion.max_tilt_radians` setting, without an alias.

Entry blends the upright reference/tolerance into the starting clip reference.
Exceeding that envelope during entry or active motion starts a blended return
to standing from the last acknowledged target and current gains. Recovery does
not require following the failed clip's tilt reference or a prescribed tilt
trajectory. The absolute upright limit must pass before standing resumes.
Logs include observation mode, tilt error, limit, and reference gravity.
Both modes retain the scalar anchor-gravity check, hardware faults, feedback
freshness, IMU plausibility, and target limits. Full-state mode additionally
enforces pose freshness/IMU agreement, estimated velocity limits, and checkpoint
absolute-height terminations. Proprioceptive mode cannot check absolute height.

Motion tilt deviation, tracking termination and inference/action failures initiate the same
blended return to standing as normal completion, using the last acknowledged
target and current gains. Failed motion outputs are never sent. The operator
phase reports the reason during the return and motion log rows retain it in
`failure`. No automatic retry occurs. Feedback/pose validation, the absolute
upright limit at handover, hardware/lease errors, and any standing-policy/return failure
still end control with damping.

Standing and motion each use their own motor/joint target limits, intersected
with the live hardware motor bounds. Blends use a transition envelope covering
both policies and the stricter per-coordinate position-step limits. A blend
does not switch to its destination limits until the last acknowledged target
is valid there, even if the configured blend duration has elapsed. Hardwared
continues independently enforcing its motor and coupled-joint limits.

Synthetic gRPC tests cover both transitions, different gains and policy periods, mounted
IMU rotation, pose failures, rejection/abort, interruptions, and read-only
preflight. They do not establish trained-policy stability on the full emulator.

## Live upper-body references

Run direct XRoboToolkit/GMR retargeting as a separate CPU producer, with the PICO bridge
already streaming on port 50051:

```bash
uv run bello-run-reference asset=bello_mini \
  xrobot.endpoint=192.168.0.216:50051
```

Producer settings live in `configs/commands/reference.yaml`. Enable
`control.following.enabled` in the Rust hardware YAML and point its endpoint at
the producer. This makes tracking available; press Y in Locomotion to turn it
on or off. It starts off on each Locomotion entry and turns off on operator
loss. Off blends arm targets back to nominal with the same speed and transition
limits used for a source outage. Operator status distinguishes availability,
enabled intent, and live reference reception. The protocol is `proto/reference.proto`: named URDF joint angles,
velocities, and three torso-command angles, not motor commands. The producer never
connects to hardwared and does not load policy weights.

The producer extracts torso orientation from forward kinematics of the same GMR
pose as the arms. It removes the nominal CAD orientation and pelvis heading,
not pelvis tilt: roll/pitch remain ground-relative lean and yaw is pelvis-relative
twist. This changes neither the upstream GMR solver nor its targets.

Rust samples one reference per tick. Torso angles are clamped to the checkpoint's
trained ranges and passed through its normal command slew before inference.
The policy retains waist/leg motor ownership; only checkpoint-held arm targets
are merged afterward. Zero trained torso ranges keep that axis neutral.
Gamepad walking/turning commands remain independent of body tracking.

Rust validates robot/joint names and numeric shape and owns arm speed limits plus
connection/outage blending. Missing, stale, or unsafe references return the arms
to the checkpoint nominal pose and slew torso commands back to zero. Malformed
messages or terminal producer errors stop control. The polling worker stores
only the latest result and cannot block the policy loop. Joint references and
policy motor ownership must be disjoint. Synthetic tests cover live following,
producer outage, malformed references, torso limits/slew, and unchanged
policy-owned targets. Update producer and runner together: available references
must contain all three torso angles. Retraining with torso commands and live
tracking/emulator commissioning are required before deploying this behavior.

Named motion entries require `source` alongside `name`, `policy`, and `settings`.
This is the clean source checkout for that individual bundle; `--source` remains
the source checkout for the main locomotion policy.
