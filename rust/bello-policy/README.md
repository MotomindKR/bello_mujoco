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
On Linux, `nix develop .#rust-runtime` supplies the native libraries.
Outside that shell, set `MUJOCO_DYNAMIC_LINK_DIR` to the directory containing
the MuJoCo 3.12.0 shared library for the build, and make that library available
to the platform loader at runtime (`LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH`).

`--telemetry` instead reads JSON objects containing `motor_position`,
`motor_velocity`, `orientation_xyzw`, `angular_velocity`, `linear_acceleration`,
and `command`. Motor arrays follow the bundle's coordinate order. Acceleration
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
kinematics, and checks the checkpoint's height/orientation/end-effector limits.
Output contains the stacked observation, raw/clipped action, and motor target.
No action low-pass filter is applied. Quaternion/vector operations use `glam`;
`npyz` reads the bundled NumPy arrays without a Python process.

Read-only hardware preflight:

```bash
cargo run --locked -p bello-policy -- --policy /path/to/policy.zip \
  --source /path/to/bello_mujoco --hardware-config rust/bello-policy/config/hardware.yaml
```

Preflight checks inventory, coordinate coverage, nominal targets/gains, IMU
frames, telemetry timestamps and freshness, group state, and IMU plausibility.
It performs no lease acquisition or motor writes.

For isolated emulator validation, add `--arm` and configure the `control` section
in the hardware YAML. Control requires an interactive terminal:
`HOLD` acquires control and ramps to nominal, `RUN` starts inference, and `IDLE`
confirms motor disable after stopping. Failures, remote abort, SIGINT, or SIGTERM
request damping first. Without an `IDLE` confirmation, it sends no explicit
disable; hardwared's watchdog and lease-expiry safety remain authoritative.

The remote-command service is not a startup dependency: the runner starts with
zero velocity commands and connects/reconnects in the background. Each connection
requires a neutral sample before accepting an armed controller;
stale/disconnected input requests standing, not motor disable. Received aborts
remain latched and malformed messages still stop control. The checkpoint owns command mapping
and slewing. Optional JSONL policy logs use a new file, never overwrite one.

Synthetic lifecycle checks cover rejection, stale feedback, remote abort,
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

## Live motion control

Pass the motion bundle to `--policy`. In a local copy of the hardware YAML,
replace `motion: null` with a motion mapping such as:

```yaml
motion:
  standing_policy: /absolute/path/to/standing.zip
  start_time_seconds: 0.0
  transition_seconds: 0.2
  max_tilt_radians: 1.22
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

Use the actual pose producer endpoint and floating-base frame. The stream must
provide world-frame linear velocity and root-local angular velocity, not just
position/orientation. Preflight checks both bundles' motor mapping and gains,
pose freshness, and pose/IMU agreement using the model's fixed IMU mounting
rotation. Both bundles must match the checked-out source commit.

`HOLD` ramps to the standing policy's nominal pose; `RUN` starts standing with
zero velocity command. Type `MOTION` to blend into the clip. Completion or the
first Ctrl-C during motion blends back to standing; another Ctrl-C stops and
requests damping. SIGTERM, remote abort, malformed/stale pose, hardware faults,
or rejected commands take the STOP path, not a standing recovery. `IDLE` remains
required for explicit disable. Standing and motion retain their own control
periods and histories. The remote is monitored for aborts, but its velocity
axes are ignored in this workflow. Upper-body following must be disabled.

Standing and motion each use their own motor/joint target limits, intersected
with the live hardware motor bounds. Blends use a transition envelope covering
both policies and the stricter per-coordinate position-step limits. A blend
does not switch to its destination limits until the last acknowledged target
is valid there, even if the configured blend duration has elapsed. Hardwared
continues independently enforcing its motor and coupled-joint limits.

Synthetic gRPC tests cover both transitions, different policy periods, mounted
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
the producer. The protocol is `proto/reference.proto`: named URDF joint angles
and velocities, not motor commands or world-space poses. The producer never
connects to hardwared and does not load policy weights.

Rust follows only checkpoint-held coordinates, validates robot/joint names and
numeric shape, and owns speed limits plus connection/outage blending. Missing,
stale, or unsafe references transition to the checkpoint nominal pose. Malformed
messages or terminal producer errors stop control. The polling worker stores
only the latest result and cannot block the policy loop. Joint references and
policy motor ownership must be disjoint. Synthetic tests cover live following,
producer outage, malformed references, and unchanged policy-owned targets;
live tracking/retargeter commissioning is separate.
