# Hardware deployment

The Python hardware adapter reconstructs a checkpoint's saved observations and
motor targets, and communicates through `bello-hardwared`'s client API. The
visualizer and gRPC transport are unchanged. Canonical hardware mappings,
directions, IDs, and limits live in pinned `bello-stack` descriptions.

## Read-only preflight

Start hardwared and the remote-command service, then:

```bash
uv run bello-run-hardware task=locomotion checkpoint_path=checkpoints/RUN \
  endpoint=http://127.0.0.1:50051 remote_endpoint=http://127.0.0.1:50054 \
  upper_body_tracking_enabled=false
```

Preflight validates the exact source commit, robot inventory, motor-coordinate
mapping, IMU frame, fresh telemetry, and nominal pose without acquiring control.
Use [Lima/vCAN](macos_vcan.md) before physical hardware. In Lima, the native Mac
remote service is reached through `host.lima.internal`, not guest loopback.

## Armed operation

Only add `arm=true` after reviewing the robot, calibration, emergency stop, and
support arrangements. Follow the prompts:

1. Support the robot and confirm `HOLD` to acquire control and ramp to nominal.
2. After the ramp and settling, confirm `RUN` to start policy control.
3. On shutdown, support the damped robot and confirm `IDLE` to disable/release.

Motion deployment additionally uses a standing checkpoint and the `MOTION`
transition. Its pose-estimator/reference requirements are checked before entry.
Checkpoint defaults and public overrides are in
[`configs/commands/hardware.yaml`](../configs/commands/hardware.yaml).

The runner continuously checks group state, command rejection, telemetry age,
finite observations/actions, orientation/acceleration plausibility, target
bounds, and control-loop deadlines. Errors follow the damp/disable/release
safety path; do not bypass hardwared position-step or thermal protections.
Optional `policy_log_path` records policy steps for diagnosis.

## Coordinates and observations

Policy actions target physical motor coordinates. Native ankle tendons encode
the mechanical coupling. Motor telemetry is converted to joint kinematics for
observations and safety, not to reinterpret policy outputs. Gains are explicit
per-motor values. IMU placement and orientation come from the robot model;
training and native evaluation use the same observation/filter functions.
Emulator pose messages report the link origin's world position/linear velocity
and body-frame angular velocity, not the offset inertial center's velocity.
Command slew comes exclusively from the checkpoint's gait scheduler. The runner
has no separate joystick slew limiter; hardwared feedback filtering is not
reapplied in the policy process.

Remote commands are semantic forward/right/yaw intent mapped into the robot's
configured command frame. Do not assume Bello and Mini share their forward axis.

## Live upper body

RTMW upper-body following requires the producer endpoint and a licensed local
`SMPLX_NEUTRAL.npz`. The live GMR bridge retains frame/shape, confidence,
freshness, monotonic-time, joint-speed, and limit checks. Temporarily missing
references blend arms to nominal; malformed/incompatible references fail.
The lower-body policy remains responsible for locomotion.

FastSAM/RTMW live integration libraries are retained. Offline motion conversion,
TWIST/TWIST2 deployment, single-joint chirp tools, and elbow-yaw keyboard jogging
are removed. Per-source revision fingerprints are replaced by the root commit.

## Independent hands

`bello-run-hands` validates RTMW hand mapping without arming by default. Explicit
arming controls Linker hands over classic CAN directly; it does not route hand
targets through hardwared. Configuration is in
[`configs/commands/hands.yaml`](../configs/commands/hands.yaml). Verify bus,
device IDs, side mapping, and emergency stop before enabling hand motion.

## ONNX

`bello-export-policy` emits one ZIP bundle with `policy.onnx`, mesh-free
`physics.xml`, and, for motion
tracking, `motion.npz`. This repository's `rust/bello-policy` provides the Rust CPU consumer.
Its JSON-line inference tool performs no hardware writes. Resumable training
state remains a separate checkpoint directory.

The Rust emulator builds from canonical robot descriptions, not the policy bundle.
Its CAN/IMU/pose process has completed an isolated Lima lifecycle with hardwared
and a synthetic ONNX hold actor. Software video and isolated transport checks
cover both robots; trained-policy stability is still pending. See [emulator usage](emulator.md).

The native policy tool supports `--telemetry` for stateful observation/inference
parity and `--hardware-config rust/bello-policy/config/hardware.yaml` for read-only
preflight. Neither mode acquires control. An explicit `--arm` enables the
experimental locomotion HOLD/RUN/IDLE lifecycle for isolated emulator validation.
Live retargeting runs separately with `bello-run-reference`; Rust consumes its
joint references and owns outage-to-nominal blending. Native motion tracking
supports standing-to-clip transitions and a first-interrupt return to standing;
malformed/stale feedback stops control. Synthetic integration checks cover these
paths; physical commissioning remains separate.
See its [usage guide](../rust/bello-policy/README.md).
