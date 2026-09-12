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

Motion telemetry is read-only: live pose streaming and the standing/motion
operator transitions are not yet connected to the native hardware path.

Read-only hardware preflight:

```bash
cargo run --locked -p bello-policy -- --policy /path/to/policy.zip \
  --source /path/to/bello_mujoco --hardware-config rust/bello-policy/config/hardware.yaml
```

Preflight checks inventory, coordinate coverage, nominal targets/gains, IMU
frames, telemetry timestamps and freshness, group state, and IMU plausibility.
It performs no lease acquisition or motor writes.

For isolated emulator validation, add `--arm` and configure the `control` section
in the hardware YAML. This locomotion-only path requires an interactive terminal:
`HOLD` acquires control and ramps to nominal, `RUN` starts inference, and `IDLE`
confirms motor disable after stopping. Failures, remote abort, SIGINT, or SIGTERM
request damping first. Without an `IDLE` confirmation, it sends no explicit
disable; hardwared's watchdog and lease-expiry safety remain authoritative.

Remote commands require a neutral sample before accepting an armed controller;
stale/disconnected input requests standing. The checkpoint owns command mapping
and slewing. Optional JSONL policy logs use a new file, never overwrite one.

Synthetic lifecycle checks cover rejection, stale feedback, remote abort,
signals, and ambiguous start timeouts. A synthetic hold actor has completed the
native policy/hardwared/MuJoCo lifecycle on isolated Lima vCAN. Trained-policy
dynamics parity and the motion operator lifecycle remain pending; keep the Python
runner for regular operation until those checks finish.
Synthetic motion telemetry checks compare observations/actions/targets against
JAX through root alignment, coupled coordinates, history reset, quaternion sign
changes, clip-end lookahead, and both BeyondMimic action mappings.
ONNX Runtime must provide the API supported by the pinned `ort` crate.

## Live upper-body references

Run RTMW/GMR retargeting as a separate CPU producer:

```bash
uv run bello-run-reference asset=bello_mini rtmw.neutral_model_path=/path/to/SMPLX_NEUTRAL.npz
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
live camera/retargeter commissioning is separate.
