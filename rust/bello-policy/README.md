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
MuJoCo adapter; it does not initialize physics during inference.
Build from the repository root with `cargo build --locked -p bello-policy`.
On Linux, `nix develop .#rust-runtime` supplies the native libraries.

`--telemetry` instead reads JSON objects containing `motor_position`,
`motor_velocity`, `orientation_xyzw`, `angular_velocity`, `linear_acceleration`,
and `command`. Motor arrays follow the bundle's coordinate order. Acceleration
is hardwared's gravity-removed local acceleration. Command is the requested
robot-frame velocity, not normalized joystick axes. Output includes the raw
observation, action, motor target, and gait state. The checkpoint owns gait
timing, directional starts, stopping buffers, command slew, and observation
history. Hardwared owns sensor filtering; this process does not filter twice.

Read-only hardware preflight:

```bash
cargo run --locked -p bello-policy -- --policy /path/to/policy.zip \
  --source /path/to/bello_mujoco --hardware-config rust/bello-policy/config/hardware.yaml
```

Preflight checks inventory, coordinate coverage, nominal targets/gains, IMU
frames, telemetry timestamps and freshness, group state, and IMU plausibility.
It performs no lease acquisition or motor writes. Armed operation, remote-command
transport, motion reference sampling, and live upper-body following still use
the Python runner until native migration and safety parity are complete.
ONNX Runtime must provide the API supported by the pinned `ort` crate.
