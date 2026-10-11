# Native hardware emulator

The setup script creates both IMU and power-board UART pairs:
`/dev/ttyIMU` / `/dev/ttyIMU-mujoco` and
`/dev/ttyPOWER` / `/dev/ttyPOWER-mujoco`.
The maintained native emulator and hardwared vCAN profiles enable a synthetic
AGX power board. Battery/rail readings are explicit fixtures, not a battery
physics model. Power data reaches the native visualizer's Power tab over the
existing hardwared RPC connection. See the
[stack power-board guide](../deps/bello-stack/docs/power_board.md) for settings,
fault injection, physical opt-in and limitations.

The emulator lives in `deps/bello-stack/components/bello-mujoco`. It uses
headless `mujoco-rs`, MuJoCo 3.12.0, and the shared RobStride protocol engine.
It builds directly from canonical URDF/hardware descriptions, independently of
any policy or deployment bundle. There is no Python emulator or compatibility
launcher.

## Run

On Linux (or inside [Lima](macos_vcan.md)), create the virtual CAN and serial
interfaces, then start the emulator:

```bash
sudo ./scripts/setup_emulator.sh start
nix develop .#rust-runtime
cargo run --locked --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  run --config deps/bello-stack/components/bello-mujoco/config/emulator_mini.yaml
```

Use `emulator_bello.yaml` for Bello. Start hardwared and the optional state
estimator/visualizer in separate terminals using the corresponding vCAN
profiles. Never run multiple emulators on the same CAN/serial endpoints.
The native emulator accepts virtual CAN interfaces only.
The setup script creates `vcan0` through `vcan3` for the four Mini buses and sets
their transmit queue lengths to 65,536 frames. Bello uses the first two buses.
This kernel setting is separate from hardwared's bounded control-batch queues;
command deadlines still apply.

Enter releases the torso guide. Enter again resets the pose and disarms motors.
Without an interactive terminal, the guide stays engaged. Ctrl-C/SIGTERM closes
CAN, serial, and pose services before rendering an optional recording.
`--duration-seconds N` bounds an unattended run.

## Configuration and physics

The [stack runtime guide](../deps/bello-stack/components/bello-mujoco/README.md)
documents profiles and recording. Runtime YAML selects interfaces, endpoints,
feedback rates, command latency, gain scaling, IMU noise, faults, and torso guide.
Model YAML selects the hardware description, timestep, initial pose, collision
exclusions, and optional payload inertias. For local overrides, copy profiles
under ignored `outputs/` and adjust their relative `model`/`hardware` paths;
do not dirty the pinned descriptions used by a policy.

Native joint/tendon transmissions supply motor positions, velocities, and
virtual-work torque mapping. The motor catalog supplies force limits. The
builder preserves source link frames and inertias and checks ownership, limits,
and nominal self-collisions at startup. Visual meshes are excluded from live
physics and loaded separately for offline video.

Command latency affects MIT targets, not enable/stop traffic. Gain scaling and
IMU noise are explicit diagnostic settings; report them when comparing runs.
The serial IMU publishes the CAD source-frame orientation and specific force;
hardwared transforms these to its configured output frame. PoseService publishes
ground-truth link-origin state independently of IMU noise.

Payload entries add spherical inertia at existing inertial-free attachment
frames. Set their mass and radius in a local model profile, not in the policy.

## Validation and recordings

```bash
cargo run --locked --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  check --config deps/bello-stack/components/bello-mujoco/config/bello_mini.yaml \
  --export-physics outputs/mini-physics.xml
```

Enable `video` in a local runtime profile as described in the stack guide.
Only released-torso poses are recorded. Rasterization uses CPU Mesa llvmpipe and
ffmpeg after transport shutdown, with explicit memory/time limits. Recording
or rendering failures fail the command; existing videos are never overwritten.
The accompanying `.recording/` directory can be rendered again with the same
native executable. Keep these generated artifacts out of Git.

Common-model Python/Rust motor replay was numerically equivalent, and isolated
CAN/serial/pose/shutdown/video checks cover both robots. This verifies the runtime
replacement, not trained-policy stability or physical hardware calibration.

For ground-truth versus state-estimator drift, run:

```bash
uv run python -m bello_mujoco.hardware.pose_metrics --duration-seconds 30
```

See [Lima networking](macos_vcan.md#network-topology) for service ports and routes.
