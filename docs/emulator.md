# RobStride emulator

The RobStride emulator presents either the canonical 55.068 kg sphere-hand
Bello or the reviewed Bello Mini hardware contract through the same motor,
IMU, and pose interfaces used around physical hardware, while native MuJoCo
supplies real-time dynamics. An optional symmetric hand payload is available
for either asset.

Public controls are documented under
[`bello-run-emulator`](commands.md#bello-run-emulator).

## Runtime

| Concern | Behavior |
| --- | --- |
| Dynamics | Native MuJoCo at a 0.0025-second timestep, paced against wall-clock time. |
| Control | MIT impedance commands received through RobStride SocketCAN, delayed by 10 ms by default, and applied through the physical transmission model. |
| Published state | Active motor feedback, EBIMU serial samples, and pelvis PoseService updates. |
| Process interface | Headless. No MuJoCo viewer or torso-control window is opened. |
| Limits | Motor-catalog torque limits and transmission ownership are always enforced. |

The emulator is intended for hardware-interface and controller integration
testing. Policy checkpoints are evaluated with `bello-eval`.

Training and native emulation target MuJoCo 3.12.0. The Rust executable in
`deps/bello-stack/components/bello-mujoco` builds its model directly from the
canonical URDF/hardware description, independently of policy bundles. It uses
published `mujoco-rs` 6.0.1 without rendering features and retains native ankle
tendons, EBIMU serial output, pose gRPC, and the interactive torso guide.

Native model profiles accept explicit payload entries (`frame`, `mass_kg`,
`radius_m`) at inertial-free attachment frames. See the
[native runtime guide](../deps/bello-stack/components/bello-mujoco/README.md).
Python actuator/torso-guide rebuilds preserve the loaded inertias without XML
rounding. Common-model Python/Rust motor-command replay agrees numerically;
this does not establish trained-policy stability through the live transport stack.

In the Linux `nix develop .#rust-runtime` shell:

```bash
cargo run --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  check --config deps/bello-stack/components/bello-mujoco/config/bello_mini.yaml

cargo run --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  run --config deps/bello-stack/components/bello-mujoco/config/emulator_mini.yaml
```

Use the corresponding `bello.yaml` / `emulator_bello.yaml` profiles for Bello.
See the [native runtime guide](../deps/bello-stack/components/bello-mujoco/README.md)
for isolated transport tests and runtime settings. The Python workflow below
remains available pending policy-driven parity and native video-workflow migration.
Never run both emulators on the same CAN/serial endpoints.

## Canonical inventory

The emulated motor contract is:

- the selected asset's `deps/bello-stack/descriptions/<asset>/hardware.yaml`:
  logical buses, motors, coordinates, directions, group membership, IMU source
  and output frames, and the canonical robot-description reference;
- `deps/bello-stack/descriptions/robstride/models.yaml`: protocol ranges and the
  voltage, current, and speed values reported through parameter registers;
- the URDF referenced by the hardware description: direct joints and
  differential-transmission ownership.

The emulator selects the configured `lower_body` and `upper_body` groups.
Differential ankles remain two coupled tendon motors per side.
Initialization validates unique
bus/CAN identities, model ranges, optional-joint coverage, complete coupling
groups, URDF transmission ownership, and torque enforcement. The command
requires a source checkout because the wheel intentionally omits hardware
topology and model catalogs.

The command defaults to `asset=bello`; select Mini with
`asset=bello_mini`. Both publish 100 Hz motor feedback, a 200 Hz EBIMU stream at
`/dev/ttyIMU-mujoco`, and PoseService on port 50052.

Mini uses the 28-motor description, reviewed directions, and differential
mapping pinned from `bello-stack`. The ROBSTRIDE_02 ankle entry currently has
protocol limits but no electrical model, so its unavailable current-limit and
bus-voltage parameter telemetry is reported as zero rather than fabricated.

## Bello Mini vCAN stack

On macOS, run this entire Linux SocketCAN workflow in the documented
[Lima vCAN environment](macos_vcan.md).

After creating the VCAN and pseudo-terminal interfaces, run these processes in
separate terminals:

```bash
uv run bello-run-emulator asset=bello_mini

cd deps/bello-stack
cargo run -p bello-hardwared -- \
  --config components/bello-hardwared/config/bello-mini-vcan.yaml

cd deps/bello-stack
cargo run -p bello-state-estimatord -- \
  --config components/bello-state-estimatord/config/bello-mini-vcan.yaml

cd deps/bello-stack
cp components/bello-visualizer/config/bello-mini.example.yaml \
  components/bello-visualizer/config/bello-mini.yaml
cargo run -p bello-visualizer -- \
  --config components/bello-visualizer/config/bello-mini.yaml
```

The maintained visualizer example and ignored local `bello-mini.yaml` use the
state estimator on port 50053. Emulator ground truth remains available on port
50052 for pose-error measurement. The visualizer continues to publish joystick
intent on port 50054.

A Mini locomotion checkpoint can then perform read-only preflight through the
emulated hardware stack:

```bash
uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=/absolute/path/to/checkpoint \
  endpoint=http://127.0.0.1:50051 \
  remote_endpoint=http://127.0.0.1:50054 \
  upper_body_tracking_enabled=false
```

The Mini policy controls 14 waist/leg joints and holds all 14 arm joints at its
checkpoint nominal pose. Add `arm=true` only after reviewing the normal
interactive HOLD/RUN safety sequence. The same command contract applies to vCAN
and physical hardware.

The public `command_latency_seconds` option delays only MIT motor-control
frames before they affect MuJoCo. Motor enable/stop, parameter access, and
other lifecycle traffic remains immediate. Set it to `0` to disable synthetic
command latency.

`motor_kp_scale` and `motor_kd_scale` multiply the effective decoded MIT gains
inside MuJoCo without changing the targets or feedback protocol. Their default
of one is the canonical emulator. They are diagnostic controls for reproducing
stiffness and damping mismatch; report both values with any comparison and do
not treat a non-default result as a calibrated hardware model.

Set `video_path` to an MP4 destination to capture a diagnostic run. State
sampling begins when Enter releases the torso guide and pauses whenever the
guide is engaged. Frames are rendered only after the emulator exits, keeping
offscreen rendering out of the real-time motor and sensor loop.

Set `end_effector_payload_mass_kg` to the carried mass at each hand. The value
uses the same asset-generation path and per-hand convention as motion training
and evaluation; for example, two 10 lb loads use
`end_effector_payload_mass_kg=4.5359237`. The default is zero.

## Linux interfaces

The emulator requires the configured SocketCAN interfaces and pseudo-terminal
IMU consumer. The Nix shell provides `ip`, `modprobe`, and `socat`;
`scripts/setup_emulator.sh` creates the default VCAN interfaces and serial pair:

```bash
sudo ./scripts/setup_emulator.sh start
sudo ./scripts/setup_emulator.sh status
```

Use `sudo ./scripts/setup_emulator.sh stop` to remove the serial pair. The VCAN
interfaces are deliberately left up.

Logical bus mappings use `name=interface` entries. The defaults map the
configured `can0` and `can1` buses to `vcan0` and `vcan1`.

## Simulation and torso guide

The commissioning guide always welds the torso/root assembly
to a world-Z slider, limiting downward motion while allowing the rest of the
dynamics to settle. Press Enter in the emulator terminal to release it. Press
Enter again to reset the robot and re-engage the guide. When standard input is
not an interactive terminal, the guide remains engaged.

## Published state

The emulator publishes:

- active RobStride feedback with simulated position, velocity, torque,
  temperature, and fault state;
- raw `source_frame` IMU orientation and specific force through the EBIMU serial
  protocol; hardwared applies the URDF transform and publishes the configured
  `output_frame`;
- world-frame pelvis translation/orientation through PoseService.

The default IMU model uses seed 0 and adds independent Gaussian noise with
per-axis standard deviations of 0.005 rad for orientation and 0.1 m/s² for
linear acceleration. PoseService remains ground-truth simulation state and is
intentionally separate from the noisy IMU path.

## State-estimator comparison

With MuJoCo ground truth on port 50052 and `bello-state-estimatord` on port
50053, collect time-aligned pose errors with:

```bash
uv run python -m bello_mujoco.hardware.pose_metrics \
  --duration-seconds 30
```

The report removes the initial XYZ translation offset before calculating
position drift, because estimator odometry and MuJoCo world position can use
different origins. It reports per-axis and total position RMSE, maximum and
final position error, raw absolute-Z RMSE, quaternion angular error, paired
sample count, and maximum pairing-time error. Keep the robot guided for a
stationary-noise baseline, then repeat while stepping to measure contact
handoff behavior.

Use the estimator profile matching the emulated robot:

```bash
cd deps/bello-stack

# Bello
cargo run -p bello-state-estimatord -- \
  --config components/bello-state-estimatord/config/bello-vcan.yaml

# Bello Mini
cargo run -p bello-state-estimatord -- \
  --config components/bello-state-estimatord/config/bello-mini-vcan.yaml
```

Mini visualizer runs use estimator output on port 50053 by default. Emulator
ground truth on port 50052 remains independent for comparison. The Mini
estimator resolves its offset, yawed IMU frame from the canonical URDF and
publishes `odom_from_pelvis_link`. Its IMU pose, ankle reductions, motor
directions, and encoder zeros remain physical-calibration requirements.
