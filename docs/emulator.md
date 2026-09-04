# RobStride emulator

The RobStride emulator presents the canonical 55.068 kg sphere-hand Bello
morphology, plus an optional symmetric hand payload, through the same motor,
IMU, and pose interfaces used around physical hardware, while native MuJoCo
supplies real-time dynamics.

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

## Canonical inventory

The emulated motor contract is:

- `deps/bello-stack/descriptions/bello/hardware.yaml`: logical buses, motors,
  coordinates, directions, group membership, and the canonical robot-description
  reference;
- `deps/bello-stack/descriptions/robstride/models.yaml`: protocol ranges and the
  voltage, current, and speed values reported through parameter registers;
- the URDF referenced by the hardware description: direct joints and
  differential-transmission ownership.

The sphere-hand emulator selects the canonical `lower_body` and `upper_body`
groups. Differential ankles remain two coupled tendon motors per side.
Initialization validates unique
bus/CAN identities, model ranges, optional-joint coverage, complete coupling
groups, URDF transmission ownership, and torque enforcement. The command
requires a source checkout because the wheel intentionally omits hardware
topology and model catalogs.

The command defaults to the canonical physical `bello` asset with a generated
pelvis IMU. Its public asset selector rejects `bello_mini` explicitly because
Mini has no calibrated hardware description yet. Bello publishes 100 Hz motor
feedback, a 200 Hz EBIMU stream at
`/dev/ttyIMU-mujoco`, and PoseService on port 50052.

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
- pelvis orientation and specific force through the EBIMU serial protocol;
- world-frame pelvis translation/orientation through PoseService.

The default IMU model uses seed 0 and adds independent Gaussian noise with
per-axis standard deviations of 0.005 rad for orientation and 0.1 m/s² for
linear acceleration. PoseService remains ground-truth simulation state and is
intentionally separate from the noisy IMU path.

## Estimator comparison

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
