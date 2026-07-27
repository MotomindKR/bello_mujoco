# RobStride HIL

The RobStride runner presents the canonical 60 kg sphere-hand Bello morphology
through the same motor, IMU, and pose interfaces used around physical hardware,
while native MuJoCo supplies dynamics.

Public modes and controls are documented under
[`bello-run-robstride`](commands.md#bello-run-robstride).

## Modes

| Mode | Interfaces | Control source | Limit source |
| --- | --- | --- | --- |
| `hil` | SocketCAN, EBIMU serial, PoseService, optional viewer | RobStride CAN commands | Motor catalog and transmission mapping |
| `prescribed_gains` | Optional viewer only | Generated nominal targets and impedance gains | Canonical MuJoCo actuator profile |

HIL is for controller and integration testing. Prescribed-gain mode is for
morphology, transmission, and gain commissioning; it is not policy evaluation
and does not open CAN, serial, or PoseService.

## Canonical inventory

The HIL motor contract is:

- `configs/robstride/motor_kinds.yaml`: motor kinds, protocol ranges, and torque
  limits;
- `configs/robstride/lower_body.yaml`: lower-body bus identities and targets;
- `configs/robstride/upper_body_sphere_hands.yaml`: upper-body inventory for the
  fixed-head, sphere-hand morphology.

Differential ankles remain two coupled tendon motors per side. Initialization
validates unique bus/CAN identities, motor-kind ranges, optional-joint coverage,
complete coupling groups, transmission ownership, and torque enforcement.

The public CLI always uses the canonical physical asset with a generated pelvis
IMU. HIL defaults to 100 Hz motor feedback, a 250 Hz EBIMU stream at
`/dev/ttyIMU-mujoco`, and PoseService on port 50052.

## Linux interfaces

HIL requires the configured SocketCAN interfaces and pseudo-terminal IMU
consumer. `scripts/setup_hil.sh` creates the default VCAN interfaces; it is not
needed for prescribed-gain mode.

Logical bus mappings use `name=interface` entries. The defaults map the
configured `can0` and `can1` buses to `vcan0` and `vcan1`.

## Simulation and torso guide

Native MuJoCo runs at a 0.0025-second physics timestep. Real-time pacing is
optional, and `max_steps` counts MuJoCo physics steps.

The commissioning guide is enabled by default and welds the torso/root assembly
to a world-Z slider, limiting downward motion while allowing the rest of the
dynamics to settle. It can be released interactively or after
`release_after_seconds`; automatic release uses simulated time. When the viewer
is active, the runtime can reset and re-engage the guide for another
commissioning cycle.

## Published state

HIL publishes:

- active RobStride feedback with simulated position, velocity, torque,
  temperature, and fault state;
- pelvis orientation and specific force through the EBIMU serial protocol;
- world-frame pelvis translation/orientation through PoseService;
- native viewer state when not headless.

The default IMU model uses seed 0 and adds independent Gaussian noise with
per-axis standard deviations of 0.005 rad for orientation and 0.1 m/s² for
linear acceleration. PoseService remains ground-truth simulation state and is
intentionally separate from the noisy IMU path.
