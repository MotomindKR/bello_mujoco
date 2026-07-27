# RobStride emulator

The RobStride emulator presents the canonical 60 kg sphere-hand Bello
morphology through the same motor, IMU, and pose interfaces used around
physical hardware, while native MuJoCo supplies real-time dynamics.

Public controls are documented under
[`bello-run-emulator`](commands.md#bello-run-emulator).

## Fixed runtime contract

The emulator has one execution mode:

| Concern | Behavior |
| --- | --- |
| Dynamics | Native MuJoCo at a 0.0025-second timestep, paced against wall-clock time. |
| Control | RobStride commands received through SocketCAN and applied through the physical transmission model. |
| Published state | Active motor feedback, EBIMU serial samples, and pelvis PoseService updates. |
| Visual interface | Native passive viewer on every run, plus the Bello torso-control window when the guide is enabled. A graphical display is required. |
| Limits | Motor-catalog torque limits and transmission ownership are always enforced. |

This fixed runtime is specifically for hardware-interface and controller
integration testing; it is not policy evaluation.

## Canonical inventory

The emulated motor contract is:

- `configs/robstride/motor_kinds.yaml`: motor kinds, protocol ranges, and torque
  limits;
- `configs/robstride/lower_body.yaml`: lower-body bus identities and targets;
- `configs/robstride/upper_body_sphere_hands.yaml`: upper-body inventory for the
  fixed-head, sphere-hand morphology.

Differential ankles remain two coupled tendon motors per side. Initialization
validates unique bus/CAN identities, motor-kind ranges, optional-joint coverage,
complete coupling groups, transmission ownership, and torque enforcement.

The command always uses the canonical physical asset with a generated pelvis
IMU. It publishes 100 Hz motor feedback, a 250 Hz EBIMU stream at
`/dev/ttyIMU-mujoco`, and PoseService on port 50052.

## Linux interfaces

The emulator requires the configured SocketCAN interfaces and pseudo-terminal
IMU consumer. `scripts/setup_emulator.sh` creates the default VCAN interfaces.

Logical bus mappings use `name=interface` entries. The defaults map the
configured `can0` and `can1` buses to `vcan0` and `vcan1`.

## Simulation and torso guide

`max_steps` counts MuJoCo physics steps and is the optional bounded-run control.

The commissioning guide is enabled by default and welds the torso/root assembly
to a world-Z slider, limiting downward motion while allowing the rest of the
dynamics to settle. It can be released from the controls window, by pressing
Enter in an interactive terminal, or after `release_after_seconds`; automatic
release uses simulated time. The controls can reset and re-engage the guide for
another commissioning cycle.

## Published state

The emulator publishes:

- active RobStride feedback with simulated position, velocity, torque,
  temperature, and fault state;
- pelvis orientation and specific force through the EBIMU serial protocol;
- world-frame pelvis translation/orientation through PoseService;
- native viewer state on every run.

The default IMU model uses seed 0 and adds independent Gaussian noise with
per-axis standard deviations of 0.005 rad for orientation and 0.1 m/s² for
linear acceleration. PoseService remains ground-truth simulation state and is
intentionally separate from the noisy IMU path.
