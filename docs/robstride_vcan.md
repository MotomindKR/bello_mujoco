# RobStride VCAN MuJoCo Runner

`bello-robstride-runner` runs the Bello full-body MuJoCo model while presenting
each configured physical actuator as a RobStride device on Linux SocketCAN. It
is intended to replace the motors in end-to-end integration tests; the existing
Motomind motor gateway talks to it with the same extended CAN frames it uses on
hardware.

Simulation uses classic CPU MuJoCo (`mujoco.MjData` and `mujoco.mj_step`). The
runner explicitly loads no MJX model and does not use JAX, Warp, or CUDA. The
optional interactive viewer may use the system graphics stack only for display;
`headless=true` avoids rendering entirely.

The implementation follows the RobStride private CAN protocol documented in
the [official product information repository](https://github.com/RobStride/Product_Information)
and is byte-compatible with the native driver in `~/motomind_platform`.

## Emulated behavior

The runner implements the platform-visible motor contract:

- 29-bit CAN 2.0 extended frames
- catalog-driven position, velocity, torque, `kp`, and `kd` encoding for the
  configured RobStride motor kinds
- device discovery, MIT/impedance control, enable, stop/fault clear, and
  mechanical-zero commands
- parameter reads and writes, including `run_mode` (`0x7005`), mechanical
  position (`0x7019`), and the parameters in the current RobStride manuals
- data-save acknowledgement and active-report enable/disable
- immediate type-2 feedback after commands and type-24 periodic active reports
- mode, fault, position, velocity, torque, and temperature feedback fields
- configured raw CAN direction for every motor

Each configured physical motor is represented by three native MuJoCo actuators
on the same joint or tendon: a position actuator, velocity actuator, and direct
motor actuator. Their controls and gains implement the RobStride
operation-control equation:

```text
torque = kd * (velocity_target - velocity)
       + kp * (position_target - position)
       + torque_feedforward
```

The three contributions are summed by MuJoCo every physics step. The target
joint or tendon has one shared `actuatorfrcrange`, updated from the motor's
`limit_torque` parameter and bounded by the motor-kind protocol range, so the
combined MIT torque is clamped once. The individual actuators are not force
clamped. RobStride applied-torque feedback is read from MuJoCo's aggregate
joint/tendon actuator-force sensor after this common clamp.

The runner models Bello's physical transmissions rather than treating every
motor as a direct MuJoCo joint. Every motor assignment explicitly names a
MuJoCo joint or tendon; Python does not infer the target from the motor name.
Each ankle differential is defined as two fixed MuJoCo tendons: motor 1
has coordinate `roll - pitch`, and motor 2 has coordinate `roll + pitch`.
MuJoCo computes the tendon positions, velocities, Jacobian, native actuator
forces, and their mapping into joint-space torque. The coupling equations
therefore live in the generated MJCF instead of the CAN layer. Generated policy
position actuators on RobStride-owned joints are replaced by the native
three-actuator motor banks. Optional waist and head axes are handled earlier:
disabled axes are changed to fixed joints in the generated URDF and therefore
never become MuJoCo actuators. Enabled axes must have RobStride motor mappings;
the runner rejects an enabled-but-unmapped or disabled-but-mapped axis.

## HIL interface setup

The default Motomind HIL configuration uses a lower-body bus and an upper-body
bus. The repository setup script creates or reuses both VCAN interfaces, sets
their transmit queues to 10,000 frames, brings them up, and starts the EBIMU
serial PTY bridge with one sudo escalation:

```bash
./scripts/setup_hil.sh
```

The script requests `sudo` automatically when needed. Run it once per boot,
before starting the runner. Inspect all HIL interfaces with:

```bash
./scripts/setup_hil.sh status
```

SocketCAN local loopback must be enabled in the client that transmits to a
userspace emulator. The accompanying change in `~/motomind_platform` leaves
`CAN_RAW_LOOPBACK` enabled while keeping `CAN_RAW_RECV_OWN_MSGS` disabled, so
the gateway does not consume its own transmissions.

## Run

From this checkout:

```bash
nix develop
./scripts/setup_hil.sh
bello-robstride-runner
```

The runner is a strict Hydra/dataclass CLI. Its default recipe is
`configs/robstride_runner/default.yaml`; its keys map directly onto
`RobstrideRunnerConfig`, and unknown keys fail before hardware interfaces are
opened. The recipe selects motor YAML files, VCAN mappings, viewer mode,
timing, torso guide, MuJoCo ankle differential tendons, and IMU emulation. The
backend and terrain are intentionally not configuration
switches: this runner always uses CPU MuJoCo and a flat plane.

Apply normal Hydra overrides as `field=value`. To use a separate complete
recipe, use Hydra's standard `--config-dir` and `--config-name` options:

```bash
bello-robstride-runner headless=true real_time=false max_steps=1000
bello-robstride-runner --config-dir /absolute/config/dir --config-name my_runner
```

The visual meshes default to a 20,000-face-per-mesh budget for responsive
interactive rendering; this does not change the general training asset
default.

The runner defaults to `end_effector_sphere_mode: both`, matching the training
morphology. Its default `upper_body_sphere_hands.yaml` mapping omits the two
wrist-yaw motors because sphere replacement removes those joint subtrees; wrist
pitch remains actuated. To run the detailed hand morphology instead, select
both the non-sphere asset and its complete upper-body motor mapping explicitly:

```bash
bello-robstride-runner \
  asset_config.end_effector_sphere_mode=none \
  'motor_configs=[robstride/lower_body.yaml,robstride/upper_body.yaml]'
```

Every prepared Bello asset contains the MuJoCo ankle transmission and uses
tendon position actuators. The runner replaces those generated ankle actuators,
along with the mapped direct-joint actuators, with the native three-actuator MIT
motor banks. This leaves one authoritative controller per physical motor while
retaining MuJoCo's tendon coordinates and Jacobian.

A pelvis-mounted MuJoCo IMU is enabled by default. The runner exposes it as an
E2BOX EBIMU-9DOFV6 on an imaginary serial port. `setup_hil.sh` starts a
background `socat` PTY bridge with these endpoints:

```text
/dev/ttyIMU         Motomind EBIMU driver endpoint
/dev/ttyIMU-mujoco  bello-robstride-runner endpoint
```

The emulator follows the EBIMU-9DOFV6 Rev1.0 serial contract: 8N1 with no flow
control, `*...\r\n` ASCII records, quaternion fields in `z,y,x,w` order, gyro
in degrees per second, acceleration in g, a wrapping millisecond timestamp,
`<...>` commands, and `<ok>`/`<er>` responses. Its stored startup layout
matches the Motomind policy default: quaternion, gyro, gravity-removed local
acceleration, timestamp, and 200 Hz output. Output configuration commands,
polling mode, Euler and binary output, temperature, velocity, and distance are
also emulated. MuJoCo remains the source of the orientation and kinematics.

Use the unmodified physical driver against the imaginary device:

```bash
ros2 launch motomind_ros_ebimu ebimu.launch.py \
  port:=/dev/ttyIMU baudrate:=460800 configure:=true
```

The serial baud setting is accepted as the real device would accept it, though
a PTY does not physically clock bits. Stop or inspect the bridge with:

```bash
./scripts/setup_hil.sh status
./scripts/setup_hil.sh stop
```

`stop` removes the serial PTY bridge and leaves the VCAN interfaces up.

Disable only the serial device connection while keeping the MuJoCo sensor in
the generated model:

```bash
bello-robstride-runner imu_serial_enabled=false
```

Disable IMU sensor generation and serial output:

```bash
bello-robstride-runner \
  asset_config.imu_config.enabled=false \
  imu_serial_enabled=false
```

The torso starts in a world-vertical guide. It cannot move sideways or rotate,
but it can descend up to 5 cm from its initial height. A MuJoCo slider joint
enforces that lower bound. In interactive mode, the runner opens a separate
**Bello controls** window next to the MuJoCo viewer. Click **Release torso** to
disable the guide permanently; the robot is then a free body governed by
gravity, contacts, and actuator commands. Closing either window stops the
runner. Override the travel with `torso_drop_limit_m=0.1` when needed.

## Motor kinds and assignments

The configuration pipeline separates reusable motor-kind behavior from robot
assignments:

```text
configs/robstride/motor_kinds.yaml
configs/robstride/lower_body.yaml
configs/robstride/upper_body_sphere_hands.yaml
configs/robstride/upper_body.yaml
```

`upper_body_sphere_hands.yaml` is the default 12-motor arm mapping.
`upper_body.yaml` additionally maps the two wrist-yaw motors for the explicit
non-sphere morphology.

`motor_kinds.yaml` is the single source for each kind's CAN field ranges,
instantaneous torque clamp, rated/peak torque, speed, voltage, phase current,
reduction ratio, and operating-temperature range. To add another kind, define
it once under `motor_kinds`; the loader rejects incomplete ranges, unknown
fields, nonpositive hardware values, and a protocol torque range that does not
match the published peak torque.

Each motor in the body files selects a kind and a simulation coordinate:

```yaml
- name: left_hip_pitch
  id: 61
  bus: can0
  kind: ROBSTRIDE_04
  direction: -1
  run_mode: mit
  target:
    type: joint
    name: left_hip_pitch_joint
```

Tendon assignments also declare which generated position actuators they own
and their required coupling group:

```yaml
- name: left_ankle_1
  id: 67
  bus: can0
  kind: ROBSTRIDE_06
  direction: -1
  run_mode: mit
  target:
    type: tendon
    name: left_ankle_motor_1
    owned_joints: [left_ankle_pitch_joint, left_ankle_roll_joint]
    coupling_group: left_ankle
    coupling_size: 2
```

Change `kind` to assign another preconfigured motor kind, or change the target
name to bind the device to another MuJoCo coordinate. Joint and tendon names
are validated against the loaded model before CAN interfaces are opened.
Coupling groups must contain exactly `coupling_size` motors.

These files are bundled into installed wheels and do not depend on
`~/motomind_platform`. Relative motor-config paths are resolved from the
bundled `configs/` root; use absolute paths for files outside it.

It maps logical `can0` to `vcan0` and logical `can1` to `vcan1`. Useful modes:

```bash
# CI or a machine without a display
bello-robstride-runner headless=true release_after_seconds=2

# Run faster than wall clock for a bounded protocol/simulation test
bello-robstride-runner headless=true real_time=false max_steps=1000

# Start free, without creating the initial torso guide
bello-robstride-runner torso_hold=false

# Use one actuator file and one interface
bello-robstride-runner \
  motor_kind_config=robstride/motor_kinds.yaml \
  'motor_configs=[robstride/lower_body.yaml]' \
  'bus_interfaces=[can0=vcan0]'
```

The default runner asset exposes only `waist_yaw_joint` from the optional
waist/head set. Waist roll, waist pitch, neck yaw, and head pitch are fixed in
the generated URDF. Configure the selection under the asset config:

```yaml
asset_config:
  joint_availability:
    waist_yaw: true
    waist_roll: false
    waist_pitch: false
    neck_yaw: false
    head_pitch: false
```

Changing an axis to `true` also requires a corresponding motor assignment in a
selected RobStride motor YAML. This is deliberately different from training
`mode: hold`: a held joint remains movable and softly PD-controlled, while a
disabled runner joint has no MuJoCo degree of freedom.

The runner validates every configured motor kind and coordinate mapping before
opening the CAN interfaces. Missing interfaces, unknown kinds, duplicate CAN
IDs, incomplete coupling groups, and missing joints or tendons fail at startup.

## Motomind gateway test

Build and source `~/motomind_platform` after the SocketCAN loopback change,
then start its lower and upper gateways with actuator settings matching the
repository-local YAML. Do not start `fake_robstride_can_bridge_node`; this
MuJoCo runner replaces it.

The normal gateway lifecycle remains unchanged:

```bash
ros2 service call /lower/enable_motors std_srvs/srv/Trigger "{}"
ros2 service call /lower/start_control std_srvs/srv/Trigger "{}"
ros2 topic echo /lower/motor_states
```

Commands published through the gateway are encoded by the production native
driver, decoded by this runner, written to native MuJoCo position, velocity,
and feedforward actuators, and returned as RobStride feedback frames for the
gateway to publish as motor states.

## Scope

This is a protocol-faithful actuator emulator, not an electrical or thermal
motor model. It clamps simulated motor torque to the catalog's peak range,
exposes catalog-rated voltage and peak phase current as device parameters,
reports a configurable constant temperature, and leaves fault bits clear. The
remaining hardware fields are descriptive and do not implement thermal
derating or an overload curve. MuJoCo supplies the rigid-body, contact,
joint-limit, and actuator-load dynamics.
