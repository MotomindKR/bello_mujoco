# Motion-tracking hardware runner

`./scripts/bello-run-motion-hardware` connects a schema-6 BeyondMimic motion
checkpoint to the existing `bello-hardwared` Python client. It is separate from
the 13-action locomotion runner and supports the current 25-action sphere-hand,
fixed-head, waist-yaw-only motion morphology.

The script intentionally remains outside `src/bello_mujoco` and is not
registered in `pyproject.toml`. Schema-6 checkpoints fingerprint every Python
module in that package and `pyproject.toml`; changing either would make an
existing checkpoint reject its runtime. Run the script from the exact checkout
recorded by the checkpoint.

## Runtime data path

```text
bello-hardwared motor/IMU telemetry ----+
                                        +-> 140-value BeyondMimic actor frame
bello-state-estimatord pelvis odometry -+   -> deterministic 25-action mean
                                            -> differential actuator transform
                                            -> complete hardwared targets
```

The runner reconstructs logical joint position and velocity through the
inverse actuator-coordinate transform. It aligns the reference clip's planar
translation and yaw to the measured pelvis pose, computes torso tracking error
through native MuJoCo forward kinematics, and uses the checkpoint's stored
observation normalization.

`bello.pose.v1.PoseService` currently publishes pose but not velocity. The
runner derives world-frame pelvis velocity from consecutive estimator samples
and applies a low-pass filter. The estimator is contact-aided local odometry,
so foot slip, flight, or prolonged single-foot balance can cause drift. Stale,
non-finite, out-of-order, excessive-speed, or IMU-inconsistent pose samples
stop the run.

The checkpoint uses an unbounded Gaussian actor. Hardware execution adds an
independent default action clamp of `[-1, 1]` and fails closed if an unbounded
mean exceeds the larger raw-action limit. Every motor target also passes the
checkpoint actuator limits, the runner's per-step target limit, and
`hardwared` command policy.

## Required services

Start `bello-hardwared` with the sphere-hand `lower_body` and `upper_body`
groups. Start the state estimator after hardware telemetry is online:

```bash
cd ~/bello_stack
cargo run -p bello-state-estimatord -- \
  --config components/bello-state-estimatord/config/bello.yaml
```

The defaults expect:

- hardware service `http://127.0.0.1:50051`;
- pose service `127.0.0.1:50052`;
- pose transform `odom_from_pelvis_link`;
- motor groups `lower_body` and `upper_body`;
- hardware IMU named `imu`; and
- at least 50 Hz group control.

Use `components/bello-hardwared/config/bello-vcan.yaml` and
`components/bello-state-estimatord/config/bello-vcan.yaml` with the complete
RobStride/vCAN emulator. Do not use a physical config merely to test the
software path.

## Read-only preflight

Keep the robot supported. From the exact checkpoint checkout, run:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  ./scripts/bello-run-motion-hardware \
  checkpoints/balance-level4-schema6-main-4741853-20260720-v3
```

Preflight restores and compiles the exact checkpoint, validates its motion and
robot contracts, connects to both read-only telemetry streams, checks all 25
coordinates, checks fresh motor/IMU/pose data, verifies pose and IMU orientation
agreement, aligns the reference, reports the required nominal-pose ramp
distance, and computes one preview action. It does not acquire a lease, enable a
group, or send a target.

Useful non-arming arguments are:

```text
--motion PATH
--endpoint URL
--pose-endpoint HOST:PORT
--start-time SECONDS
--max-duration SECONDS
--max-action-abs VALUE
--max-nominal-arrival-error RADIANS
--kp-scale VALUE
--kd-scale VALUE
```

A relocated motion file is accepted only if its complete schema and content
hash match the checkpoint.

## Arming

First complete the read-only preflight and an armed vCAN run. Physical use
requires a supported initial reference pose, a spotter, a tested emergency
stop, clear space, and explicit operator confirmation. The arming flags are:

```text
--arm
--operator-confirmation I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

On arming, the runner acquires both group leases, starts from measured targets,
ramps to the robot's nominal joint pose, realigns the clip, and only then
executes the deterministic actor at 50 Hz. The runner verifies measured joints
are within the configured arrival tolerance after the ramp. Motion completion,
tracking termination, stale or invalid telemetry, pose/IMU disagreement,
excessive motion, unsafe group state, missed deadlines, rejected commands, lost
acknowledgements, lease loss, `Ctrl-C`, or the duration limit exits the control
contexts. Their teardown stops, disables, and releases both groups.

This runner is an integration path, not evidence that a particular checkpoint
is safe for an unsupported physical robot. Evaluate the checkpoint in native
MuJoCo and validate the complete runner against vCAN before physical use.
