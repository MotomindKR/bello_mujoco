# Physical locomotion policy runner

`bello-run-locomotion-hardware` connects a trained locomotion policy to the
blocking `bello_hardware` Python client in `~/bello_stack`. The runner is for the
current sphere-hand Bello inventory only. It does not support the standard
29-actuator training morphology or motion-tracking checkpoints.

The data path is:

```text
hardwared telemetry -> physical-to-logical transmission -> 55-value actor frame
                    -> ten-frame stack -> deterministic policy mean
                    -> logical target -> physical differential-ankle target
                    -> hardwared position/impedance command
```

The runner imports the binding at runtime instead of adding the neighboring
checkout to this repository's lock file.

## Train a compatible checkpoint

The hardware recipe inherits the maintained locomotion algorithm and fixes the
four coordinates absent from the physical inventory: waist roll, waist pitch,
neck yaw, and head pitch. Waist yaw remains actuated, hands are spheres, and the
policy retains its 13 logical leg/waist actions.

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name hardware_locomotion
```

The runner rejects checkpoints whose task, checkpoint schema, runtime
fingerprints, observation/action layouts, policy distribution, sphere-hand
mode, or joint availability differ. A standard `locomotion.yaml` checkpoint is
therefore intentionally incompatible.

## Read-only preflight

Start `bello-hardwared` with its physical sphere-hand topology, keep the robot
supported, and run:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  bello-run-locomotion-hardware \
  checkpoint_path=checkpoints/hardware_locomotion_seed0
```

This is the default mode. It loads and compiles the policy on CPU before
connecting, then verifies:

- exactly the configured `lower_body` and `upper_body` groups and all 25
  checkpoint actuator coordinates;
- group control rates of at least 50 Hz and an initial `IDLE` state;
- online motor and `imu` inventory status;
- complete, finite, fresh position, velocity, quaternion, gyro, and
  gravity-removed local-acceleration telemetry;
- an upright robot within 0.75 rad of the checkpoint nominal motor pose; and
- a command within the velocity bounds stored in the checkpoint.

The read-only path does not acquire a lease, enable a motor group, or send a
target. Correct IMU mounting and axis convention cannot be inferred from the
inventory: the configured EBIMU must report the pelvis frame expected by the
training asset. At rest and upright it should report identity orientation,
near-zero angular velocity, and near-zero gravity-removed local acceleration.

## Arm and run

Physical execution requires a spotter, a tested physical emergency stop, a
supported initial pose, clear space, and deliberate arming. Begin with a small
forward command:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  bello-run-locomotion-hardware \
  checkpoint_path=checkpoints/hardware_locomotion_seed0 \
  'hardware.command=[0.2,0.0,0.0]' \
  hardware.arm=true \
  hardware.operator_confirmation=I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

The runner acquires both group leases, starts each group at its measured
position, ramps to the nominal pose over three seconds, and then executes the
deterministic actor at 50 Hz. Logical ankle position and velocity feedback is
reconstructed by inverting the model's actuator-coordinate transform. Policy
targets use the forward transform, so the two physical tendon motor targets
remain consistent with training. Motor gains also come from the generated
checkpoint morphology; `hardware.kp_scale` and `hardware.kd_scale` default to
`1.0`. They may reduce gains for a guarded commissioning run but cannot exceed
the trained gains.

By default an armed run lasts at most ten seconds. `Ctrl-C`, stale or non-finite
telemetry, excessive tilt, excessive IMU motion, a group fault/state change,
missed loop deadlines, rejected commands, lost acknowledgements, or an expired
lease exits the control contexts. The binding then stops, disables, and releases
both groups. The runner disables transparent telemetry reconnects so a broken
connection cannot silently resume actuation.

Useful guarded overrides include:

```text
hardware.max_duration_seconds
hardware.ramp_seconds
hardware.max_target_step_radians
hardware.max_tilt_radians
hardware.max_motor_age_ms
hardware.max_imu_age_ms
hardware.kp_scale
hardware.kd_scale
```

All safety limits must remain positive. The runner's checks complement rather
than replace the position limits, command step policy, command timeout, lease
expiry, motor health handling, and fault damping configured in
`bello-hardwared`.
