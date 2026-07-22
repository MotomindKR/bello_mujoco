# Physical locomotion policy runner

`bello-run-locomotion-hardware` connects a trained locomotion policy to the
blocking `bello_hardware` Python client in `~/bello_stack`. The runner is for the
current sphere-hand Bello inventory only. It does not support the standard
29-actuator training morphology or motion-tracking checkpoints.

The data path is:

```text
hardwared telemetry -> physical-to-logical transmission -> 51-value actor frame
                    -> ten-frame stack -> checkpoint observation normalization
                    -> deterministic bounded policy mean
                    -> logical target -> physical differential-ankle target
                    -> hardwared position/impedance command
```

The runner imports the binding at runtime instead of adding the neighboring
checkout to this repository's lock file.

## Train a compatible checkpoint

The hardware recipe inherits the maintained locomotion algorithm and fixes the
four coordinates absent from the physical inventory: waist roll, waist pitch,
neck yaw, and head pitch. Waist yaw remains actuated, hands are spheres, and the
policy retains its 13 logical leg/waist actions. It samples standing for half of
new commands, uses a longer PPO credit horizon, and prepares the robot at an
explicit total mass of 60 kg by uniformly scaling all link masses and inertia
tensors. The policy is phase-free;
landing rewards remain active during measured residual motion, the stronger
zero-command dual-foot-contact reward acts immediately, and dense
height/orientation penalties lead into the terminal fall penalty.
Vertical-velocity and locomotion-gated flight penalties discourage the short
synchronized hopping gait, while touchdown air-time credit remains deliberately
small.
Its scheduled deterministic video is a ten-second zero-command rollout, making
a failed neutral stance visible before deployment.

```bash
uv run bello-train-ppo \
  --config-dir "$PWD/experiments/train" \
  --config-name hardware_locomotion
```

The runner rejects checkpoints whose task, checkpoint schema, assets, generated
scenes, dependency lock, observation/action layouts, policy distribution,
sphere-hand mode, or joint availability differ. Runner-only implementation
changes are allowed for evaluation and deployment; training resume retains the
exact implementation check. A standard `locomotion.yaml` checkpoint is therefore
intentionally incompatible.

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
- an upright robot, while reporting its maximum deviation from the checkpoint
  nominal motor pose for operator inspection; and
- a command within the velocity bounds stored in the checkpoint.

The read-only path does not acquire a lease, enable a motor group, or send a
target. Correct IMU mounting and axis convention cannot be inferred from the
inventory: the configured EBIMU must report the pelvis frame expected by the
training asset. At rest and upright it should report identity orientation,
near-zero angular velocity, and near-zero gravity-removed local acceleration.
The policy runner reconstructs accelerometer specific force from that
gravity-removed acceleration and the reported orientation, matching the
MuJoCo observation used for training. At rest and upright, the reconstructed
policy input is approximately `+9.80665 m/s^2` along local Z.

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

The runner does not reject an initial pose based on its distance from nominal.
It acquires both group leases, starts each group at its measured position,
ramps to the nominal pose over ten seconds, and then executes the
deterministic actor at 50 Hz. Independent motor groups are started and sent
targets concurrently so neither group's command watchdog waits behind the
other group's acknowledgement. By default, the inverted left-stick Y commands
forward velocity, left-stick X commands lateral velocity, and inverted
right-stick X commands yaw. The deadzone is rescaled and each axis uses the
asymmetric bounds stored in the checkpoint. A missing or disconnected gamepad
immediately commands zero; Start / Options exits and disables both groups.
Set `hardware.gamepad.enabled=false` to use the fixed `hardware.command` value.

With the default `hardware.wait_for_start_button=true`, completing the ramp does
not start policy inference. The runner continues sending the nominal hold
target while the operator releases the external torso support. Once the robot
is free and settled, A / Cross resets the policy history and starts inference;
Start / Options aborts without entering the policy. A button press made during
the ramp is not latched, so it must be released and pressed again at the prompt.
Fixed-command operation requires `hardware.wait_for_start_button=false` together
with `hardware.gamepad.enabled=false`.

The repository-level gamepad launcher remains available as a compact deployment
entry point:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  /path/to/current/repo/scripts/run_locomotion_hardware_gamepad.py \
  --checkpoint-path checkpoints/hardware_locomotion_seed0 \
  --arm \
  --operator-confirmation I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

It uses the same checkpoint validation and deployment-side gamepad command
source as the installed runner.
When developing the local Python binding itself, add `--no-cache` before
`--with` so uv rebuilds the current source. Armed execution requires the
binding's asynchronous command-receipt API; the runner rejects an older cached
binding before acquiring a lease.

The compatibility launcher also exposes command-validity, acknowledgement,
and loop-lag limits for an explicitly isolated emulator. Keep the checked-in
physical defaults for real hardware.

Logical ankle position and velocity feedback is
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
hardware.gamepad.controller_index
hardware.gamepad.deadzone
hardware.wait_for_start_button
```

All safety limits must remain positive. The runner's checks complement rather
than replace the position limits, command step policy, command timeout, lease
expiry, motor health handling, and fault damping configured in
`bello-hardwared`.
