# Physical hardware

`bello-run-hardware` is the sole physical deployment command:

```bash
uv run --no-cache \
  --with ../bello-stack/components/bello-hardware-python \
  bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/policy/final
```

The default is a read-only preflight: it loads and validates the checkpoint,
inventory, telemetry, IMU, morphology, layouts, command bounds, and generated
transmission without acquiring a control lease.

## Arming

Use a spotter and working E-stop. Arming requires:

```text
arm=true
operator_confirmation=I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

The operator-facing controls are endpoint, task command or motion source,
duration, start/ramp behavior, gain scales, gamepad behavior, arming, and
confirmation. Telemetry age, tilt, acceleration, angular velocity, command
validity, deadline lag, and target-step limits are reviewed internal safety
constants.

For a fixed locomotion command without a gamepad:

```bash
uv run --no-cache \
  --with ../bello-stack/components/bello-hardware-python \
  bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/policy/final \
  command='[0.5,0.0,0.0]' \
  gamepad_enabled=false \
  wait_for_start_button=false \
  duration_seconds=60 \
  arm=true \
  operator_confirmation=I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

Set `duration_seconds=null` to run until operator abort or a safety failure.

## Motion

Motion deployment uses motor/IMU telemetry from `bello-hardwared` and absolute
pelvis pose from `bello-state-estimatord` PoseService, normally at
`127.0.0.1:50052`. IMU orientation is an independent agreement check; root
translation is never reconstructed by double-integrating acceleration.

```bash
uv run --no-cache \
  --with ../bello-stack/components/bello-hardware-python \
  bello-run-hardware \
  task=motion \
  checkpoint_path=checkpoints/motion/final \
  motion_path=motions/local/reference.npz
```

The runner fuses observation construction, stacking, normalization, actor
execution, bounded action mapping, and tracking diagnostics into one JIT call.
Structured timing reports separate inference, command submission, total loop
work, maxima, and accumulated deadline lag.

Any arming failure stops every control group. Read-only preflight and shutdown
behavior are covered by hardware contract tests.
