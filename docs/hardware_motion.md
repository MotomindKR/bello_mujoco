# Physical motion-tracking policy runner

`bello-run-motion-hardware` connects a schema-6 motion-tracking checkpoint to
the blocking `bello_hardware` Python client in `~/bello_stack`. It supports the
current fixed-waist, fixed-head, 25-coordinate sphere-hand morphology and both
maintained motion formulations. Locomotion checkpoints and older checkpoint
schemas are rejected.

The runner reconstructs logical joint position and velocity through the inverse
physical actuator transform, including both differential ankles. It uses the
pelvis IMU for orientation and angular velocity. Because the current hardware
API has no absolute pelvis position or velocity source, root translation is a
local estimate initialized from the selected reference frame and integrated
from gravity-removed local IMU acceleration. This is appropriate for guarded,
short balance-clip commissioning; it is not an absolute localization system and
must not be used for long or traveling motions without adding external pose
feedback.

## Read-only preflight

Keep the robot supported and start `bello-hardwared` with the physical
sphere-hand topology. Then run:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  bello-run-motion-hardware \
  checkpoint_path=checkpoints/asap_balance_level4/final
```

The default invocation compiles deterministic CPU inference and validates the
checkpoint task, formulation, motion hash, named observation/action layouts,
normalization state, 25-coordinate hardware inventory, differential
transmission, group rates and state, online motors and IMU, telemetry freshness,
initial pose, and tilt. It acquires no lease and sends no target.

If the clip was relocated, point to it explicitly. Its complete schema and
content hash must still match the checkpoint:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  bello-run-motion-hardware \
  checkpoint_path=checkpoints/asap_balance_level4/final \
  hardware.motion_path=/data/video_single_foot_balance_level4_filter_amass.npz
```

## Arm and run

Physical execution requires a spotter, a tested physical emergency stop, a
supported starting pose, and clear space. The selected clip frame is aligned to
the measured initial pelvis yaw and a local XY origin. The runner starts both
groups at measured positions, ramps to the robot's nominal joint pose over
three seconds, and only then evaluates the deterministic actor at 50 Hz:

```bash
uv run --with ~/bello_stack/components/bello-hardware-python \
  bello-run-motion-hardware \
  checkpoint_path=checkpoints/asap_balance_level4/final \
  hardware.arm=true \
  hardware.operator_confirmation=I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

The run stops at the earlier of the clip end and
`hardware.max_duration_seconds`. `Ctrl-C`, stale/non-finite telemetry, excess
tilt or IMU motion, excessive inertial-estimator velocity or reference error, a
group fault/state change, missed control deadlines, command rejection, lost
acknowledgements, or lease expiry exits the control contexts. The binding then
stops, disables, and releases both groups.

`hardware.start_time_seconds` selects another clip frame. Guarded commissioning
may reduce `hardware.kp_scale` or `hardware.kd_scale`, but neither can exceed the
trained gains. The runner's checks complement rather than replace the position,
step, timeout, lease, health, and fault-damping policy in `bello-hardwared`.
