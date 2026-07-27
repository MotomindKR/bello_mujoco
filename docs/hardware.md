# Physical hardware

`bello-run-hardware` is the sole physical policy deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware).

The runner has two task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and fixed or gamepad velocity command | `bello-hardwared` |
| Motion | Motor state, pelvis IMU, absolute pelvis pose, reference motion, and previous action | `bello-hardwared` plus `bello-state-estimatord` PoseService |

The `bello-hardware` Python binding is distributed with `bello-stack`, not this
repository. Supply that local component through uv as documented in the command
reference.

## Read-only preflight

`arm=false` is the default. Preflight:

1. restores and validates the checkpoint document and deployment state;
2. prepares the exact robot asset bound into checkpoint metadata;
3. reconstructs observation, action, and physical-coordinate layouts;
4. compiles policy inference;
5. connects to the hardware daemon without acquiring a control lease;
6. validates motor groups, coordinate ownership, IMU presence, group state, and
   fresh finite telemetry;
7. evaluates one preview policy step and reports the resolved contract.

No motor target is sent in this mode.

## Arming contract

Arming requires a spotter, a working E-stop, `arm=true`, and the exact
confirmation:

```text
operator_confirmation=I_HAVE_A_SPOTTER_AND_ESTOP_READY
```

The confirmation is intentionally not configurable or inferred. Armed control
requires the asynchronous hardware binding so target acknowledgements do not
block the policy-rate loop.

Safety failures are terminal and unwind the acquired control sessions.
Explicit completion, gamepad abort, and keyboard interruption stop and disable
the groups. The external hardware daemon remains the final authority for
command validity, leases, and device faults.

## Locomotion mode

Locomotion checkpoints must use the sphere-hand physical morphology, active
waist yaw, nonprivileged actor observations, and a bounded tanh-Gaussian policy.
The runner controls the 13 policy coordinates and supplies nominal targets for
the remaining physical coordinates.

The default ten-second nominal-pose ramp is target-step limited. With the
default gamepad settings, the runner then holds until A/Cross starts the policy.
Left-stick vertical/horizontal commands forward/lateral velocity, right-stick
horizontal commands yaw, and Start/Options aborts. Controller disconnection
commands zero.

A fixed command is active only when gamepad input is disabled. Disabling the
gamepad also requires disabling the start-button gate.

## Motion mode

Motion deployment restores the reference contract from the checkpoint.
`motion_path=null` reuses the saved path; a relocated archive is accepted only
when its schema and content identity match.

The pose stream supplies world-frame pelvis translation and orientation,
normally from PoseService at `127.0.0.1:50052`. IMU orientation is retained as
an independent agreement check. Root translation is never estimated by
double-integrating acceleration.

After its default three-second ramp, motion mode verifies nominal-pose arrival,
Pose/IMU orientation agreement, reference bounds, and initial policy output,
then starts without a gamepad gate. Gamepad, velocity-command, and
wait-for-start fields are locomotion-only.

Motion inference fuses observation construction, stacking, normalization, actor
execution, bounded action mapping, and diagnostics into one JIT call. Timing
reports distinguish inference, asynchronous command submission, total loop
work, maxima, and accumulated deadline lag.

## Runtime safety checks

Both armed modes continuously check:

- expected control-group state and hardware receipt errors;
- motor and IMU sample freshness;
- finite position, velocity, orientation, acceleration, and action values;
- quaternion normalization, robot tilt, angular velocity, and acceleration;
- policy target bounds and maximum per-step target movement;
- accumulated control-loop deadline lag.

Motion adds pose age/frame validation, pose/IMU agreement, base-velocity bounds,
reference completion, and raw-versus-applied action bounds.

`duration_seconds` covers only the post-ramp policy loop. Ramp time and the
locomotion start wait are not included. A `null` duration runs until operator
exit, reference completion, interruption, or a safety failure.
