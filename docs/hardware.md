# Physical hardware

`bello-run-hardware` is the sole physical policy deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware).

The runner has two task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and fixed or gamepad velocity command | `bello-hardwared` |
| Motion | Motor state, pelvis IMU, absolute pelvis pose, reference motion, and previous action | `bello-hardwared` plus `bello-state-estimatord` PoseService |

The `bello-hardware` Python binding is installed from the pinned
`deps/bello-stack` uv workspace member. Hardware execution requires a source
checkout; installed wheels contain neither that binding nor the hardware
description. The root Git commit pins the stack gitlink, and exact-source checks
require the recursive submodule checkout to be initialized, clean, and at that
commit. The checkpoint does not record daemon or firmware revisions.

An exact checkout also restores that revision's hardware runner and safety
checks. Review old runner code and its guards before using `arm=true`; a later
safety fix is not inherited automatically by a historical checkpoint.

## Read-only preflight

`arm=false` is the default. Preflight:

1. requires a clean checkout and, unless explicitly overridden, the
   checkpoint's exact Git commit;
2. restores and validates checkpoint integrity and policy state;
3. prepares the robot asset described by checkpoint metadata;
4. reconstructs observation, action, and physical-coordinate layouts;
5. compiles policy inference;
6. connects to the hardware daemon without acquiring a control lease;
7. validates motor groups, coordinate ownership, IMU presence, group state, and
   fresh finite telemetry;
8. evaluates one preview policy step and reports the resolved configuration.

No motor target is sent in this mode.

With `allow_git_commit_mismatch=true`, the current checkout's runner and
generated assets are used without adaptation. Run an unarmed preflight and
review the source differences before arming.

## Arming contract

Arming requires a spotter, a working E-stop, and `arm=true`. Armed control
requires the asynchronous hardware binding so target acknowledgements do not
block the policy-rate loop.

After preflight, both task modes use the same interactive sequence:

1. The operator supports the robot and types the exact word `HOLD`.
2. The runner acquires control and ramps to the checkpoint nominal pose over
   ten seconds by default.
3. The runner continuously refreshes the nominal target and checks telemetry
   and nominal-pose error bounds while the operator releases external support
   and lets the robot settle. Direct coordinates use 0.25 rad; the
   load-bearing differential ankle motor coordinates use 0.35 rad.
4. The operator types the exact word `RUN`, immediately before policy control.

Any other input fails closed. The `RUN` prompt is read asynchronously so motor
targets, command receipts, and safety checks continue while the terminal waits
for the operator.

Both tasks use the proportional and derivative gains stored in checkpoint
metadata.

Safety failures are terminal and unwind the acquired control sessions.
Explicit completion, gamepad abort, and keyboard interruption stop and disable
the groups. The external hardware daemon remains the final authority for
command validity, leases, and device faults.

## Locomotion mode

Locomotion checkpoints must use the sphere-hand physical morphology, active
waist yaw, nonprivileged actor observations, and a bounded tanh-Gaussian policy.
The runner controls the 13 policy coordinates and supplies nominal targets for
the remaining physical coordinates.

After the shared `HOLD`, ramp, and `RUN` sequence, left-stick
vertical/horizontal commands forward/lateral velocity, right-stick horizontal
commands yaw, and Start/Options aborts. Controller disconnection commands zero.
The abort is level-sensitive, including when Start/Options is already held
during startup.

A fixed command is active only when gamepad input is disabled.

## Motion mode

Motion deployment restores the reference contract from the checkpoint.
`motion_path=null` reuses the saved path; a relocated archive is accepted only
when its schema and content identity match.

The pose stream supplies world-frame pelvis translation and orientation,
normally from PoseService at `127.0.0.1:50052`. IMU orientation is retained as
an independent agreement check. Root translation is never estimated by
double-integrating acceleration.

Motion uses the same default ten-second ramp, bounded nominal-pose arrival, and
manual `HOLD` and `RUN` confirmations as locomotion. It additionally verifies
Pose/IMU orientation agreement, reference bounds, and initial policy output.
Gamepad and velocity-command fields are locomotion-only.

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
- bounded checkpoint nominal-pose error from ramp completion through `RUN`;
- policy target bounds;
- accumulated control-loop deadline lag.

Motion adds pose age/frame validation, pose/IMU agreement, base-velocity bounds,
reference completion, and raw-versus-applied action bounds.

`duration_seconds` covers only the post-`RUN` policy loop. Ramp time and both
manual confirmation waits are not included. A `null` duration runs until
operator exit, reference completion, interruption, or a safety failure.

The runner sends policy targets without applying its own target-step limiter.
The pinned hardware daemon remains authoritative and rejects any motor target
that moves more than 0.5 rad from its previously accepted target.
