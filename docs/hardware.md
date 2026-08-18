# Physical hardware

`bello-run-hardware` is the sole physical policy deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware).

The runner has two task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and normalized remote velocity intent | `bello-hardwared` plus the visualizer's RemoteCommandService |
| Motion | Motor state, pelvis IMU, absolute pelvis pose, reference motion, and previous action | `bello-hardwared` plus `bello-state-estimatord` PoseService |

Both modes run policy inference on JAX's CPU backend. The command overrides
inherited JAX platform selectors before importing either runner, and exposes no
hardware backend option.

The `bello-hardware` Python binding is installed from the pinned
`deps/bello-stack` uv workspace member. Hardware execution requires a source
checkout; installed wheels contain neither that binding nor the hardware
description. The root Git commit pins the stack gitlink, and exact-source checks
require the recursive submodule checkout to be initialized, clean, and at that
commit. The checkpoint does not record daemon or firmware revisions.

The runner requires the pinned daemon and Python binding's synchronous hardware
states API. Each policy iteration requests one complete frame containing the
latest motor, pelvis IMU, and group samples. It does not build observations by
draining the individual-event telemetry stream, and no background frame stream
can accumulate behind Python. The runner conservatively adds the complete RPC
round-trip duration to every daemon-reported sample age before applying its
configured motor, IMU, group, and sample-span limits. A stale or incoherent
frame causes a fail-closed damped shutdown.

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
6. requires a fresh sample from the visualizer's remote-command service for
   locomotion, then connects to the hardware daemon without acquiring a control
   lease;
7. validates motor groups, coordinate ownership, IMU presence, group state, and
   fresh finite complete state frames;
8. evaluates a live preview step for locomotion, or a standing-policy preview
   for motion, and reports the resolved configuration. Motion-reference
   alignment and termination checks begin only when transitioning from the
   active standing policy into motion tracking.

No motor target is sent in this mode.

With `allow_git_commit_mismatch=true`, the current checkout's runner and
generated assets are used. Evaluation and hardware loading may discard the
retired `end_effector_sphere_mode` metadata field and disabled selectors for
joints that are now fixed in the canonical URDF. The loader rejects a retired
joint selector if it was enabled. No other checkpoint metadata is adapted. Run
an unarmed preflight and review the source differences before arming.

## Arming contract

Arming requires a spotter, a working E-stop, and `arm=true`. Armed control
requires the asynchronous hardware binding so target acknowledgements do not
block the policy-rate loop.

After preflight, both task modes begin with the same interactive sequence:

1. The operator supports the robot and types the exact word `HOLD`. The remote
   abort remains active while the runner waits and prevents control acquisition.
2. The runner acquires control and ramps to the checkpoint nominal pose over
   ten seconds by default.
3. The runner continuously refreshes the nominal target and checks telemetry
   and nominal-pose error bounds while the operator releases external support
   and lets the robot settle. Direct coordinates use 0.25 rad; the
   load-bearing differential ankle motor coordinates use 1.00 rad.
4. The operator types the exact word `RUN`, immediately before locomotion
   policy control.

Any other input fails closed. The `RUN` prompt is read asynchronously so motor
targets, command receipts, and safety checks continue while the terminal waits
for the operator.

Both tasks use the proportional and derivative gains stored in checkpoint
metadata.

Safety failures are terminal for the current run. A daemon command timeout or
lease loss puts the affected group in `DAMPED`; the runner reports the group's
cause and detail and does not resume automatically. Cleanup attempts to stop
every group that is still `RUNNING`, even if stopping another group fails. If
any group is `DAMPED`, it remains damped until the operator supports the robot
and types the exact word `IDLE`; only those damped groups are then explicitly
disabled. Lease release preserves `DAMPED` and `FAULT` states, so an exception
or failed confirmation does not automatically undamp the robot. Explicit
completion, remote-controller abort, and keyboard interruption use the same
stop-confirm-disable sequence. A hard fault or lost daemon connection can
prevent part of this handshake. The external hardware daemon remains the final
authority for command validity, leases, damping, and device faults.

## Locomotion mode

Locomotion checkpoints must use the sphere-hand physical morphology, active
waist yaw, nonprivileged actor observations, and a bounded tanh-Gaussian policy.
The runner controls the 13 policy coordinates and supplies nominal targets for
the remaining physical coordinates.

The physical controller is attached to the host running `bello-visualizer`.
Its RemoteCommandService streams normalized forward, lateral, and yaw intent;
the hardware runner scales that intent through the checkpoint's command bounds
and minimum speeds, mapping lateral to `lin_vel_x` and forward to `lin_vel_y`.
In Bello's locomotion frame, `+X` is lateral/right and `+Y` is forward.
The runner advances the same checkpointed gait clock used by training and
native evaluation. A standing command resets it and supplies a zero-valued
three-component gait observation in the current actor frame; older moving
frames age out of the configured observation stack normally.
Configure the visualizer to listen on port 50054 and supply its plaintext URL
as `remote_endpoint`, for example
`remote_endpoint=http://visualizer-host:50054`.

Left-stick vertical/horizontal controls forward/lateral velocity, right-stick
horizontal controls yaw, and R1 aborts. The controller must be
physically neutral before nonzero intent is accepted after controller or gRPC
reconnection. Controller disconnection or a stale service stream commands
zero; the runner reconnects automatically and requires neutral again. A fresh
service sample is required before the runner connects to hardware. R1
is level-sensitive and latched by the client, including when it is held during
startup. Ctrl-C remains the terminal fallback. Hardware locomotion has no
native-controller or fixed-command mode.

## Motion mode

Motion deployment requires both a motion checkpoint in `checkpoint_path` and a
locomotion checkpoint in `standing_checkpoint_path`. The locomotion checkpoint
runs with a zero command before and after motion tracking. Both checkpoints must
use the same ordered physical-coordinate mapping.

The motion checkpoint restores the reference contract.
`motion_path=null` reuses the saved path; a relocated archive is accepted only
when its schema and content identity match.

The pose stream supplies world-frame pelvis translation, estimator-state linear
velocity, orientation, and a precise source timestamp, normally from PoseService
at `127.0.0.1:50053`. The runner never differentiates received positions. IMU
orientation is retained as an independent agreement check. Root translation is
never estimated by double-integrating acceleration.

Motion first uses the standing checkpoint's default ten-second ramp, bounded
nominal-pose arrival, and manual `HOLD` and `RUN` confirmations. `RUN` starts
standing locomotion. The runner continues sending standing-policy targets while
waiting for the exact `MOTION` confirmation, then blends from live standing
policy targets to live motion-policy targets. Reference completion blends back
to live standing-policy targets and standing locomotion resumes. Pressing
Ctrl-C while motion tracking is active performs the same transition back to
standing; pressing Ctrl-C again while standing enters damped shutdown. Each
transition takes 0.2 seconds by default; it does not force the unsupported
robot through either checkpoint's static nominal pose. Pose/IMU orientation
agreement, reference bounds, and initial policy output remain checked. Remote
command fields are locomotion-only.

Motion inference fuses observation construction, stacking, normalization, actor
execution, bounded action mapping, and diagnostics into one JIT call. Periodic
state/timing output is disabled by default; set `status_interval_seconds` to a
positive interval to enable it. Phase transitions and a final motion timing
summary are always reported.

## Runtime safety checks

Both armed modes continuously check:

- expected control-group state and hardware receipt errors;
- motor and IMU sample freshness;
- finite position, velocity, orientation, acceleration, and action values;
- quaternion normalization, robot tilt, angular velocity, and acceleration;
- bounded checkpoint nominal-pose error from ramp completion through `RUN`;
- policy target bounds;
- accumulated control-loop deadline lag.

Motion adds pose age/frame/source-time validation, pose/IMU agreement,
estimator-provided base-velocity bounds,
reference completion, and raw-versus-applied action bounds.

`duration_seconds` covers the complete post-`RUN` policy flow. Ramp time and the
`HOLD`/`RUN` confirmation waits are not included. A `null` duration runs until
operator exit, interruption, or a safety failure. Motion reference completion
returns to standing locomotion rather than ending the process.

The runner sends policy targets without applying its own target-step limiter.
The pinned hardware daemon remains authoritative and rejects any motor target
that moves more than 2.0 rad from its previously accepted target.
