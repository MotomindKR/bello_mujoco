# Physical hardware

`bello-run-hardware` is the sole physical policy deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware).

The runner has three task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and fixed or gamepad velocity command | `bello-hardwared` |
| Motion | Motor state, pelvis IMU, absolute pelvis pose, reference motion, and previous action | `bello-hardwared` plus `bello-state-estimatord` PoseService |
| TWIST2 | Motor state, pelvis IMU, current and historical proprioception, and a live retargeted pose | `bello-hardwared` plus `fastsam3dbody-serve` |

All modes run policy inference on JAX's CPU backend. The command overrides
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
6. connects to the hardware daemon without acquiring a control lease;
7. validates motor groups, coordinate ownership, IMU presence, group state, and
   fresh finite complete state frames;
8. evaluates one preview policy step and reports the resolved configuration.

No motor target is sent in this mode.

With `allow_git_commit_mismatch=true`, the current checkout's runner and
generated assets are used without adaptation. Run an unarmed preflight and
review the source differences before arming.

## Arming contract

Arming requires a spotter, a working E-stop, and `arm=true`. Armed control
requires the asynchronous hardware binding so target acknowledgements do not
block the policy-rate loop.

After preflight, all task modes begin with the same interactive sequence:

1. The operator supports the robot and types the exact word `HOLD`.
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
completion, gamepad abort, and keyboard interruption use the same
stop-confirm-disable sequence. A hard fault or lost daemon connection can
prevent part of this handshake. The external hardware daemon remains the final
authority for command validity, leases, damping, and device faults.

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
agreement, reference bounds, and initial policy output remain checked. Gamepad
and velocity-command fields are locomotion-only.

Motion inference fuses observation construction, stacking, normalization, actor
execution, bounded action mapping, and diagnostics into one JIT call. Periodic
state/timing output is disabled by default; set `status_interval_seconds` to a
positive interval to enable it. Phase transitions and a final motion timing
summary are always reported.

## TWIST2 mode

TWIST2 deployment requires a typed `twist2` checkpoint, the pinned
`deps/Fast-SAM-3D-Body` and `deps/GMR` revisions, and an operator-supplied
licensed `SMPLX_NEUTRAL.npz`. The neutral model is read locally only to derive
the exact zero-pose SMPL-X body joints used in normal standing; it is never
copied into a checkpoint or Git. The live server must advertise schema 1,
`smplx_body_22`, the exact standard 22-joint order and hierarchy, OpenCV camera
coordinates, `xyzw` quaternions, and the reviewed learned converter. A legacy
native-MHR stream is rejected.

Run the pinned server from its own environment. Copy and calibrate one of its
service examples, retain `output_representation: smplx_body_22` and
`person_selection: track`, and use the reserved local port 50055 when it shares
a host with the other Bello services:

```text
cd deps/Fast-SAM-3D-Body
nix develop
mkdir -p ~/.config/fastsam3dbody
cp examples/service.pinhole.yaml ~/.config/fastsam3dbody/service.yaml
# Set grpc.listen to 127.0.0.1:50055 and replace camera calibration values.
fastsam3dbody-serve --config ~/.config/fastsam3dbody/service.yaml --check-config
fastsam3dbody-serve --config ~/.config/fastsam3dbody/service.yaml
```

Keep that local service YAML outside the submodule so the deployment checkout
remains clean for exact-source hardware validation.

Then perform read-only Bello preflight from the repository root:

```text
uv run bello-run-hardware \
  task=twist2 \
  checkpoint_path=checkpoints/twist2_hardware \
  smplx_neutral_model_path=/absolute/path/to/SMPLX_NEUTRAL.npz
```

The default camera rotation assumes a level, forward-looking camera: camera
forward becomes world forward, camera right becomes world right, and camera
down becomes world down. On each target acquisition the adapter captures the
initial pelvis horizontal origin, standing foot height, and heading. Subsequent
translation and vertical motion are preserved. A differently mounted camera
must supply a calibrated `fastsam_camera_to_world_quaternion_xyzw`; disabling
initial heading or floor alignment is explicit.

Preflight connects to Fast-SAM, validates server readiness, tracking mode, and
the stream schema, derives the neutral reference through GMR, and compiles the
policy before connecting to the robot. It does not require Fast-SAM to be
transmitting a qualified person target. The live gRPC consumer and GMR solver
run in background threads with capacity-one/latest-only semantics; the 50 Hz
control loop never blocks on either service. Frame age is the server's complete
pipeline latency plus the configured transport budget and local time since
receipt.

Armed operation follows `HOLD`, a ten-second ramp to the GMR-retargeted SMPL-X
neutral joint pose, and `RUN`; the same neutral target is held while waiting for
`RUN`. `RUN` starts the policy with a 0.2-second blend from that pose to the
neutral SMPL-X standing policy output, then continues commands at 50 Hz. Only
after exact `MOTION` confirmation does the runner require a confidence- and
latency-qualified person target. It keeps sending neutral-standing commands
while waiting up to `fastsam_reference_ready_timeout_seconds`, then transitions
to the current live GMR reference. If that timeout expires, neutral standing
continues and the operator can type `MOTION` to retry.
Occlusion, ambiguity, low confidence, excessive latency, invalid retargeted
motion, or stream failure returns the robot to neutral standing or initiates a
damped shutdown according to severity. Fast-SAM `LOST` remains terminal for
that target until its `ResetTarget` RPC is invoked; target reacquisition is not
performed silently by the Bello runner.

## Runtime safety checks

All armed modes continuously check:

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

TWIST2 adds exact Fast-SAM stream semantics, tracking and required-joint
confidence, pipeline/reference age, source sequence and timestamp monotonicity,
SMPL-X quaternion validity, GMR joint velocity/acceleration and joint limits,
retargeted root/body velocity, and raw Gaussian action bounds.

`duration_seconds` covers the complete post-`RUN` policy flow. Ramp time and the
`HOLD`/`RUN` confirmation waits are not included. A `null` duration runs until
operator exit, interruption, or a safety failure. Motion reference completion
returns to standing locomotion rather than ending the process.

The runner sends policy targets without applying its own target-step limiter.
The pinned hardware daemon remains authoritative and rejects any motor target
that moves more than 2.0 rad from its previously accepted target.
