# Physical hardware

`bello-run-hardware` is the sole physical policy deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware).

The runner has three task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and normalized remote velocity intent; live GMR pose targets the upper body outside the policy | `bello-hardwared`, the visualizer's RemoteCommandService, and `fastsam3dbody-serve` |
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
6. requires a fresh sample from the visualizer's remote-command service and five
   qualified live Fast-SAM/GMR poses for locomotion, then connects to the hardware
   daemon without acquiring a control lease;
7. validates motor groups, coordinate ownership, IMU presence, group state, and
   fresh finite complete state frames;
8. evaluates a live lower-body-policy/upper-body-pose preview for locomotion, or
   a standing-policy preview for motion, and reports the resolved configuration.
   Motion-reference alignment and termination checks begin only when
   transitioning from the active standing policy into motion tracking.

No motor target is sent in this mode.

With `allow_git_commit_mismatch=true`, the current checkout's runner and
generated assets are used. Evaluation and hardware loading may discard the
retired `end_effector_sphere_mode` metadata field and disabled selectors for
joints that are now fixed in the canonical URDF. The loader rejects a retired
joint selector if it was enabled. Locomotion checkpoints created before the PPO
configuration split receive the explicit historical optimizer, value-loss,
observation-normalization, and MLP-architecture defaults. Later locomotion
reward and domain-randomization fields are restored with their historical
disabled or identity values. No other checkpoint metadata is adapted. Run an
unarmed preflight and review the source differences before arming.

## Arming contract

Arming requires a spotter, a working E-stop, and `arm=true`. Armed control
requires the asynchronous hardware binding so target acknowledgements do not
block the policy-rate loop.

After preflight, all task modes begin with the same interactive sequence:

1. The operator supports the robot and types the exact word `HOLD`. The remote
   abort remains active while the runner waits and prevents control acquisition.
2. The runner acquires control and ramps to the task's standing target over ten
   seconds by default. Locomotion combines the checkpoint nominal lower body with
   the repository-owned GMR standing arms.
3. The runner continuously refreshes the nominal target and checks telemetry
   and nominal-pose error bounds while the operator releases external support
   and lets the robot settle. Direct coordinates use 0.25 rad; the
   load-bearing differential ankle motor coordinates use 1.00 rad.
4. The operator types the exact word `RUN`, immediately before locomotion
   policy control.

Any other input fails closed. The `RUN` prompt is read asynchronously so motor
targets, command receipts, and safety checks continue while the terminal waits
for the operator.

All tasks use the proportional and derivative gains stored in checkpoint
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

Motor commands remain valid for 80 ms. Their asynchronous gRPC acknowledgement
may arrive up to 500 ms after submission; this longer client-side receipt window
does not extend command validity or delay the daemon's command-timeout damping.

## Locomotion mode

Locomotion checkpoints must use the sphere-hand physical morphology, active
waist yaw, nonprivileged actor observations, and a bounded tanh-Gaussian policy.
The joystick command path remains unchanged: RemoteCommandService supplies
forward, lateral, and yaw intent to the locomotion actor, which remains
authoritative over its 13 waist and leg coordinates. The same Fast-SAM -> SMPL-X
-> pinned GMR pipeline used by TWIST2 supplies only the other 12 coordinates:
the bilateral shoulder, elbow, and wrist joints. The live pose does not enter
the locomotion actor observation and cannot replace joystick intent or any
policy coordinate.

Locomotion therefore also requires the pinned `deps/Fast-SAM-3D-Body` and
`deps/GMR` revisions and an operator-supplied licensed `SMPLX_NEUTRAL.npz`.
Start and calibrate the server as described under [TWIST2 mode](#twist2-mode),
then include the model path in locomotion preflight:

```text
uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/hardware_locomotion \
  remote_endpoint=http://visualizer-host:50054 \
  smplx_neutral_model_path=/absolute/path/to/SMPLX_NEUTRAL.npz
```

Before contacting `bello-hardwared`, preflight requires five distinct fresh
confidence-, latency-, GMR-, and upper-body-limit-qualified poses by default.
The supported ramp and `RUN` hold use canonical GMR standing arms. After `RUN`,
upper-body tracking begins automatically while joystick commands continue to
control the trained gait. No separate `MOTION` confirmation is used.

Live upper-body changes are limited to `3*pi` rad/s at the 50 Hz control rate,
including the first pose and source recovery, and outgoing targets remain inside
the model actuator range. Locomotion adds the independently configurable
`left_elbow_yaw_offset_radians` and `right_elbow_yaw_offset_radians` to their
respective GMR targets, then clips the adjusted values to the corresponding
target ranges; both offsets default to zero. After `RUN`, the terminal also
accepts nonblocking 5-degree jogs: `q`/`a` increase/decrease the left offset and
`e`/`d` increase/decrease the right offset. The configured offsets are the
starting values, and each jogged target uses the same clipping and slew-rate
limits. Briefly rejected frames leave the
last safe reference usable
until its total age reaches 1000 ms. When no fresh safe pose remains, the runner
reports the loss and returns only the arms toward GMR standing at the same
bounded rate; joystick locomotion continues. A later safe pose resumes arm
following. A terminal gRPC/GMR worker failure still causes fail-closed damped
shutdown.

The physical controller is attached to the host running `bello-visualizer`.
Its RemoteCommandService streams normalized forward, lateral, and yaw intent;
the hardware runner scales that intent through the checkpoint's command bounds
and minimum speeds, mapping lateral to `lin_vel_x` and forward to `lin_vel_y`.
In Bello's locomotion frame, `+X` is lateral/right and `+Y` is forward.
The runner advances the same checkpointed gait clock used by training and
native evaluation. A standing command resets it and supplies a zero-valued
three-component gait observation in the current actor frame; older moving
frames age out of the configured observation stack normally. Each policy
sample is clipped to the checkpoint action bound, converted directly to motor
targets, and fed back as `last_action`, matching training and native evaluation
exactly. Smoothness comes from the trained action-rate and action-acceleration
objectives rather than a deployment-only command filter.
Configure the visualizer to listen on port 50054 and supply its plaintext URL
as `remote_endpoint`, for example
`remote_endpoint=http://visualizer-host:50054`.

Left-stick vertical/horizontal controls forward/lateral velocity, right-stick
horizontal controls yaw, and R1 aborts. The controller must be
physically neutral before nonzero intent is accepted after controller or gRPC
reconnection. Controller disconnection or a stale service stream commands
zero. A stale stream remains subscribed and requires neutral again when fresh
samples resume; an ended RPC reconnects on the existing channel. A fresh
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
velocity, orientation, filtered root-local angular velocity, and a precise source
timestamp, normally from PoseService at `127.0.0.1:50053`. The runner never
differentiates received poses. State-frame IMU orientation remains an independent
agreement check, and its raw angular velocity and acceleration remain subject to
hardware safety limits. Root translation is never estimated by double-integrating
acceleration.

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

## TWIST2 mode

TWIST2 deployment requires a typed `twist2` checkpoint, the pinned
`deps/Fast-SAM-3D-Body` and `deps/GMR` revisions, and an operator-supplied
licensed `SMPLX_NEUTRAL.npz`. The neutral model is read locally only to derive
and validate the server's exact SMPL-X body-joint contract; it is never copied
into a checkpoint or Git. Normal standing applies a small repository-owned,
hand-authored relaxed-standing pose to that skeleton and retargets it through
the same pinned GMR solver as live motion. The result uses the dataset builder's
positive-flex knee treatment to solve the whole body, then retains the relaxed
GMR upper body while replacing the legs with the policy checkpoint's canonical
Bello stance, levels the root, and places the lower sole on the floor. This
keeps normal standing inside the stance distribution learned by the policy and
avoids persistent stepping. It does not use SMPL-X's mathematical zero pose,
whose arms extend sideways.
The live server must advertise schema 1,
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
translation is passed through the shared causal support-foot estimator before
GMR: confident planted feet anchor root XY, support changes use speed/height
hysteresis and blending, and unsupported phases preserve filtered source
translation. This is the exact estimator replayed by the Fast-SAM recording
importer, so the policy does not train on a cleaner root contract than it sees
live. Vertical motion is preserved separately by GMR's contact-aware grounding.
A differently mounted camera must supply a calibrated
`fastsam_camera_to_world_quaternion_xyzw`; disabling initial heading or floor
alignment is explicit.

Preflight connects to Fast-SAM, validates server readiness, tracking mode, and
the stream schema, retargets the canonical SMPL-X standing sample, and compiles
the policy before connecting to the robot. After connection, it also requires
every model actuator range to fit inside the live daemon inventory range. Tiny
text-serialization differences of at most `1e-5` radians are accepted, then the
effective command range is tightened to the exact model/daemon intersection; a
larger model/daemon limit mismatch fails before a control lease is acquired. It
does not require Fast-SAM to be
transmitting a qualified person target. The live gRPC consumer and GMR solver
run in background threads with capacity-one/latest-only semantics; the 50 Hz
control loop never blocks on either service. Frame age is the server's complete
pipeline latency plus the configured transport budget and local time since
receipt. Live targets use the same pinned knee-pole thresholds, reachability
constraint, joint velocity/acceleration bounds, and foot-clearance geometry as
offline retargeting. The shared root estimator's causal contact state grounds
supporting feet while preserving airborne vertical motion.

Armed operation follows `HOLD`, a ten-second ramp to the hybrid normal-standing
target described above, and `RUN`; that target is held while waiting for `RUN`.
`RUN` starts the policy with a 0.2-second blend from that pose to its
normal-standing policy output, then continues commands at 50 Hz. Only
after exact `MOTION` confirmation does the runner require five distinct
confidence-, latency-, and robot-limit-qualified person targets by default. It
keeps sending normal-standing commands while waiting up to
`fastsam_reference_ready_timeout_seconds`, then transitions to the current live
GMR reference. If that timeout expires, normal standing continues and the
operator can type `MOTION` to retry.

Hardware command-stream submission is pipelined across one control tick. Before
enqueuing the current target, the runtime resolves the preceding tick's
submission and checks completed acknowledgement errors; the current queue
submission then overlaps state acquisition, policy inference, and pacing for
the next tick. Targets remain ordered and none are coalesced or dropped. Explicit
drains resolve both outstanding submissions and acknowledgements; shutdown
waits for submission workers before releasing the command streams.

A rejected frame caused by brief occlusion, ambiguity, low confidence,
excessive latency, or an invalid retargeted pose does not replace the last safe
reference or reset GMR. The last safe reference remains usable only until its
total age reaches the fixed 250 ms fail-closed bound; persistent loss then
returns the robot to normal standing. A changed tracking identity, backwards or
reset source timeline, Fast-SAM `LOST`, or `UNLOCKED` state clears the cached
reference and re-anchors both coordinates and GMR immediately. `LOST` remains
terminal for that server target until its `ResetTarget` RPC is invoked; target
reacquisition is not performed silently by the Bello runner.

## Runtime safety checks

All armed modes continuously check:

- expected control-group state and hardware receipt errors;
- motor and IMU sample freshness;
- finite position, velocity, orientation, acceleration, and action values;
- quaternion normalization, robot tilt, angular velocity, and acceleration;
- bounded checkpoint nominal-pose error from ramp completion through `RUN`;
- policy target bounds;
- per-iteration control-loop deadline lag.

Motion adds pose age/frame/source-time validation, pose/IMU agreement,
estimator-provided base-velocity bounds,
reference completion, and raw-versus-applied action bounds.

Locomotion adds exact Fast-SAM stream semantics, tracking and required-joint
confidence, pipeline/reference age, source sequence and timestamp monotonicity,
SMPL-X quaternion validity, GMR joint velocity/acceleration and hard upper-body
joint limits, and a fixed upper-body target slew bound. These checks affect only
the 12 arm targets; the joystick-conditioned locomotion policy continues
controlling the waist and legs. A temporarily unavailable or stale reference
blends the arms back to GMR standing over
`upper_body_follow.connection_transition_duration_seconds`; a recovered
reference uses the same blend before resuming direct tracking. Locomotion
continues throughout either transition, while a terminal source failure remains
fatal.

TWIST2 adds exact Fast-SAM stream semantics, tracking and required-joint
confidence, pipeline/reference age, source sequence and timestamp monotonicity,
SMPL-X quaternion validity, GMR joint velocity/acceleration and hard URDF joint
limits, retargeted root/body velocity, and raw Gaussian action bounds. Reference
poses are observations and may use the full hard joint range seen in the packed
training data; outgoing policy targets remain bounded by the narrower actuator
and live-daemon command ranges. The body-frame IMU gyro is passed directly into
proprioception without a second orientation transform.
Policy observations retain the raw previous action used during training. Motor
commands preserve the training mapping: they add `0.5 * action` to the model
default, apply the actuator-coordinate transform, and clip each physical motor
to the live/model range intersection. Direct-joint targets are not projected in
logical space. For each differential ankle only, the runner then finds the
closest two-motor target whose resolved pitch and roll remain inside the hard
URDF limits. It leaves an already feasible ankle target bit-for-bit unchanged
and fails closed if the projected pair is still invalid. This minimal coupled
projection prevents independently legal ankle motor targets from resolving to
an illegal ankle pitch or roll without imposing the training soft limits on the
rest of the policy output.

`duration_seconds` covers the complete post-`RUN` policy flow. Ramp time and the
`HOLD`/`RUN` confirmation waits are not included. A `null` duration runs until
operator exit, interruption, or a safety failure. Motion reference completion
returns to standing locomotion rather than ending the process.

The runner sends policy targets without applying its own target-step limiter.
The pinned hardware daemon remains authoritative and rejects any motor target
that moves more than 2.0 rad from its previously accepted target.
