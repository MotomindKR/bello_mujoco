# Physical hardware

`bello-run-hardware` is the sole physical **policy** deployment interface. Its
complete public fields and defaults are in the
[command reference](commands.md#bello-run-hardware). The independent
[`bello-run-hands`](commands.md#bello-run-hands) process can send RTMW-derived
targets directly to Linker O6/L6 hands; it does not control robot joints or
acquire a `bello-hardwared` lease.

The runner has three task modes:

| Task | Policy input | External services |
| --- | --- | --- |
| Locomotion | Motor state, pelvis IMU, previous action, and normalized remote velocity intent; live GMR pose targets the upper body outside the policy | `bello-hardwared`, the visualizer's RemoteCommandService, and the RTMW-SMPLX-Teleop gRPC service |
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

Physical locomotion deployment supports `bello` and `bello_mini`. The runner
reconstructs Mini checkpoints against Mini's own reviewed 28-motor description;
it never borrows Bello's topology or GMR profile. Live RTMW/GMR follows the 12
held arm coordinates on Bello or all 14 held arm coordinates on Mini.

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

## Standalone Linker hand control

`bello-run-hands` consumes the pinned RTMW service's 52 SMPL-X joint positions
and ten fingertip positions. It constructs a handed palm-local frame for each
side and maps the observed thumb and finger geometry directly to six controller
axes. The target order is thumb flexion, thumb abduction, index, middle, ring,
and pinky. This is a standalone landmark controller: it does not use GMR, a
locomotion checkpoint, the policy process, or `bello-hardwared` command RPCs.

The supported devices are Linker O6 and L6 hands using 11-bit **classic CAN**
at 1 Mbit/s. Right and left commands use standard identifiers `0x27` and
`0x28`. Do not connect this command to a HITBOT
`eHand-6-R-CFD-N-8F`: that hand's documented interface is a different CAN-FD
protocol with a 5 Mbit/s BRS data phase, 32-byte payloads, and identifiers
`0x11`/`0x12`. The frame layout and actuator order are not interchangeable.
The wire contract was reviewed against
[`linker-bot/linkerhand-python-sdk`](https://github.com/linker-bot/linkerhand-python-sdk)
at revision `0cc0585b97214b2cc4a9a5afcc84aee9f414e0e8`; this repository uses a
small direct SocketCAN implementation and does not import or vendor that SDK.

The hands share the existing `can1` upper-body network with RobStride motors.
The pinned hardware daemon reads only extended RobStride frames, while each
hand socket installs an exact standard-ID receive filter. The hand process does
not create a CAN interface, call `ip link`, change bitrate or MTU, or bring a
link up or down. Consequently it can be started and stopped without modifying
the running policy or daemon. It does add its position writes and periodic
read-only feedback queries to the physical bus; monitor CAN error counters when
commissioning the combined network.

Before arming, inspect the current configuration from the repository root:

```text
deps/bello-stack/scripts/setup-can.sh --check
```

This is a read-only check of both `can0` and `can1`. It requires the links to be
up with classic-CAN MTU 16, CAN FD disabled, and a 1 Mbit/s bitrate. The normal
`sudo deps/bello-stack/scripts/setup-can.sh` mode takes both links down while
configuring them; never run normal mode while `bello-hardwared`, a hand
controller, or the robot is operating.

The default hand command is an RTMW-only preflight:

```text
uv run bello-run-hands
```

With `arm=false`, it validates the pinned protobuf and stream metadata, service
status, height prior, timestamps, latency, tracking identity, confidence, and
hand geometry, then returns after one acceptable frame. It never opens a CAN
socket, sends a hand probe, or changes the hand. Therefore a successful
unarmed preflight does not prove that either physical hand is present; use
`setup-can.sh --check` for the non-mutating link check.

Only after confirming the physical model, unobstructed motion, and a working
power cutoff should the separate controller be armed:

```text
uv run bello-run-hands protocol=linker_o6 arm=true calibration_confirmed=true
```

Armed startup validates RTMW first, then opens a filtered socket per enabled
side and issues a one-byte, read-only position query to each hand. Position
writes begin only after those probes answer. The first target is low-pass and
slew limited from the measured starting positions. The default raw calibration
follows Linker's convention of `255` for open/extended and `0` for closed;
calibrate each of the six endpoints before relying on full-range tracking.
Armed mode rejects `calibration_confirmed=false`; setting it to `true` asserts
that the configured endpoints were checked on the actual model and side, but
does not perform calibration automatically.

Fresh targets are written at no more than 30 Hz. Invalid landmarks are gated
per side, so one reliable hand may continue while writes to the other stop. A
tracking loss, stale target, RTMW outage, or invalid target never causes a
disable, fully-open, fully-closed, or nominal command: the process simply stops
new position writes for the unavailable side. The reviewed O6/L6 classic-CAN
interface has no disable command equivalent to the HITBOT CAN-FD action code,
so inventing one would be unsafe. Firmware behavior after command silence and
the external hand power cutoff remain outside this controller.

Retryable RTMW transport failures reconnect automatically, but only a newly
accepted fresh frame can resume target writes. Before the first resumed write,
the controller drains old feedback, performs a new read-only position probe,
and reseeds its low-pass and slew limits from that physical position. Clicking
**Stop stream** in the RTMW GUI pauses publication while camera capture,
inference, visualization, gRPC, and existing subscriptions remain alive. Once
the last received target exceeds `max_reference_age_ms`, the hand controller
stops writing. Frames produced during the pause are discarded rather than
queued; **Start stream** resumes with newly captured frames only.

Stopping `bello-run-hands` closes its own filtered sockets. It neither stops
`bello-hardwared` nor damps, disables, or changes state of any RobStride group.
See the [command reference](commands.md#bello-run-hands) for every confidence,
latency, filtering, calibration, side-selection, and duration field.

## Read-only preflight

`arm=false` is the default. Preflight:

1. requires a clean checkout and, unless explicitly overridden, the
   checkpoint's exact Git commit;
2. restores and validates checkpoint integrity and policy state;
3. prepares the robot asset described by checkpoint metadata;
4. reconstructs observation, action, and physical-coordinate layouts;
5. compiles policy inference;
6. requires a fresh sample from the visualizer's remote-command service and,
   when upper-body tracking is enabled, validates the RTMW service contract and
   initializes GMR from an offline standing sample, then connects to the
   hardware daemon without acquiring a control lease; no tracked person is
   required yet;
7. validates motor groups, coordinate ownership, the hardwared IMU output
   frame, group state, and fresh finite complete state frames;
8. evaluates a lower-body-policy preview with checkpoint-nominal arms for
   locomotion, or a standing-policy preview for motion, and reports the
   resolved configuration.
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
   seconds by default. Bello locomotion uses the checkpoint's complete nominal
   pose regardless of whether live upper-body tracking is enabled.
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

## Single-joint chirp diagnostic

`scripts/joint_chirp.py` applies a small exponential frequency sweep to one
logical joint while holding every other coordinate in the selected motor groups
at its measured starting position. It uses the same inventory checks, complete
state frames, telemetry limits, asynchronous acknowledged commands, loop
deadlines, and damped cleanup as the policy runner. Ankle pitch and roll are
translated through the two-motor differential; the script never treats one
ankle motor as an independently movable joint.

Select the reviewed robot contract with `--asset`. Both `bello` and
`bello_mini` support read-only preflight and armed joint-chirp diagnostics.

Run the read-only preflight first. It connects to either physical
`bello-hardwared` or the VCAN emulator, verifies that both groups are `IDLE`,
checks the selected joint and complete proposed sweep against logical and motor
limits, and sends no command:

```text
uv run python scripts/joint_chirp.py \
  --asset bello \
  --joint left_hip_roll_joint \
  --endpoint http://127.0.0.1:50051
```

Only run an armed sweep with the robot restrained or safely supported, a
spotter, and a working E-stop. Start with a smaller amplitude when testing a new
joint or load:

```text
uv run python scripts/joint_chirp.py \
  --asset bello \
  --joint left_hip_roll_joint \
  --amplitude 0.01 \
  --start-frequency 0.5 \
  --end-frequency 10 \
  --duration 30 \
  --output outputs/hardware_identification/left_hip_roll.jsonl \
  --arm
```

Armed operation requires the exact confirmations `HOLD`, `CHIRP`, and `IDLE`.
The first acquires control at the measured pose. The script refreshes that pose
while waiting for `CHIRP`. Completion or failure enters the normal damped state,
which is preserved until the supported robot is explicitly returned to `IDLE`.

The JSONL log records the complete inventory and experiment configuration, then
one record per command cycle. Each cycle contains logical and physical targets,
reconstructed logical position and velocity, command-scheduling/deadline
timestamps, raw motor position, daemon-filtered motor velocity, torque, faults
and sample time, group command count/control tick/jitter, IMU samples,
state-frame sample span, and RPC round-trip time. The state in a record is
sampled after submission of its target is scheduled and that cycle's control
deadline passes. The Python client does not expose the exact daemon/CAN transmit
instant; use the group command count together with the monotonic timestamps,
rather than JSONL row number alone, when estimating command/response delay.

## Locomotion mode

Locomotion checkpoints must use a registered hardware morphology, active waist
yaw, nonprivileged actor observations, and a bounded tanh-Gaussian policy. The
joystick command path remains unchanged: RemoteCommandService supplies forward,
lateral, and yaw intent to the locomotion actor. The actor remains authoritative
over Bello's 13 or Mini's 14 waist and leg coordinates. RTMW-SMPLX-Teleop's
`smplx52_landmark_fit` output passes through the selected robot's pinned GMR
profile and supplies only the held arm coordinates: 12 on Bello or all 14 on
Mini. The live pose does not enter the locomotion actor observation and cannot
replace joystick intent or any policy coordinate. Fast-SAM is not a supported
locomotion pose source; it remains specific to TWIST2.

Bello Mini's live profile maps its complete seven-DoF arms from the derived
shoulder, elbow, and palm orientations. Its arm position costs are disabled:
the robot's fixed links determine elbow and wrist positions, avoiding
infeasible landmark-offset lengths. Palm orientation directly drives the wrist
yaw, pitch, and roll targets. Mini has no elbow-yaw joints, so the Bello-only
elbow-yaw offsets and terminal keyboard jog are unavailable.
Hardwared first applies the canonical URDF rotation from the upside-down
physical `imu_sensor_link` into the logical `imu_link`; the runner then rotates
that logical sample into the checkpoint's pelvis observation frame. The sensor
origin is offset from the pelvis origin, so existing checkpoints retain an
unmodeled lever-arm acceleration residual. This remains a known limitation of
the current Mini checkpoint rather than a deployment gate.

Run the pinned `deps/RTMW-SMPLX-Teleop` producer with its gRPC service enabled.
Preflight requires that submodule to be clean and exactly at the reviewed
gitlink, then checks the protobuf schema, 52-joint order, hierarchy, coordinate
semantics, stream instance, and producer configuration hash. Slow consumers
skip intermediate frames instead of accumulating transport latency. Producer
monotonic timestamps are checked against one another; receipt age is measured
on the runner's local clock, so the producer and runner may use different
hosts. Both processes must use the same licensed neutral SMPL-X model. RTMW
projects reconstructed upper-body directions onto that model's canonical bone
lengths before publishing, matching native neutral-SMPL-X clips. The measured
person height remains a depth-reconstruction parameter only:
`rtmw_human_height_m` must equal RTMW's
`reconstruction.subject_height_m` (both default to 1.75 m), while
`smplx_human_height_m` independently remains at GMR's calibrated 1.66 m
neutral-source convention:

```text
cd ~/bello_mujoco/deps/RTMW-SMPLX-Teleop
uv run rtmw-teleop \
  --grpc-listen 0.0.0.0:50055
```

Then run locomotion, pointing `rtmw_endpoint` at that producer:

```text
cd ~/bello_mujoco
uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_locomotion \
  endpoint=http://robot-host:50051 \
  remote_endpoint=http://visualizer-host:50054 \
  rtmw_endpoint=rtmw-host:50055 \
  smplx_neutral_model_path=/absolute/path/to/SMPLX_NEUTRAL.npz
```

To run locomotion with fixed upper-body targets instead, disable tracking. This
skips RTMW, SMPL-X, and GMR pose setup and holds every arm joint at the
checkpoint's nominal training target (12 for Bello, 14 for Mini):

```text
uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_locomotion \
  remote_endpoint=http://visualizer-host:50054 \
  upper_body_tracking_enabled=false
```

With tracking enabled, startup requires a compatible running RTMW service but
does not require a tracked person or live pose before contacting
`bello-hardwared`. The ramp, `RUN` hold, and tracking-loss fallback use the
robot's nominal arm pose; a GMR-retargeted standing sample cannot redefine that
safety pose. After `RUN`, the first fresh schema-, validity-, latency-, GMR-,
and upper-body-limit-qualified pose starts arm following by default. Tracking
loss resets that qualification and returns the arms toward nominal through the
same bounded transition. The consumer accepts only RTMW schema 1,
`smplx52_landmark_fit`,
the exact face-free 52-joint hierarchy, calibrated metric OpenCV camera
coordinates, and the versioned rotation semantics
`derived_tree_consistent_landmark_ik;canonical_smplx_upper_body_projection_v1;anatomical_elbow_hinge_arm_chain_v2;palm_frame_wrist_orientation_v2;lower_body_hand_shape_and_unobserved_axial_twist_approximate`.
The producer projects its smoothed upper-body tree to canonical neutral-SMPL-X
bone lengths before deriving rotations, while moving each hand rigidly with its
corrected wrist. This rejects older running producers that lack the same
geometry, anatomical arm-frame, and palm-frame contracts even if their protobuf
schema is otherwise compatible.
The producer's confidence-aware SMPL-X landmark filter uses a 500 ms smoothing
time constant by default and runs before canonical projection and orientation
IK. Producer pose and height-anchor confidence gates default to 0.05. A missed
detector update keeps the last associated box output-eligible for the tracker's
configured grace period, while detection retries on every frame. Height-prior
reconstruction holds the last valid same-track root depth for up to two seconds
when its face or foot anchors disappear; a new track or an initially invalid
depth still produces no output. Root-range filtering remains independently
configured.
The fixed `live_upper_body` GMR task profile disables the independent arm
position costs and constrains the shoulder, elbow, and terminal wrist
orientations. The shoulder and elbow frames use the observed
shoulder-elbow-wrist chain and an anatomical elbow hinge; palm landmarks define
only the terminal wrist orientation. Fixed robot links therefore determine arm
positions without an infeasible position target competing with palm
orientation or duplicated palm twist at the elbow.
The pelvis rotation must have two-vector constraint rank and a residual no
greater than 35 degrees. Shoulder frames use a side-specific anatomical
elbow-hinge prior when the arm is bent far enough; nearly straight or folded
arms remain rank one and transport their preceding twist. Unreliable spine,
shoulder, elbow, and palm rotations fall back to the nearest usable ancestor
instead of suppressing the entire pose. The default confidence thresholds are
zero, so the producer's explicit validity masks remain authoritative.
Root-relative/uncalibrated output and prerecorded video timestamps are rejected.
No separate `MOTION` confirmation is used.

Initial heading alignment removes only the subject's yaw relative to the
canonical SMPL-X-to-world pelvis basis; the fixed SMPL-X frame rotation itself
is not mistaken for subject heading.

RTMW accepts source intervals through the configurable
`rtmw_max_source_interval_seconds` limit (700 ms by default). Live GMR uses the
profile's fixed solver-pass count for each accepted source frame: Mini uses
three passes to converge its orientation-only arm tasks, while the universal
offline profile remains at one. The pass count does not depend on wall-clock
gaps, so solver work cannot create a feedback loop that reduces the consumed
RTMW frame rate. Live upper-body changes are limited to `3*pi` rad/s at the 50
Hz control rate, including the first pose and source recovery, and outgoing
targets remain inside the model actuator range.
On Bello, locomotion adds the independently configurable
`left_elbow_yaw_offset_radians` and `right_elbow_yaw_offset_radians` to their
respective GMR targets, then clips the adjusted values to the corresponding
target ranges; both offsets default to zero. After `RUN`, the terminal also
accepts nonblocking 5-degree jogs: `q`/`a` increase/decrease the left offset and
`e`/`d` increase/decrease the right offset. The configured offsets are the
starting values, and each jogged target uses the same clipping and slew-rate
limits. Briefly rejected frames leave the last safe reference usable until its
total age reaches 1000 ms. When no fresh safe pose remains, the runner
reports the loss and returns only the arms toward their nominal pose at the same
bounded rate; joystick locomotion continues. A later safe pose resumes arm
following. A malformed RTMW record or terminal GMR worker failure still causes
fail-closed damped shutdown. A clean RTMW stream end or transient gRPC outage is
instead a source discontinuity: the runner clears the live pose epoch, returns
the arms toward nominal, and keeps locomotion active while reconnecting with
bounded backoff. Every replacement producer is checked against the full stream
contract and service status before its frames are accepted. A new stream
instance and reset sequence are allowed only through that reconnect path, and
the normal reference-qualification and transition limits apply before arm
following resumes.

The physical controller is attached to the host running `bello-visualizer`.
Its RemoteCommandService streams normalized forward, lateral, and yaw intent;
the hardware runner maps that intent into the selected robot's pelvis frame and
caps each supported direction at `joystick_command_scale` times its checkpoint
outer limit while preserving trained inner bounds and minimum moving speeds.
The resulting planar and yaw requests are slew-limited at
`joystick_linear_slew_rate_mps2` and
`joystick_yaw_slew_rate_rps2`, respectively. The defaults cap full stick at 60%
of each trained outer limit and change commands at 1.0 m/s² or 1.0 rad/s².
Bello maps right/forward to `+X`/`+Y`; Bello Mini maps right/forward to
`-Y`/`+X`. Hardwared publishes Mini's calibrated logical `imu_link` frame, and
the runner performs only its fixed logical-IMU-to-pelvis rotation before
constructing the policy observation.
The runner advances the same checkpointed gait clock used by training and
native evaluation, initialized at phase zero. A neutral stick, stale service,
disconnected controller, or unarmed intent supplies a zero standing target. The
current gait target remains active until the next scheduled half-cycle boundary,
then the effective command slews to zero. Moving requests, including direction
reversals, slew from the actual effective command while the active gait phase is
preserved through the artificial near-zero crossing. Periodic status
reports the effective `command` and keeps `standing_transition_pending` active
while either the gait boundary or the final ramp to zero is pending. Each policy
sample is clipped to the checkpoint action bound, converted directly to motor
targets, and fed back as `last_action`, matching training and native evaluation
exactly. The joystick slew limiter changes only the velocity command in the
policy observation; motor targets remain unfiltered.

The deployed `bello-hardwared` applies each canonical URDF source-to-output IMU
rotation before filtering orientation, angular velocity, and linear
acceleration with timestamp-aware 10 Hz first-order low-pass filters. Bello
uses 200 Hz EBIMU output and Mini uses 100 Hz; the locomotion runner samples the
latest state at 50 Hz. Motor velocity is independently filtered at 10 Hz. Joint
position and locomotion action targets remain unfiltered.
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

Hardware and emulator topology is resolved from the robot selected by checkpoint or
command configuration. The robot contract points at the pinned bello-stack hardware
description and motor-model catalog; startup validates that every enabled robot
actuator has a compatible physical mapping.

The motion checkpoint restores the reference contract.
`motion_path=null` reuses the saved path; a relocated archive is accepted only
when its schema and content identity match.

The pose stream supplies world-frame pelvis translation, estimator-state linear
velocity, orientation, filtered root-local angular velocity, and a precise source
timestamp, normally from PoseService at `127.0.0.1:50053`. The runner never
differentiates received poses. State-frame IMU orientation remains an independent
agreement check, and its hardwared-filtered angular velocity and acceleration
remain subject to hardware safety limits. Root translation is never estimated by
double-integrating acceleration.

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
the same robot-profiled pinned GMR solver as live motion. The configured standing
joints are then replaced with the policy checkpoint's canonical stance before the
root is leveled and the lower sole is placed on the floor. This keeps normal
standing inside the stance distribution learned by the policy and avoids persistent
stepping. It does not use SMPL-X's mathematical zero pose, whose arms extend
sideways.
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
receipt. The selected robot motion profile supplies GMR's source, target, task,
IK configuration, landmark set, feet, and output-body order. Velocity limiting,
solver passes, collision geometry, and foot clearance remain owned by the pinned
GMR configuration; offline conversion additionally uses its versioned smoothing
and protected-body settings. Knee-pole data emitted offline is quality-control
metadata, not a second target transformation. The shared root estimator's causal
contact state grounds supporting feet while preserving airborne vertical motion.

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

Locomotion adds exact RTMW protobuf and stream metadata checks, tracking and
required-joint confidence, derived-rotation confidence and tree consistency,
pipeline/transport/reference age, source sequence and timestamp monotonicity,
required rank-two rotation observability and residual, reference joint speed
and hard upper-body joint limits with a nonzero live solver-boundary margin,
and a fixed upper-body target slew bound. Invalid terminal-joint rotations use
the nearest valid ancestor only outside the required upper-body chain. These
checks affect only the 12 Bello or 14 Mini arm targets; the joystick-conditioned
locomotion policy continues controlling the waist and legs. A temporarily
unavailable or stale reference, including an RTMW transport outage, blends the
arms back to their checkpoint-nominal pose over
`upper_body_follow.connection_transition_duration_seconds`; a recovered
reference uses the same blend before resuming direct tracking. Locomotion
continues throughout either transition. Reconnection does not relax source
validation: incompatible stream metadata, malformed records, a
producer-reported terminal failure, or a terminal GMR worker failure remains
fatal.

TWIST2 adds exact Fast-SAM stream semantics, tracking and required-joint
confidence, pipeline/reference age, source sequence and timestamp monotonicity,
SMPL-X quaternion validity, reference joint speed and hard URDF joint limits,
retargeted root/body velocity, and raw Gaussian action bounds. Reference
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
