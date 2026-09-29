# Bello Mini policy console

The Electron console supports macOS and Ubuntu/Debian on x64 (amd64) and arm64. The interface
uses Motomind typography and surfaces, English labels, a full-width dashboard
with a compact edge inset, and Light/Dark/System appearance settings. System is
the default and follows operating-system changes; Light and Dark override it.
Pretendard and the Mini GLBs are bundled for offline use.

## Views

The compact **Overview** keeps the robot, controller guidance, motion
progress and feedback together. The 3D view fills its card; mode and runner status
appear in the top bar. Header fields reserve fixed widths so digits and status
changes do not shift adjacent components. IMU and pose ages, position coordinates and an
expandable motor readout sit at the bottom-right of the 3D view.
The top bar keeps only the runner mode on the left, with power, temperature,
services, and settings grouped at the right. The mode appears as one large word; its tooltip carries phase
and fault details. Service indicators report freshness, including XRoboToolkit
body tracking. Tracking shows Live only while body poses are fresh; receiving
service packets alone does not make it Live. It also shows battery percentage and
motor-rail voltage/current on every tab. Full power-board readings remain in
Hardware status.
An icon-only fullscreen toggle beside Settings enters and exits fullscreen.
The top bar stays visible while scrolling any view. Tracking uses the same
small edge inset as Overview.
Missing readings remain `--`; outdated readings retain their values with a stale
label. Power alerts appear beside the header readings; critical power flags and
board faults also appear in the fault summary.
The status tabs provide detailed read-only views:

- **Hardware:** all motor positions, velocities, torques, temperatures, feedback
  ages, fault bits and mode status; group state, component health, power-board
  readings, hardware inventory and the latest 100 command-policy events.
- **Pose:** complete estimator and IMU fields, including frames, source sequence,
  orientation, velocities and sample ages.
- **Operator:** runner state, device status, locally sampled axes/buttons and
  the last sequence written to the operator stream.
- **Services:** each endpoint, freshness, full error text, last receipt time,
  connection attempts and message counts, plus display/controller settings.

Search matches field names and values. Status views always update live. Each tab
opens directly onto its search field and data, without a repeated page heading or
endpoint subtitle. Switching tabs resets the search. Arrow keys, Home and End
navigate the tabs.
Detailed tables scroll independently on smaller screens. Floating-point readings
show at most three decimal places, with trailing zeros removed; hover over a
detailed table value for the original precision. Integer counters and timestamps
retain their full precision. Display formatting does not alter telemetry or control.
Command-policy event history and counters reset when connection settings change.

## Connections and ownership

| Connection from Electron main | Default | Purpose |
| --- | --- | --- |
| hardwared | `10.42.0.1:50051` | `ListHardware`, `WatchTelemetry`; read-only |
| State estimator | `10.42.0.1:50053` | `WatchPose`; display only |
| Policy runner | `10.42.0.1:50055` | `OperatorService/Exchange`; controller samples and status |
| XRoboToolkit tracking | `10.42.0.1:50052` | `PoseStream/StreamPoses`; native body and hand display only |

Each connection has its own receipt/freshness indicator. Stale motor values are
marked and the last pose remains visible. Loss of a display stream never creates
an operator command. A full-state motion's runner-side estimator connection
still enforces its own feedback safety checks independently of this display.
Power readings use each board's `staleTimeoutMs` from hardware inventory, plus
the server's stale flag and the elapsed time since receipt. They do not use the
faster motor-feedback freshness limit. Disconnection marks power readings stale.

The renderer is sandboxed with no Node access. A narrow preload bridge exposes
snapshots, the fixed robot asset URL, settings, fullscreen, fault recovery,
selection and hold-to-abort actions. SDL and all gRPC clients run in
Electron main. Only the runner acquires hardware leases and writes motor targets.
The Kiss3d visualizer remains a diagnostic client and has no joystick server.
The former port 50054 remote protocol and headless remote process are retired.

## Start

Start hardwared and, when needed, the estimator. Pass a locomotion bundle to
`--policy`; its zero-velocity operation supplies Stand. Configure optional named
BeyondMimic bundles in the hardware YAML before starting the runner:

```yaml
motions:
  - name: Wave
    policy: /absolute/path/to/wave.zip
    source: /absolute/path/to/wave-source
    settings:
      start_time_seconds: 0.0
      transition_seconds: 0.2
      max_tilt_error_radians: 0.5
      max_raw_action_abs: 5.0
  - name: Bow
    policy: /absolute/path/to/bow.zip
    source: /absolute/path/to/bow-source
    settings:
      start_time_seconds: 0.0
      transition_seconds: 0.2
      max_tilt_error_radians: 0.5
      max_raw_action_abs: 5.0
control:
  operator_listen: 0.0.0.0:50055
```

Merge these entries into a copy of the complete
[hardware configuration](../rust/bello-policy/config/hardware.yaml). The example
motions are proprioceptive. Full-state bundles require `settings.pose`; see the
[native runtime](../rust/bello-policy/README.md). Every bundle is loaded and
validated against the source checkout, motor mapping, limits, gains and required
feedback before acquisition. Names must be unique and cannot be `Stand`,
`Locomotion`, `Motion`, or `Tracking`. Optional buffered SONIC Tracking uses the
same runner and transition lifecycle; see [buffered Tracking](#buffered-tracking-hil).

```bash
nix develop .#rust-runtime
cargo run --locked -p bello-policy -- --policy outputs/standing.zip --source . \
  --hardware-config outputs/operator.yaml --arm
```

Without `--arm`, hardware mode runs read-only preflight and exits. With `--arm`,
the operator service starts Disarmed; no lease or motor control begins until
the entry chord. Commission with vCAN before physical deployment.

```bash
cd console
npm ci
npm start
```

Open the **Settings** icon in the console header to set the robot IP or hostname
(default `10.42.0.1`). **Save** applies the address to all four
services, preserving their ports. Expand **Individual service addresses** to
configure separate hosts or ports. Addresses are validated before saving;
connection and freshness indicators report whether each service is reachable.
Hover or focus a service indicator to expand its full error message.
Changing an endpoint ends the previous operator session, clears its displayed
telemetry, and reconnects immediately. Release controller inputs before using the new session.
Settings persist in Electron's per-user application data directory as
`config.json` and load on the next launch.

**Controller prompts** offers PlayStation, Xbox and Switch styles with previews.
Saving only this preference keeps all service streams and the operator lease
open. The style changes button artwork and labels, using JulioCacko's
[Free Input Prompts Pack 1.4](https://juliocacko.itch.io/free-input-prompts), bundled
locally under CC0. It does not remap SDL input: confirm is Cross on PlayStation
and A on Xbox/Switch; abort is L1 + R1, LB + RB, or L + R respectively.
`controllerStyle` accepts `playstation`, `xbox`, or `switch`; the shipped default
is `xbox`. Prompt variants and both 3D views follow the selected appearance.
The Settings appearance selector saves `theme: light | dark | system` without
reconnecting services or ending the operator session.

For an explicit configuration file, copy `console/config.json` to a local file and set `BELLO_CONSOLE_CONFIG` to its
absolute path to change endpoints, controller index, deadzone, pose frames or
display freshness. This file takes precedence over per-user settings, and UI
changes are saved to it. Endpoints use `HOST:PORT`, without an HTTP prefix. The runner
listener uses `control.operator_listen`. Only one operator stream can own the
session; a second receives `RESOURCE_EXHAUSTED` and shows **In use** in the
Runner indicator. Hover or focus it for the explanation. The server acquires
the lease atomically before accepting any samples. Valid samples renew it even
when no gamepad is connected. Closing the stream releases the lease; 250 ms
without samples expires it. A rejected console cannot change the current
owner's commands or neutral gate. The console retries once per second and a
new owner must release all controls before operating.

## Controller sequence

| State | Input | Result |
| --- | --- | --- |
| Disarmed | Release all sticks/buttons, then Back + Start for 2 seconds | Acquire control and ramp to Hold |
| Crane Hold, ramp complete | A for 1 second | Stand |
| Stand | D-pad left/right | Select Stand, Locomotion, Motion, or configured Tracking |
| Stand, Motion selected | D-pad up/down | Select a named motion in the dropdown |
| Motion dropdown open | Press A | Accept the highlighted motion and close the menu |
| Stand | A for 1 second with neutral sticks | Enter the selected mode or motion |
| Locomotion | Left stick and right horizontal stick | Forward/lateral velocity and yaw command |
| Locomotion, following configured | Press Y (Triangle on PlayStation) | Toggle upper-body tracking |
| Locomotion | A for 1 second with neutral sticks | Stand |
| Tracking, reference buffer ready | A for 1 second with neutral sticks | Blend back to Stand |
| Named motion | Completion | Blend back to Stand |
| Named motion | Policy inference/action, tilt deviation, or tracking termination failure | Blend back to Stand; retain control |
| Any state | L1 + R1 for 1 second | Latched abort and Fault |
| Fault, after cleanup | Console **Clear fault**, with neutral inputs | Recover hardware groups and return to Disarmed |

Presses must be continuous and released before repeating. Named motions cannot
start from Hold, Locomotion or another motion. Status includes the selection,
active phase, progress and fault text. A reconnect requires neutral input before
accepting new movement. Samples are sequenced and monotonic; invalid values or
ordering close the session. Silence for 250 ms expires ownership.
The runner treats each normalized axis within ±0.05 as zero, using the same
neutral range for motion and the entry/confirmation checks. Small residual
input cannot activate the policy's minimum walking speed. The checkpoint's
gait stop transition can finish its remaining steps after a moving command
returns to zero.

Before acquisition, connection loss leaves the runner Disarmed. Hold retains
its current target (a ramp pauses). Locomotion zeros its velocity command and
returns to Stand. A healthy named motion finishes and returns to Stand even if
the console disappears. Tracking blends back to Stand if the controller session or
retargeted reference is lost. Control has no run-duration limit. Abort and feedback/lease faults
take priority. Remove `control.max_duration_seconds` from older hardware profiles. The first Ctrl-C during a motion requests a blended return;
another Ctrl-C or SIGTERM stops control. Stop cleanup damps and
releases leases automatically. A group already in Fault retains its daemon safety
response when its lease is released. The console's **Clear fault** action travels
through the active operator stream; only the runner obtains recovery leases.
Recovery must pass hardware guards and fresh Idle feedback. Failure leaves the
fault visible with the recovery error. Successful recovery resets policy history
and requires neutral input and a new Back+Start chord; it never enables motors.
Time-limit faults can be cleared through the same explicit action.

## Buffered Tracking (HIL)

Tracking follows the full-body XRoboToolkit/GMR reference with a configured
delay. This first version targets HIL commissioning. It shares the motion
transition, joint/motor limits, gain blending, fault handling and lease owner;
the console never sends joint targets. Upper-body following remains a separate
Locomotion option. The console's direct XRoboToolkit stream is display-only.

Run `bello-run-reference` on the robot service host, then add this explicit
section to the runner's hardware YAML (`tracking: null` leaves the mode absent):

```yaml
tracking:
  policy: /absolute/path/to/sonic.zip
  source: /absolute/path/to/sonic-checkpoint-source
  endpoint: http://127.0.0.1:50056
  rpc_timeout_seconds: 0.1
  poll_period_seconds: 0.02
  max_reference_age_ms: 100.0
  max_sample_gap_seconds: 0.1
  delay_seconds: 1.05
  transition_seconds: 1.0
  max_tilt_error_radians: 0.5
  max_raw_action_abs: 5.0
```

Continue using the locomotion bundle for `--policy`. The SONIC source checkout
must be clean and match its exported checkpoint revision; both policies must
agree on robot, motor coordinates, logical joints and transmission. Models load
and warm before any lease is acquired. No prerecorded clip is selected for live
Tracking.

Reach Stand, select Tracking with left/right, and wait for **Ready**. Hold A for
one second with neutral sticks to enter; release and hold A again to return.
The card shows buffered time and configured delay. The runner consumes ten future
reference frames, interpolating joint positions/velocities and root orientation.
The delay must cover `9 * future_dt + control_timestep + max_sample_gap_seconds`.
It never extrapolates, and the retained window is bounded. Input age includes
producer age and time since the request began. Duplicate sequences do not extend
the buffer; stale data, gaps or producer restarts require a fresh full window.

Reference loss, malformed references, inference failure and motion tilt deviation
blend back to Stand without releasing control. Reconnection cannot automatically
reenter Tracking. L1+R1 held for one second still damps the robot and latches Fault;
hardware feedback and lease faults retain that priority. Physical deployment and
live headset quality require separate commissioning; synthetic lifecycle and
prerecorded checkpoint parity do not establish either.

## Build and verify

Build on the target OS so the SDL native module matches Electron and the host:

```bash
cd console
npm ci
npm test
npm run make:mac       # macOS host: DMG and ZIP
npm run make:linux     # Linux host: AppImage (requires squashfs-tools)
```

Forge rebuilds the native module and keeps SDL's native library outside ASAR.
Artifacts go to ignored `console/out/make`. macOS packages are unsigned local
commissioning builds. Release signing/notarization requires deployment credentials.
CI builds all four OS/architecture combinations on native runners. Pushing a
`console-v<version>` tag triggers the build; its version must match
`console/package.json`. After all four packaged smoke tests pass, CI publishes
the installers and `SHA256SUMS` to a GitHub Release using
`console/RELEASE_NOTES.md`. Installers also remain downloadable Actions artifacts. See the
[console README](../console/README.md) for tag commands and required deploy keys.
`npm test` launches Electron against loopback mock services, checks direct
read-only hardware calls, stream independence, native SDL loading, GLB loading,
system-theme changes and a 375px layout. Set `BELLO_CONSOLE_EXECUTABLE` to a
packaged executable to run the same check against an installed build.

A physical controller and the UMPC mapping still require target-device testing.
Synthetic protocol tests cannot establish trained-policy stability. Follow
[hardware commissioning](hardware.md) and [Lima/vCAN](macos_vcan.md).

See the [verification record](policy_console_verification.md) for completed checks
and target-host checks still required.

### Controller guidance

Overview combines control state, mode selection, motion progress and faults in
one card. Compact button hints show the next action and hold duration. The two
stick diagrams show local normalized controller position. Velocity labels come
from the runner's requested Locomotion command, in operator forward/right axes
(m/s) and counterclockwise turn (rad/s). They show `--` outside Locomotion or
when runner status is stale; these are commands, not measured robot speeds.

### Damping on abort and faults

A named motion that fails inference, produces invalid actions, or exceeds its
reference-tilt deviation or tracking termination thresholds returns to Stand through the normal target and
gain blend. The runner reports the return reason in its phase and records it in
the policy log. The failed motion is not retried automatically. Hardware/lease
faults, stale or invalid feedback, and failures
of the standing policy or return transition still damp the robot. A return to
Stand requires healthy feedback throughout the transition.
The failed clip's tilt reference does not gate that recovery blend. The absolute
upright limit is checked before handing control back to Stand.

Console stream closure, timeout, malformed samples, and operator-service failure
use the same non-aborting disconnect path. Hold retains its last target (and
pauses its ramp); Stand continues; Locomotion zeros its command and returns to
Stand; healthy named motions finish and return to Stand. Hardware leases stay
with the runner. A previously confirmed explicit abort remains latched.

Hold the Overview **Hold to abort** button for one second, or hold **L1 + R1**
together for one second on the controller. The console button works without a gamepad. Releasing
early, losing focus, or losing the runner connection cancels the button hold.
Both paths request a runner abort and release hardware leases with damping,
without disabling the motors. The runner keeps a latched Fault for explicit
recovery. **Clear fault** performs the existing hardware recovery/reset and
returns to Disarmed; a new entry chord is required.

Bello Mini's physical and vCAN hardware profiles request damping for every
fault cause, including temperature and reported motor faults. Damping uses
zero position gain, zero requested velocity/torque, and the configured damping
gain, so stale host feedback does not force motor disable. Failed damping
transmission remains an error, not a substitute disable command. Power loss,
bus failure, and a drive's internal protections can prevent damping from taking
effect. Explicit hardware disable and daemon shutdown still disable motors.

### Motion selection and upper-body tracking

The highlighted dropdown row shows the controller's A/Cross prompt. A press
accepts that motion and closes the menu; release it, then hold again for one
second to start. Clicking a motion also selects and closes. D-pad up/down opens
the list again. Menu state and confirmation consumption belong to the runner.

With `control.following.enabled: true`, Y (Triangle on PlayStation) toggles
upper-body tracking during Locomotion. It starts Off on each entry; leaving
Locomotion, losing the operator, or faulting turns it Off. Holding Y cannot
repeat the toggle. The right-stick panel reports Off, Waiting, or Live. Off
blends the held joints back to nominal and slews torso commands to neutral.
Named motions and tracking may be configured together, with exclusive runtime
ownership. The reference producer still runs independently on port 50056.

### Direct tracking view

The Tracking tab connects from Electron main directly to the XRoboToolkit/PICO
tracking bridge's `PoseStream/StreamPoses` service. It displays the native
`xrobot_body`, `xrobot_left_hand`, and `xrobot_right_hand` fields in metres and
Y-up coordinates, including joint orientation rays. It does not reconstruct
SMPL-X or read pose data through the policy runner. Orbit, zoom, and Center pose
operate only on the display. Invalid joints are hidden; stale poses are muted.
Body and hands have independent ages and validity. Set the service address in
Settings under XRoboToolkit; no XRoboToolkit service is supplied by the HIL
emulator (its port 50052 hosts the emulator API).

The motion dropdown appears only when Motion is selected. Locomotion shows the
Upper-body teleop switch in the same slot, operable by click or Y after entering
Locomotion with a ready controller. Stand leaves that slot empty. Reference
transport failures, including HTTP/2 errors, return upper-body targets smoothly
to nominal while the reference client retries; they do not damp the robot.
Malformed reference payloads and producer application errors remain faults.

The desktop Overview fills the available window height. At 16:9 fullscreen
sizes from 1280 x 720 upward, the robot view and controller card fit without
page scrolling. Shorter desktop screens use compact stick indicators; small
windows and mobile layouts remain scrollable so controls stay accessible.
The controller/status card has fixed geometry across modes and faults. Mode
choices share the full row equally, with left/right hints at its outer edges.
Long fault details scroll within their panel.

Named bundles appear inside one Motion mode. Horizontal and vertical selection
wrap once per press. Navigation cancels an in-progress A hold; release A before
confirming the new selection. The dropdown retains its selection across modes.
The top bar shows the hottest fresh motor reading and its name above 60 C.
Otherwise it shows green Temp OK only when all motor telemetry is fresh and
available; missing or stale readings show Temp --. Hot readings are red. The indicator has
fixed width, with the full motor name available on hover.

Each named motion requires its own `source` checkout at the bundle's exact clean
commit. `--source` validates the locomotion bundle. This allows independently
trained policies to share one runner without rewriting bundle provenance.

The battery glyph fills proportionally to the reported state of charge. Missing
charge telemetry leaves the glyph empty and the numeric value at `--`.
Fresh charge above 20% colors the battery green, above 10% through 20% amber,
and 10% or below red. Missing or stale charge leaves the icon neutral.
Beside the percentage, a green bolt means charging, a crossed bolt means not
charging, and a question mark means the charging status is unavailable. Hover
for the status label. This uses the power board's fresh `BATTERY_CHARGING` info
flag; battery percentage and charging flags have independent freshness.

Mode buttons and motion dropdown items also accept clicks in Stand. Clicks only
select; the runner echoes the accepted selection and still requires a fresh
one-second confirm hold to enter it. Selection works without a gamepad, but it
never bypasses the controller connection or reconnect-neutrality gates for
control. Unknown choices and selections outside Stand are rejected. Clicking or
D-pad navigation cancels confirmation; release confirm before starting again.
Small arrow hints light up with their corresponding D-pad presses. Entry,
confirm, and shoulder-abort prompts fill using runner-reported hold progress.
Releasing either shoulder or losing samples resets an unfinished abort timer.
The on-screen abort button keeps its separate one-second pointer/keyboard hold.

## Assisted front-down startup

The runner supports two startup choices, selected with D-pad left/right or the
console's **Startup** buttons while Disarmed. **Crane** remains the default:
Back+Start for two seconds ramps to nominal Hold, then A for one second enters
Stand.

Enable **Front-down** by setting `control.lying_start` in the hardware YAML. Its
angles are logical joint radians, resolved through the bundle's motor mapping,
including differential ankles. This example is a commissioning preset, not a
validated physical lift pose:

```yaml
control:
  lying_start:
    hip_pitch_radians: -1.0
    knee_radians: 2.0
    straighten_speed_rps: 0.1
    ankle_engage_seconds: 2.0
```

Merge this into the full configuration. `null` leaves only Crane available.
Both hips must be more flexed and both knees more bent than the policy's nominal
pose. Configuration is rejected before lease acquisition if required Mini joints
are missing, mappings are invalid, or the pose exceeds policy/hardware limits.
The maximum configured straightening speed is 0.5 rad/s; ankle engagement takes
1 to 60 seconds.

| Phase | Operator action | Runner behavior |
| --- | --- | --- |
| Disarmed | Select Front-down; hold Back+Start for 2 s | Acquire control; ramp to bent hips/knees over `ramp_seconds`. |
| Lift | Lift and balance the supported robot upright | Hold bent legs. All ankle actuators have zero position gain, damping gain and feedforward torque. |
| Confirm balance | Release sticks; hold A for 1 s | Accept only while torso tilt is within the normal hardware limit. IMU tilt does not measure foot contact or balance; the operator confirms support. |
| Straighten | Push left stick up; release to pause | Move toward nominal hips/knees at up to the configured joint speed. Stick deflection scales speed; tighter motor step limits reduce it further. Other axes and downward input do not move the pose. Ankles remain free. |
| Upright | Release sticks; hold A for 1 s | Gradually restore ankle gains and targets from measured ankle positions, then enter the standing policy. |

The initial lying ramp and supported lift defer only the absolute upright-tilt
check. The normal tilt limit resumes after balance confirmation. Motor/IMU
freshness, velocity/acceleration limits, command bounds, lease checks and damping
abort remain active throughout. A tilted balance confirmation keeps the robot in
bent Hold. A held A cannot skip from balance confirmation into Stand.

Operator loss freezes startup progress and retains the current target/gains.
Reconnect requires neutral input. No display connection drives this sequence.
L1+R1 for one second, or the console's hold-to-abort button, damps the robot in
any startup phase. These are assisted setup positions; the runner does not
perform an autonomous get-up from the floor.
