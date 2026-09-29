# Console and Mini visual verification

Verified locally on macOS arm64 on 2026-09-29. The visual stack update is commit
`3458fb5dd0f07029f317ed0bd4182c773cb84a4e`; queue and cleanup fixes advance it to
`61406e7`, then fault damping fixes advance it to `c5e2520`. The console branch
pins that revision. The sections below record checks as implementation progressed;
the final maintenance audit records the isolated branch's current validation.

## Completed

| Check | Result |
| --- | --- |
| `uv lock --check`, protobuf regeneration check, Ruff checks and formatting | Passed |
| `./scripts/test.sh` | Passed; opt-in integration and unavailable licensed-model cases retain their normal exclusions |
| `cargo test --locked -p bello-policy --lib` | 29 passed |
| Native controller lifecycle suite | 21 cases passed across focused runs, including lease renewal failure and cleanup |
| Native motion control suite | 49 cases passed across focused runs, including two named motions, single-session ownership, reconnect neutrality and phase-dependent connection loss |
| Stack MuJoCo and visualizer library/integration tests on macOS | 48 passed; Linux-only vCAN cases are excluded on macOS |
| Native Mini MuJoCo `check` and physics export | Passed, 28 actuators and 48.58 kg |
| Python Mini and full Bello model compilation | Passed, physics/visual masses agree and runtime mass/collision validation passes |
| Canonical Mini comparison | All nonvisual XML data identical to the previous stack revision |
| Asset integrity | All 28 GLBs match supplied files byte-for-byte; every visual path and STL companion exists |
| Viewer inspection | Neutral and bent poses rendered in Electron and Kiss3d; embedded materials, head and hands visible |
| `npm test`, source and packaged macOS app | Passed: direct streams, independent hardware/estimator loss, native SDL loading, sequenced samples, GLB loading, system-theme changes, AA text contrast and 375px layout |
| `npm run make:mac` | DMG and ZIP built for arm64 |
| Connection settings | UI validation, cancel, shared host changes, individual runner changes, previous stream shutdown and saved settings after restart passed against local mock services |
| Verbose status tabs | Hardware/pose/operator/service fields, search, keyboard navigation, mobile width and display pause while controller samples continue verified with mock streams |
| 3D feedback overlay | IMU/pose ages and XYZ shown at bottom-right; expandable motor values and stale feedback verified; separate Robot state card removed |
| Power telemetry and compact header | Header battery percentage and motor-rail values, missing charge, critical power flags, stale readings, icon-only settings and hover/focus service errors verified with mock streams |
| Header robot status | Mode remains visible across tabs, updates on runner changes and is marked stale after runner loss |
| `npm audit --omit=dev` | No reported production dependency vulnerabilities |

Native lifecycle tests use loopback synthetic hardware and synthetic policies.
They verify runner transitions and safety handling without commanding a robot.
The two-motion test exercises Disarmed, Hold, Stand, Locomotion, Wave, Bow and
Fault, with disconnects before acquisition and during each controlled phase.

The full development dependency audit reports advisories in Forge's packaging
dependency tree, including its archive utilities. They are excluded from the
packaged application; the production-only audit is separate from that finding.

Generated screenshots, the native physics export and installers are in ignored
`console/out/`. The reproducible Kiss3d renderer is the stack's
`render_description` example. Do not commit these generated artifacts.

## Lima commissioning and console follow-up

The `bello-vcan` Lima guest now runs the native Mini emulator, hardwared,
estimator and runner. Ports 50051, 50052, 50053 and 50055 forward to localhost.
The locomotion, Lunge and Squat bundles passed their exact-source provenance
checks. Read-only preflight saw 28 motors, live pose and synthetic power telemetry.
The virtual robot completed its Hold ramp and remained in Hold for a 60-second
observation after the CAN queue fix. Taking `vcan1` down produced a latched
hardware fault and stopped the emulator; cleanup released ownership without the
previous invalid-state error chain. Recovery was rejected while feedback was
missing. After restoring the emulator, recovery returned groups to Idle. A
subsequent operator abort was cleared through the packaged console to Disarmed.
Screenshots and commissioning helpers are under ignored `outputs/console-hil/`.

Follow-up checks passed:

- Controller sampling regression tests reproduce synchronous SDL close events
  during both axes and button reads. Both yield neutral disconnected samples,
  and a replacement controller can be sampled without restarting the app.
- 30 runner unit tests, including neutral-only clearing after cleanup, duplicate
  clear rejection, abort during recovery, and requiring a new entry chord.
- 12 CAN unit tests, including bounded enqueue waiting and deadline expiry.
- 16 hardware-client integration tests, including lease release from Fault
  without automatic recovery or disable.
- Electron smoke checks for failed/successful clear requests, exactly one
  sequenced clear sample per click, fixed header geometry when values and mode
  change, full-card 3D bounds, and entering/exiting native fullscreen.
- Power readings use inventory's 3-second board timeout; a live packaged check
  observed 40 consecutive fresh snapshots. Float display is capped at three
  decimal places with raw detailed values on hover.
- Both vCAN transmit queue lengths are 65,536; the emulator setup script retains
  that value. Hardwared's internal queue remains bounded with its original
  transmission deadline.

## Still required on target hosts

- Linux x86_64 Debian packaging and packaged smoke test. The new console workflow
  defines macOS and Ubuntu build/test jobs; those CI jobs were not run locally.
- The complete Stand/Locomotion/named-motion sequence through Linux vCAN and
  the MuJoCo emulator; the Lima checks above cover Hold, faults and recovery.
- Physical gamepad mapping, long presses and hot-plug behavior on the macOS
  controller and Linux UMPC. SDL now detects a Nintendo Switch Pro Controller;
  manual mapping and hot-plug acceptance remain to be completed.
- Trained-policy commissioning and physical operation remain governed by the
  existing hardware deployment gate. No physical motors were commanded.

The earlier text-free lunge video is a separate deliverable.

Operator lease checks also cover renewing ownership with the gamepad unplugged,
rejecting another client's abort samples, clean stream release, silence expiry,
and neutral input on takeover. The console smoke test covers the Runner
`In use` indicator and automatic reconnection after the lease becomes available.
The current suite passes 34 runner tests and four controller sampling tests.
The nominal-pose deviation gate and its configuration have been removed from
the runner for both HIL and real runs.
Neutral-axis regression coverage checks both signs and the exact neutral
boundary, independent axes, and return to zero. Values accepted as neutral
cannot reach the policy's minimum-moving-speed mapping.
A live Lima regression entered Hold, Stand, and Locomotion, then supplied
normalized input `[0.00874, -0.00874, 0.00874]` for two seconds. All 40 sampled
policy log entries contained zero requested velocity, zero gait velocity, and
no pending gait steps. The test returned to Stand and released the torso guide.

## Unlimited runs and controller guidance

The run-duration field and enforcement have been removed. The Lima runner was
rebuilt and restarted with a migrated profile; startup reached Disarmed. Its
34 unit tests pass. Synthetic lifecycle tests now explicitly terminate healthy
sessions after checking behavior instead of relying on the removed timer; those
Python integration cases were not rerun in this pass.

The combined Overview card shows state-specific controller instructions, local
button and stick feedback, motion phase progress, and fault recovery. Seven
console unit tests pass, including instruction priority for stale connections,
recovery, neutral input, and transitions. The Electron smoke suite passes with
light/dark contrast, mobile width, full-card model bounds, read-only display
streams, connection changes, and fault recovery checks.
The rebuilt macOS arm64 package also passes the smoke suite and loads its
bundled SDL module. The updated repository app was launched against Lima.

Horizontal mode selection uses D-pad left/right, with wraparound and one change
per press. Up/down only participate in neutral-input checks. The suite now
passes 36 runner tests and eight console tests, including both horizontal
directions, held buttons, simultaneous directions, and reconnect neutrality.
The Electron smoke suite and macOS package build also pass. Both running
Lima and desktop components were updated together.

The concise controller card uses two stick-position diagrams with runner-supplied
command velocities. 37 runner tests pass, including projection into operator
forward/right axes; eight console unit tests and the Electron smoke suite pass.
Smoke coverage verifies m/s and rad/s fields through the operator stream and
hides command values outside Locomotion. Light/dark contrast and mobile width
checks continue to pass.

Fixed controller-card geometry is checked across Disarmed, Hold ramp/holding,
Locomotion, named motion and long faults at desktop and 375px width. Mouse and
keyboard short holds cancel abort; a full hold sends one sequenced request.
The packaged macOS console is rebuilt with the narrow abort-hold IPC.

Fault damping changes are pinned in bello-stack commit `c5e2520`. Tests pass:
39 runner tests, 20 client unit tests, 18 client integration tests, and 113
hardwared tests. These cover ready Damped groups, console abort without a
gamepad, damping release without disable, failed-start damping, and zero-Kp
damping without host feedback.

The Motion mode groups Lunge/Squat (or other loaded bundles) into one dropdown.
41 runner tests pass, including vertical wraparound and restarting A confirmation
after navigation. Ten console unit tests and packaged macOS smoke checks pass.
The smoke suite checks grouped mode labels, dropdown selection updates, the strict
60 C temperature threshold, stale temperature status, and fixed indicator bounds.
Gamepad hints use Lucide glyphs and shoulder/face button shapes.

A live Lima abort request without a connected gamepad left both lower_body and
upper_body in GROUP_STATE_DAMPED. A SIGSTOP-based watchdog probe did not stop the
runner process in this environment and did not induce a watchdog fault; it is not
counted as live fault verification. Fault responses are covered by the hardwared
and client tests listed above. All four Lima services were restarted after the
probe, restoring the upright startup guide and Disarmed runner. vcan0 and vcan1
retain queue lengths of 65536.

Controller prompts now use 54 unmodified PNGs from JulioCacko Input Prompts 1.4,
with source paths and hashes recorded alongside the assets. All prompts render at
24 px and switch between original Default/Light variants with the system theme.
General Settings offers PlayStation, Xbox and Switch previews, plus robot
connection fields. Eleven console unit tests and the rebuilt macOS packaged
smoke suite pass. The suite verifies each style, saved selection after restart,
cancel behavior, intact operator ownership and service streams during style-only
changes, and loading the bundled artwork in the packaged app. Desktop light and
375px dark Settings screenshots were reviewed.

The card title is removed while its status and controller connection remain.
The current prompt variants are retained at 24 px. Temperature uses green
Temp OK and red for the hottest motor above 60 C; its normal detail is tooltip
only. Battery icon fill follows reported charge, including an empty unknown
state. Packaged smoke checks cover these states and fixed header geometry.

HIL locomotion now uses W&B run `gyungmin/bello-mujoco/w3orivmd`, checkpoint
`bello_mini_locomotion_seed0-update-106496:v10`, the last uploaded checkpoint of
`bello_mini_locomotion_measured_mass_a8d9b9f_seed0_resume7168`. It was exported
from clean source `a8d9b9fe1fd653faa7314fc7cfc4e74be146d6cb` with its pinned
bello-stack dependency. The original export lacks torso-command metadata. An
explicit deployment adaptation adds zero torso ranges and unit slew rates;
ONNX graph, weights, observations, physics and source commit are unchanged.
The adapted bundle SHA-256 is
`6931bfd43ef8d97692d0dfc7c4a5999ed122c197ca3da482e84bd688ffbaa219`.
The local artifact provenance is recorded in
`outputs/requested-policy-selection.json`.

Each named motion now declares its own exact source checkout, allowing the
existing Lunge/Squat bundles at ab58dd2 to remain loaded. 41 runner unit tests
pass. Native ONNX validation and HIL hardware preflight passed before the runner
was restarted with the new versioned bundle and source checkout. No automatic
Locomotion trial was started.

Final packaged smoke passed with proportional battery fill (unknown, 1%,
82.5%), green/red temperature states and the removed card heading. The refreshed
live console was visually checked with Switch prompts and 75% charge. A feedback
sample-span fault occurred during the user's HIL trial after policy startup;
Clear fault succeeded and the final observed runner state was Disarmed, with
all three streams live. The 40 ms sample-span guard remains in effect.

## Click selection, hold feedback, compact status, and main integration

Merged remote `origin/main` through `d582d2a` into `feat/sonic-transfer` with a
fast-forward, then restored the ongoing work and its original staged gitlinks.
Resolved overlaps retain the operator service, power logging, direct XRoboToolkit
tracking, and systemd CAN initialization. Pre-merge file backups and a named stash
remain available locally.

The rebuilt macOS package passes the full mock-service Electron smoke test:
clicking mode/motion buttons sends sequenced selection requests, highlights follow
the echoed runner selection, unknown requests and selection outside Stand are
rejected, and the console still calls only ListHardware/WatchTelemetry on hardwared.
The same test checks hold-progress fills, fixed card/header geometry, centered
Temp OK, proportional battery fill, removal of repeated status headers and pause
controls, theme switching, native SDL load, and 375 px layout.

46 runner unit tests pass, including one-second continuous shoulder abort,
release/gap cancellation, abort priority, selection without actuation, fresh
confirmation after clicks, and reconnect neutrality. 11 console unit tests and
95 targeted Python tests pass (systemd, XRoboToolkit hands, and command configuration).
Protobuf regeneration and git diff whitespace checks pass. The macOS DMG and ZIP
were rebuilt. No physical controller was attached during packaged verification.

Live HIL: the torso guide was released as requested. The emulator subsequently
exited on its existing 100 ms wall-clock lag guard. Full service restart passed
read-only policy preflight and the rebuilt console briefly showed all three
streams Live, 75% charge, 48 V / 1 A, Temp OK, and Disarmed. The emulator timing
failure recurred, including after a trial of SCHED_RR priority 10 on its main
thread. Scheduling priority is therefore not established as a fix. No feedback
or timing guard was loosened. At the end of this verification the emulator is
stopped; hardwared, estimator, and the new runner remain running, with the runner
Disarmed. Stable HIL runtime is still unresolved on this host.

## Motion recovery, controller menus, and direct tracking

Recoverable named-motion termination, inference, and action failures now blend
back to Stand without releasing control. Feedback, tilt-envelope, command,
lease, and standing-policy failures remain damping faults. Synthetic integration
tests exercise failures during entry and active motion, repeated selection,
target-step bounds, successful completion, abort after recovery, rejection,
stale feedback, and unsafe return tilt. The menu now consumes an A press to
accept and close; starting requires release and another full hold.

Y toggles configured upper-body following in Locomotion. Synthetic reference
tests verify smooth off/on transitions, reference outage, and invalid reference
fault handling. Named motions can coexist with following configuration; only
one controls upper-body targets at a time. The held-button/phase/configuration
checks are covered by runner unit tests.

Validation: 47 Rust unit tests, 19 focused synthetic integration cases across
the motion and following suites, and 15 console unit tests pass. The unsupported
proprioceptive absolute-height case is skipped. Lifecycle fixtures now wait for
acknowledged Stand and settled targets before terminating the runner, and use
the one-second abort chord. Protobuf consistency, Python formatting/lint, and
diff whitespace checks pass.

Both development and packaged macOS Electron smoke tests pass. Coverage includes
dropdown selection/close/reopen, selected-row controller prompt, independent
direct XRoboToolkit streaming and stale detection, native SDL loading, fixed
geometry, and motor-rail label/value alignment. Mode buttons share equal widths
with left/right hints at the outer edges of the selector. The tracking view was
rendered with synthetic native body poses. No physical controller/headset stream or Linux
desktop package was validated in this update. macOS DMG/ZIP were rebuilt.

Lima's runner was rebuilt and restarted, with following configured as available
but initially Off. Emulator restart and read-only hardware preflight passed;
the live console showed Disarmed, all three robot services Live, and no active
fault. Tracking waits for the configured XRoboToolkit bridge at 10.42.0.1:50052;
HIL does not provide that service or a live reference producer. The existing
emulator wall-clock lag issue has not been changed by this work.

## 2026-09-29: App icon and architecture builds

The Bello face PNG is now the desktop icon. macOS packaging generates ICNS
representations from the source; the packaged ICNS hash matches the generated
file. Linux AppImages contain the PNG, desktop entry, and native SDL module.

Built an arm64 macOS DMG and an arm64 Linux AppImage. Packaged smoke tests pass
on macOS (536 operator samples) and Ubuntu 24.04 in Lima (3953 samples), including
native SDL loading. Linux tests launch the extracted AppImage's AppRun, with
SwiftShader under Xvfb; the normal application retains its default renderer.
Physical gamepads, FUSE mounting, and x64 executables were not tested here.

The Actions workflow passes actionlint. It builds DMG/AppImage artifacts on
native x64 and arm64 runners for each OS and validates `console-v<version>` tags
against the package version. No tag was pushed and no remote CI run was started.

## 2026-09-29: Motion recovery and connection resilience

Motion-reference tilt deviations now blend back to Stand during entry and active
motion. The return does not enforce the failed clip's tilt trajectory; the
absolute upright limit remains required at handover. Tests cover repeated
recovery, target/gain continuity, and subsequent explicit abort. Reference
HTTP/2 transport errors now take the reconnect/nominal-return path, including
the tonic Unknown status produced by a malformed HTTP/2 peer. Producer
application errors and invalid reference payloads still fault.

Audited console transport closure, silence/lease expiry, invalid samples,
reconnection, and operator-service failure. None creates an abort. Hold ramps
pause at the acknowledged target, Hold/Stand retain control, Locomotion zeros
the command and returns to Stand, and named motions finish. Explicit aborts
remain latched. Independent display streams never participate in those gates.

Validation: 49 Rust unit tests; 19 focused motion/disconnect integration cases;
three following/outage/invalid-reference integration cases. Packaged macOS
smoke passes (533 samples, native SDL loaded, one attached controller), including
conditional dropdown/teleop display, fixed geometry, Light/Dark/System settings,
system changes, and preservation of the operator session across appearance
changes. The HTTP/2 regression explicitly verifies tonic Code::Unknown.

Restarted emulator, hardwared, estimator, and updated runner in Lima. Read-only
preflight passed and both vCAN queue lengths remain 65536. The runner starts
Disarmed. No reference producer is listening on 127.0.0.1:50056; following waits
and retries. No physical robot recovery was tested.

## 2026-09-29: Compact status and window fixes

Fault details now scroll separately from the recovery button and release-input
hint. Packaged macOS checks verify that both controls remain inside the fault
panel with long messages at desktop and 375-pixel widths. The XRoboToolkit badge
also contains its complete stale label at 375 pixels.

The native window is resizable, with its minimum height reduced from 620 to 360
pixels. Packaged checks resize the native window to 800 x 480 and 1000 x 700.
Battery tests cover green/amber/red thresholds, stale neutral color, charging,
not charging, and unavailable charging flags without shifting header geometry.
The packaged smoke passed with native SDL loaded and one controller attached.

## 2026-09-29: Maintenance audit and branch isolation

Console work is isolated on `feat/bello-mini-console` in a separate worktree.
The original directory remains on `feat/sonic-transfer` with its existing SONIC
work uncommitted. SONIC training, configuration, runtime implementation, and the
gear_sonic gitlink are excluded from the console commit.

The renderer now separates the header, settings, robot view, and fault panel
from application state. Electron shares one sender check for privileged IPC and
one runner-readiness check. Display snapshot derivation is a pure module with
independent tests; connection ownership stays in the main process. Three.js
teardown disposes geometry, materials, textures, and environment resources, and
guards asynchronous mesh loading after unmount. The runner shares its neutral
input predicate between reconnect and fault recovery. Obsolete styles and remote
command naming were removed. A pinned formatter and CI formatting check cover
the console source.

Validation on this isolated branch:

- 19 console unit tests, formatting, UI build, and packaged macOS smoke passed.
  The smoke run received 642 operator samples and loaded the native SDL module
  with one controller attached. The arm64 DMG and ZIP were rebuilt.
- 48 Rust unit tests and the Linux release build passed.
- Native control/motion integration completed with 82 passed, one skipped, and
  one failure caused by a test binding the live runner's port 50055. The fixture
  now chooses an unused operator port; both affected parametrized cases passed
  when rerun. No additional integration failures remained.
- The GLB-to-STL URDF adapter test, lockfile check, protobuf consistency check,
  targeted Python lint/format checks, and Git whitespace checks passed.
- Production dependency audit reported no advisories. Development packaging
  dependencies still have transitive advisories; no incompatible dependency
  overrides were applied.

This audit did not restart the live HIL services or command physical motors.
Linux and x64 installers were not rebuilt in this maintenance pass; the earlier
Linux arm64 result above and the CI matrix describe their separate coverage.

## 2026-09-29: SONIC merge and buffered Tracking

Merged the committed, validated SONIC history from
`experiment/sonic-amass-all-20260929` (`34e90d4`) into the isolated console branch.
The original workspace and its uncommitted SONIC work remain untouched.

Tracking uses a bounded delayed reference window and the existing motion task
transition lifecycle. The full-body and upper-body reference payloads come from
one GMR sample. Operator status exposes readiness, age, buffer duration and delay;
the console adds a fourth mode only when configured. Reference failures return
to Stand; hardware faults and the one-second abort chord still damp.

Validation:

- 55 Rust unit tests and the Linux release build passed, including interpolation,
  sequence duplication/restart, stale/gapped references, bounded retention and
  the Tracking confirmation gate.
- Native synthetic SONIC parity passed for 25/50 Hz clips, history resets and clip
  boundaries. Synthetic Tracking lifecycle passed for lost, stale and malformed
  references, console lease expiry, reconnect neutrality, rebuffering/re-entry,
  return to Stand, and abort damping.
- Across native SONIC/control/motion integration, 90 cases passed and one skipped
  after correcting the new fixture and replacing a racy PTY-EOF/process-poll
  assertion with `wait()`. The six initially failing cases passed on rerun.
- Console unit tests and macOS SDL/WebGL smoke passed with four modes and stable
  controller geometry. Source format, protobuf consistency and lockfile checks
  passed. The full default CPU suite passed: 790 passed, five skipped.
- Requested W&B run `7qvvdxi1`, display name
  `sonic-amass-upright-v1-kr-20260928-r5-sampler`, exported from update 6000,
  artifact `gyungmin/bello-mujoco/sonic-amass-upright-v1-kr-20260928-r5-sampler_20260928_132430_423700:v59`.
  Export retained its exact clean source revision
  `9d6dcf0c23d5235439e5f252090115aa92e9a3f8` and verified corpus. Linux native
  bundle validation and eight telemetry samples across two held-out clips passed.
  Comparing native actions with the restored JAX actor gave maximum absolute error
  `1.8914703370231933e-06`.

These checks do not establish live-headset retargeting quality or physical SONIC
stability. No physical motor commands were sent. Checkpoints, observations,
source snapshots and generated installers remain ignored local artifacts.
Read-only preflight against the existing Lima stack loaded the locomotion,
named-motion and real SONIC bundles together, but stopped at hardwared's existing
`control lease lost; damping applied` fault. No trained-policy physics HIL run
is claimed by this record.

## 2026-09-29: Assisted startup review and console 0.2.0

Reviewed startup transitions, lease recovery, controller/IPC boundaries, the
shared motion/Tracking backend, and desktop packaging. The runner offers Crane
and explicitly configured Front-down startup. Front-down keeps ankle gains and
feedforward at zero until the operator confirms the final Stand transition.

Review fixes:

- Startup phases use an enum internally, with wire labels defined in one place.
- Straightening duration accounts for both logical joint speed and motor step
  limits. This prevents a delayed target from continuing after stick release.
- Fault recovery still requires Idle, while allowing a supported flat torso.
  Startup readiness is cleared on fault/recovery.
- Removed the console's fabricated Crane fallback; startup choices come from
  the runner. Shared motion backends use one allocation each at initialization
  instead of a large inline enum. Clippy findings were resolved.
- Console 0.2.0 adds a tag-only release job. It requires all four native desktop
  jobs, checks that each supplies exactly one installer, and publishes distinct
  architecture names and SHA-256 checksums. The asset collection/checksum script
  was exercised locally with publishing mocked.

Validation includes the full default CPU suite (790 passed, five skipped),
58 Rust unit tests, a Linux release build, warning-free library Clippy, 28 native
startup/control cases, seven Motion/Tracking lifecycle regressions, and a
separate flat-torso fault recovery check. The native
startup test covers differential ankles, balance confirmation, reconnect
neutrality, stick release under tight step limits, gradual ankle engagement,
abort damping, and invalid configuration before acquisition. Console validation
includes 21 unit tests and the macOS Electron SDL/WebGL smoke with fixed startup
card geometry. Production npm dependencies reported zero audit advisories.
Lockfile, generated protobuf, Python lint/format, console format and whitespace
checks passed.

These tests use loopback hardware and synthetic policies. Front-down pose angles
remain opt-in configuration; no physical lift or trained-policy physics HIL
validation is claimed. Platform installers are independently verified by the
release workflow before publication.

## 2026-09-29: Console 0.2.3 recovery and policy continuity

- Abort changes the one-second stop button to Hold to idle after damping cleanup.
  A separate one-second L1+R1 hold uses the same runner recovery path. Continued
  holds cannot cross from abort to Idle or from recovery into another abort.
- Stand/Locomotion changes and disconnect-to-Stand preserve policy history,
  previous actions, gait state and torso command smoothing.
- Assisted startup is now Back-down (face upward). The loopback startup fixture
  begins with the corresponding flat-torso orientation. Hip and knee flexion
  signs, balance confirmation, free ankles and paced straightening are retained.
- macOS packaging targets arm64 only; Linux retains x64 and arm64.

Validation: all 60 Rust unit tests and eight Linux native integration cases
passed. The native continuity regression first failed against the previous
runner when its previous-action observation jumped to zero on a mode change.
With the fix, repeated Stand/Locomotion changes and moving-session disconnect
match an uninterrupted replay of the same measured inputs. The startup tests
include damping-to-Idle recovery followed by re-entry to Hold and Stand.
All 25 console unit tests and the macOS Electron smoke passed, including short
holds, action changes during a held button, recovery failure/retry and Back-down
selection. These are synthetic control checks, not physical stability tests.

## SONIC neutral return (console 0.2.4)

The runner now generates a bounded smoothstep reference to the standing bundle's
nominal pose inside SONIC, followed by an independently timed Stand handover.
Synthetic tests cover normal return, stale/malformed/missing references, console
loss, settling timeout, abort during return, rebuffering and explicit re-entry.
They check reference convergence, retained SONIC observation history, full SONIC
targets/gains during neutral return and the short handover duration. Three
existing named-motion/connection-loss cases remain passing.

Validation: 61 Rust unit tests, strict Clippy, 10 native lifecycle/regression cases,
25 console unit tests and the macOS Electron smoke test. One expanded native run
hit the existing hardware-freshness guard before Tracking entry; that isolated
case passed on rerun without changing freshness limits. These checks validate
control flow and synthetic observations, not physical SONIC balance or tracking.
