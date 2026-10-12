# Policy control and deployment gate

`rust/bello-policy` is the only policy-control runner. It uses CPU ONNX Runtime
and the pinned `bello-stack` hardware client. Python remains responsible for
training, evaluation, export, live retargeting, and independent hand control;
there is no Python policy runner or compatibility launcher.

**Physical policy deployment is not yet commissioned.** Synthetic inference and
safety-lifecycle checks pass, but these do not establish trained-policy stability.
Before physical deployment, a trained bundle from the exact clean refactor
commit must pass CPU evaluation and isolated Lima/vCAN standing, walking,
transitions, perturbations, and shutdown checks. Motion deployment also requires
clip entry/completion/interruption checks. Review logs and videos for oscillation,
rejections, stale observations, and timing failures. Then separately verify
physical calibration, emergency stop, and support arrangements. Do not bypass
hardwared protections. This is a commissioning requirement, not a new artifact
version or compatibility flag; `--arm` alone is not evidence of approval.

## Export and read-only preflight

From the bundle's exact clean source checkout:

```bash
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
nix develop .#rust-runtime
cargo run --locked -p bello-policy -- --policy outputs/policy.zip --source . \
  --hardware-config rust/bello-policy/config/hardware.yaml
```

Start hardwared first. The [policy console](policy_console.md) connects directly
to hardwared, the estimator and the runner. The runner starts Disarmed and owns
all transitions; see the console guide for the controller sequence and loss
behavior. Only the runner acquires leases or sends motor commands.
Preflight checks source,
inventory, coordinate coverage, nominal targets/gains, IMU frames, group state,
and telemetry freshness without acquiring control or writing motor targets.
Use [Lima/vCAN](macos_vcan.md) for commissioning, not physical CAN.

## Target limits

The runner constrains every HOLD, RUN and live-tracking target against both motor
and URDF joint position bounds. Coupled motors move together: an unsafe request is
shortened along the line from the feasible nominal pose, then from the previous
accepted target to respect the hardware description's maximum position step.
Unrelated motors are not slowed. This is a hard envelope, not action smoothing.
Hardwared independently rejects motor, joint and step violations; it never
silently modifies an accepted command. Nonfinite requests fail.

Startup feedback may lie outside a bound by up to two Robstride position-encoding
intervals; joint-space tolerance is propagated through the transmission. This
allows quantization around zero without widening any command limit. HOLD starts
from a feasible target projected from that measurement, checks its step from
feedback, and ramps from that corrected target. Larger feedback violations fail
before acquiring control. The original feedback is never modified; the
`startup_target` log event records it alongside the initial command. Read-only
preflight performs the same check and reports the proposed startup target.
For emulator limit overshoot, the hardware YAML can explicitly add
`startup_feedback_margin_radians: 0.002` per motor coordinate. Its default is
zero, leaving the quantization-only check unchanged. This allowance propagates
through coupled-joint mappings; it does not change motor/joint command bounds,
the maximum first step, or hardwared's independent feedback checks. It is not
an action clamp or a way to bypass a hardwared fault. Do not copy this emulator
override into a physical-robot configuration without commissioning it.

Bundles carry exact URDF joint bounds, the motor-to-joint transform, motor bounds
and `position_step_limits` from the pinned hardware description. Live hardware
motor bounds can only tighten the bundle envelope. A stricter local hardwared
step setting can still reject a request and requires a matching export.
The ONNX `motor_target` output is a proposal; the runner applies the final envelope
after live upper-body following. Training and CPU evaluation use the same
connected-coordinate limiting rule. Older trained weights have not been trained
with this rule and require recommissioning even when packaged for the new runner.

JSONL `command.motor_target` records the submitted target;
`command.requested_motor_target` records the proposal. `target_limited` events
and periodic warnings report intervention; `command_rejected` events retain
the failed request. Repeated limiting is a diagnostic, not proof of stability.

For local settings, copy the hardware YAML to ignored `outputs/` and point
`--hardware-config` at that file. Keep the tracked checkout clean. In Lima,
forward runner port 50055 to Mac loopback and set the console runner endpoint
to `127.0.0.1:50055`. The reference endpoint, limits, and logging controls live in
the [Rust configuration](../rust/bello-policy/config/hardware.yaml).

Hold starts from measured joint positions and ramps to the bundle's nominal
pose. Distance from that pose does not gate startup, acquisition, or Hold in
either HIL or real runs. Joint and motor position limits, command slew limits,
feedback freshness, tilt checks, and hardware leases still apply. Remove the
retired `max_nominal_error_radians`, `max_coupled_nominal_error_radians`, and
`nominal_error_overrides_radians` control fields from machine-local YAML copies;
unknown configuration fields are rejected.

## Isolated emulator control

Add `--arm` for an authorized commissioning run. The two-second Back+Start chord
enters Hold, then A for one second enters Stand. Select modes with D-pad left/right and
confirm with A. L2 + R2 aborts. Stop cleanup damps and releases leases.
See the [console guide](policy_console.md) for the complete state sequence.

Named BeyondMimic bundles are preloaded from `motions` and start only from Stand.
Full-state bundles require `motions[].settings.pose`; proprioceptive bundles
require it to be omitted. Motion gravity-angle limits are configured under
`motions[].settings.max_tilt_error_radians`. The top-level `max_tilt_radians`
protects Hold, Stand and Locomotion. Feedback faults and abort take priority over
motion completion. SONIC physical arming remains outside this release.
See [observation modes](motion_tracking.md#observation-modes) and the
[native runtime guide](../rust/bello-policy/README.md).

The checkpoint owns normalization, motor targets, gait timing, command slew,
and observation history. Motor feedback is converted to joint kinematics for
observations. CAD frames define the IMU transform. Hardwared owns sensor filters;
the policy runner does not apply another action/observation low-pass filter.
The [emulator](emulator.md) owns its robot model independently of policy bundles.

## Live references and hands

See the [PICO startup guide](pico_startup.md) for process order, Ubuntu service
commands, local port assignments, and proportional hand control.

The live path consists of three processes:

```text
PICO -> pico-smplx-bridge :50052 -> bello-run-reference (GMR) :50056 -> Rust policy runner
                              -> bello-run-hands -> Linker CAN
```

Body following uses the bridge's typed `xrobot_body`: 24 named global poses,
positions in metres and unit WXYZ orientations, in native right-handed Y-up
tracking space. The caller applies upstream GMR's proper rotation
`(x, y, z) -> (x, -z, y)` once. No reference-pose rotation subtraction, neutral
SMPL-X skeleton, mesh, floor correction, or reconstructed wrist position is used.
The producer publishes held-arm joint references and torso lean/twist from the
same retargeted robot pose. Torso roll/pitch are ground-relative; yaw is relative
to pelvis heading, after removing the nominal CAD frame orientation. Rust clamps
these commands to trained ranges and uses the checkpoint's torso slew before
locomotion inference. It does not copy waist joints directly. Gamepad velocity
commands remain independent.

Enable `control.following.enabled` with a newly trained torso-conditioned bundle;
leave `control.torso_command` neutral (manual torso overrides are disabled while
following). Rust retains motor ownership and safety limits. Tracking loss
transitions arms toward nominal and slews torso commands to neutral. Malformed
messages or terminal producer errors stop control. The producer and runner must
be updated together; available references require all three torso angles.

Start the updated tracking bridge first, then use
`bello-run-reference asset=bello_mini xrobot.endpoint=HOST:50052`.
The old `smplx.*` body-command fields are removed; there is no legacy fallback.
A bridge without the typed body field fails explicitly. PICO's own body-tracker
calibration is still required; the bridge's SMPL-X T-pose calibration is not.

Both consumers use `pico.smplx.v1.PoseStream/StreamPoses` from pinned
`deps/pico-smplx-bridge/pose_stream.proto`. Bello imports only generated
protobufs, not the bridge's SDK or server. Hands use `xrobot.endpoint` and
consume typed `xrobot_left_hand` / `xrobot_right_hand`: 26 named native PICO
landmarks per side, position validity flags and independent sample age. The
mapper derives palm-local finger flexion and thumb adduction directly, retaining
six-axis actuator calibration, smoothing, slew limits and recovery reseeding.
No SMPL-X rotations, neutral model or reconstructed skeleton are used. Missing
body tracking does not invalidate hands; a stale or invalid hand suspends only
that side. Missing typed fields require a bridge upgrade, with no legacy fallback.
The bridge no longer generates SMPL-X pose fields; update producer and consumers
together. `smplx.*` is no longer accepted by the hand command.

For body tracking, freshness uses the larger of body-sample and packet age,
plus `xrobot.transport_latency_budget_ms` and local monotonic elapsed time.
This budget is an allowance, not measured network latency. The bridge rejects
frozen bodies even when outer packets advance. With firmware that sends zero
body timestamps, it conservatively uses pose changes to detect freezing.
`xrobot.max_source_age_ms` rejects stale frames,
`max_source_interval_seconds` detects clock gaps,
`rpc_timeout_seconds` reconnects silent streams, and
`connect_timeout_seconds` bounds first-message waits.
`max_rate_hz` and optional `tls_root_certificate_path` /
`tls_server_name_override` are under `xrobot` for both commands.
Hand freshness similarly uses the larger of packet and hand-sample age. The
current headset client has no hand timestamps, so unchanged samples conservatively
expire even if body packets continue; a perfectly static hand can also expire.
Both paths retain latest-only bounded delivery and discontinuity barriers.

The live profile is `xrobot_to_bello_mini.json`; the SMPL-X profile remains
for offline datasets and FastSAM, not as another live PICO option. Raw-source
frame offsets are transferred through the common G1 link frames; upper-body
position costs are zero and orientation costs follow the upstream xrobot
baseline. These are initial live profiles, not newly commissioned hardware
settings. The Mini high-five trial is deferred to the operator.

The upstream GMR solver is unchanged. The caller initializes the first scaled
root target and configured joint seed once per sequence, then makes one upstream
retarget call per frame. No synthetic standing solve precedes the raw input.
Runtime safety checks and the policy's existing reference limits are unchanged.

Exercise raw pose delivery, real GMR and reference RPC without a licensed model
or hardware:

```bash
JAX_PLATFORMS=cpu uv run pytest -q -m integration tests/test_xrobot_reference_integration.py
```

This is a synthetic integration check, not a live tracking or physical trial.

`bello-run-hands` independently controls Linker hands over classic CAN, not
hardwared motor targets. It defaults to read-only validation; see
[hands.yaml](../configs/commands/hands.yaml). Hand arming is separate from policy
arming and requires its own bus, device, calibration, and safety review.

Armed operation exposes the [hand control API](hand_control.md) at
`127.0.0.1:50057`. Hands start suspended. Each hand has an exclusive renewable
lease selecting application targets, XRoboToolkit tracking, or suspended writes.
Tracking cannot overwrite application targets. Lease expiry never opens the
hand or switches back to tracking. Position feedback is available; contact,
grasp confirmation, and force control are not exposed by this service.

All six hand axes use proportional targets between their calibrated open and
closed positions: a half-flexed finger requests the midpoint. The
`smoothing_time_constant_ms` low-pass filter and `max_raw_rate_per_second` slew
limit apply to finger flexion and thumb abduction alike. After tracking loss,
control reseeds from measured hand positions and limits the first recovered
command to one control period of movement. `control_rate_hz` sets the command
rate (5 Hz by default).

The optional runner `barbell` configuration exposes the dedicated console mode;
see [Barbell mode](barbell_mode.md) for policy sources, arm poses, hand leases,
confirmation gates, and cancellation behavior.
