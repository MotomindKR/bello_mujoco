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

Start hardwared first. The gamepad/remote-command service may start later: the
policy runner starts with zero velocity commands and connects in the background.
Missing or stale remote input requests standing, not motor disable; it is not an
emergency stop. On connection or reconnection, an armed neutral sample is required
before accepting movement. The interactive HOLD/RUN/IDLE sequence is unchanged.
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
set `control.remote.endpoint` to `http://host.lima.internal:50054` for the native
Mac visualizer. The reference endpoint, limits, and logging controls live in
the [Rust configuration](../rust/bello-policy/config/hardware.yaml).

Before acquiring control, the runner checks measured motor positions against
the bundle's nominal pose. `control.max_nominal_error_radians` defaults to
0.5 radians for direct coordinates; coupled coordinates use
`control.max_coupled_nominal_error_radians` (1.0 radian). Existing machine-local
YAML copies retain their own values and must be updated separately.

## Isolated emulator control

Add `--arm` only for an explicitly authorized commissioning run. Follow the
interactive sequence: `HOLD` acquires control and ramps to nominal; `RUN` starts
inference; after stopping, `IDLE` explicitly disables motors. Errors and
interruptions request damping first. Without `IDLE`, hardwared's watchdog and
lease-expiry safety remain authoritative.

Motion control uses a motion bundle plus a standing bundle and a fresh pose
stream. `MOTION` blends from standing into the clip. Completion or the first
Ctrl-C during motion returns to standing; another interrupt stops control.
Malformed/stale feedback, rejected commands, remote abort, and SIGTERM stop
control instead of attempting a standing recovery. See the
[native runtime guide](../rust/bello-policy/README.md) for settings and commands.

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
PICO -> pico-smplx-bridge :50051 -> bello-run-reference (GMR) :50056 -> Rust policy runner
                              -> bello-run-hands -> Linker CAN
```

Body following uses the bridge's typed `xrobot_body`: 24 named global poses,
positions in metres and unit WXYZ orientations, in native right-handed Y-up
tracking space. The caller applies upstream GMR's proper rotation
`(x, y, z) -> (x, -z, y)` once. No reference-pose rotation subtraction, neutral
SMPL-X skeleton, mesh, floor correction, or reconstructed wrist position is used.
GMR publishes joint references; Rust retains motor ownership, limits and
outage-to-nominal blending. Malformed messages or terminal producer errors stop
control.

Start the updated tracking bridge first, then use
`bello-run-reference asset=bello_mini xrobot.endpoint=HOST:50051`.
The old `smplx.*` body-command fields are removed; there is no legacy fallback.
A bridge without the typed body field fails explicitly. PICO's own body-tracker
calibration is still required; the bridge's SMPL-X T-pose calibration is not.

Both consumers use `pico.smplx.v1.PoseStream/StreamPoses` from pinned
`deps/pico-smplx-bridge/pose_stream.proto`. Bello imports only generated
protobufs, not the bridge's SDK or server. Existing SMPL-X fields remain for the
viewer and independent finger control. Hands still use `smplx.endpoint` and
`smplx.neutral_model_path`, require calibrated valid finger rotations, and do not
require valid body tracking. Licensed model files remain external.

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
`tls_server_name_override` remain under `xrobot` (under `smplx` for hands).
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

All six hand axes use proportional targets between their calibrated open and
closed positions: a half-flexed finger requests the midpoint. The
`smoothing_time_constant_ms` low-pass filter and `max_raw_rate_per_second` slew
limit apply to finger flexion and thumb abduction alike. After tracking loss,
control reseeds from measured hand positions and limits the first recovered
command to one control period of movement. `control_rate_hz` sets the command
rate (5 Hz by default).
