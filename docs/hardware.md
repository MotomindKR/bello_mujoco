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

Start hardwared and the remote-command service first. Preflight checks source,
inventory, coordinate coverage, nominal targets/gains, IMU frames, group state,
and telemetry freshness without acquiring control or writing motor targets.
Use [Lima/vCAN](macos_vcan.md) for commissioning, not physical CAN.

For local settings, copy the hardware YAML to ignored `outputs/` and point
`--hardware-config` at that file. Keep the tracked checkout clean. In Lima,
set `control.remote.endpoint` to `http://host.lima.internal:50054` for the native
Mac visualizer. The reference endpoint, limits, and logging controls live in
the [Rust configuration](../rust/bello-policy/config/hardware.yaml).

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

The live path consists of three processes:

```text
PICO -> pico-smplx-bridge :50051 -> bello-run-reference (GMR) :50056 -> Rust policy runner
                              -> bello-run-hands -> Linker CAN
```

`bello-run-reference` reconstructs a neutral SMPL-X skeleton from streamed pose
parameters, runs GMR, and publishes named URDF joint references. It never opens
motor IO. Rust owns arm following, target speed/limit checks, and the transition
to nominal when references are unavailable. Malformed messages or terminal
producer errors stop control. The separate FastSAM adapter remains available.

Both Bello consumers use the `pico.smplx.v1.PoseStream/StreamPoses` contract in
the pinned `deps/pico-smplx-bridge/pose_stream.proto`. Any server implementing
that contract can supply poses. The submodule supplies the schema, and
`scripts/protobuf.sh` generates Bello's bindings with its own locked toolchain.
Bello does not install the bridge with pip or import its PICO SDK/server code.
Run the bridge separately on the tracking PC as described in its README.

Set `smplx.endpoint` and `smplx.neutral_model_path` in either command. The licensed
`SMPLX_NEUTRAL.npz` model is external and is not included in either repository.
FK uses the model's zero-shape skeleton: root orientation, 21 body rotations,
and 15 rotations per hand, all axis-angle radians. Hands follow SMPL-X's
index/middle/pinky/ring/thumb order. Jaw and eyes are neutral; shape and facial
expression are not streamed. Root translation is calibrated pelvis displacement
in metres, offset by the neutral model's standing pelvis height. The input is
right-handed, Y-up; GMR receives the existing Bello Z-up basis. Fingertips use
neutral mesh landmarks attached rigidly to their distal bones, not full mesh
skinning.

Body following requires calibrated, valid body tracking. Each enabled hand is
accepted independently when calibrated and all 15 joint-valid flags are set;
body tracking is not required for hand control. Invalid components are never
interpreted as a valid zero pose. Latest-only delivery does not replay a backlog
or refresh duplicate observations. Device/calibration changes, sequence resets,
source clock discontinuities, and reconnects reset downstream tracking state.

Freshness uses the producer's packet age plus `smplx.transport_latency_budget_ms`
and elapsed local monotonic time. The budget is an allowance, not measured
one-way network latency; source timestamps belong to the headset clock.
`smplx.max_source_age_ms` rejects stale input, `smplx.max_source_interval_seconds`
detects gaps, and `smplx.rpc_timeout_seconds` reconnects silent subscriptions.
`smplx.connect_timeout_seconds` bounds the wait for the first message after
each connection attempt.
`smplx.max_rate_hz` requests a bounded stream rate. Optional
`smplx.tls_root_certificate_path` and `smplx.tls_server_name_override` configure
TLS; the bridge's default local deployment uses plaintext gRPC.

To exercise both RPC boundaries with the real model and GMR, without hardware:

```bash
BELLO_SMPLX_MODEL=/path/to/SMPLX_NEUTRAL.npz \
  uv run pytest -q -m integration tests/test_smplx_reference_integration.py
```

This uses a synthetic pose producer and a local reference service. It validates
numerical retargeting and named joint delivery, not physical robot behavior.

`bello-run-hands` independently controls Linker hands over classic CAN, not
hardwared motor targets. It defaults to read-only validation; see
[hands.yaml](../configs/commands/hands.yaml). Hand arming is separate from policy
arming and requires its own bus, device, calibration, and safety review.
