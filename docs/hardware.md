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

`bello-run-reference` runs RTMW/GMR retargeting separately and publishes named
joint references. It requires the live producer and a licensed SMPL-X model.
Rust owns arm following, target speed/limit checks, and the transition to nominal
when references are unavailable. Malformed messages or terminal producer errors
stop control. FastSAM/RTMW integration libraries and their live shape/frame,
confidence, and freshness checks remain available.

`bello-run-hands` independently controls Linker hands over classic CAN, not
hardwared motor targets. It defaults to read-only validation; see
[hands.yaml](../configs/commands/hands.yaml). Hand arming is separate from policy
arming and requires its own bus, device, calibration, and safety review.
