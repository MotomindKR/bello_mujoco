# PICO tracking and hand-control startup

The tracking PC runs XRoboToolkit and `pico-smplx-bridge`. Bello's reference
producer and hand controller subscribe independently. The Rust policy runner
consumes the reference producer and hardwared telemetry.

| Process | Endpoint in this same-host example | Purpose |
| --- | --- | --- |
| PICO bridge | `127.0.0.1:50051` | Raw XRoboToolkit body and SMPL-X hand poses |
| `bello-run-reference` | `127.0.0.1:50056` | GMR joint references |
| `bello-hardwared` | `127.0.0.1:50055` | Robot telemetry and motor control |
| Bello visualizer | `127.0.0.1:50054` | Remote controller intent |
| PICO viewer | `http://192.168.0.216:8080/` | Human SMPL-X preview |

Hardwared also defaults to port 50051. Use 50055 in its machine-local profile
when it shares the PICO host, and point its clients at 50055. On a separate
robot host it can keep 50051; clients must use that host's address. The PICO
viewer and Bello visualizer serve different purposes.

## Tracking PC

The existing `gmmyung-ubuntu` installation has these user services. These names
describe that deployment; cloning Bello does not install them:

```bash
ssh gmmyung@gmmyung-ubuntu
systemctl --user start xrobotoolkit.service pico-smplx.service pico-smplx-viewer.service
systemctl --user is-active xrobotoolkit.service pico-smplx.service pico-smplx-viewer.service
```

In the headset, open XRoboToolkit, connect **PC Service** to `192.168.0.216`,
then enable **Head**, **Hand**, **Full Body**, and **Send**. Keep the calibrated
body trackers awake. Use the index-finger trigger to activate UI buttons. Open
the PICO viewer URL above to inspect tracking before connecting robot control.

Inspect producer logs with:

```bash
journalctl --user -u pico-smplx.service -n 50 --no-pager
```

For a different tracking PC, install the bridge using its
[README](../deps/pico-smplx-bridge/README.md) and substitute its endpoints below.
The bridge loads the external PICO SDK; Bello uses only its gRPC schema.

## Body references

In each Bello Python terminal on Ubuntu:

```bash
cd /home/gmmyung/bello_mujoco-smplx-grpc
nix develop
export SMPLX_MODEL=/home/gmmyung/GMR/assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

The licensed model stays outside Git. Start the body reference producer in one
terminal and leave it running:

```bash
uv run bello-run-reference asset=bello_mini \
  xrobot.endpoint=127.0.0.1:50051
```

Update and restart the tracking bridge before using this command: it must
publish typed `xrobot_body` data. Body following no longer accepts `smplx.*`
arguments and does not need `SMPLX_MODEL`; keep that model for the hand command.
On the Jetson, run Python commands with `env -u LD_LIBRARY_PATH uv run ...`
if the Nix library path conflicts with the system glibc. Do not apply this
workaround to the Rust runtime shell.

This is the GMR service, listening on port 50056. It opens no motor IO. Set
`retarget.human_height_m` to the operator's height if it differs from the 1.66 m
default. No additional GMR server is needed.

## Linker hands

In a second Bello Python terminal, first validate tracking:

```bash
uv run bello-run-hands \
  smplx.endpoint=127.0.0.1:50051 \
  smplx.neutral_model_path="$SMPLX_MODEL" \
  duration_seconds=10
```

The default `arm=false` opens no CAN device and exits when an acceptable hand
frame arrives. It fails if no acceptable frame arrives within ten seconds.

For hand motion, confirm the actual Linker model, an already configured classic
CAN interface, and both hands' calibrated open/closed endpoints in
[hands.yaml](../configs/commands/hands.yaml). Pass local overrides on the CLI.
For a confirmed O6 setup on `can1`, the command is:

```bash
uv run bello-run-hands \
  smplx.endpoint=127.0.0.1:50051 \
  smplx.neutral_model_path="$SMPLX_MODEL" \
  protocol=linker_o6 can_interface=can1 \
  arm=true calibration_confirmed=true
```

Use `protocol=linker_l6` for L6 hardware. `calibration_confirmed=true` asserts
that the supplied endpoint values have actually been checked. The hand process
does not configure the CAN bus. It is independent of hardwared and policy
arming; stop it separately with Ctrl-C.

All six axes now follow partial bends proportionally. Defaults are 5 Hz control,
100 ms smoothing, and a 510 raw-units/second slew limit. Tracking loss suspends
position writes; recovery reads physical positions before resuming bounded
movement. This does not imply that stopping opens the hand.

## Hardware and policy runner

Start the configured emulator or robot hardware stack first. Use the
[emulator guide](emulator.md) and [Lima setup](macos_vcan.md) for simulation.
Physical buses, serial devices, and calibration require a machine-specific
hardwared profile; the checked-in Mini profile is provisional.

If hardwared shares the tracking PC, set its local profile to:

```yaml
grpc:
  listen:
    type: tcp
    address: 127.0.0.1:50055
```

Preserve the rest of the profile. If copying it into `outputs/`, resolve its
`description` path relative to the original profile, then use the absolute path
in the copy. From the Bello checkout, start that profile with:

```bash
nix develop .#rust-runtime --command cargo run --locked \
  --manifest-path deps/bello-stack/Cargo.toml -p bello-hardwared -- \
  --config /absolute/path/to/machine-hardwared.yaml
```

Start the [Bello visualizer](../deps/bello-stack/components/bello-visualizer/README.md)
with its machine-local profile and a connected gamepad. For the same-host
example, pass `--hardware-endpoint http://127.0.0.1:50055`; its remote service
listens on 50054. The Mini example also expects the state estimator on 50053.
Keep the visualizer's **Commanding** toggle off while the policy owns motors.
It may start after the policy runner: absent/stale gamepad commands request
standing, and connection/reconnection requires neutral input before movement.
The policy's HOLD/RUN confirmations are still required.

Copy the policy settings into ignored output and edit the copy:

```bash
mkdir -p outputs/pico
cp rust/bello-policy/config/hardware.yaml outputs/pico/policy-hardware.yaml
```

Set these fields, retaining the other defaults:

```yaml
endpoint: http://127.0.0.1:50055
control:
  following:
    enabled: true
    endpoint: http://127.0.0.1:50056
  remote:
    endpoint: http://127.0.0.1:50054
```

Run read-only preflight with a policy bundle and its **exact clean source
checkout**:

```bash
nix develop .#rust-runtime --command cargo run --locked -p bello-policy -- \
  --policy /absolute/path/to/policy.zip \
  --source /absolute/path/to/matching-source-checkout \
  --hardware-config outputs/pico/policy-hardware.yaml
```

An older bundle does not become compatible by pointing `--source` at new main.
Use its matching checkout or export a bundle from a checkpoint trained at the
new source revision. See [export and deployment](hardware.md).

For a commissioned control run, add `--arm`, then use the interactive `HOLD`
and `RUN` sequence. After stopping, `IDLE` explicitly disables motors. Live
following applies to checkpoint-held upper-body coordinates; motion-bundle
mode requires following to be disabled. Physical policy commissioning remains
pending as described in the deployment guide.

Only the PICO processes above are installed as Ubuntu services. The Bello
reference producer, hands, hardwared, visualizer, and policy runner are separate
foreground processes in this workflow.
