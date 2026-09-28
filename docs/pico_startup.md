# PICO tracking and hand-control startup

For the installed Jetson/tracking-PC system, use the
[systemd workflow](systemd.md). Do not run the foreground daemon
examples below alongside their managed services. The remaining examples explain
individual commands and alternative deployments.

The tracking host (Jetson or a separate PC) runs XRoboToolkit and `pico-smplx-bridge`. Bello's reference
producer and hand controller subscribe independently. The Rust policy runner
consumes the reference producer and hardwared telemetry.

| Process | Endpoint in this same-host example | Purpose |
| --- | --- | --- |
| PICO bridge | `127.0.0.1:50052` | Native XRoboToolkit body and hand poses |
| `bello-run-reference` | `127.0.0.1:50056` | GMR joint references |
| `bello-hardwared` | `127.0.0.1:50051` | Robot telemetry and motor control |
| Bello visualizer | `127.0.0.1:50054` | Remote controller intent |

The receiver listens on port 63901; its SDK uses local port 60061. The bridge
uses 50052 so hardwared can retain 50051 on the same host. No viewer is needed.

## Tracking host

Install the receiver and pinned bridge with the [systemd installer](systemd.md#tracking-receiver-and-bridge):

```bash
systemctl --user start bello-tracking.target
systemctl --user is-active xrobotoolkit.service pico-smplx.service
```

In the headset, connect XRoboToolkit **PC Service** to `10.42.0.1` when using
Bello-Jetson, or to the tracking host's LAN address. Enable **Head**, **Hand**,
**Full Body**, and **Send**. Keep PICO-calibrated body trackers awake and bare
hands visible. The bridge does not require a separate SMPL-X calibration.

Inspect producer logs with:

```bash
journalctl --user -u pico-smplx.service -n 50 --no-pager
```

The bridge loads the external PICO SDK. Bello consumers use typed gRPC messages
from its [pinned protocol](../deps/pico-smplx-bridge/pose_stream.proto).

## Body references

In each Bello Python terminal:

```bash
cd ~/bello_mujoco
nix develop
```

Start the body reference producer in one terminal and leave it running:

```bash
uv run bello-run-reference asset=bello_mini \
  xrobot.endpoint=127.0.0.1:50052
```

Update and restart the tracking bridge before using this command: it must
publish typed `xrobot_body` data. Body following no longer accepts `smplx.*`
arguments. Hand control also uses native tracking, with no SMPL-X model.
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
  xrobot.endpoint=127.0.0.1:50052 \
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
  xrobot.endpoint=127.0.0.1:50052 \
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

Use a machine-local hardware profile with port 50051. If copying it into
`outputs/`, resolve its `description` path relative to the original profile.
From the Bello checkout, start that profile with:

```bash
nix develop .#rust-runtime --command cargo run --locked \
  --manifest-path deps/bello-stack/Cargo.toml -p bello-hardwared -- \
  --config /absolute/path/to/machine-hardwared.yaml
```

Start the [Bello visualizer](../deps/bello-stack/components/bello-visualizer/README.md)
with its machine-local profile and a connected gamepad. For the same-host
example, pass `--hardware-endpoint http://127.0.0.1:50051`; its remote service
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
endpoint: http://127.0.0.1:50051
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

The systemd workflow also manages the reference producer, hands,
hardwared, and estimator with systemd. Only the policy runner and Bello
visualizer remain manual; see the [daemon guide](systemd.md).
