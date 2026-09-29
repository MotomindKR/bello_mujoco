# Robot and tracking daemon workflow

Use the same operator account for installation and `systemctl --user` commands.
These targets **start automatically at boot**, without login, using user
lingering. They survive SSH logout. The policy runner starts Disarmed and waits
for the console's gamepad Hold chord. The **Bello visualizer stays manual**.
No PICO viewer is installed. Both targets can run on the Jetson;
a separate tracking PC remains an explicit deployment option.

| Host | Target | Services |
| --- | --- | --- |
| Jetson (or a separate tracking PC) | `bello-tracking.target` | `xrobotoolkit`, `pico-smplx` |
| `nvidia@192.168.0.229` | `bello-robot.target` | `bello-hardwared`, `bello-state-estimatord`, `bello-reference`, **`bello-hands`**, `bello-policy` |

Starting the robot target starts **armed hand control**, independently of the
body policy, including after reboot. Keep hands clear of people and objects
before powering on or rebooting. Hardwared starts body
groups IDLE; the policy service acquires no motor lease until Back+Start is held
for two seconds. Stopping hands suspends CAN
writes; it does **not** open or release the fingers.

## Daily checks and manual start

The targets start on boot. The `start` commands below also work after an
operator stops a target; they do not restart services already running.

On the tracking host:

```bash
systemctl --user start bello-tracking.target
systemctl --user is-active xrobotoolkit pico-smplx
```

In the headset, open XRoboToolkit and connect **PC Service** to the tracking
host. When connected to Bello-Jetson, use `10.42.0.1`; its router-side address is
`192.168.0.229`. Enable **Head**, **Hand**, **Full Body**, and **Send**. Keep the
calibrated body trackers awake. Inspect bridge status and run unarmed hand
preflight before robot control; see [PICO startup](pico_startup.md).

On the Jetson, check physical buses/devices, clear the hands, then start:

```bash
systemctl status onboard-can.service --no-pager
ip -details link show can0
ip -details link show can1
systemctl --user start bello-robot.target
systemctl --user is-active bello-hardwared bello-state-estimatord bello-reference bello-hands bello-policy
systemctl --user --failed
```

**Check all five robot services.** An active target only
means startup was requested, not that every daemon is healthy or tracking is
valid. Check each service and fresh telemetry before entering Hold. The
root-owned `onboard-can.service` calls the pinned bello-stack
`scripts/setup-can.sh` at boot. Hardwared and hands use its read-only `--check`
before starting; user services never reconfigure buses or require root.

| Endpoint | Owner |
| --- | --- |
| Tracking host `:63901` | XRoboToolkit PC Service |
| Tracking host `0.0.0.0:50052` | Typed native body and hand bridge |
| Jetson `:50051` | Hardwared |
| Jetson `:50053` | State estimator |
| Jetson `127.0.0.1:50056` | GMR joint references (`bello-reference`) |
| Jetson `:50055` | Runner operator stream |

Installed units and their referenced configs are authoritative. Keep these
unauthenticated services on the trusted robot LAN. Body GMR and hand control do not need a
licensed SMPL-X model or bridge reference-pose calibration. `bello-reference` is the
GMR server, not an additional process to launch.

## Console and policy service

On the Ubuntu tracking PC, open the packaged Bello Mini console with a gamepad.
Configure its hardware, estimator and runner endpoints to the Jetson ports
50051, 50053 and 50055. See [console setup](policy_console.md).

The optional `bello-run-visualizer` launcher remains a Kiss3d diagnostic tool.
Keep **Commanding OFF** while the policy owns motors. Its configuration is
`~/.config/bello/ubuntu-visualizer.yaml`; it hosts no gamepad service.

The installer generates `bello-policy.service` with the selected locomotion
bundle, its exact clean source checkout, and the runner hardware/control YAML.
It runs the current `bello-policy` binary with `--arm`, which enables the operator
server on port 50055; it does not acquire control until the gamepad Hold chord.
Named motions, buffered SONIC Tracking, upper-body following and assisted
Front-down startup remain configured in that YAML. Use absolute bundle/source
paths in its `motions` and `tracking` entries.

Support the robot before entering Hold. Use Back+Start for two seconds to enter
Hold, then A for one second to enter Stand. Select modes with D-pad left/right
and confirm with A. Hold L1 + R1 for one second to abort and damp. Reconnection
requires neutral controller input. See [console controls](policy_console.md).

The service starts after the four supporting daemons and stops before them.
Ordering does not imply telemetry readiness: the runner validates feedback at
startup. Startup/process failures retry with systemd's rate limit; each new
process starts Disarmed and requires a new neutral input and Hold chord.
Runtime faults remain visible in the console for explicit fault clearing.
Reference or console connection loss does not trigger a systemd restart.

Each service invocation creates a unique run directory inside systemd's
`LogsDirectory=bello-policy`. Set `control.policy_log_path: policy.jsonl` for
separate, non-overwriting logs. Other paths in the runner YAML must be absolute.
The journal prints the run directory. No builds, policy downloads or dependency
installs happen on service start. Policy/source compatibility is still validated
by the runner; the installer never rewrites checkpoint metadata.

For status and an explicit stop:

```bash
systemctl --user status bello-policy --no-pager
systemctl --user stop bello-policy
```

The existing `~/.local/bin/bello-run-policy` manual launcher remains available
for the separately installed `deploy/jetson-policy.yaml` profile and its pinned
061440 / 2973427 bundle. It is not used by the service installer. Stop the policy
service before using this manual alternative; do not run two policy processes:

```bash
~/.local/bin/bello-run-policy         # Read-only preflight
~/.local/bin/bello-run-policy --arm   # Manual operator server
```

Manual runs still log under `~/bello_mujoco/outputs/policy_logs/` and check the
four supporting daemons. See [hardware deployment](hardware.md) for
commissioning constraints.

## Stop, logs, and recovery

First stop the policy service (or interrupt a manual runner) and wait for damping
and lease cleanup. Verify it has released body control. Then, with hands clear:

```bash
# Jetson
systemctl --user stop bello-policy
systemctl --user stop bello-robot.target \
  bello-policy bello-hardwared bello-state-estimatord bello-reference bello-hands
# Tracking host
systemctl --user stop bello-tracking.target xrobotoolkit pico-smplx
```

Stopping a target stops all members; listing them explicitly also waits for
their shutdown jobs to finish. This is not an emergency-stop system.
Do not stop/restart hardwared or the robot target under an armed policy.

```bash
systemctl --user status bello-hands --no-pager
journalctl _UID="$(id -u)" _SYSTEMD_USER_UNIT=bello-hands.service -n 80 --no-pager
journalctl _UID="$(id -u)" _SYSTEMD_USER_UNIT=bello-hardwared.service \
  _SYSTEMD_USER_UNIT=bello-state-estimatord.service \
  _SYSTEMD_USER_UNIT=bello-reference.service \
  _SYSTEMD_USER_UNIT=bello-policy.service -f
# Tracking host:
journalctl --user -u xrobotoolkit -u pico-smplx -f
```

The Jetson uses a volatile system journal, so `journalctl --user` can report no
files. The explicit user/unit filters above work with that journal; the operator
account already has journal access through `adm`. Jetson logs do not survive a
reboot; save needed diagnostics before rebooting.

The estimator stays alive without hardwared, suppresses stale poses, and
reconnects. GMR and hands tolerate missing tracking: body following returns
toward nominal in the policy; hands suspend writes, re-read physical positions,
then resume bounded movement on fresh tracking. Missing tracking is not proof
that a process is unhealthy.

Daemon crashes restart automatically, subject to a restart-rate limit,
**except hands**. A crashed hand process requires an operator to inspect the
error and clear the workspace before `systemctl --user restart bello-hands`.
If rate-limited, fix the cause, then `systemctl --user reset-failed UNIT` and
`systemctl --user start UNIT`. Starting an already-active target does not
necessarily restart a failed member; address that member explicitly.

## Installation and updates

The maintained guide and installer are this repository's `docs/systemd.md` and
`scripts/install_systemd.py`. Run the installer from the repository so its
`deploy/systemd/` templates are available.

The manual launchers are installed in `~/.local/bin/`. Their maintained sources
are `scripts/run_jetson_policy.sh` and `scripts/run_ubuntu_visualizer.sh`; the
machine profiles live in `deploy/jetson-policy.yaml` and
`deploy/ubuntu-visualizer.yaml`. Install the matching script and YAML on each
host; these are deployment files, not changes to the pinned policy checkout.
The policy service instead uses `scripts/run_policy_service.sh` from the
installer's `--checkout`; no manual launcher installation is needed for it.

### Jetson CAN setup

The installer's `robot` role also installs and boot-enables the root-owned
Jetson CAN integration from `deploy/systemd/`. Run it as the operator account;
it uses `sudo` for system files and the system manager:

- `bello-can-modules.conf`: loads `can`, `can_raw`, and `mttcan` at boot.
- `onboard-can.service.in`: renders `onboard-can.service`, which waits for both
  devices, verifies onboard controller identity, restores the existing AGX
  Orin pinmux, then runs bello-stack's
  canonical setup script. CAN bitrate/recovery/queue settings belong to that
  script, not a second local implementation.
- `user-can-ordering.conf`: orders the installing operator's user manager after
  CAN setup.
  CAN failure does not prevent login for repairs; hardwared/hand startup checks
  reject interfaces that fail the canonical read-only check. The estimator
  remains independent and can reconnect when hardwared becomes available.

The installer writes `/etc/modules-load.d/bello-can.conf`,
`/etc/systemd/system/onboard-can.service`, and
`/etc/systemd/system/user@<operator-uid>.service.d/50-bello-can.conf`. The CAN
service uses the selected `--checkout` for its mount dependency and setup
script. These deployment-owned system files are replaced on subsequent robot
installations. The script configures only `can0` and `can1`, at 1 Mbps classic
CAN, with 100 ms bus-off recovery and a 64-frame transmit queue.

The previous `/usr/local/sbin/setup-onboard-can` implementation is no longer
used. Its migration backup and the old unit are under `/var/backups/bello-can.*`.
The pinned bello-stack script remains the CAN configuration source.

Installing/reloading units does not change current CAN settings. They take
effect on the next safe boot. **Never restart `onboard-can` or run the setup
script without `--check` while policy or hands are active**: it takes both buses
down. The Jetson pinmux unit is hardware-specific, not for the tracking PC or
arbitrary CAN adapters. Read-only inspection is safe:

```bash
bash ~/bello_mujoco/deps/bello-stack/scripts/setup-can.sh --check
ip -details link show can0
ip -details link show can1
```

### Daemon units

The installer writes user units, verifies them, reloads systemd, and enables
the target at boot (not each member individually). For `robot`, it also verifies,
installs, and enables the CAN system service. It **does not start, stop,
or restart processes immediately**. Stop
unmanaged copies before starting the target: one owner per device/port.

### Tracking receiver and bridge

Initialize the pinned submodules and enter the Nix environment. On Ubuntu
22.04 or 24.04, the tracking role selects the native ARM64 or x86-64 XRoboToolkit
v1.0.0 distribution and verifies its pinned SHA256. ARM64 uses the headless
package; x86-64 uses only the headless service executable from its Ubuntu package.
It extracts the distribution under `~/.local/share/bello/tracking/` without
running vendor desktop installation scripts or requiring sudo. An isolated
uv-managed Python 3.12 environment installs the submodule's hashed `requirements.lock`.
Neither dependency installation nor builds happen when services start.

```bash
nix develop
python scripts/install_systemd.py tracking --checkout "$PWD" --dry-run
python scripts/install_systemd.py tracking --checkout "$PWD"
sudo loginctl enable-linger "$USER"
systemctl --user start bello-tracking.target
```

This generates `xrobotoolkit.service` and `pico-smplx.service`, including the
architecture-specific SDK path and receiver-before-bridge ordering. The viewer
is not installed. Missing tracking is allowed at startup; the bridge reports
it explicitly. `--dry-run` prints units, download URL, checksum and environment
paths without downloading, changing systemd or writing files. Repeated installs
reuse the verified distribution and matching dependency environment.

The default bridge listener is `0.0.0.0:50052`, exposing native tracking on all
IPv4 interfaces and leaving hardwared's `50051` free. Onboard GMR and hand
clients can use `127.0.0.1:50052`. For a separate tracking host, set the robot
installer's `--tracking-endpoint` to that host's reachable IP and port 50052.
Pass `--bridge-listen HOST_IP:50052`
to restrict the listener to one interface, or `--bridge-listen 127.0.0.1:50052`
for local clients only. Repeat `--bridge-listen` for additional addresses.
`--device-id` selects a particular
headset; `--tracking-dir` sets the installation directory; `--uv` selects the
setup-time uv executable. Unsupported distributions/architectures fail explicitly.

The installer refuses to overwrite unmanaged service units. To migrate the old
Ubuntu installation, stop the tracking target and its services, back up the
old units and lifecycle drop-ins, remove those old definitions, and rerun the
installer. Remove any old `pico-smplx-viewer.service` and its autostart links.
The new target contains only receiver and bridge. Old SMPL-X consumers must be
updated with the bridge: live frames now carry native body and hand poses,
without generating the old SMPL-X rotation fields.

### Robot daemons

On the Jetson, prepare Python dependencies with `nix develop` and build native
daemons once (not on each service start):

```bash
cd /home/nvidia/bello_mujoco
nix develop .#rust-runtime --command cargo build --locked --release \
  --manifest-path deps/bello-stack/Cargo.toml \
  -p bello-hardwared -p bello-state-estimatord
nix develop .#rust-runtime --command cargo build --locked --release -p bello-policy
python3 scripts/install_systemd.py robot \
  --checkout /home/nvidia/bello_mujoco \
  --policy /home/nvidia/bello_mujoco/outputs/locomotion.zip \
  --policy-source /home/nvidia/policy-source \
  --policy-config /home/nvidia/.config/bello/jetson-policy.yaml \
  --tracking-endpoint 127.0.0.1:50052 \
  --human-height 1.73 --confirm-hand-calibration
sudo loginctl enable-linger "$USER"
```

Set height to the operator's actual height. The calibration flag asserts that
O6 endpoints in `configs/commands/hands.yaml` were physically checked; this
deployment uses O6 on `can1`. Do not reuse it blindly for another hand model.
Supply the actual deployed bundle, its matching clean source checkout, and the
reviewed runner YAML; all three policy options are required. `--checkout` is the
current runtime checkout and can differ from the older `--policy-source`.
The `--runtime-bin` directory must contain all three Rust binaries.
Pass `--hardware-config` / `--estimator-config` for machine-local YAMLs,
`--runtime-bin` for a different binary directory, or `--dry-run` to inspect
without installing or invoking sudo; the preview includes the CAN system files.
Python services use the prepared venv without Nix's
`LD_LIBRARY_PATH` (avoiding the Jetson glibc conflict); Rust services enter the
pinned `rust-runtime` Nix shell. Startup does no build or dependency install.

Lingering starts the user manager at boot and keeps it alive after logout.
Both targets are enabled in its `default.target`; existing CAN device setup
remains enabled. Check with `systemctl --user is-enabled bello-robot.target`
and `systemctl --user is-enabled bello-tracking.target` on their service hosts.
To prevent startup on the next boot, disable the relevant target with
`systemctl --user disable TARGET`. That does not stop running services; use the
safe shutdown sequence above if needed. Re-running the installer re-enables
boot startup. Hands still require an explicit restart after a process crash.

To update code or units, complete policy shutdown and stop the robot target, update
the checkout/dependencies, rebuild, rerun the installer, and deliberately start
the target. `systemctl --user cat UNIT` shows its effective command. Never update
running Python sources or overwrite running native binaries. Restart tracking
only with the robot safe, since references may be lost and recovered.
