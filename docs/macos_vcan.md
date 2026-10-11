# macOS Lima vCAN environment

macOS does not provide Linux SocketCAN. This workflow runs the Bello Mini vCAN
emulator, hardware services, and policy in an Ubuntu Lima guest while mounting
the repository from the Mac. The graphical visualizer can run natively on
macOS. Nix supplies both toolchains, and the helper creates `vcan0`, `vcan1`,
`/dev/ttyIMU`, and `/dev/ttyIMU-mujoco` in the guest.

This environment is for emulated policy commissioning only. Do not substitute a
physical CAN configuration for the Bello Mini vCAN configurations shown below.
The supplied VZ/virtiofs profile targets Apple Silicon and requires macOS 13 or
newer; Intel Macs are not supported by this profile.

## Create the guest

Install Lima on the Mac:

```bash
brew install lima
```

Use a standalone clone with initialized recursive submodules. Linked Git
worktrees are intentionally rejected because their Git metadata lives outside
the mounted repository directory.

```bash
git clone --recurse-submodules \
  https://github.com/MotomindKR/bello_mujoco.git
cd bello_mujoco
git submodule sync --recursive
git submodule update --init --recursive
```

Create the Ubuntu guest, build the Linux Nix environment, and verify it:

```bash
./scripts/lima_vcan.sh start
./scripts/lima_vcan.sh bootstrap
./scripts/lima_vcan.sh doctor
```

The first `start` downloads the Ubuntu image and installs Nix in the guest.
`bootstrap` verifies or repairs that guest-local Nix installation, then downloads
and builds the Rust/CAN and Python tooling environment, so its first run can take
several minutes. A transient installer download failure does not require deleting
the VM: rerun `bootstrap`, which downloads the complete installer with retries
before executing it. `doctor` must report both vCAN links, both IMU
pseudo-terminals, and the JAX CPU backend before continuing.

The networking contract is versioned in the Lima template. If `start` reports
that an existing instance uses obsolete networking, recreate only the VM and
then rebuild its guest-local caches:

```bash
./scripts/lima_vcan.sh delete
./scripts/lima_vcan.sh start
./scripts/lima_vcan.sh bootstrap
```

The repository on the Mac is not deleted.

The repository is mounted at `/workspace/bello_mujoco`. Every process that uses
SocketCAN must run inside this guest: macOS cannot see the guest's `vcan0` or
`vcan1`. Keep the emulator, hardware daemon, policy runner, and optional state
estimator in the guest. `./scripts/lima_vcan.sh run ...` enters the guest Nix
shell and preserves an interactive terminal. The visualizer is the exception:
it can run as a native macOS application because it communicates over gRPC and
does not use SocketCAN directly.

Lima records the clone's absolute host path when the instance is created. If
the clone is moved or renamed, delete and recreate the instance so the mount
does not continue pointing at the old path.

## Network topology

Lima forwards guest services only to Mac loopback. The console connects directly
to guest hardwared, estimator and runner services. Port 50054 is unused.

| Service | Actual listener | Client route |
| --- | --- | --- |
| Hardware daemon | Lima guest, `0.0.0.0:50051` | Mac visualizer: `http://127.0.0.1:50051` |
| MuJoCo ground-truth pose | Lima guest, `0.0.0.0:50052` | Mac: `http://127.0.0.1:50052` |
| State estimator | Lima guest, `0.0.0.0:50053` | Mac visualizer: `http://127.0.0.1:50053` |
| Runner operator stream | Lima guest, `0.0.0.0:50055` | Mac console: `127.0.0.1:50055` |

`host.lima.internal` is Lima's stable guest-side name for the Mac. The guest
Nix shell adds it to `NO_PROXY` so a host proxy cannot intercept local gRPC
traffic. A Mac process bound to `127.0.0.1` remains reachable through this Lima
host gateway while staying unavailable on the Mac's LAN interfaces. SSH is
managed separately by Lima; use `./scripts/lima_vcan.sh shell` rather than
opening or forwarding an SSH port manually.

## Checkpoint and checkout requirements

Hardware-policy loading normally requires the clean, exact Git revision
recorded by the checkpoint. Before starting the processes, check both the root
checkout and every recursive submodule:

```bash
git status --short
git submodule status --recursive
```

There is no Git-commit mismatch override.
Export a deployment bundle from the checkpoint's exact source revision:

```bash
uv run bello-export-policy checkpoint_path=checkpoints/RUN output_path=outputs/policy.zip
```

The ignored `outputs/` directory is visible in the guest without dirtying the
checkout. The native runner consumes the bundle, not a training checkpoint
directory. Physical deployment remains gated on [trained-policy commissioning](hardware.md).

The `lima-vcan` Nix shell keeps platform-specific state off the shared source
tree. Its Python environment and interpreter live below
`~/.local/state/bello-mujoco/lima-vcan` in the guest. Warp, Cargo, and Python
caches live below `~/.cache/bello-mujoco/lima-vcan`, and uv downloads use the
guest's normal `~/.cache/uv`. These paths disappear if the Lima instance is
deleted.

## Run the Bello Mini stack

Open a separate macOS Terminal window for each long-running command. Run the
commands from the repository root on the Mac; the helper starts each process in
`/workspace/bello_mujoco` inside the guest.

Before starting any process, check that no unrelated Mac process owns the four
active TCP ports. This command should print nothing:

```bash
lsof -nP \
  -iTCP:50051 -iTCP:50052 -iTCP:50053 -iTCP:50055 \
  -sTCP:LISTEN
```

### Terminal 1: MuJoCo emulator

```bash
./scripts/lima_vcan.sh run \
  nix develop .#rust-runtime --command cargo run --locked \
  --manifest-path deps/bello-stack/Cargo.toml -p bello-mujoco -- \
  run --config deps/bello-stack/components/bello-mujoco/config/emulator_mini.yaml
```

The torso guide starts engaged. Leave this terminal open and do not press Enter
yet. Enter releases the guide; pressing Enter again resets the robot and
re-engages it.

### Terminal 2: hardware daemon

Use the vCAN configuration, not a physical hardware configuration:

```bash
./scripts/lima_vcan.sh run bash -c \
  'cd deps/bello-stack && cargo run --locked -p bello-hardwared -- \
    --config components/bello-hardwared/config/bello-mini-vcan.yaml'
```

This makes the hardware service reachable on guest loopback port 50051.

### Terminal 3: state estimator

The locomotion policy does not require the state estimator, but the native
visualizer configuration below uses its pelvis pose on port 50053. Start it
when using Terminal 4 or when collecting estimator output:

```bash
./scripts/lima_vcan.sh run bash -c \
  'cd deps/bello-stack && cargo run --locked -p bello-state-estimatord -- \
    --config components/bello-state-estimatord/config/bello-mini.yaml'
```

The estimator uses the same robot profile for physical hardware and emulation;
only hardwared selects the CAN and serial transports. MuJoCo ground-truth pose
remains available independently on port 50052.

### Terminal 4: policy console

On the Mac, connect a mapped gamepad and start the Electron console:

```bash
cd console
npm ci
npm start
```

The default endpoints use Mac loopback ports 50051, 50053 and 50055. See the
[console guide](policy_console.md) for configuration and packaged builds.
The Kiss3d visualizer is optional diagnostic tooling. Keep its direct group
control disabled during policy runs; it no longer supplies gamepad commands.

### Terminal 5: read-only policy preflight

Copy the native hardware profile to ignored local configuration:

```bash
mkdir -p outputs
cp rust/bello-policy/config/hardware.yaml outputs/policy-hardware.yaml
```

In that YAML, keep `control.operator_listen: 0.0.0.0:50055` and
`control.following.enabled: false` unless the reference producer is running.
Run preflight before every armed simulated run:

```bash
./scripts/lima_vcan.sh run nix develop .#rust-runtime --command \
  cargo run --locked -p bello-policy -- \
  --policy outputs/policy.zip --source . \
  --hardware-config outputs/policy-hardware.yaml
```

Without `--arm`, the runner validates the bundle, policy, services, motor
inventory, IMU frame, and a complete state sample. It acquires no control lease
and sends no motor command. Do not continue unless its result contains
`"hardware_preflight": "passed"` and `"armed": false`.

## Run an armed simulation

Keep Terminals 1, 2, and Terminal 4 running. Start a bounded
simulated run in Terminal 5:

```bash
./scripts/lima_vcan.sh run nix develop .#rust-runtime --command \
  cargo run --locked -p bello-policy -- \
  --policy outputs/policy.zip --source . \
  --hardware-config outputs/policy-hardware.yaml --arm
```

Use the same local hardware YAML and command authority validated during preflight.

Use the controller sequence:

1. With the torso guide engaged, release all controls, then hold Back+Start for
   two seconds. The runner acquires control and ramps to Hold.
2. Wait for phase `holding`, then press Enter in the emulator terminal to release
   its guide. Let the simulated robot settle.
3. Hold A for one second to enter Stand. D-pad left/right selects Locomotion or a named
   motion; another one-second A press confirms it.
4. Use L2 + R2 to abort. There is no run-duration limit. The runner damps and
   releases leases; Ctrl-C exits the final status display.

The emulator's Enter key only controls its torso guide. Test two named motions,
short/repeated presses, rejected transitions, abort and reconnect neutrality.
Disconnect each console display stream independently: it must not create a
control command. Loss of the operator stream preserves Hold, returns Locomotion
to Stand and allows a healthy named motion to finish. Feedback/lease faults remain authoritative. See [console behavior](policy_console.md).

## Inspect and remove the guest

After stopping every long-running process with Ctrl-C, inspect or stop the VM:

```bash
./scripts/lima_vcan.sh status
./scripts/lima_vcan.sh stop
```

`stop` removes the IMU pseudo-terminals and powers down the VM. A later `start`
reuses the VM disk and guest caches. Permanently delete the instance only when
those guest-local caches are no longer needed:

```bash
./scripts/lima_vcan.sh delete
```

Deletion removes the Lima VM disk but does not remove the host repository. Set
`BELLO_LIMA_INSTANCE` on every helper invocation to use a non-default instance
name.
