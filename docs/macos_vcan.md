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

The first `start` downloads the Ubuntu image and installs Nix in the guest;
`bootstrap` downloads and builds the Rust/CAN and Python policy environment, so
its first run can take several minutes. `doctor` must report both vCAN links,
both IMU pseudo-terminals, and the JAX CPU backend before continuing.

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

Lima forwards guest services only to Mac loopback. Port 50054 is deliberately
not forwarded from guest to host because it belongs to the native visualizer.

| Service | Actual listener | Client route |
| --- | --- | --- |
| Hardware daemon | Lima guest, `0.0.0.0:50051` | Mac visualizer: `http://127.0.0.1:50051` |
| MuJoCo ground-truth pose | Lima guest, `0.0.0.0:50052` | Mac: `http://127.0.0.1:50052` |
| State estimator | Lima guest, `0.0.0.0:50053` | Mac visualizer: `http://127.0.0.1:50053` |
| Visualizer remote commands | Mac, `127.0.0.1:50054` | Guest policy: `http://host.lima.internal:50054` |
| Reserved service | Lima guest port 50055, when used | Mac: `http://127.0.0.1:50055` |

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

Do not use `allow_git_commit_mismatch=true` as part of the normal Lima workflow.
Place a local checkpoint under the ignored `checkpoints/` directory so it is
both visible in the guest and does not dirty the checkout. The commands below
use `checkpoints/bello_mini_locomotion` as an example; replace it with the
checkpoint's actual path.

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
  -iTCP:50051 -iTCP:50052 -iTCP:50053 -iTCP:50054 \
  -sTCP:LISTEN
```

### Terminal 1: MuJoCo emulator

```bash
./scripts/lima_vcan.sh run \
  uv run bello-run-emulator asset=bello_mini
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
when using Terminal 4A or when collecting estimator output:

```bash
./scripts/lima_vcan.sh run bash -c \
  'cd deps/bello-stack && cargo run --locked -p bello-state-estimatord -- \
    --config components/bello-state-estimatord/config/bello-mini-vcan.yaml'
```

MuJoCo ground-truth pose remains available independently on port 50052.

### Terminal 4A: native macOS visualizer

Use this option for visualization, gamepad commands, or walking tests. On the
Mac, create the ignored local visualizer configuration once:

```bash
cp deps/bello-stack/components/bello-visualizer/config/bello-mini.example.yaml \
  deps/bello-stack/components/bello-visualizer/config/bello-mini.yaml
```

In `bello-mini.yaml`, change only the remote listener from
`0.0.0.0:50054` to `127.0.0.1:50054`. Its hardware and external-pose endpoints
already use Mac loopback ports 50051 and 50053. Then start the native binary:

```bash
nix develop .#macos-visualizer
cd deps/bello-stack
cargo run --locked -p bello-visualizer -- \
  --config components/bello-visualizer/config/bello-mini.yaml
```

The dedicated shell supplies Rust 1.95 and `protoc`, and keeps Cargo downloads
and build output below the Mac's cache directory instead of the shared source
tree. Keep the visualizer's direct group control disabled while the policy owns
the lower-body control lease; use its monitoring and gamepad remote-command
functions during a policy run.

### Terminal 4B: zero-only guest remote command

Locomotion requires a fresh `RemoteCommandService` sample even when upper-body
tracking is disabled. The internal headless service verifies that `vcan0` and
`vcan1` are real vCAN links, binds only to `127.0.0.1:50054`, and continuously
publishes armed zero velocity intent:

```bash
./scripts/lima_vcan.sh run \
  uv run python -m bello_mujoco.hardware.headless_remote
```

This service cannot request forward, lateral, or yaw motion. It is suitable for
read-only preflight and zero-command standing tests only. To command walking,
use Terminal 4A instead. Do not run the native visualizer and the headless
remote service at the same time: they are alternative command authorities, and
a mistaken endpoint can select the wrong one.

### Terminal 5: read-only policy preflight

Run preflight before every armed simulated run:

```bash
./scripts/lima_vcan.sh run uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_mini_locomotion \
  endpoint=http://127.0.0.1:50051 \
  remote_endpoint=http://host.lima.internal:50054 \
  upper_body_tracking_enabled=false
```

That remote endpoint selects the native Mac visualizer. When intentionally
using the zero-only guest service from Terminal 4B, replace it with
`remote_endpoint=http://127.0.0.1:50054`.

The default `arm=false` validates the checkpoint, policy, services, motor
inventory, IMU frame, and a complete state sample. It acquires no control lease
and sends no motor command. Do not continue unless its result contains
`'hardware_preflight': 'passed'` and reports that no lease or command was
issued.

## Run an armed simulation

Keep Terminals 1, 2, and exactly one Terminal 4 option running. Start a bounded
simulated run in Terminal 5:

```bash
./scripts/lima_vcan.sh run uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_mini_locomotion \
  endpoint=http://127.0.0.1:50051 \
  remote_endpoint=http://host.lima.internal:50054 \
  upper_body_tracking_enabled=false \
  duration_seconds=30 \
  arm=true
```

When using Terminal 4B instead, replace the armed command's remote endpoint
with `remote_endpoint=http://127.0.0.1:50054`, just as in preflight.

Follow the interactive sequence exactly:

1. With the emulator's torso guide still engaged, type the exact word `HOLD` in
   Terminal 5. The runner acquires control and performs its nominal-pose ramp.
2. When Terminal 5 displays the `RUN` prompt, switch to Terminal 1 and press
   Enter once to release the torso guide.
3. Let the simulated robot settle, return to Terminal 5, and type the exact word
   `RUN` to start the policy.
4. Allow the bounded run to finish, or press Ctrl-C in Terminal 5 to enter the
   normal stop sequence. Follow any resulting `IDLE` confirmation prompt.

`HOLD` and `RUN` belong in the policy-runner terminal. The bare Enter used to
release or reset the torso belongs only in the emulator terminal. When using
Terminal 4B, the headless remote keeps the commanded velocity at zero throughout
this sequence, so it tests standing and the complete vCAN policy path but not
commanded locomotion. When using Terminal 4A, keep the gamepad neutral through
the hold-and-release sequence and command motion only after the run starts.

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
