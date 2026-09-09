# macOS Lima vCAN environment

macOS does not provide Linux SocketCAN. This workflow runs the complete Bello
Mini vCAN stack in an Ubuntu Lima guest while mounting the repository from the
Mac. Nix supplies the project toolchain inside the guest, and the helper creates
`vcan0`, `vcan1`, `/dev/ttyIMU`, and `/dev/ttyIMU-mujoco`.

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

The repository is mounted at `/workspace/bello_mujoco`. Every process that uses
SocketCAN must run inside this guest: macOS cannot see the guest's `vcan0` or
`vcan1`. Keep the emulator, hardware daemon, remote-command service, policy
runner, and optional state estimator in the guest as well so their loopback
endpoints refer to the same machine. `./scripts/lima_vcan.sh run ...` enters the
guest Nix shell and preserves an interactive terminal.

Lima records the clone's absolute host path when the instance is created. If
the clone is moved or renamed, delete and recreate the instance so the mount
does not continue pointing at the old path.

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
  'cd deps/bello-stack && cargo run -p bello-hardwared -- \
    --config components/bello-hardwared/config/bello-mini-vcan.yaml'
```

This makes the hardware service reachable on guest loopback port 50051.

### Terminal 3: optional state estimator

The locomotion policy does not require the state estimator. Start it only when
collecting or comparing estimator output on port 50053:

```bash
./scripts/lima_vcan.sh run bash -c \
  'cd deps/bello-stack && cargo run -p bello-state-estimatord -- \
    --config components/bello-state-estimatord/config/bello-mini-vcan.yaml'
```

MuJoCo ground-truth pose remains available independently on port 50052.

### Terminal 4: zero-only remote command

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
run the graphical Bello visualizer with a supported controller instead; the
headless Lima template deliberately has no display.

### Terminal 5: read-only policy preflight

Run preflight before every armed simulated run:

```bash
./scripts/lima_vcan.sh run uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_mini_locomotion \
  endpoint=http://127.0.0.1:50051 \
  remote_endpoint=http://127.0.0.1:50054 \
  upper_body_tracking_enabled=false
```

The default `arm=false` validates the checkpoint, policy, services, motor
inventory, IMU frame, and a complete state sample. It acquires no control lease
and sends no motor command. Do not continue unless its result contains
`'hardware_preflight': 'passed'` and reports that no lease or command was
issued.

## Run an armed zero-command simulation

Keep Terminals 1, 2, and 4 running. Start a bounded simulated run in Terminal 5:

```bash
./scripts/lima_vcan.sh run uv run bello-run-hardware \
  task=locomotion \
  checkpoint_path=checkpoints/bello_mini_locomotion \
  endpoint=http://127.0.0.1:50051 \
  remote_endpoint=http://127.0.0.1:50054 \
  upper_body_tracking_enabled=false \
  duration_seconds=30 \
  arm=true
```

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
release or reset the torso belongs only in the emulator terminal. The headless
remote keeps the commanded velocity at zero throughout this sequence, so this
tests standing and the complete vCAN policy path but not commanded locomotion.

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
