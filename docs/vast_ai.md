# Vast.ai Training

`bello-vast-train` owns one complete Vast.ai training lifecycle. It resolves a
remote Git branch to an exact commit, rents a matching instance, clones and
verifies that commit, installs the locked environment, runs PPO with W&B, and
destroys the instance before the local command returns.

The controller is intentionally blocking. Keep its local process alive for the
duration of training. Transient SSH interruptions are recovered as described
below, but the controller still needs network access to monitor and eventually
tear down the instance.

## Credentials

Store the Vast.ai and W&B API keys with the Python `keyring` command:

```bash
keyring set vast api_key
keyring set wandb api_key
```

For a private GitHub repository, also store a read-only token:

```bash
keyring set github bello_mujoco_read_token
```

`VAST_API_KEY`, `WANDB_API_KEY`, and `GITHUB_TOKEN` are explicit environment
overrides for automation. Secrets are read before rental, sent to the remote
bootstrap over SSH standard input, and never written to `.vast/` state or
printed in commands. `WANDB_PROJECT` defaults to `bello-mujoco`; set
`WANDB_ENTITY` when the project belongs to a team.

The Vast account must have an SSH public key registered, and the corresponding
private key must be available through the local SSH agent or normal SSH config.
Use an instance-management-only Vast API key where possible.

## Launch

Choose a committed remote branch, a repository-relative Hydra YAML recipe, and
an explicit maximum total hourly price:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion_hard.yaml \
  --max-price 1.25
```

Omit `--branch` in an interactive terminal to fetch and select from `origin`.
The pipeline never trains uncommitted local changes. It records the selected
commit before rental and aborts if the branch moves before the remote clone is
verified.

Pass Hydra overrides after `--`:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion_hard.yaml \
  --max-price 1.25 \
  -- \
  config.seed=2 \
  video_config.enabled=false
```

The controller applies these mandatory overrides after user overrides:

```text
dry_run=false
wandb_config.enabled=true
wandb_config.mode=online
checkpoint_config.enabled=true
checkpoint_config.upload_wandb_artifact=true
```

To launch the matched direct-joint, differential ±18, and differential ±36
short experiments concurrently in a tiled tmux window, run:

```bash
scripts/run_vast_differential_ablations.sh main
```

The launcher creates one detached worktree per controller so each parallel Vast
instance has independent lifecycle state. Set `MAX_PRICE` to change the
per-instance hourly ceiling.

Consequently a user override cannot silently disable W&B or final checkpoint
artifact upload. Training configs from older branches must support the current
W&B and checkpoint fields or their setup validation will fail.

Use `--dry-run` to validate the branch and config and print the three offer
queries without reading credentials or renting anything:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion_hard.yaml \
  --max-price 1.25 \
  --dry-run
```

Use `--yes` only for intentional noninteractive operation. Branch, recipe, and
maximum price remain explicit.

To reproduce the pinned, otherwise unchanged MuJoCo Playground Berkeley
Humanoid baseline, use the dedicated launcher. It checks out Playground commit
`87a4bf98f1806adefd240d72dd53a2c3ceeb2f0d` on the rented machine and runs its
native Warp PPO entry point:

```bash
uv run bello-vast-playground \
  --branch main \
  --seed 0 \
  --num-timesteps 150000000 \
  --max-price 1.25
```

The Bello morphology port is `experiments/train/locomotion_berkeley.yaml` and
uses the normal `bello-vast-train` launcher. Its 2,048-environment batch and
joint-specific action scales are deliberate adaptations for Bello. The command
distribution, gait clock, rollout horizon, and PPO hyperparameters are retained;
the Bello recipe additionally uses explicit physical-joint and policy-action
safety margins.

## Offer Selection

The default profile requires:

```text
GPU:                 1 x RTX 5090
verified/rentable:   true
reliability:         >= 0.98
host RAM:            >= 64 GB
disk:                >= 80 GB
CUDA compatibility: >= 13.0
direct SSH ports:    >= 1
contract:            on-demand
```

The requested 80 GB allocation is included in the Vast search price. Returned
offers are checked again against `--max-price` before rental.

Location is a tiered preference. The controller first searches Korea, then the
default Asia list (`JP SG TW HK CN VN TH MY ID PH IN`), then globally. It stops
at the first tier with matches, sorts that tier by DL performance per dollar,
prints up to five candidates, and asks for confirmation of the best candidate
and exact hourly cost. Vast's search query accepts host RAM in GB, so the preset
queries `cpu_ram>=64`; returned offers report usable RAM in MB, where local
validation accepts at least `64000`. This accounts for nominal 64 GB hosts that
expose slightly less than 64 GiB.

Nested tyro flags customize the preset. For example:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config configs/train/rtx5090_4096x256.yaml \
  --max-price 1.50 \
  --offer.min-cpu-ram-gb 96 \
  --offer.disk-gb 120 \
  --offer.asia-countries JP SG TW HK IN TH
```

## Remote Setup and Output

The controller starts the remote bootstrap as a detached process before it
installs the required EGL/FFmpeg system libraries and `uv`, then runs:

```bash
uv python install 3.12
uv sync --frozen
```

It verifies that JAX selects a GPU and runs a one-environment training dry-run
before starting the requested recipe. The detached job stores its log and exit
status under `/workspace/.bello-vast-jobs/<run-name>/` on the rented instance.
When Vast advertises a public IP and direct port mapping, the controller tests
that endpoint before falling back to Vast's SSH proxy URL. This avoids waiting
for a broken proxy while the instance's direct SSH service is already ready.
The controller fetches bounded log chunks over short SSH connections, so a lost
SSH session does not stop setup or training. It reconnects for up to
`--ssh-timeout-seconds` of continuous SSH unavailability and resumes output from
the last received byte. If SSH remains unavailable beyond that timeout, the
pipeline fails and tears down the instance as usual.
Detachment protects against an SSH client or proxy interruption; it cannot keep
training alive if the rented host or container itself stops.

No local run log is retained. Once W&B initializes, its metrics, console output,
videos, and successful final checkpoint artifact are the durable run output. A
failure before W&B initialization is visible only in the terminal. The remote
job log is removed with the instance during teardown.

## Teardown and Recovery

As soon as Vast returns an instance ID, the controller writes ignored lifecycle
state to `.vast/active_instance.json`. A `finally` block destroys the instance
after success, setup failure, training failure, Ctrl-C, or handled `SIGTERM`,
then removes the state only after destruction is confirmed. There is no
keep-instance option.

Only one controller may run from a checkout at a time. If a previous process
left active state, the next launch first destroys that recorded instance and
refuses to rent another one if cleanup cannot be confirmed.

Local-controller cleanup cannot run after local power loss, `SIGKILL`, or a
complete network outage. If teardown retries fail, the command exits nonzero,
keeps `.vast/active_instance.json`, and prints the exact instance ID and manual
command:

```bash
vastai destroy instance <INSTANCE_ID>
```

Check the Vast console after an abnormal local termination instead of assuming
that billing stopped.
