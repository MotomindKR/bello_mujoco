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
  --train-config experiments/train/locomotion.yaml \
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
  --train-config experiments/train/locomotion.yaml \
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

Consequently a user override cannot silently disable W&B or checkpoint artifact
upload. Add `checkpoint_config.every_updates=<N>` after `--` when a long run
should publish exact update-boundary recovery snapshots as well as its final
checkpoint. Training configs from older branches must support the current W&B
and checkpoint fields or their setup validation will fail.

Use `--dry-run` to validate the branch and config and print the three offer
queries without reading credentials or renting anything:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion.yaml \
  --max-price 1.25 \
  --dry-run
```

Use `--yes` only for intentional noninteractive operation. Branch, recipe, and
maximum price remain explicit. Override the standard recipe's budget for a
longer run:

```bash
uv run bello-vast-train \
  --branch main \
  --train-config experiments/train/locomotion.yaml \
  --max-price 1.25 \
  -- config.total_timesteps=1800000000
```

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
  --train-config experiments/train/locomotion.yaml \
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

The controller enforces two billing-safety deadlines. The complete remote job
may run for at most `--job-timeout-seconds` (12 hours by default), and its log
must advance at least once every `--progress-timeout-seconds` (30 minutes by
default). Every SSH probe, start, and status subprocess is bounded by the
remaining deadline. In addition, the instance `onstart` command launches an
absolute-deadline self-delete watchdog using Vast's restricted per-container
credentials. It remains armed until normal instance destruction, protecting
against controller power loss or `SIGKILL`. Increase the deadlines explicitly
for an intentionally longer job; the progress timeout must not exceed the hard
job timeout. The watchdog can only start after the container starts; durable
pending-label reconciliation remains the protection during pre-start
provisioning.

No local run log is retained. Once W&B initializes, its metrics, console output,
videos, periodic checkpoint artifacts (when configured), and successful final
checkpoint artifact are the durable run output. A failure before W&B
initialization is visible only in the terminal. The remote job log is removed
with the instance during teardown.

## Teardown and Recovery

Immediately before every create API call, the controller writes a pending label
atomically to `.vast/active_instance.json` and fsyncs it. A returned or
reconciled instance ID atomically transitions that state to active. A `finally`
block destroys the instance after success, setup failure, training failure,
Ctrl-C, or handled `SIGTERM`, `SIGHUP`, or `SIGQUIT`, then removes the state only
after destruction is confirmed. There is no keep-instance option.

An ambiguous create response retains the pending label and is reconciled by
that unique label before another offer can be rented. If the instance is not yet
visible, later invocations keep refusing another rental rather than discarding
the recovery record. Legacy active-ID state remains readable specifically for
billing recovery. Transient or malformed API responses never count as
successful teardown: destruction is confirmed only by an explicit
destroyed/not-found response or by the instance being absent from a successful,
well-formed instance listing.

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
