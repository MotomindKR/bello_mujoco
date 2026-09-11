# Vast.ai training

`bello-vast-train` manages one branch-pinned, billing-bounded remote training
lifecycle. Its fields, defaults, and validation are documented in the
[command reference](commands.md#bello-vast-train).

## Selection contract

Before rental, the launcher:

1. fetches current `origin` branches;
2. resolves the selected branch to one immutable commit;
3. verifies the selected profile, `.gitmodules`, pinned `deps/bello-stack`
   gitlink, `.python-version`, `pyproject.toml`, and `uv.lock` in that commit;
4. verifies any motion archive is repository-tracked in that commit;
5. validates any local TWIST/TWIST2 dataset and typed teacher checkpoint;
6. generates a unique timestamped run/lifecycle label;
7. builds offer queries from the maintained hardware policy.

Motion jobs require an explicit tracked clip. TWIST and TWIST2 jobs instead
require an explicit local schema-3 dataset. A student additionally requires one explicit
typed teacher checkpoint. No dataset or teacher is selected implicitly.

The default offer policy performs one global search for an RTX 5090 with at
least 64 GB CPU RAM, 80 GB disk, 0.98 reliability, CUDA 13, verification,
rentability, and direct SSH. `max_price` is a required upper bound on total
hourly price.

The default training experiment is `bello_locomotion`. Select
`experiment=bello_mini_locomotion` to train Mini locomotion, or
`experiment=bello_mini_standing` for its standing-only policy. The standing
experiment composes the shared production recipe, Mini robot profile, and
standing overlay. Mini checkpoints can use the vCAN emulator or physical
locomotion runner, subject to the runner's normal checkpoint and safety gates.

## Lifecycle

Non-dry execution:

1. acquires the `.vast` lifecycle lock and reconciles recorded state;
2. selects and confirms one compatible offer;
3. records a creating state before requesting the instance;
4. records the active instance ID as soon as it is known;
5. waits for instance readiness and its required direct SSH mapping;
6. transfers explicit TWIST inputs with resumable `rsync --append-verify` and
   verifies every file's SHA-256 on the instance;
7. recursively clones the exact root commit and pinned submodules, then performs
   a frozen workspace dependency sync;
8. verifies the JAX GPU backend and a headless render-and-encode path;
9. runs a bounded training preflight;
10. starts the real job in a detached process, records PID/status/log files, and
   streams bounded log/status updates;
11. relies on the configured W&B run for checkpoint artifacts;
12. attempts and verifies instance teardown after success, error, timeout, or
    handled interruption; unconfirmed destruction retains durable state and
    fails closed.

The remote bootstrap uses OSMesa for native diagnostic videos so rendering does
not depend on a provider's NVIDIA EGL stack or consume training VRAM. Before the
bounded rollout, it renders one frame and encodes it to MP4. The rollout disables
W&B; non-TWIST preflights also disable their longer configured videos because the
dedicated render probe has already exercised that path. Locomotion preflights
also require a completed PPO update, restore the saved checkpoint, check finite
policy parameters, and render 25 native frames from the restored policy before
starting the full run. The real job forces W&B
on and assigns the resolved run/checkpoint names. Every produced
checkpoint records the same full commit selected by the launcher, and W&B logs
that commit with the run configuration.

## TWIST transfer

The launch machine must provide `rsync` and direct outbound SSH. The remote
bootstrap installs its own `rsync` before transfer. Dataset files and checkpoint
contents remain outside Git and are copied only to dedicated paths under
`/workspace`. Interrupted transfers retry with partial-file continuation. A
remote SHA-256 mismatch prevents training and triggers normal teardown.
TWIST2 warm starts transfer an explicitly selected typed checkpoint directory;
the remote trainer restores policy and normalization only and records the new
dataset fingerprint in every output checkpoint.

TWIST and TWIST2 offers are ranked by proximity to Korea (`KR`, `JP`, `TW`,
`HK`, then `SG`) before marketplace value. After SSH becomes ready, the launcher uploads a
16 MiB probe and requires at least 5 MiB/s. A slower instance is destroyed and
the next nearby offer is tried before the multi-gigabyte dataset is transferred.
These controls can be overridden with `preferred_countries`,
`min_upload_mib_per_second`, and `route_probe_mib`.

Teacher example:

```bash
uv run bello-vast-train \
  experiment=twist_teacher \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  branch=feature/twist-teacher-student \
  max_price=1.25 \
  duration_hours=36
```

Student example:

```bash
uv run bello-vast-train \
  experiment=twist_student \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  teacher_checkpoint_path=/path/to/teacher/final \
  branch=feature/twist-teacher-student \
  max_price=1.25 \
  duration_hours=60
```

One-stage TWIST2 example:

```bash
uv run bello-vast-train \
  experiment=twist2 \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  branch=feature/twist-teacher-student \
  max_price=1.25 \
  duration_hours=60
```

TWIST2 transfer example:

```bash
uv run bello-vast-train \
  experiment=twist2 \
  dataset_path=/home/gmmyung/amass/twist_dataset/index.json \
  warm_start_checkpoint_path=/path/to/previous/final \
  allow_git_commit_mismatch=true \
  total_timesteps=491520000 \
  learning_rate=0.00005 \
  branch=feature/twist-teacher-student \
  run_name=twist2_fastsam_finetune_seed1 \
  max_price=1.25 \
  duration_hours=24
```

## Time bounds

`duration_hours` is the remote-job timeout, not a request to train for exactly
that duration. The selected profile's timestep budget still determines normal
completion and may finish earlier.

The watchdog also includes readiness, SSH, input-transfer, and teardown grace
windows. With the six-hour default, the upper bound is the six-hour job window
plus 15 minutes for readiness, 15 minutes for SSH, one hour for input transfer,
and 15 minutes of final watchdog grace.
Progress must be observed at least every 30 minutes or before the shorter job
deadline.

## Credentials

Non-dry execution requires:

- `VAST_API_KEY` or keyring entry `vast/api_key`;
- `WANDB_API_KEY` or keyring entry `wandb/api_key`;
- optionally `GITHUB_TOKEN` or keyring entry
  `github/bello_mujoco_read_token` when either the root repository or its
  submodules cannot be cloned anonymously.

Credentials are read from the supported environment/keyring entries and are
not interpolated into diagnostic command previews.

## Dry run and confirmation

Dry run still fetches and validates Git selection, then prints the offer query.
It does not read credentials, query marketplace results,
provision an instance, or print the complete remote command protocol.

Interactive rental asks for confirmation after showing the chosen offer.
`skip_confirmation=true` is reserved for explicitly unattended operation.

## Durable state and recovery

The launcher records creating/active state under `.vast/` before and after
instance creation. Do not delete that directory while a rental may exist.

If launch is interrupted, rerun the same workflow. Reconciliation examines the
recorded instance and label, handles explicit not-found/destroyed state, retries
teardown, and refuses ambiguous cleanup rather than silently risking an active
billed instance.
