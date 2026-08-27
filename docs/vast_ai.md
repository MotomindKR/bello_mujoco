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
5. validates any local TWIST/TWIST2 dataset and typed teacher checkpoint, or
   the canonical SONIC split files and pinned release artifact;
6. generates a unique timestamped run/lifecycle label;
7. builds offer queries from the maintained hardware policy.

Motion jobs require an explicit tracked clip. TWIST, TWIST2, and SONIC jobs
instead require an explicit local dataset. A student additionally requires one
explicit typed teacher checkpoint; SONIC requires the exact exported NVIDIA
release artifact. No dataset, checkpoint, or artifact is selected implicitly.

The default offer policy performs one global search for an RTX 5090 with at
least 128 GB CPU RAM, 80 GB disk, 0.98 reliability, CUDA 13, verification,
rentability, and direct SSH. `max_price` is a required upper bound on total
hourly price.

## Lifecycle

Non-dry execution:

1. acquires the `.vast` lifecycle lock and reconciles recorded state;
2. selects and confirms one compatible offer;
3. records a creating state before requesting the instance;
4. records the active instance ID as soon as it is known;
5. waits for instance readiness and its required direct SSH mapping;
6. transfers explicit dataset/checkpoint/artifact inputs with resumable
   `rsync --append-verify` and verifies every file's SHA-256 on the instance;
7. clones the exact root commit, initializes the runtime-required pinned
   `bello-stack` submodule, then performs a frozen workspace dependency sync;
8. verifies the JAX GPU backend;
9. runs a bounded training preflight;
10. starts the real job in a detached process, records PID/status/log files, and
   streams bounded log/status updates;
11. relies on the configured W&B run for checkpoint artifacts;
12. for SONIC, evaluates every held-out clip and retrieves a hash-verified
    report and compact summary before teardown;
13. attempts and verifies instance teardown after success, error, timeout, or
    handled interruption; unconfirmed destruction retains durable state and
    fails closed.

The preflight uses a small fixed rollout and disables W&B. Non-TWIST preflights
also disable video; TWIST does not expose that setting. The real job forces W&B
on and assigns the resolved run/checkpoint names. Every produced
checkpoint records the same full commit selected by the launcher, and W&B logs
that commit with the run configuration.

## Dataset transfer

The launch machine must provide `rsync` and direct outbound SSH. The remote
bootstrap installs its own `rsync` before transfer. Dataset files and checkpoint
contents remain outside Git and are copied only to dedicated paths under
`/workspace`. Interrupted transfers retry with partial-file continuation. A
remote SHA-256 mismatch prevents training and triggers normal teardown.
TWIST2 warm starts transfer an explicitly selected typed checkpoint directory;
the remote trainer restores policy and normalization only and records the new
dataset fingerprint in every output checkpoint.

Dataset-backed offers are ranked by proximity to Korea (`KR`, `JP`, `TW`,
`HK`, then `SG`) before marketplace value. After SSH becomes ready, the
launcher uploads a 256 MiB sustained-route probe and requires at least 5 MiB/s. A slower
instance is destroyed and the next nearby offer is tried before the
multi-gigabyte dataset is transferred. These controls can be overridden with
`preferred_countries`, `min_upload_mib_per_second`, and `route_probe_mib`.

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

SONIC Any2Any comparison-milestone example. With 4,096 environments and 24
rollout steps, `786,432,000` transitions are 8,000 PPO updates:

```bash
uv run bello-vast-train \
  experiment=sonic \
  dataset_path=/home/gmmyung/amass/sonic_dataset_v7_author_fidelity/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  total_timesteps=786432000 \
  num_envs=4096 \
  checkpoint_every_updates=500 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=12 \
  skip_confirmation=true
```

The remote job first benchmarks three complete PPO updates at the requested
SONIC environment count while sampling GPU utilization, peak VRAM, power, and
elapsed time. The real run starts only when that production-scale benchmark
succeeds. It uploads periodic and final checkpoints through W&B, evaluates all
accepted held-out clips, and downloads `benchmark.json`,
`sonic_evaluation.json`, `summary.json`, and their checksum manifest to
`outputs/vast/<run-name>/` before destroying the instance. Other experiment
types retain the fixed 32-environment,
768-transition preflight.

To continue an 8,000-update checkpoint, download its W&B artifact and pass the
artifact directory as `resume_path`. The budget is absolute. This example
extends the run to the maintained 30,000-update budget while retaining the
exact 4,096-environment batch contract:

```bash
uv run bello-vast-train \
  experiment=sonic \
  dataset_path=/home/gmmyung/amass/sonic_dataset_v7_author_fidelity/index.json \
  sonic_artifact_path=artifacts/sonic/default_release \
  resume_path=outputs/wandb_artifacts/sonic_edge_4096 \
  allow_git_commit_mismatch=true \
  total_timesteps=2949120000 \
  num_envs=4096 \
  checkpoint_every_updates=500 \
  branch=feature/sonic-transfer \
  max_price=1.25 \
  duration_hours=24 \
  skip_confirmation=true
```

The launcher rejects the resume before rental if its dataset fingerprint,
release hashes, environment count, or absolute budget is
incompatible. The remote trainer then enforces the complete observation,
action, robot, PPO, and simulator contract while restoring the array state.
Remote training initializes only the pinned `deps/bello-stack` submodule needed
by the package workspace and robot assets. Tool-only GMR and Fast-SAM
submodules are not downloaded onto paid training instances.
After a detached job starts, a temporary SSH outage no longer tears down its
instance. The launcher keeps reconnecting until the job deadline while the
remote watchdog retains the final bounded-lifetime teardown. During an outage,
the launcher also polls the Vast instance state. A confirmed `exited`,
`offline`, or absent instance ends reconnection and triggers lifecycle cleanup
immediately.

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
