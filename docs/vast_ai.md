# Vast.ai training

`bello-vast-train` rents one instance, clones a pinned root commit and its
required `bello-stack` submodule, and starts detached training. The Vast host owns
shutdown; your computer only launches and monitors it.

```bash
uv run bello-vast-train \
  experiment=bello_mini_locomotion \
  branch=your-training-branch \
  max_price=0.60 \
  duration_hours=40
```

Use `dry_run=true` to validate source selection and print the offer query without
renting. Normal execution asks for confirmation. Use `skip_confirmation=true`
only for explicitly approved unattended launches. See
[command fields](commands.md#bello-vast-train) and `configs/commands/vast_train.yaml`.

## Remote lifecycle

The instance's onstart hook installs a supervisor before dependency setup.
Training starts only when it is armed. The detached job installs the CUDA
training dependency group, checks GPU training and software rendering, runs a short PPO preflight, then
starts the selected profile with W&B enabled.
Motion preflight also restores the saved policy, renders a short CPU evaluation,
and exports its deployment bundle before the full run can start.
SONIC preflight performs the same restore/render/export gate after downloading
the versioned W&B corpus artifact and the exact Hugging Face checkpoint revision.

- Completion, setup/training failure, or a disappearing job process triggers
  automatic destruction.
- `duration_hours` bounds the detached job, including setup and preflight.
  The host enforces it even without the local monitor.
- `shutdown_grace_seconds` (default 1200) reserves time before that deadline
  for a clean stop. The host writes `stop_requested` in `BELLO_JOB_DIR`; PPO
  finishes its current update, skips new evaluation videos, saves the final
  checkpoint and finishes W&B uploads. W&B records `training/stop_reason` as
  `budget_deadline`. Zero disables the advance request. The grace must be less
  than the job duration; the hard timeout still applies if an update, video,
  save or upload stalls. Choose a grace longer than the slowest update/video
  plus checkpoint upload time. A budget stop does not imply the target number
  of transitions was reached.
- An absolute host deadline also covers readiness, SSH, job time, and teardown
  grace, including a launch that never reaches the job-start step.
- Deletion uses Vast's instance-scoped credentials and retries until confirmed.
  Your account API key is not sent to the host supervisor.

Closing your terminal, pressing Ctrl-C locally, losing SSH, or shutting down
your computer does **not** stop training or delete the instance. Local
no-progress/reconnect timeouts only stop monitoring. Check W&B or the Vast
console for the continuing job; explicitly destroy the instance there to cancel.

The absolute deadline starts at rental creation. API outages can delay actual
destruction; a host that never starts cannot run its supervisor. Check the Vast
console if startup fails. This is not a provider-level billing guarantee.

## Logs and recovery

On the instance:

- `/tmp/bello-vast-supervisor.log`: shutdown decisions and retries.
- `/workspace/.bello-vast/<run-name>/output.log`: setup and training output.
- `/workspace/.bello-vast/<run-name>/status`: final job exit code.

Checkpoints must finish uploading to W&B before the job exits. Teardown deletes
the instance's local disk; the brief final-log grace is not an upload mechanism.

SSH retries reuse the detached job under a host-side launch lock. An ambiguous
rental response is reconciled using the exact instance label, never blindly
retried against a second offer. There is no local lifecycle journal or recovery
across separate launcher invocations. Rerunning starts a **new rental**, not an
attachment to the old one. Inspect the console after an interrupted launch.

## Source and credentials

Push the intended branch before launching. The remote checkout must match the
root commit selected locally. Motion jobs accept a local `.npz` path, including
absolute paths and ignored files. The clip need not be committed. Before starting
the detached job, the launcher streams that one file over SSH into the run's
directory outside the source checkout and verifies its SHA-256. Interrupted
transfers retry within the SSH startup timeout; partial or mismatched uploads
cannot start training. An already-started job's clip cannot be replaced. Keep
the local file unchanged until upload completes. Local disconnection before
upload/start finishes does not complete a launch; the host's absolute deadline
still bounds the rental.

```bash
uv run bello-vast-train experiment=motion branch=your-training-branch \
  motion_path=/path/to/local/clip.npz max_price=0.60 duration_hours=40
```

For Bello Mini, select `experiment=bello_mini_motion` and supply a Mini-retargeted
clip. This uses the same verified upload and host-side teardown workflow.

Relative motion paths resolve from the local invocation directory. `dry_run=true`
checks the local file and source selection without uploading or renting.
The transfer checksum is not a checkpoint compatibility contract. Training
checkpoints and deployment bundles already contain their exact motion copy.
Other local datasets and historical checkpoint bundles are not uploaded or
migrated by the launcher.

SONIC jobs use immutable remote inputs instead of SSH-uploaded motion files:

```bash
uv run bello-vast-train experiment=bello_mini_sonic_transfer \
  branch=feat/sonic-transfer max_price=0.60 duration_hours=40 \
  motion_dataset_artifact=ENTITY/PROJECT/mini-sonic:v0 \
  pretrained_revision=6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  evaluation_clips='[held_out_walk,held_out_dance]'
```

The host initializes the pinned `deps/gear_sonic` inspection submodule only for
this experiment. It downloads the artifact and NVIDIA files outside Git, then
records their resolved artifact version and checkpoint SHA-256 in every saved
checkpoint.

### SONIC sizing

Use a **single RTX 5090 (32 GB), at least 64 GB host RAM, and 200 GB local disk**
as the initial eight-hour-corpus candidate. Prefer at least 12 CPU cores and a
CUDA-13-capable driver. The launcher currently requires one GPU; multiple
consumer cards do not combine their VRAM for this training implementation.
The corrected 2,048-environment smoke measured 16,049 MiB whole-device peak on
the local 5090, but only with the four small integration clips. Benchmark the
full corpus before increasing the environment count. A 96 GB RTX PRO 6000 is
an optional larger-memory tier, not a demonstrated requirement for this task.
See [measured results and the unresolved stability finding](sonic_validation.md).

Hardware filters and environment count can be made explicit without changing
the experiment profile:

```bash
uv run bello-vast-train experiment=bello_mini_sonic_transfer \
  branch=feat/sonic-transfer dry_run=true num_envs=2048 \
  offer.gpu_name=RTX_5090 offer.min_gpu_ram_gb=30 \
  offer.min_cpu_ram_gb=64 offer.disk_gb=200 \
  motion_dataset_artifact=ENTITY/PROJECT/mini-sonic:v0 \
  pretrained_revision=6733128a3d8a523b1418b06bca3cdf61c8b0987f \
  evaluation_clips='[held_out_walk,held_out_dance]'
```

The example validates source/configuration without renting. Use the finalized
artifact and complete held-out list, and obtain a separate rental budget before
launch. Offer availability and hourly rates are live marketplace data, not
fixed training costs.
Inspect the returned offer's CPU allocation separately; CPU-core count is not
currently a configurable launcher filter.

Non-dry execution reads:

- `VAST_API_KEY` or keyring `vast/api_key`;
- `WANDB_API_KEY` or keyring `wandb/api_key`;
- optional `GITHUB_TOKEN` or keyring `github/bello_mujoco_read_token` for private
  repositories/submodules.

Credentials travel to the job through SSH standard input, not command arguments.
Offers respect `max_price` and country preferences. The price cap is hourly,
not a total bill including transfer fees.

Offers must also remain available through the full host deadline: job duration,
readiness, SSH startup, and teardown grace. The launcher filters by the contract
end date, rechecks it immediately before renting, and prints the expiry in UTC.
The job timeout cannot extend a provider's rental contract.
