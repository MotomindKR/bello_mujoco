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

- Completion, setup/training failure, or a disappearing job process triggers
  automatic destruction.
- `duration_hours` bounds the detached job, including setup and preflight.
  The host enforces it even without the local monitor.
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

Relative motion paths resolve from the local invocation directory. `dry_run=true`
checks the local file and source selection without uploading or renting.
The transfer checksum is not a checkpoint compatibility contract. Training
checkpoints and deployment bundles already contain their exact motion copy.
Other local datasets and historical checkpoint bundles are not uploaded or
migrated by the launcher.

Non-dry execution reads:

- `VAST_API_KEY` or keyring `vast/api_key`;
- `WANDB_API_KEY` or keyring `wandb/api_key`;
- optional `GITHUB_TOKEN` or keyring `github/bello_mujoco_read_token` for private
  repositories/submodules.

Credentials travel to the job through SSH standard input, not command arguments.
Offers respect `max_price` and country preferences. The price cap is hourly,
not a total bill including transfer fees.
