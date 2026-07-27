# Vast.ai training

`bello-vast-train` manages one branch-pinned, billing-bounded remote training
lifecycle. Its fields, defaults, and validation are documented in the
[command reference](commands.md#bello-vast-train).

## Selection contract

Before rental, the launcher:

1. fetches current `origin` branches;
2. resolves the selected branch to one immutable commit;
3. verifies the selected profile, `pyproject.toml`, and `uv.lock` in that
   commit;
4. verifies any motion archive is repository-tracked in that commit;
5. generates a unique timestamped run/lifecycle label;
6. builds offer queries from the maintained hardware policy.

Motion jobs require an explicit tracked clip. Ignored local datasets are never
uploaded implicitly.

The default offer policy performs one global search for an RTX 5090 with at
least 64 GB CPU RAM, 80 GB disk, 0.98 reliability, CUDA 13, verification,
rentability, and direct SSH. `max_price` is a required upper bound on total
hourly price.

## Lifecycle

Non-dry execution:

1. acquires the `.vast` lifecycle lock and reconciles recorded state;
2. selects and confirms one compatible offer;
3. records a creating state before requesting the instance;
4. records the active instance ID as soon as it is known;
5. waits for instance readiness and an SSH route, preferring the direct mapping
   and falling back to the Vast proxy;
6. clones the exact commit and performs a frozen dependency sync;
7. verifies the JAX GPU backend;
8. runs a bounded training preflight;
9. starts the real job in a detached process, records PID/status/log files, and
   streams bounded log/status updates;
10. relies on the configured W&B run for checkpoint artifacts;
11. attempts and verifies instance teardown after success, error, timeout, or
    handled interruption; unconfirmed destruction retains durable state and
    fails closed.

The preflight uses a small fixed rollout and disables W&B and videos. The real
job forces W&B on and assigns the resolved run/checkpoint names. Every produced
checkpoint records the same full commit selected by the launcher, and W&B logs
that commit with the run configuration.

## Time bounds

`duration_hours` is the remote-job timeout, not a request to train for exactly
that duration. The selected profile's timestep budget still determines normal
completion and may finish earlier.

The watchdog also includes readiness and SSH windows plus teardown grace. With
the six-hour default, the upper bound is the six-hour job window plus 15 minutes
for readiness, 15 minutes for SSH, and 15 minutes of final watchdog grace.
Progress must be observed at least every 30 minutes or before the shorter job
deadline.

## Credentials

Non-dry execution requires:

- `VAST_API_KEY` or keyring entry `vast/api_key`;
- `WANDB_API_KEY` or keyring entry `wandb/api_key`;
- optionally `GITHUB_TOKEN` or keyring entry
  `github/bello_mujoco_read_token` when the remote repository cannot be cloned
  anonymously.

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
