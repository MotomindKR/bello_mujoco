# Vast.ai training

`bello-vast-train` performs one branch-pinned, billing-safe training lifecycle:

```bash
uv run bello-vast-train \
  experiment=hardware_locomotion \
  branch=main \
  max_price=1.25 \
  duration_hours=6 \
  run_name=hardware-locomotion
```

The launcher:

1. verifies the profile and commit on `origin/<branch>`;
2. selects one compatible offer under the price limit;
3. records durable state under `.vast/`;
4. provisions and verifies the locked GPU environment;
5. runs a small profile preflight;
6. starts detached training and streams progress;
7. uploads configured checkpoints through W&B;
8. destroys the instance on success, failure, timeout, or interruption.

Use `dry_run=true` to inspect selection and commands without renting. Use
`yes=true` only for unattended operation.

Motion jobs must explicitly reference a repository-tracked clip:

```bash
uv run bello-vast-train \
  experiment=motion \
  motion_path=motions/asap_fixed_head/video_single_foot_balance_level4_filter_amass.npz \
  branch=main \
  max_price=1.25
```

Ignored local datasets are never uploaded implicitly. Additional training
command overrides can be passed through the `overrides` list, but maintained
numerical changes belong in the canonical profile rather than one-off jobs.

Do not delete `.vast` while a rental may still exist. If a process is
interrupted, rerun the same command so lifecycle reconciliation can inspect and
tear down the recorded instance.
