# Testing And Audit Checklist

## Standard Gate

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

Before committing larger env/training changes, also run:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env --step
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  --no-dry-run \
  --config.total-timesteps 4 \
  --config.num-envs 2 \
  --config.num-steps 2 \
  --config.num-minibatches 1 \
  --config.update-epochs 1
```

## Focused Test Sets

Asset pipeline:

```bash
uv run pytest tests/test_assets_pipeline.py tests/test_model.py -q
```

Environment and rewards:

```bash
uv run pytest tests/test_env.py tests/test_rewards.py -q
```

PPO:

```bash
uv run pytest tests/test_ppo.py tests/test_cartpole.py tests/test_cartpole_ppo_integration.py -q
```

Checkpoint and tracking:

```bash
uv run pytest tests/test_checkpoint.py tests/test_tracking.py -q
```

## Smoke Contracts

Default full-body smoke:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env --step
```

Expected:

```text
backend: gpu
mjx_impl: warp
action_size: 35
observation_size: 149
```

Disable actuator force to recover the previous 114-dim observation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env \
  --env-config.observation.no-actuator-force \
  --step
```

Lower-body smoke:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env \
  --asset-config.asset lower_body \
  --step
```

Upper-body smoke with foot/base terms disabled:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env \
  --asset-config.asset upper_body \
  --reward-config.foot-contact.no-enabled \
  --reward-config.foot-slip.no-enabled \
  --reward-config.foot-clearance.no-enabled \
  --reward-config.track-lin-vel-xy.no-enabled \
  --reward-config.track-ang-vel-z.no-enabled \
  --reward-config.lin-vel-z.no-enabled \
  --reward-config.ang-vel-xy.no-enabled \
  --reward-config.flat-orientation.no-enabled \
  --reward-config.base-height.no-enabled \
  --done-config.base-height.no-enabled \
  --done-config.roll-pitch.no-enabled \
  --step
```

## Audit Checklist Before Longer Training

1. Confirm working tree only contains intended changes:

   ```bash
   git status --short
   git diff --stat
   git diff --check
   ```

2. Confirm asset choices:

   ```bash
   uv run bello-prepare-assets --help | rg -e "--asset|full_body|lower_body|upper_body"
   ```

3. Confirm env shape:

   ```bash
   XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-smoke-env --step
   ```

4. Confirm one PPO update:

   ```bash
   XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
     --no-dry-run \
     --config.total-timesteps 4 \
     --config.num-envs 2 \
     --config.num-steps 2 \
     --config.num-minibatches 1 \
     --config.update-epochs 1
   ```

5. Confirm docs match CLI:

   ```bash
   uv run bello-train-ppo --help | rg "env-config|reward-config|done-config|command-config"
   ```
