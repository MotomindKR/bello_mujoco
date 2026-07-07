# Training, Checkpointing, And Evaluation

## Dry Run

`bello-train-ppo` defaults to dry-run mode:

```bash
bello-train-ppo
```

Dry-run loads the env, initializes policy state, resets a vectorized batch, and prints shape/runtime metadata without training.

## Tiny PPO Smoke

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run bello-train-ppo \
  --no-dry-run \
  --config.total-timesteps 4 \
  --config.num-envs 2 \
  --config.num-steps 2 \
  --config.num-minibatches 1 \
  --config.update-epochs 1
```

This verifies the end-to-end PPO path without doing meaningful learning.

## PPO Implementation

The PPO loop is local and follows the PureJaxRL style:

- rollout collection with `jax.lax.scan`
- vectorized env stepping through `env.step_batch`
- GAE with reverse scan
- shuffled minibatch updates
- clipped policy objective
- value loss
- entropy metric
- Optax Adam with global grad clipping
- optional linear learning-rate annealing

The policy is an Equinox actor-critic with a diagonal Gaussian action distribution.

## Checkpointing

Enable final Orbax checkpoint saving:

```bash
bello-train-ppo \
  --no-dry-run \
  --checkpoint-config.enabled \
  --checkpoint-config.directory checkpoints \
  --checkpoint-config.name flat-smoke
```

Saved checkpoint layout:

```text
checkpoints/<name>/
  metadata.json
  state/
```

`metadata.json` includes:

- PPO config
- asset config
- reward config
- command config
- done config
- env config
- warp config
- observation size
- action size

## W&B

Enable online W&B logging:

```bash
bello-train-ppo \
  --no-dry-run \
  --wandb-config.enabled \
  --wandb-config.project bello-mujoco
```

Offline logging:

```bash
bello-train-ppo \
  --no-dry-run \
  --wandb-config.enabled \
  --wandb-config.mode offline
```

The run config includes PPO, asset, reward, command, done, env, checkpoint, warp, observation size, action size, MJX implementation, XML path, and checkpoint path. Scalar PPO metrics and reward components are logged against environment steps.

## Evaluation Video

Render a checkpoint rollout:

```bash
bello-eval-policy \
  --checkpoint-path checkpoints/flat-smoke \
  --output-path outputs/flat-smoke.mp4 \
  --steps 250
```

Evaluation uses the checkpoint metadata to reconstruct the env and policy shape.

Default camera mode follows `bello_root`:

```bash
bello-eval-policy \
  --checkpoint-path checkpoints/flat-smoke \
  --camera-config.mode follow
```

Use a fixed camera:

```bash
bello-eval-policy \
  --checkpoint-path checkpoints/flat-smoke \
  --camera-config.mode fixed
```

Log evaluation video to W&B:

```bash
bello-eval-policy \
  --checkpoint-path checkpoints/flat-smoke \
  --output-path outputs/flat-smoke.mp4 \
  --log-wandb \
  --wandb-config.project bello-mujoco
```

## CartPole PPO Smoke

```bash
bello-ppo-cartpole-smoke
```

This uses a small pure-JAX continuous-action CartPole env to validate the PPO hot path independently from MuJoCo model compilation and mesh collision.
