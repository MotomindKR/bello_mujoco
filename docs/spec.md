# Repository Spec

## Purpose

`bello_mujoco` provides a MuJoCo/MJX-Warp simulation environment, generated Bello robot assets, reward/done scaffolding, and a PureJaxRL-style PPO training loop.

The project is a training scaffold, not a polished locomotion policy. It is designed so reward terms, done terms, observations, assets, terrain, and training runtime can be configured from CLI dataclasses.

## Package And Environment

- Python package: `bello-mujoco`
- Python version: `>=3.12,<3.13`
- Environment manager: uv
- Shell manager: Nix flake
- GPU backend: JAX CUDA wheels plus MuJoCo MJX-Warp

Nix only provides `uv` and `git`. Python and Python packages are resolved by uv.

## Entry Points

```text
bello-prepare-assets          Generate cached robot URDF/MJCF assets.
bello-smoke-env               Load env, reset, optionally run one JIT step.
bello-benchmark-env           Benchmark vectorized env stepping.
bello-train-ppo               Dry-run or run PPO training.
bello-eval-policy             Render checkpoint rollout to MP4, optionally W&B.
bello-ppo-cartpole-smoke      PPO smoke on a tiny pure-JAX continuous CartPole.
```

## Default Full-Body Runtime Contract

```text
asset: full_body
terrain: plane
simulator: MJX-Warp
nq: 42
nv: 41
nu: 35
action_size: 35
observation_size: 149
control_decimation: 10
max_episode_steps: 1000
```

Observation layout:

```text
base_ang_vel[3]
projected_gravity[3]
command[3]
joint_pos[35]
joint_vel[35]
actuator_force[35]
last_action[35]
```

Action layout:

```text
policy actuator residuals in MuJoCo actuator order
```

By default, every actuator is in `policy` mode. Per-joint actuator config can switch joints to `hold`, which removes them from the policy action vector while keeping them simulated and observable depending on `ObservationConfig`.

## Source Assets

Tracked source assets:

```text
assets/source/bello/full_body/
assets/source/bello/lower_body/
assets/source/bello/upper_body/
```

Valid asset names:

```text
full_body
lower_body
upper_body
```

## Generated Assets

Generated assets are cached outside git by default:

```text
/tmp/bello-mujoco-assets/<config-hash>/<asset>/
```

The cache key includes asset settings, mesh budgets, collision mode, end-effector mode, neighbor exclusion depth, and actuator config. Asset generation uses a file lock per cache directory.

## Hot Path

The simulation and PPO rollout/update hot path is JAX-based:

- `BelloLocomotionEnv.step`
- `BelloLocomotionEnv.step_batch`
- PPO rollout collection with `jax.lax.scan`
- GAE with reverse `jax.lax.scan`
- policy/value evaluation with Equinox/JAX
- optimization with Optax

Host-side pieces include asset generation, W&B logging, checkpoint metadata, and evaluation video rendering.
