from __future__ import annotations

import jax
import tyro

from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.sim import DEFAULT_WARP_NACONMAX, DEFAULT_WARP_NJMAX, MjxImpl
from bello_mujoco.train.ppo import PPOConfig, init_train_state, make_train


def main(
    config: PPOConfig = PPOConfig(),
    dry_run: bool = True,
    jit: bool = True,
    mjx_impl: MjxImpl = "warp",
    warp_naconmax: int | None = DEFAULT_WARP_NACONMAX,
    warp_naccdmax: int | None = None,
    warp_njmax: int | None = DEFAULT_WARP_NJMAX,
) -> None:
    env = BelloLocomotionEnv.load(
        mjx_impl=mjx_impl,
        warp_naconmax=warp_naconmax,
        warp_naccdmax=warp_naccdmax,
        warp_njmax=warp_njmax,
    )
    rng = jax.random.PRNGKey(config.seed)
    if dry_run:
        state, _ = init_train_state(rng, env.observation_size, env.action_size, config)
        batch_state, batch_obs = env.reset_batch(config.num_envs)
        print(
            {
                "backend": jax.default_backend(),
                "mjx_impl": env.model.mjx_impl,
                "obs_size": env.observation_size,
                "action_size": env.action_size,
                "num_envs": config.num_envs,
                "num_steps": config.num_steps,
                "num_updates": config.num_updates,
                "batch_obs_shape": tuple(batch_obs.shape),
                "batch_action_shape": tuple(batch_state.last_action.shape),
                "params_leaves": len(jax.tree.leaves(state.model)),
            }
        )
        return

    train = make_train(config, env)
    train = jax.jit(train) if jit else train
    out = train(rng)
    metrics = jax.tree_util.tree_map(lambda x: x[-1] if getattr(x, "ndim", 0) > 0 else x, out["metrics"])
    print(metrics)


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
