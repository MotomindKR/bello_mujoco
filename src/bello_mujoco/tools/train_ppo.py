from __future__ import annotations

import jax
import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.sim import (
    DEFAULT_CONTACT_BUDGET_PER_ENV,
    DEFAULT_WARP_NJMAX,
    WarpConfig,
)
from bello_mujoco.train.ppo import PPOConfig, init_train_state, make_train


def main(
    config: PPOConfig = PPOConfig(),
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    dry_run: bool = True,
    jit: bool = True,
    contact_budget_per_env: int = DEFAULT_CONTACT_BUDGET_PER_ENV,
    warp_naconmax: int | None = None,
    warp_naccdmax: int | None = None,
    warp_njmax: int | None = DEFAULT_WARP_NJMAX,
) -> None:
    warp = WarpConfig(
        contact_budget_per_env=contact_budget_per_env,
        naconmax=warp_naconmax,
        naccdmax=warp_naccdmax,
        njmax=warp_njmax,
    )
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=warp,
        batch_size=config.num_envs,
    )
    rng = jax.random.PRNGKey(config.seed)
    if dry_run:
        state, _ = init_train_state(rng, env.observation_size, env.action_size, config)
        batch_state, batch_obs = env.reset_batch(config.num_envs)
        print(
            {
                "backend": jax.default_backend(),
                "mjx_impl": "warp",
                "warp_naconmax": env.model.warp_naconmax,
                "warp_njmax": env.model.warp_njmax,
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
    metrics = jax.tree_util.tree_map(
        lambda x: x[-1] if getattr(x, "ndim", 0) > 0 else x, out["metrics"]
    )
    print(metrics)


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
