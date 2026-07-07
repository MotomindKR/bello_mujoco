from __future__ import annotations

import jax
import jax.numpy as jnp
import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.sim import (
    DEFAULT_CONTACT_BUDGET_PER_ENV,
    DEFAULT_WARP_NJMAX,
    WarpConfig,
)


def main(
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    step: bool = False,
    contact_budget_per_env: int = DEFAULT_CONTACT_BUDGET_PER_ENV,
    warp_njmax: int | None = DEFAULT_WARP_NJMAX,
) -> None:
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=WarpConfig(
            contact_budget_per_env=contact_budget_per_env, njmax=warp_njmax
        ),
    )
    state, obs = env.reset()
    obs.block_until_ready()
    reward = jnp.asarray(0.0)
    done = jnp.asarray(False)
    if step:
        step_fn = jax.jit(lambda s, a: env.step(s, a))
        state, obs, reward, done, _ = step_fn(
            state, jnp.zeros(env.action_size, dtype=jnp.float32)
        )
        obs.block_until_ready()
        reward.block_until_ready()
        done.block_until_ready()

    print(
        {
            "backend": jax.default_backend(),
            "mjx_impl": "warp",
            "nq": env.model.mj_model.nq,
            "nv": env.model.mj_model.nv,
            "nu": env.model.mj_model.nu,
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "observation_shape": tuple(obs.shape),
            "reward_mean": float(jnp.mean(reward)),
            "done_mean": float(jnp.mean(done.astype(jnp.float32))),
        }
    )


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
