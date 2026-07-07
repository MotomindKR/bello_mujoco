from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import (
    WarpConfig,
)


def main(
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    env_config: BelloEnvConfig = BelloEnvConfig(),
    reward_config: RewardConfig = RewardConfig(),
    command_config: CommandConfig = CommandConfig(),
    done_config: DoneConfig = DoneConfig(),
    batch_size: int = 512,
    step_count: int = 5,
    warp: WarpConfig = WarpConfig(),
) -> None:
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=warp,
        batch_size=batch_size,
        env_config=env_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
    )
    state, obs = env.reset_batch(batch_size)
    obs.block_until_ready()
    actions = jnp.zeros((batch_size, env.action_size), dtype=jnp.float32)
    step = jax.jit(lambda s, a: env.step_batch(s, a))
    state, obs, reward, done, _ = step(state, actions)
    obs.block_until_ready()
    reward.block_until_ready()
    done.block_until_ready()

    step_seconds: list[float] = []
    for _ in range(step_count):
        start = time.perf_counter()
        state, obs, reward, done, _ = step(state, actions)
        obs.block_until_ready()
        reward.block_until_ready()
        done.block_until_ready()
        step_seconds.append(time.perf_counter() - start)

    print(
        {
            "backend": jax.default_backend(),
            "mjx_impl": "warp",
            "batch_size": batch_size,
            "terrain": env_config.terrain.mode,
            "decimation": env_config.decimation,
            "max_episode_steps": env_config.max_episode_steps,
            "warp_naconmax": env.model.warp_naconmax,
            "warp_naccdmax": env.model.warp_naccdmax,
            "warp_njmax": env.model.warp_njmax,
            "nq": env.model.mj_model.nq,
            "nv": env.model.mj_model.nv,
            "nu": env.model.mj_model.nu,
            "action_size": env.action_size,
            "observation_size": env.observation_size,
            "observation_shape": tuple(obs.shape),
            "reward_mean": float(jnp.mean(reward)),
            "done_mean": float(jnp.mean(done.astype(jnp.float32))),
            "step_seconds": step_seconds,
            "env_steps_per_second": [batch_size / seconds for seconds in step_seconds],
        }
    )


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
