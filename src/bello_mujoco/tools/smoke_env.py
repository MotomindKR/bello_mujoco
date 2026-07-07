from __future__ import annotations

from dataclasses import dataclass

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.sim import (
    WarpConfig,
)


@dataclass(frozen=True)
class SmokeEnvConfig:
    asset_config: AssetPrepConfig = AssetPrepConfig()
    env_config: BelloEnvConfig = BelloEnvConfig()
    reward_config: RewardConfig = RewardConfig()
    command_config: CommandConfig = CommandConfig()
    done_config: DoneConfig = DoneConfig()
    step: bool = False
    warp: WarpConfig = WarpConfig()


def main(
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    env_config: BelloEnvConfig = BelloEnvConfig(),
    reward_config: RewardConfig = RewardConfig(),
    command_config: CommandConfig = CommandConfig(),
    done_config: DoneConfig = DoneConfig(),
    step: bool = False,
    warp: WarpConfig = WarpConfig(),
) -> None:
    env = BelloLocomotionEnv.load(
        asset_config,
        warp=warp,
        env_config=env_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
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
            "terrain": env_config.terrain.mode,
            "decimation": env_config.decimation,
            "max_episode_steps": env_config.max_episode_steps,
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
    hydra_entrypoint()


@hydra.main(
    version_base=None,
    config_path=config_path("smoke_env"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    cfg = structured_config(SmokeEnvConfig, config)
    main(
        asset_config=cfg.asset_config,
        env_config=cfg.env_config,
        reward_config=cfg.reward_config,
        command_config=cfg.command_config,
        done_config=cfg.done_config,
        step=cfg.step,
        warp=cfg.warp,
    )


if __name__ == "__main__":
    entrypoint()
