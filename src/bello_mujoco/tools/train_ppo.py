from __future__ import annotations

from dataclasses import replace

import jax
import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.envs.locomotion import BelloLocomotionEnv, DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import (
    DEFAULT_CONTACT_BUDGET_PER_ENV,
    DEFAULT_WARP_NJMAX,
    WarpConfig,
)
from bello_mujoco.tracking import (
    AimConfig,
    close_run,
    create_aim_run,
    log_config,
    track_metric_history,
)
from bello_mujoco.train.ppo import PPOConfig, init_train_state, make_train


def main(
    config: PPOConfig = PPOConfig(),
    asset_config: AssetPrepConfig = AssetPrepConfig(),
    reward_config: RewardConfig = RewardConfig(),
    command_config: CommandConfig = CommandConfig(),
    done_config: DoneConfig = DoneConfig(),
    aim_config: AimConfig = AimConfig(),
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
    env = replace(
        env,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
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
                "aim_enabled": aim_config.enabled,
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
    run = create_aim_run(aim_config)
    try:
        log_config(
            run,
            {
                "ppo": config,
                "asset": asset_config,
                "reward": reward_config,
                "command": command_config,
                "done": done_config,
                "warp": warp,
                "env": {
                    "obs_size": env.observation_size,
                    "action_size": env.action_size,
                    "mjx_impl": "warp",
                    "xml_path": env.model.xml_path,
                },
            },
        )
        track_metric_history(
            run,
            out["metrics"],
            samples_per_update=config.num_envs * config.num_steps,
            context={"phase": "train"},
        )
    finally:
        close_run(run)
    print(metrics)


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
