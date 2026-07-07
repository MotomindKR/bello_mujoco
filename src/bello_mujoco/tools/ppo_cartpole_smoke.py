from __future__ import annotations

import jax
import jax.numpy as jnp
import tyro

from bello_mujoco.envs.cartpole import ContinuousCartPoleEnv
from bello_mujoco.train.ppo import PPOConfig, make_train


def run_cartpole_ppo(config: PPOConfig) -> dict[str, float | int | list[float] | str]:
    env = ContinuousCartPoleEnv()
    train = jax.jit(make_train(config, env))
    out = train(jax.random.PRNGKey(config.seed))
    rewards = out["metrics"]["mean_reward"]
    episode_returns = out["metrics"]["mean_episode_return"]
    max_episode_returns = out["metrics"]["max_episode_return"]
    losses = out["metrics"]["loss"]
    trace_idx = jnp.linspace(
        0, config.num_updates - 1, min(config.num_updates, 12), dtype=jnp.int32
    )
    return {
        "backend": jax.default_backend(),
        "updates": config.num_updates,
        "obs_size": env.observation_size,
        "action_size": env.action_size,
        "first_mean_reward": float(rewards[0]),
        "last_mean_reward": float(rewards[-1]),
        "first_episode_return": float(episode_returns[0]),
        "last_episode_return": float(episode_returns[-1]),
        "max_episode_return": float(jnp.max(max_episode_returns)),
        "last_loss": float(losses[-1]),
        "episode_return_trace": [float(episode_returns[i]) for i in trace_idx],
    }


def main(
    seed: int = 0,
    total_timesteps: int = 16_384,
    num_envs: int = 64,
    num_steps: int = 64,
    update_epochs: int = 4,
    num_minibatches: int = 4,
    learning_rate: float = 3.0e-4,
    hidden_size: int = 64,
    hidden_depth: int = 2,
    ent_coef: float = 0.0,
    init_log_std: float = -0.5,
) -> None:
    config = PPOConfig(
        seed=seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        update_epochs=update_epochs,
        num_minibatches=num_minibatches,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        hidden_depth=hidden_depth,
        ent_coef=ent_coef,
        init_log_std=init_log_std,
        anneal_lr=True,
    )
    print(run_cartpole_ppo(config))


def entrypoint() -> None:
    tyro.cli(main)


if __name__ == "__main__":
    entrypoint()
