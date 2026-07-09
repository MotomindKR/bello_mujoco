from __future__ import annotations

from dataclasses import dataclass

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from bello_mujoco.envs.cartpole import ContinuousCartPoleEnv
from bello_mujoco.hydra_config import config_path, structured_config
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.train.ppo import LearningRateSchedule, PPOConfig, make_train


@dataclass(frozen=True)
class CartPoleSmokeConfig:
    seed: int = 0
    total_timesteps: int = 16_384
    num_envs: int = 64
    num_steps: int = 64
    update_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3.0e-4
    anneal_lr: bool = True
    schedule: LearningRateSchedule | None = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_value: bool = False
    hidden_size: int = 64
    hidden_depth: int = 2
    ent_coef: float = 0.0
    init_log_std: float = -0.5
    max_log_std: float = 2.0
    obs_stack_size: int = 1

    def ppo_config(self) -> PPOConfig:
        return PPOConfig(
            seed=self.seed,
            total_timesteps=self.total_timesteps,
            num_envs=self.num_envs,
            num_steps=self.num_steps,
            update_epochs=self.update_epochs,
            num_minibatches=self.num_minibatches,
            learning_rate=self.learning_rate,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            normalize_value=self.normalize_value,
            policy=ActorCriticConfig(
                hidden_size=self.hidden_size,
                hidden_depth=self.hidden_depth,
                init_log_std=self.init_log_std,
                max_log_std=self.max_log_std,
                obs_stack_size=self.obs_stack_size,
            ),
            anneal_lr=self.anneal_lr,
            schedule=self.schedule,
        )


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
        "max_episode_return": float(jnp.nanmax(max_episode_returns)),
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
    anneal_lr: bool = True,
    schedule: LearningRateSchedule | None = None,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    normalize_value: bool = False,
    hidden_size: int = 64,
    hidden_depth: int = 2,
    ent_coef: float = 0.0,
    init_log_std: float = -0.5,
    max_log_std: float = 2.0,
    obs_stack_size: int = 1,
) -> None:
    print(
        run_cartpole_ppo(
            CartPoleSmokeConfig(
                seed=seed,
                total_timesteps=total_timesteps,
                num_envs=num_envs,
                num_steps=num_steps,
                update_epochs=update_epochs,
                num_minibatches=num_minibatches,
                learning_rate=learning_rate,
                anneal_lr=anneal_lr,
                schedule=schedule,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                normalize_value=normalize_value,
                hidden_size=hidden_size,
                hidden_depth=hidden_depth,
                ent_coef=ent_coef,
                init_log_std=init_log_std,
                max_log_std=max_log_std,
                obs_stack_size=obs_stack_size,
            ).ppo_config()
        )
    )


def entrypoint() -> None:
    hydra_entrypoint()


@hydra.main(
    version_base=None,
    config_path=config_path("cartpole_smoke"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    print(run_cartpole_ppo(structured_config(CartPoleSmokeConfig, config).ppo_config()))


if __name__ == "__main__":
    entrypoint()
