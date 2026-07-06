import pytest

from bello_mujoco.tools.ppo_cartpole_smoke import run_cartpole_ppo
from bello_mujoco.train.ppo import PPOConfig


@pytest.mark.integration
def test_cartpole_ppo_reaches_max_score() -> None:
    config = PPOConfig(
        seed=3,
        total_timesteps=8_388_608,
        num_envs=128,
        num_steps=128,
        update_epochs=4,
        num_minibatches=8,
        learning_rate=1.0e-3,
        hidden_size=64,
        hidden_depth=2,
        ent_coef=0.0,
        init_log_std=-1.0,
        anneal_lr=True,
    )

    metrics = run_cartpole_ppo(config)

    assert metrics["last_episode_return"] >= 500.0
    assert metrics["max_episode_return"] >= 500.0
