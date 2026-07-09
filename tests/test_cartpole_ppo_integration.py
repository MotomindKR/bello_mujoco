import pytest

from bello_mujoco.tools.ppo_cartpole_smoke import CartPoleSmokeConfig, run_cartpole_ppo


@pytest.mark.integration
def test_cartpole_ppo_reaches_max_score() -> None:
    config = CartPoleSmokeConfig(
        seed=3,
        total_timesteps=8_388_608,
        num_envs=128,
        num_steps=128,
        num_minibatches=8,
        learning_rate=1.0e-3,
        init_log_std=-1.0,
    ).ppo_config()

    metrics = run_cartpole_ppo(config)

    assert metrics["last_episode_return"] >= 499.99
    assert metrics["max_episode_return"] >= 499.99
