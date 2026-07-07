import jax.numpy as jnp

from bello_mujoco.envs.cartpole import ContinuousCartPoleEnv


def test_cartpole_reset_and_step_shapes() -> None:
    env = ContinuousCartPoleEnv()
    state, obs = env.reset_batch(8)
    next_state, next_obs, reward, done, info = env.step_batch(
        state, jnp.zeros((8, env.action_size))
    )

    assert obs.shape == (8, env.observation_size)
    assert next_obs.shape == (8, env.observation_size)
    assert next_state.step_count.shape == (8,)
    assert reward.shape == (8,)
    assert done.shape == (8,)
    assert info["episode_return"].shape == (8,)
