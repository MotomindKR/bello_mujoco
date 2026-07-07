import jax.numpy as jnp
import pytest

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.contracts import FRAME_OBS_DIM
from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def test_reset_shapes(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    state, obs = env.reset()

    assert env.action_size == 35
    assert env.observation_size == FRAME_OBS_DIM
    assert obs.shape == (env.observation_size,)
    assert state.last_action.shape == (env.action_size,)
    assert int(state.step_count) == 0


def test_observation_is_finite_after_reset(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    _, obs = env.reset()

    assert obs.shape == (env.observation_size,)
    assert jnp.all(jnp.isfinite(obs))


def test_reset_batch_shapes(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    batch_size = 4
    state, obs = env.reset_batch(batch_size)

    assert obs.shape == (batch_size, env.observation_size)
    assert state.last_action.shape == (batch_size, env.action_size)
    assert state.command.shape == (batch_size, 3)
    assert state.step_count.shape == (batch_size,)


def test_reset_batch_broadcasts_command(full_body_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    command = jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32)
    state, obs = env.reset_batch(3, command=command)

    assert obs.shape == (3, env.observation_size)
    assert jnp.all(state.command == command)


def test_observation_size_tracks_loaded_model_action_size(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)

    assert env.observation_size == 9 + 3 * env.action_size


def test_locomotion_env_rejects_non_full_body_config() -> None:
    with pytest.raises(ValueError, match="requires asset='full_body'"):
        BelloLocomotionEnv.load(AssetPrepConfig(asset="lower_body"))
