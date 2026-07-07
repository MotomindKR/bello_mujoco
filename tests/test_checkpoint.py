from pathlib import Path

import jax
import jax.numpy as jnp

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.checkpoint import (
    CheckpointMetadata,
    load_checkpoint_metadata,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from bello_mujoco.envs.locomotion import DoneConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.terrain import TerrainConfig
from bello_mujoco.train.ppo import PPOConfig, apply_model, init_train_state


def test_training_checkpoint_round_trips_model_and_metadata(tmp_path: Path) -> None:
    config = PPOConfig(
        seed=11,
        hidden_size=8,
        hidden_depth=1,
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )
    train_state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    metadata = CheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(output_root=tmp_path / "assets"),
        reward=RewardConfig(),
        command=CommandConfig(lin_vel_x=(-0.2, 0.3)),
        done=DoneConfig(min_base_height=0.7),
        terrain=TerrainConfig(mode="rough", seed=3, hfield_resolution=17),
        warp=WarpConfig(contact_budget_per_env=12),
        obs_size=4,
        action_size=2,
    )
    obs = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    expected_mean, _, expected_value = apply_model(train_state.model, obs)

    save_training_checkpoint(tmp_path / "ckpt", train_state, metadata)
    restored_state, restored_metadata = restore_training_checkpoint(tmp_path / "ckpt")
    restored_mean, _, restored_value = apply_model(restored_state.model, obs)
    disk_metadata = load_checkpoint_metadata(tmp_path / "ckpt")

    assert jnp.allclose(restored_mean, expected_mean)
    assert jnp.allclose(restored_value, expected_value)
    assert restored_metadata.terrain.seed == 3
    assert restored_metadata.command.lin_vel_x == (-0.2, 0.3)
    assert disk_metadata.asset.output_root == tmp_path / "assets"
