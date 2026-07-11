import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp

import bello_mujoco.checkpoint as checkpoint
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    ValueNormMetadata,
    timestamped_checkpoint_name,
    load_checkpoint_metadata,
    restore_training_checkpoint,
    save_training_checkpoint,
    value_norm_metadata,
)
from bello_mujoco.policy import ActorCriticConfig, apply_policy
from bello_mujoco.envs.locomotion import (
    BaseHeightDoneConfig,
    BelloEnvConfig,
    DoneConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.sim import WarpConfig
from bello_mujoco.terrain import TerrainConfig
from bello_mujoco.train.ppo import PPOConfig, ValueNormState, init_train_state


def test_training_checkpoint_round_trips_model_and_metadata(tmp_path: Path) -> None:
    config = PPOConfig(
        seed=11,
        policy=ActorCriticConfig(hidden_size=8, hidden_depth=1),
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
        done=DoneConfig(base_height=BaseHeightDoneConfig(min_height=0.7)),
        env=BelloEnvConfig(
            decimation=4,
            max_episode_steps=123,
            terrain=TerrainConfig(mode="rough", seed=3, hfield_resolution=17),
        ),
        warp=WarpConfig(contact_budget_per_env=12),
        obs_size=4,
        action_size=2,
        value_norm=ValueNormMetadata(mean=1.0, var=4.0, count=8.0),
    )
    obs = jnp.asarray([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    expected_mean, _, expected_value = apply_policy(train_state.policy, obs)

    save_training_checkpoint(tmp_path / "ckpt", train_state, metadata)
    restored_state, restored_metadata = restore_training_checkpoint(tmp_path / "ckpt")
    restored_mean, _, restored_value = apply_policy(restored_state.policy, obs)
    disk_metadata = load_checkpoint_metadata(tmp_path / "ckpt")

    assert jnp.allclose(restored_mean, expected_mean)
    assert jnp.allclose(restored_value, expected_value)
    assert restored_metadata.env.decimation == 4
    assert restored_metadata.env.max_episode_steps == 123
    assert restored_metadata.env.terrain.seed == 3
    assert restored_metadata.command.lin_vel_x == (-0.2, 0.3)
    assert disk_metadata.asset.output_root == tmp_path / "assets"
    assert disk_metadata.value_norm == ValueNormMetadata(mean=1.0, var=4.0, count=8.0)


def test_training_checkpoint_saves_relative_path_as_absolute(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    saved = {}

    class FakeCheckpointer:
        def save(self, path, item, *, force):
            saved["path"] = path
            saved["force"] = force
            saved["item"] = item

    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FakeCheckpointer()
    )
    config = PPOConfig(
        seed=3,
        policy=ActorCriticConfig(hidden_size=8, hidden_depth=1),
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )
    train_state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    metadata = CheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=WarpConfig(),
        obs_size=4,
        action_size=2,
    )

    save_training_checkpoint(Path("relative_ckpt"), train_state, metadata)

    assert saved["path"].is_absolute()
    assert saved["path"] == tmp_path / "relative_ckpt" / checkpoint.STATE_DIR
    assert saved["force"] is False


def test_timestamped_checkpoint_name_appends_local_time() -> None:
    name = timestamped_checkpoint_name(
        CheckpointConfig(name="locomotion_hard_seed0"),
        PPOConfig(seed=7),
        timestamp=datetime(2026, 7, 12, 1, 30, 45, 123456),
    )

    assert name == "locomotion_hard_seed0_20260712_013045_123456"


def test_timestamped_checkpoint_name_uses_seed_default() -> None:
    name = timestamped_checkpoint_name(
        CheckpointConfig(),
        PPOConfig(seed=7),
        timestamp=datetime(2026, 7, 12, 1, 30, 45, 123456),
    )

    assert name == "ppo_seed_7_20260712_013045_123456"


def test_training_checkpoint_refuses_existing_path(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    class FakeCheckpointer:
        def save(self, path, item, *, force):
            calls.append((path, item, force))

    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FakeCheckpointer()
    )
    config = PPOConfig(
        seed=3,
        policy=ActorCriticConfig(hidden_size=8, hidden_depth=1),
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )
    train_state, _ = init_train_state(jax.random.PRNGKey(0), 4, 2, config)
    metadata = CheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=WarpConfig(),
        obs_size=4,
        action_size=2,
    )
    ckpt_path = tmp_path / "existing_ckpt"
    ckpt_path.mkdir()

    try:
        save_training_checkpoint(ckpt_path, train_state, metadata)
    except FileExistsError as error:
        assert "timestamped" in str(error)
    else:
        raise AssertionError("expected existing checkpoint path to be refused")

    assert calls == []


def test_training_checkpoint_restore_uses_local_restore_args(
    tmp_path: Path, monkeypatch
) -> None:
    config = PPOConfig(
        seed=5,
        policy=ActorCriticConfig(hidden_size=8, hidden_depth=1),
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )
    metadata = CheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=WarpConfig(),
        obs_size=4,
        action_size=2,
    )
    ckpt_path = tmp_path / "ckpt"
    ckpt_path.mkdir()
    (ckpt_path / checkpoint.METADATA_FILE).write_text(
        json.dumps(checkpoint.to_builtin(metadata), indent=2, sort_keys=True)
    )
    restore_args_sentinel = object()
    calls = {}

    def fake_construct_restore_args(item):
        calls["restore_args_item"] = item
        return restore_args_sentinel

    class FakeCheckpointer:
        def restore(self, path, *, item, restore_args):
            calls["path"] = path
            calls["item"] = item
            calls["restore_args"] = restore_args
            return item

    monkeypatch.setattr(
        checkpoint.ocp.checkpoint_utils,
        "construct_restore_args",
        fake_construct_restore_args,
    )
    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FakeCheckpointer()
    )

    _, restored_metadata = restore_training_checkpoint(ckpt_path)

    assert calls["path"] == ckpt_path.resolve() / checkpoint.STATE_DIR
    assert calls["restore_args_item"] is calls["item"]
    assert calls["restore_args"] is restore_args_sentinel
    assert restored_metadata == metadata


def test_value_norm_metadata_converts_runner_state_arrays() -> None:
    metadata = value_norm_metadata(
        ValueNormState(
            mean=jnp.asarray(1.5, dtype=jnp.float32),
            var=jnp.asarray(2.5, dtype=jnp.float32),
            count=jnp.asarray(32.0, dtype=jnp.float32),
        )
    )

    assert metadata == ValueNormMetadata(mean=1.5, var=2.5, count=32.0)
