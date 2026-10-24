from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

from bello_mujoco.configuration import supermimic_training_profile
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.sonic.corpus import MANIFEST, file_sha256
from bello_mujoco.supermimic import checkpoint
from bello_mujoco.supermimic.collection import CollectorState, RunnerState
from bello_mujoco.supermimic.config import LearningConfig, ObservationConfig
from bello_mujoco.supermimic.model import NetworkConfig
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.optimization import initialize_learner
from bello_mujoco.supermimic.replay import ReplayBuffer
from bello_mujoco.supermimic.sampling import SamplingState
from bello_mujoco.training.supermimic import DatasetConfig, SuperMimicTrainingRunConfig
from test_supermimic_learning import transition


def test_chunked_checkpoint_is_readable_by_unmodified_equinox(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "_TRANSFER_CHUNK_BYTES", 256)
    tree = (
        jnp.arange(240, dtype=jnp.float32).reshape((20, 12)),
        np.arange(200, dtype=np.int32).reshape((20, 10)).T,
        jnp.array(True),
        3,
    )
    path = tmp_path / "chunked.eqx"
    eqx.tree_serialise_leaves(path, tree, filter_spec=checkpoint._serialise_leaf)
    restored = eqx.tree_deserialise_leaves(path, tree)
    for actual, expected in zip(
        jax.tree.leaves(restored), jax.tree.leaves(tree), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "require_repository_commit", lambda _: None)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / MANIFEST).write_text('{"fixture":true}\n')
    config = LearningConfig(
        num_envs=2,
        total_timesteps=128,
        warmup_steps=2,
        replay_steps=8,
        window_length=2,
        model_batches=1,
        model_batch_size=4,
        imagined_batch_size=2,
        imagination_horizon=2,
        critic_epochs=1,
        critic_minibatches=1,
    )
    networks = NetworkConfig(
        dynamics_widths=(8,), actor_widths=(8,), critic_widths=(8,), latent_size=4
    )
    observations = ObservationConfig(history_frames=2, future_frames=2)
    env = SimpleNamespace(
        action_size=1,
        codec=ObservationCodec(observations, jnp.zeros(1)),
        kinematics=SimpleNamespace(body_names=("root",)),
    )
    learner = initialize_learner(env, config, networks, 6, jax.random.PRNGKey(0))
    collector = CollectorState(
        jnp.ones(2),
        jnp.zeros((2, env.codec.sample_size)),
        jnp.zeros((2, 2, env.codec.sample_size)),
        jnp.zeros(2, jnp.int32),
        jnp.zeros(2, jnp.int32),
        jnp.ones(2),
        jnp.asarray(3),
    )
    replay = ReplayBuffer.empty(transition(0), 8).insert(transition(0))
    sampler = SamplingState(jnp.array([1.0]), jnp.array([1]), jnp.array([1.0]))
    runner = RunnerState(
        learner,
        collector,
        replay,
        sampler,
        jax.random.PRNGKey(3),
        jnp.asarray(2),
        jnp.asarray(1),
    )
    run = SuperMimicTrainingRunConfig(
        ModelConfig(asset="bello_mini", actuator_mode="policy"),
        DatasetConfig(corpus),
        config=config,
        networks=networks,
        observations=observations,
    )
    return runner, run, file_sha256(corpus / MANIFEST)


def test_full_checkpoint_restores_replay_optimizers_rng_and_sampling(
    continuation, tmp_path
):
    runner, config, digest = continuation
    path = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(
        path, runner, config, git_commit="0" * 40, dataset_digest=digest
    )
    template = jax.tree.map(jnp.zeros_like, runner)
    restored = checkpoint.restore_checkpoint(
        path,
        template,
        replace(config, config=replace(config.config, total_timesteps=256)),
        git_commit="0" * 40,
        dataset_digest=digest,
    )
    for actual, expected in zip(
        jax.tree.leaves(restored), jax.tree.leaves(runner), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(FileExistsError):
        checkpoint.save_checkpoint(
            path, runner, config, git_commit="0" * 40, dataset_digest=digest
        )
    assert checkpoint.checkpoint_metadata(path)["actor_updates"] == 1


def test_checkpoint_rejects_changed_contract_or_corrupted_state(continuation, tmp_path):
    runner, config, digest = continuation
    path = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(
        path, runner, config, git_commit="0" * 40, dataset_digest=digest
    )
    changed = replace(config, config=replace(config.config, learning_rate=0.001))
    with pytest.raises(ValueError, match="configuration changed"):
        checkpoint.restore_checkpoint(
            path, runner, changed, git_commit="0" * 40, dataset_digest=digest
        )
    with (path / "state.eqx").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.restore_checkpoint(
            path, runner, config, git_commit="0" * 40, dataset_digest=digest
        )


def test_profiles_decode_frozen_task_configs_and_preserve_motion_gains():
    from bello_mujoco.configuration import motion_training_profile

    run = supermimic_training_profile(Path("motions/local"))
    motion = motion_training_profile(name="bello_mini_motion")
    assert run.asset_config.actuator_config == motion.asset_config.actuator_config
    smoke = supermimic_training_profile(
        Path("motions/local"), name="bello_mini_supermimic_smoke"
    )
    assert smoke.config.total_timesteps < run.config.total_timesteps
    assert (
        smoke.config.model_warmup_updates
        < smoke.config.total_timesteps / smoke.config.num_envs
    )
    assert (
        smoke.environment.control.command_delay_seconds
        == run.environment.control.command_delay_seconds
    )
    assert smoke.environment.randomization.joint_origin_range == (0.0, 0.0)
    lafan = supermimic_training_profile(
        Path("motions/local"), name="bello_mini_supermimic_lafan"
    )
    for name in (
        "num_envs",
        "replay_steps",
        "model_batches",
        "model_batch_size",
        "imagined_batch_size",
    ):
        assert getattr(lafan.config, name) == getattr(run.config, name)
    vector_steps = lafan.config.total_timesteps // lafan.config.num_envs
    assert (
        vector_steps - lafan.config.warmup_steps
    ) * lafan.config.update_to_data_ratio == 14000
    assert lafan.max_training_seconds is None
    assert lafan.environment.control.command_delay_seconds == 0
    assert not lafan.environment.filtering.enabled
    assert not lafan.environment.latency.enabled
