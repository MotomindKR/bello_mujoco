from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from bello_mujoco.contracts import ActionLayout
from bello_mujoco.envs.twist import TwistSamplingConfig, TwistSimulationConfig
from bello_mujoco.envs.twist2 import (
    BelloTwist2Env,
    TWIST2_UPSTREAM_REVISION,
    Twist2EnvConfig,
    Twist2ObservationContract,
    Twist2ParityContract,
    default_twist2_asset_config,
)
from bello_mujoco.motion.twist2_dataset import (
    Twist2DatasetAudit,
    audit_twist2_dataset,
    require_released_twist2_dataset,
)
from bello_mujoco.policy import ActorCriticConfig, apply_policy, init_actor_critic
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.checkpoint import (
    ObservationNormMetadata,
    Twist2CheckpointMetadata,
    TwistDatasetContract,
    ValueNormMetadata,
    _metadata_from_dict,
    _validate_metadata_layouts,
    validate_warm_start_metadata,
)
from bello_mujoco.training.ppo import PPOConfig
from bello_mujoco.training.ppo import (
    init_observation_norm_state,
    normalize_critic_observation,
    normalize_observation,
)
from bello_mujoco.training.tracking import to_builtin
from bello_mujoco.training.twist2 import (
    Twist2TrainingRunConfig,
    validate_twist2_training_config,
)


def test_twist2_uses_canonical_unscaled_robot_mass() -> None:
    assert default_twist2_asset_config().target_total_mass_kg is None


JOINT_NAMES = tuple(f"joint_{index}" for index in range(25))
FINGERPRINT = "sha256:" + "a" * 64


def policy_config() -> ActorCriticConfig:
    return ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        action_distribution="gaussian",
        architecture="twist2",
        layer_norm_final_hidden=True,
    )


def test_twist2_observation_contract_matches_bello_release_dimensions() -> None:
    contract = Twist2ObservationContract(JOINT_NAMES, randomization_size=55)

    assert contract.command_frame_size == 31
    assert contract.proprioception_frame_size == 80
    assert contract.privileged_reference_frame_size == 73
    assert contract.private_information_size == 94
    assert contract.actor_observation_size == 1252
    assert contract.critic_observation_size == 1634
    assert contract.actor_layout.size == 1252
    assert contract.critic_layout.size == 1634
    assert Twist2ParityContract().upstream_revision == TWIST2_UPSTREAM_REVISION
    assert Twist2ParityContract().current_offsets == (0,)
    assert Twist2ParityContract().raw_action_clip == 10.0
    assert Twist2ParityContract().action_delay_warmup_policy_steps == 1_200_000
    assert Twist2ParityContract().root_push_interval_policy_steps == 200
    assert Twist2ParityContract().root_push_phase_mode == "per_environment_reset"
    assert not Twist2ParityContract().gravity_randomization_executed


def test_twist2_reset_randomizes_reference_joint_positions_like_release() -> None:
    env = object.__new__(BelloTwist2Env)
    object.__setattr__(
        env,
        "model",
        SimpleNamespace(
            policy_qpos_indices=jnp.arange(7, 32, dtype=jnp.int32),
            policy_ctrl_indices=jnp.arange(25, dtype=jnp.int32),
        ),
    )
    reference = jnp.concatenate(
        (jnp.asarray((0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0)), jnp.ones(25))
    )

    randomized = jax.jit(env._reset_qpos)(reference, jax.random.PRNGKey(7))
    deterministic = env._reset_qpos(reference, None)

    assert randomized[2] == pytest.approx(0.85)
    assert jnp.all(randomized[7:] >= 0.8)
    assert jnp.all(randomized[7:] < 1.2)
    assert not jnp.array_equal(randomized[7:], reference[7:])
    assert jnp.array_equal(deterministic[7:], reference[7:])


def test_twist2_root_pushes_use_random_per_environment_phases() -> None:
    env = object.__new__(BelloTwist2Env)
    object.__setattr__(env, "config", Twist2EnvConfig())
    phases = env._sample_root_push_phase(jax.random.PRNGKey(11), (4096,))

    assert phases.shape == (4096,)
    assert jnp.all(phases >= 0)
    assert jnp.all(phases < 200)
    assert jnp.unique(phases).size > 1

    state = SimpleNamespace(
        total_step=jnp.zeros((200,), dtype=jnp.int32),
        root_push_phase=jnp.arange(200, dtype=jnp.int32),
    )
    first_due = env._root_push_due(state)
    second_due = env._root_push_due(
        SimpleNamespace(
            total_step=state.total_step + 1,
            root_push_phase=state.root_push_phase,
        )
    )

    assert jnp.sum(first_due) == 1
    assert first_due[199]
    assert jnp.sum(second_due) == 1
    assert second_due[198]


def test_twist2_actor_and_critic_shapes_and_training_dropout() -> None:
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        1252,
        25,
        policy_config(),
        critic_obs_size=1634,
    )
    actor_obs = jnp.zeros((2, 1252), dtype=jnp.float32)
    critic_obs = jnp.zeros((2, 1634), dtype=jnp.float32)

    deterministic, log_std, value = jax.jit(apply_policy)(policy, actor_obs, critic_obs)
    stochastic, _, _ = jax.jit(
        lambda key: apply_policy(policy, actor_obs, critic_obs, network_rng=key)
    )(jax.random.PRNGKey(1))

    assert deterministic.shape == (2, 25)
    assert log_std.shape == (2, 25)
    assert value.shape == (2,)
    assert not jnp.array_equal(deterministic, stochastic)
    assert jnp.all(policy.actor.offset_zero_encoder.layer1.bias == 0.0)
    assert jnp.all(policy.actor.offset_zero_encoder.layer2.bias == 0.0)
    assert jnp.all(policy.actor.offset_zero_encoder.layer3.bias == 0.0)
    assert jnp.all(policy.actor.backbone.hidden[0].bias == 0.0)
    assert jnp.all(policy.actor.backbone.output.bias == 0.0)


def test_twist2_actor_normalization_matches_release_and_critic_is_raw() -> None:
    config = PPOConfig(
        normalize_observations=True,
        normalize_critic_observations=False,
        observation_norm_epsilon_mode="std_offset",
        observation_norm_eps=1.0e-4,
    )
    state = init_observation_norm_state(2)
    observation = jnp.asarray([1.0, -2.0], dtype=jnp.float32)

    assert normalize_observation(observation, state, config) == pytest.approx(
        observation / 1.0001
    )
    assert jnp.array_equal(
        normalize_critic_observation(observation, state, config), observation
    )


def test_twist2_training_contract_is_one_stage_and_has_no_teacher() -> None:
    config = Twist2TrainingRunConfig(
        config=PPOConfig(
            total_timesteps=4096 * 24,
            num_envs=4096,
            num_steps=24,
            normalize_observations=True,
            normalize_critic_observations=False,
            observation_norm_update_mode="rollout",
            observation_norm_update_updates=3_000,
            observation_norm_epsilon_mode="std_offset",
            observation_norm_eps=1.0e-4,
            observation_norm_clip=1.0e9,
            policy=policy_config(),
        ),
        sampling=TwistSamplingConfig(
            base_distribution="manifest_weight",
            curriculum_enabled=True,
            curriculum_gamma=0.01,
            curriculum_min_difficulty=1.0,
            curriculum_max_difficulty=10.0,
        ),
    )

    validate_twist2_training_config(config)
    assert "teacher_checkpoint_path" not in config.__dataclass_fields__
    with pytest.raises(ValueError, match="architecture=twist2"):
        validate_twist2_training_config(
            replace(
                config,
                config=replace(
                    config.config,
                    policy=replace(config.config.policy, architecture="twist_student"),
                ),
            )
        )


def test_twist2_dataset_audit_does_not_mislabel_amass_only_data(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "source_id": "amass",
                        "sequence_id": "walk",
                        "split": "train",
                        "sampling_weight": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = audit_twist2_dataset(index_path)

    assert audit.clip_count == 1
    assert audit.train_clip_count == 1
    assert audit.evaluation_clip_count == 0
    assert audit.source_counts == (("amass", 1),)
    assert audit.train_source_counts == (("amass", 1),)
    assert audit.unit_manifest_weights
    assert audit.train_unit_manifest_weights
    assert not audit.matches_released_training_composition
    with pytest.raises(ValueError, match="released-dataset parity"):
        require_released_twist2_dataset(index_path)


def test_typed_twist2_checkpoint_round_trips_strict_metadata() -> None:
    contract = Twist2ObservationContract(JOINT_NAMES, randomization_size=55)
    ppo = PPOConfig(
        total_timesteps=4096 * 24,
        num_envs=4096,
        num_steps=24,
        normalize_observations=True,
        policy=policy_config(),
    )
    actor_norm = ObservationNormMetadata(
        mean=(0.0,) * contract.actor_observation_size,
        var=(1.0,) * contract.actor_observation_size,
        count=1.0,
    )
    critic_norm = ObservationNormMetadata(
        mean=(0.0,) * contract.critic_observation_size,
        var=(1.0,) * contract.critic_observation_size,
        count=1.0,
    )
    metadata = Twist2CheckpointMetadata(
        ppo=ppo,
        asset=default_twist2_asset_config(),
        warp=WarpConfig(),
        obs_size=contract.actor_observation_size,
        critic_obs_size=contract.critic_observation_size,
        action_size=25,
        actor_observation_layout=contract.actor_layout,
        critic_observation_layout=contract.critic_layout,
        action_layout=ActionLayout(JOINT_NAMES),
        simulation=TwistSimulationConfig(),
        twist2_env=Twist2TrainingRunConfig().twist2_env,
        sampling=TwistSamplingConfig(
            base_distribution="manifest_weight",
            curriculum_max_difficulty=10.0,
        ),
        observation_contract=contract,
        parity_contract=Twist2ParityContract(),
        dataset=TwistDatasetContract(
            index_path=Path("dataset/index.json"),
            schema_version=3,
            dataset_fingerprint=FINGERPRINT,
            robot_fingerprint=FINGERPRINT,
            gmr_revision="b" * 40,
            joint_names=JOINT_NAMES,
            body_names=contract.key_body_names,
        ),
        dataset_audit=Twist2DatasetAudit(
            clip_count=1,
            train_clip_count=1,
            evaluation_clip_count=0,
            source_counts=(("amass", 1),),
            train_source_counts=(("amass", 1),),
            unit_manifest_weights=True,
            train_unit_manifest_weights=True,
            matches_released_training_composition=False,
        ),
        robot_fingerprint=FINGERPRINT,
        source_revision="c" * 40,
        value_norm=ValueNormMetadata(mean=0.0, var=1.0, count=1.0),
        observation_norm=actor_norm,
        critic_observation_norm=critic_norm,
    )

    _validate_metadata_layouts(metadata)
    restored = _metadata_from_dict(
        to_builtin(metadata),
        strict=True,
        require_all_fields=True,
        path="metadata",
    )

    transferred = replace(
        metadata,
        ppo=replace(metadata.ppo, total_timesteps=2 * 4096 * 24, learning_rate=5e-5),
        dataset=replace(
            metadata.dataset,
            index_path=Path("new-dataset/index.json"),
            dataset_fingerprint="sha256:" + "d" * 64,
        ),
        dataset_audit=replace(
            metadata.dataset_audit,
            clip_count=2,
            train_clip_count=2,
            source_counts=(("amass", 2),),
            train_source_counts=(("amass", 2),),
        ),
        source_revision="d" * 40,
    )
    validate_warm_start_metadata(
        metadata,
        transferred,
        allow_source_revision_mismatch=True,
    )
    with pytest.raises(ValueError, match="dataset"):
        validate_warm_start_metadata(
            metadata,
            replace(
                transferred,
                dataset=replace(
                    transferred.dataset,
                    robot_fingerprint="sha256:" + "e" * 64,
                ),
            ),
            allow_source_revision_mismatch=True,
        )

    assert restored == metadata
    assert isinstance(restored, Twist2CheckpointMetadata)


def test_twist2_history_matches_source_reset_update_order() -> None:
    env = object.__new__(BelloTwist2Env)
    previous_history = jnp.arange(2 * 10 * 3, dtype=jnp.float32).reshape((2, 10, 3))
    frames = jnp.asarray(((100.0, 101.0, 102.0), (200.0, 201.0, 202.0)))
    previous = SimpleNamespace(observation_history=previous_history)
    current = SimpleNamespace(
        motion_state=SimpleNamespace(episode_length=jnp.asarray((1, 2)))
    )

    stored, observed = env._student_history_transition_batch(previous, current, frames)
    shifted = jnp.concatenate((previous_history[:, 1:], frames[:, None, :]), axis=1)

    assert jnp.array_equal(observed, shifted)
    assert jnp.array_equal(stored[0], jnp.repeat(frames[0, None, :], 10, axis=0))
    assert jnp.array_equal(stored[1], shifted[1])
