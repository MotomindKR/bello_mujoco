from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bello_mujoco.training.checkpoint as checkpoint
import bello_mujoco.repository as source
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    LocomotionCheckpointMetadata,
    MotionCheckpointMetadata,
    load_checkpoint_metadata,
    restore_evaluation_checkpoint,
    restore_runner_checkpoint,
    save_training_checkpoint,
    timestamped_checkpoint_name,
    validate_checkpoint_options,
)
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    BelloState,
    DoneConfig,
)
from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.motion.clip import MotionClip
from bello_mujoco.policy import (
    ActorCriticConfig,
    apply_policy,
    policy_observation_size,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.ppo import (
    ObservationNormState,
    PPOConfig,
    RunnerState,
    ValueNormState,
    init_runner_state,
)

ACTOR_LAYOUT = ObservationLayout((("actor_observation", 4),))
CRITIC_LAYOUT = ObservationLayout((("critic_observation", 5),))
ACTION_LAYOUT = ActionLayout(("action_0", "action_1"))
GIT_COMMIT = "1" * 40


@pytest.fixture(autouse=True)
def exact_source_checkout(monkeypatch: pytest.MonkeyPatch):
    state = {"commit": GIT_COMMIT}

    def require_commit(expected: str, *, allow_mismatch: bool = False) -> None:
        source.validate_repository_commit(expected)
        current = state["commit"]
        if expected != current:
            if allow_mismatch:
                return
            raise ValueError(
                "checkpoint source commit does not match this checkout: "
                f"checkpoint={expected}, current={current}"
            )

    monkeypatch.setattr(
        checkpoint,
        "capture_repository_commit",
        lambda: state["commit"],
    )
    monkeypatch.setattr(checkpoint, "require_repository_commit", require_commit)
    return state


class FakeEnv:
    task = "locomotion"
    config = BelloEnvConfig()
    action_clip = 1.0
    ppo_stagger_initial_step_count = False
    observation_layout = ACTOR_LAYOUT
    critic_observation_layout = CRITIC_LAYOUT
    action_layout = ACTION_LAYOUT
    observation_size = ACTOR_LAYOUT.size
    critic_observation_size = CRITIC_LAYOUT.size
    action_size = ACTION_LAYOUT.size

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[BelloState, jax.Array]:
        del command, rng
        zeros = jnp.zeros((batch_size,), dtype=jnp.float32)
        foot_zeros = jnp.zeros((batch_size, 2), dtype=jnp.float32)
        state = BelloState(
            data={
                "empty": jnp.zeros((batch_size, 0), dtype=jnp.float32),
                "position": jnp.zeros((batch_size, 3), dtype=jnp.float32),
            },
            last_action=jnp.zeros((batch_size, self.action_size), dtype=jnp.float32),
            last_action_delta=jnp.zeros(
                (batch_size, self.action_size), dtype=jnp.float32
            ),
            tracking_ang_vel_z=zeros,
            actor_feedback_history=jnp.zeros((batch_size, 1, 0), dtype=jnp.float32),
            actor_feedback_delay_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            foot_contact_debounce_time=foot_zeros,
            last_foot_contact=jnp.zeros((batch_size, 2), dtype=bool),
            command=jnp.zeros((batch_size, 3), dtype=jnp.float32),
            gait_phase=zeros,
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_return=zeros,
            episode_track_lin_vel_xy=zeros,
            episode_length=jnp.zeros((batch_size,), dtype=jnp.int32),
            last_episode_return=zeros,
            last_episode_track_lin_vel_xy=zeros,
            last_episode_length=jnp.zeros((batch_size,), dtype=jnp.int32),
        )
        return state, jnp.zeros((batch_size, self.observation_size), dtype=jnp.float32)

    def critic_observation_batch(self, state: BelloState) -> jax.Array:
        return jnp.zeros(
            (state.step_count.shape[0], self.critic_observation_size),
            dtype=jnp.float32,
        )

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array:
        del rng
        return jnp.zeros((batch_size,), dtype=jnp.int32)


class FakeTaskStateEnv(FakeEnv):
    def initial_task_state(self) -> jax.Array:
        return jnp.asarray((9.0, 9.0, 9.0), dtype=jnp.float32)


class FakeMotionEnv(FakeEnv):
    task = "motion_tracking"

    def reset_batch(
        self,
        batch_size: int,
        command: jax.Array | None = None,
        rng: jax.Array | None = None,
    ) -> tuple[FakeMotionState, jax.Array]:
        state, observation = super().reset_batch(batch_size, command, rng)
        failure_ema = jnp.asarray((0.125, 0.25, 0.5), dtype=jnp.float32)
        return (
            FakeMotionState(
                data=state.data,
                last_action=state.last_action,
                command=state.command,
                sampling_failure_ema=jnp.broadcast_to(
                    failure_ema, (batch_size, failure_ema.size)
                ),
                step_count=state.step_count,
                episode_return=state.episode_return,
                episode_length=state.episode_length,
            ),
            observation,
        )


@chex.dataclass(frozen=True)
class FakeMotionState:
    data: dict[str, jax.Array]
    last_action: jax.Array
    command: jax.Array
    sampling_failure_ema: jax.Array
    step_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array


def ppo_config() -> PPOConfig:
    return PPOConfig(
        seed=11,
        policy=ActorCriticConfig(
            hidden_dims=(8,),
            obs_stack_size=2,
        ),
        total_timesteps=24,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )


def metadata(config: PPOConfig, tmp_path: Path) -> CheckpointMetadata:
    return LocomotionCheckpointMetadata(
        ppo=config,
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(lin_vel_x=(-0.2, 0.3)),
        done=DoneConfig(),
        env=BelloEnvConfig(decimation=4, max_episode_steps=123),
        warp=WarpConfig(contact_budget_per_env=12),
        obs_size=ACTOR_LAYOUT.size * config.policy.obs_stack_size,
        action_size=ACTION_LAYOUT.size,
        actor_observation_layout=ACTOR_LAYOUT,
        critic_observation_layout=CRITIC_LAYOUT,
        action_layout=ACTION_LAYOUT,
        critic_obs_size=CRITIC_LAYOUT.size * config.policy.obs_stack_size,
    )


def synthetic_motion_clip() -> MotionClip:
    num_frames = 3
    num_bodies = 2
    qpos = jnp.zeros((num_frames, 7 + ACTION_LAYOUT.size), dtype=jnp.float32)
    qpos = qpos.at[:, 3].set(1.0)
    body_quat = jnp.zeros((num_frames, num_bodies, 4), dtype=jnp.float32)
    body_quat = body_quat.at[:, :, 0].set(1.0)
    return MotionClip(
        fps=50.0,
        joint_names=ACTION_LAYOUT.names,
        body_names=("pelvis_link", "torso_link"),
        source_sha256="1" * 64,
        qpos=qpos,
        qvel=jnp.zeros((num_frames, 6 + ACTION_LAYOUT.size), dtype=jnp.float32),
        body_pos=jnp.zeros((num_frames, num_bodies, 3), dtype=jnp.float32),
        body_quat=body_quat,
        body_lin_vel=jnp.zeros((num_frames, num_bodies, 3), dtype=jnp.float32),
        body_ang_vel=jnp.zeros((num_frames, num_bodies, 3), dtype=jnp.float32),
        foot_contacts=jnp.zeros((num_frames, 2), dtype=bool),
    )


def motion_metadata(
    config: PPOConfig,
    motion_path: Path,
    clip: MotionClip,
    state: RunnerState,
) -> CheckpointMetadata:
    return checkpoint.motion_metadata_with_runner_state(
        MotionCheckpointMetadata(
            ppo=config,
            asset=AssetPrepConfig(),
            warp=WarpConfig(contact_budget_per_env=12),
            obs_size=ACTOR_LAYOUT.size * config.policy.obs_stack_size,
            critic_obs_size=CRITIC_LAYOUT.size * config.policy.obs_stack_size,
            action_size=ACTION_LAYOUT.size,
            actor_observation_layout=ACTOR_LAYOUT,
            critic_observation_layout=CRITIC_LAYOUT,
            action_layout=ACTION_LAYOUT,
            motion_env=MotionTrackingEnvConfig(motion_path=motion_path),
            motion=checkpoint.motion_clip_contract(motion_path, clip),
        ),
        state,
    )


def runner_state(config: PPOConfig, env: FakeEnv) -> RunnerState:
    state, _, _ = init_runner_state(jax.random.PRNGKey(7), env, config)
    batch_size = config.num_envs
    env_state = state.env_state.replace(
        data={
            "empty": jnp.zeros((batch_size, 0), dtype=jnp.float32),
            "position": jnp.arange(batch_size * 3, dtype=jnp.float32).reshape(
                batch_size, 3
            ),
        },
        last_action=jnp.full((batch_size, env.action_size), 0.25, dtype=jnp.float32),
        command=jnp.full((batch_size, 3), 0.5, dtype=jnp.float32),
        step_count=jnp.arange(batch_size, dtype=jnp.int32) + 9,
        episode_return=jnp.asarray([1.5, 2.5], dtype=jnp.float32),
    )
    return state._replace(
        update=jnp.asarray(3, dtype=jnp.int32),
        env_state=env_state,
        obs_stack=jnp.arange(state.obs_stack.size, dtype=jnp.float32).reshape(
            state.obs_stack.shape
        ),
        critic_obs_stack=jnp.arange(
            state.critic_obs_stack.size, dtype=jnp.float32
        ).reshape(state.critic_obs_stack.shape),
        rng=jax.random.PRNGKey(99),
        learning_rate=jnp.asarray(7.5e-5, dtype=jnp.float32),
        value_norm_state=ValueNormState(
            mean=jnp.asarray(-1.25, dtype=jnp.float32),
            var=jnp.asarray(2.5, dtype=jnp.float32),
            count=jnp.asarray(17.0, dtype=jnp.float32),
        ),
        observation_norm_state=ObservationNormState(
            mean=jnp.linspace(-1.0, 1.0, state.observation_norm_state.mean.shape[-1]),
            var=jnp.linspace(1.0, 2.0, state.observation_norm_state.var.shape[-1]),
            count=jnp.asarray(19.0, dtype=jnp.float32),
        ),
        critic_observation_norm_state=ObservationNormState(
            mean=jnp.linspace(
                -2.0, 2.0, state.critic_observation_norm_state.mean.shape[-1]
            ),
            var=jnp.linspace(
                2.0, 3.0, state.critic_observation_norm_state.var.shape[-1]
            ),
            count=jnp.asarray(23.0, dtype=jnp.float32),
        ),
    )


def checkpoint_document(
    metadata: CheckpointMetadata, state: RunnerState
) -> dict[str, Any]:
    return checkpoint._checkpoint_document(
        metadata,
        checkpoint._train_checkpoint_item(state),
        checkpoint._continuation_checkpoint_item(state),
        GIT_COMMIT,
    )


def save_checkpoint(
    path: Path, state: RunnerState, metadata: CheckpointMetadata
) -> None:
    save_training_checkpoint(
        path,
        state,
        metadata,
        GIT_COMMIT,
    )


def write_checkpoint_document(path: Path, document: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / checkpoint.METADATA_FILE).write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(path), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_tree_equal(actual: Any, expected: Any) -> None:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_checkpoint_round_trips_eval_and_exact_runner_state(tmp_path: Path) -> None:
    env = FakeEnv()
    config = ppo_config()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "checkpoint"

    save_checkpoint(path, state, checkpoint_metadata)
    evaluation_state, eval_metadata, normalization = restore_evaluation_checkpoint(path)
    restored, static_policy, optimizer, resume_metadata = restore_runner_checkpoint(
        path, env, checkpoint_metadata
    )
    document = json.loads((path / checkpoint.METADATA_FILE).read_text(encoding="utf-8"))

    assert_tree_equal(restored, state)
    assert eval_metadata == resume_metadata == checkpoint_metadata
    assert_tree_equal(normalization.value, state.value_norm_state)
    assert_tree_equal(normalization.observation, state.observation_norm_state)
    assert_tree_equal(
        normalization.critic_observation,
        state.critic_observation_norm_state,
    )
    assert load_checkpoint_metadata(path) == checkpoint_metadata
    assert callable(optimizer.init)
    observation = jnp.arange(checkpoint_metadata.obs_size, dtype=jnp.float32)
    critic_observation = jnp.arange(
        checkpoint_metadata.critic_obs_size, dtype=jnp.float32
    )
    restored_policy = eqx.combine(restored.params, static_policy)
    expected_policy = eqx.combine(state.params, static_policy)
    assert_tree_equal(
        apply_policy(evaluation_state.policy, observation, critic_observation),
        apply_policy(expected_policy, observation, critic_observation),
    )
    assert_tree_equal(
        apply_policy(restored_policy, observation, critic_observation),
        apply_policy(expected_policy, observation, critic_observation),
    )
    assert int(evaluation_state.update) == 3
    assert (path / checkpoint.STATE_DIR).is_dir()
    assert (path / checkpoint.CONTINUATION_DIR).is_dir()
    assert document["git_commit"] == GIT_COMMIT
    assert document["update"] == 3
    assert document["metadata_sha256"].startswith("sha256:")
    assert document["train_layout_sha256"].startswith("sha256:")
    assert document["continuation_layout_sha256"].startswith("sha256:")


def test_checkpoint_round_trips_nonempty_task_state(tmp_path: Path) -> None:
    env = FakeTaskStateEnv()
    config = ppo_config()
    state = runner_state(config, env)._replace(
        task_state=jnp.asarray((8.5, 7.0, 3.0), dtype=jnp.float32)
    )
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "task_state_checkpoint"

    save_checkpoint(path, state, checkpoint_metadata)
    restored, _, _, _ = restore_runner_checkpoint(path, env, checkpoint_metadata)

    np.testing.assert_array_equal(restored.task_state, state.task_state)


def test_motion_checkpoint_rejects_locomotion_environment_task(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    clip = synthetic_motion_clip()
    motion_path = tmp_path / "motion.npz"
    motion_path.write_bytes(b"motion checkpoint fixture")
    motion_env = FakeMotionEnv()
    state = runner_state(config, motion_env)
    checkpoint_metadata = motion_metadata(config, motion_path, clip, state)
    path = tmp_path / "motion_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)

    restored, _, _, restored_metadata = restore_runner_checkpoint(
        path, motion_env, checkpoint_metadata
    )

    assert_tree_equal(restored, state)
    assert restored_metadata == checkpoint_metadata
    assert restored_metadata.value_norm == checkpoint.value_norm_metadata(
        restored.value_norm_state
    )
    assert restored_metadata.observation_norm == checkpoint.observation_norm_metadata(
        restored.observation_norm_state
    )
    assert (
        restored_metadata.critic_observation_norm
        == checkpoint.observation_norm_metadata(restored.critic_observation_norm_state)
    )
    np.testing.assert_array_equal(
        restored.env_state.sampling_failure_ema,
        np.broadcast_to(
            np.asarray(
                restored_metadata.motion_sampling_failure_ema,
                dtype=np.float32,
            ),
            restored.env_state.sampling_failure_ema.shape,
        ),
    )

    with pytest.raises(
        ValueError,
        match="checkpoint task 'motion_tracking'.*environment task 'locomotion'",
    ):
        restore_runner_checkpoint(path, FakeEnv(), checkpoint_metadata)


def test_motion_checkpoint_rejects_normalization_and_sampler_snapshot_tampering(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    motion_env = FakeMotionEnv()
    state = runner_state(config, motion_env)
    motion_path = tmp_path / "motion.npz"
    motion_path.write_bytes(b"motion checkpoint fixture")
    checkpoint_metadata = motion_metadata(
        config,
        motion_path,
        synthetic_motion_clip(),
        state,
    )
    path = tmp_path / "tampered_motion_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    original_document = json.loads(metadata_path.read_text(encoding="utf-8"))

    normalization_tamper = json.loads(json.dumps(original_document))
    normalization_tamper["metadata"]["observation_norm"]["mean"][0] += 0.5
    normalization_tamper["metadata_sha256"] = checkpoint._fingerprint(
        normalization_tamper["metadata"]
    )
    metadata_path.write_text(json.dumps(normalization_tamper), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="normalization state does not match.*observation_norm_state",
    ):
        restore_evaluation_checkpoint(path)

    sampler_tamper = json.loads(json.dumps(original_document))
    sampler_tamper["metadata"]["motion_sampling_failure_ema"][0] += 0.125
    sampler_tamper["metadata_sha256"] = checkpoint._fingerprint(
        sampler_tamper["metadata"]
    )
    metadata_path.write_text(json.dumps(sampler_tamper), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="adaptive-sampler state does not match its metadata snapshot",
    ):
        restore_runner_checkpoint(path, motion_env, checkpoint_metadata)


def test_motion_checkpoint_without_observation_latency_loads_as_disabled(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    motion_env = FakeMotionEnv()
    state = runner_state(config, motion_env)
    motion_path = tmp_path / "motion.npz"
    motion_path.write_bytes(b"motion checkpoint fixture")
    checkpoint_metadata = motion_metadata(
        config,
        motion_path,
        synthetic_motion_clip(),
        state,
    )
    path = tmp_path / "legacy_motion_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["motion_env"].pop("observation_latency")
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert not restored.motion_env.observation_latency.enabled
    assert restored.motion_env.observation_latency.delay_steps == (0, 0)


def test_motion_checkpoint_without_split_ppo_fields_loads_historical_defaults(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    motion_env = FakeMotionEnv()
    state = runner_state(config, motion_env)
    motion_path = tmp_path / "motion.npz"
    motion_path.write_bytes(b"motion checkpoint fixture")
    checkpoint_metadata = motion_metadata(
        config,
        motion_path,
        synthetic_motion_clip(),
        state,
    )
    path = tmp_path / "legacy_motion_ppo_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name in (
        "adam_epsilon",
        "value_loss_scale",
        "normalize_critic_observations",
        "observation_norm_update_mode",
        "observation_norm_update_updates",
        "observation_norm_epsilon_mode",
    ):
        document["metadata"]["ppo"].pop(name)
    document["metadata"]["ppo"]["policy"].pop("architecture")
    document["metadata"]["ppo"]["policy"].pop("layer_norm_final_hidden")
    document["metadata"]["ppo"]["policy"].pop("sonic_lora_rank", None)
    document["metadata"]["ppo"]["policy"].pop("sonic_lora_alpha", None)
    document["metadata"]["ppo"]["policy"].pop("sonic_train_log_std", None)
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert restored.ppo.adam_epsilon == pytest.approx(1.0e-5)
    assert restored.ppo.value_loss_scale == pytest.approx(0.5)
    assert restored.ppo.normalize_critic_observations
    assert restored.ppo.observation_norm_update_mode == "online"
    assert restored.ppo.observation_norm_update_updates is None
    assert restored.ppo.observation_norm_epsilon_mode == "variance_floor"
    assert restored.ppo.policy.architecture == "mlp"
    assert not restored.ppo.policy.layer_norm_final_hidden
    assert restored.ppo.policy.sonic_lora_rank == 32
    assert restored.ppo.policy.sonic_lora_alpha == pytest.approx(32.0)
    assert restored.ppo.policy.sonic_train_log_std


def test_locomotion_checkpoint_without_observation_latency_loads_as_disabled(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_locomotion_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["env"].pop("observation_latency")
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert not restored.env.observation_latency.enabled
    assert restored.env.observation_latency.delay_steps == (0, 0)


def test_locomotion_checkpoint_without_split_ppo_fields_loads_historical_defaults(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_locomotion_ppo_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name in (
        "adam_epsilon",
        "value_loss_scale",
        "normalize_critic_observations",
        "observation_norm_update_mode",
        "observation_norm_update_updates",
        "observation_norm_epsilon_mode",
        "symmetry_loss_coef",
    ):
        document["metadata"]["ppo"].pop(name)
    document["metadata"]["ppo"]["policy"].pop("architecture")
    document["metadata"]["ppo"]["policy"].pop("layer_norm_final_hidden")
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert restored.ppo.adam_epsilon == pytest.approx(1.0e-5)
    assert restored.ppo.value_loss_scale == pytest.approx(0.5)
    assert restored.ppo.normalize_critic_observations
    assert restored.ppo.observation_norm_update_mode == "online"
    assert restored.ppo.observation_norm_update_updates is None
    assert restored.ppo.observation_norm_epsilon_mode == "variance_floor"
    assert restored.ppo.symmetry_loss_coef == 0.0
    assert restored.ppo.policy.architecture == "mlp"
    assert not restored.ppo.policy.layer_norm_final_hidden


def test_checkpoint_without_symmetry_loss_loads_with_loss_disabled(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_checkpoint_without_symmetry_loss"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["ppo"].pop("symmetry_loss_coef")
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert restored.ppo.symmetry_loss_coef == 0.0


def test_locomotion_checkpoint_without_joint_force_reward_loads_as_disabled(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_locomotion_joint_force_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    for name in (
        "action_acceleration",
        "joint_force_peak",
        "touchdown_impact",
        "stand_still",
        "stand_contact",
        "stand_foot_heading",
    ):
        document["metadata"]["reward"]["weights"].pop(name)
    document["metadata"]["reward"].pop("smoothness")
    document["metadata"]["reward"]["tracking"].pop("ang_vel_z_lpf_cutoff_hz")
    document["metadata"]["reward"]["standing"].pop("foot_heading_deadband")
    document["metadata"]["reward"]["standing"].pop("foot_heading_scale")
    for name in (
        "joint_force_soft_limit_body_weights",
        "joint_force_top_k",
        "joint_force_apply_while_standing",
        "touchdown_force_soft_limit_body_weights",
        "touchdown_vertical_speed_scale",
    ):
        document["metadata"]["reward"]["gait"].pop(name)
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    assert restored.reward.weights.action_acceleration == 0.0
    assert restored.reward.weights.joint_force_peak == 0.0
    assert restored.reward.weights.touchdown_impact == 0.0
    assert restored.reward.weights.stand_still == 0.0
    assert restored.reward.weights.stand_contact == 0.0
    assert restored.reward.weights.stand_foot_heading == 0.0
    assert restored.reward.gait.joint_force_soft_limit_body_weights == 1.0
    assert restored.reward.gait.joint_force_top_k == 12
    assert not restored.reward.gait.joint_force_apply_while_standing
    assert restored.reward.smoothness.ankle_action_weight == 1.0
    assert restored.reward.smoothness.waist_action_weight == 1.0
    assert restored.reward.tracking.ang_vel_z_lpf_cutoff_hz == pytest.approx(5.0)
    assert restored.reward.standing.foot_heading_deadband == pytest.approx(
        np.deg2rad(5.0)
    )
    assert restored.reward.standing.foot_heading_scale == pytest.approx(
        np.deg2rad(15.0)
    )


def test_locomotion_checkpoint_without_new_randomization_fields_loads_as_noops(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_locomotion_randomization_checkpoint"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    randomization = document["metadata"]["env"]["model_randomization"]
    for name in (
        "added_base_mass_range",
        "base_com_offset_range",
        "friction_range",
        "actuator_strength_range",
        "gravity_fraction_range",
    ):
        randomization.pop(name)
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    _, restored, _ = restore_evaluation_checkpoint(path)

    randomization_config = restored.env.model_randomization
    assert randomization_config.added_base_mass_range == (0.0, 0.0)
    assert randomization_config.base_com_offset_range == (0.0, 0.0)
    assert randomization_config.friction_range == (1.0, 1.0)
    assert randomization_config.actuator_strength_range == (1.0, 1.0)
    assert randomization_config.gravity_fraction_range == (0.0, 0.0)


def test_locomotion_checkpoint_without_gait_phase_contract_is_rejected(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    env = FakeEnv()
    state = runner_state(config, env)
    checkpoint_metadata = metadata(config, tmp_path)
    path = tmp_path / "legacy_locomotion_without_gait_phase"
    save_checkpoint(path, state, checkpoint_metadata)
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["metadata"]["env"].pop("gait")
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="predates the gait-phase observation"):
        restore_evaluation_checkpoint(path)


def test_motion_contract_allows_relocation_and_rejects_content_or_schema_tampering(
    tmp_path: Path,
) -> None:
    clip = synthetic_motion_clip()
    original_path = tmp_path / "original" / "motion.npz"
    original_path.parent.mkdir()
    original_path.write_bytes(b"stable motion bytes")
    contract = checkpoint.motion_clip_contract(original_path, clip)
    relocated_path = tmp_path / "relocated" / "motion.npz"
    relocated_path.parent.mkdir()
    relocated_path.write_bytes(original_path.read_bytes())

    relocated_contract = checkpoint.validate_motion_clip_contract(
        relocated_path, clip, contract
    )

    assert relocated_contract == replace(contract, path=relocated_path)

    relocated_path.write_bytes(b"tampered motion bytes")
    with pytest.raises(ValueError, match="content_sha256"):
        checkpoint.validate_motion_clip_contract(relocated_path, clip, contract)

    relocated_path.write_bytes(original_path.read_bytes())
    with pytest.raises(ValueError, match="fps"):
        checkpoint.validate_motion_clip_contract(
            relocated_path,
            replace(clip, fps=60.0),
            contract,
        )


def test_checkpoint_round_trips_real_mjx_continuation_state(
    tmp_path: Path, full_body_asset: PreparedAsset
) -> None:
    config = ppo_config()
    asset_config = AssetPrepConfig()
    env_config = BelloEnvConfig()
    reward_config = RewardConfig()
    command_config = CommandConfig()
    done_config = DoneConfig()
    warp_config = WarpConfig()
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        batch_size=config.num_envs,
        env_config=env_config,
        reward_config=reward_config,
        command_config=command_config,
        done_config=done_config,
        warp=warp_config,
    )
    state, _, _ = init_runner_state(jax.random.PRNGKey(7), env, config)
    checkpoint_metadata = LocomotionCheckpointMetadata(
        ppo=config,
        asset=asset_config,
        reward=reward_config,
        command=command_config,
        done=done_config,
        env=env_config,
        warp=warp_config,
        obs_size=policy_observation_size(env.observation_size, config.policy),
        critic_obs_size=policy_observation_size(
            env.critic_observation_size, config.policy
        ),
        action_size=env.action_size,
        actor_observation_layout=env.observation_layout,
        critic_observation_layout=env.critic_observation_layout,
        action_layout=env.action_layout,
    )
    path = tmp_path / "mjx_checkpoint"

    save_checkpoint(path, state, checkpoint_metadata)
    restored, _, _, restored_metadata = restore_runner_checkpoint(
        path, env, checkpoint_metadata
    )

    assert_tree_equal(restored, state)
    assert restored_metadata == checkpoint_metadata
    assert restored.env_state.data.act.shape == (config.num_envs, 0)


def test_checkpoint_config_and_timestamped_name() -> None:
    config = CheckpointConfig(
        name="run", every_updates=25, resume_path=Path("previous/update_000025")
    )

    assert (
        timestamped_checkpoint_name(
            config,
            PPOConfig(seed=7),
            timestamp=datetime(2026, 7, 12, 1, 30, 45, 123456),
        )
        == "run_20260712_013045_123456"
    )
    with pytest.raises(ValueError, match="path component"):
        validate_checkpoint_options(CheckpointConfig(name="../run"))
    with pytest.raises(ValueError, match="requires resume_path"):
        validate_checkpoint_options(CheckpointConfig(allow_git_commit_mismatch=True))
    validate_checkpoint_options(
        CheckpointConfig(
            warm_start_path=Path("previous/final"),
            allow_git_commit_mismatch=True,
        )
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_checkpoint_options(
            CheckpointConfig(
                resume_path=Path("previous/update_000025"),
                warm_start_path=Path("previous/final"),
            )
        )


def test_checkpoint_metadata_upgrade_adds_optional_critic_learning_rate() -> None:
    legacy = {"task": "twist2", "ppo": {"learning_rate": 3.0e-4}}

    upgraded = checkpoint._with_checkpoint_metadata_defaults(legacy)

    assert upgraded["ppo"] == {
        "learning_rate": 3.0e-4,
        "critic_learning_rate": None,
        "symmetry_loss_coef": 0.0,
    }
    assert "critic_learning_rate" not in legacy["ppo"]


def test_repository_commit_requires_origin_tracked_clean_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_git(tmp_path, "init", "-b", "main")
    run_git(tmp_path, "config", "user.name", "Checkpoint Test")
    run_git(tmp_path, "config", "user.email", "checkpoint@example.com")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("checkpoints/\n", encoding="utf-8")
    stack_source = tmp_path.parent / f"{tmp_path.name}-stack"
    stack_source.mkdir()
    run_git(stack_source, "init", "-b", "main")
    run_git(stack_source, "config", "user.name", "Checkpoint Test")
    run_git(stack_source, "config", "user.email", "checkpoint@example.com")
    (stack_source / "description.txt").write_text("pinned\n", encoding="utf-8")
    run_git(stack_source, "add", "description.txt")
    run_git(stack_source, "commit", "-m", "description")
    run_git(
        tmp_path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(stack_source),
        "deps/bello-stack",
    )
    run_git(tmp_path, "add", "tracked.txt", ".gitignore", ".gitmodules")
    run_git(tmp_path, "commit", "-m", "initial")
    run_git(
        tmp_path,
        "remote",
        "add",
        "origin",
        "https://github.com/MotomindKR/bello_mujoco.git",
    )
    run_git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    monkeypatch.setattr(source, "repository_root", lambda: tmp_path)

    commit = source.capture_repository_commit()

    assert commit == run_git(tmp_path, "rev-parse", "HEAD")
    assert source.require_tracked_repository_path(tracked) == Path("tracked.txt")
    assert source.resolve_repository_path(Path("tracked.txt")) == tracked
    assert source.require_checkpoint_output_directory(tmp_path.parent) == (
        tmp_path.parent
    )
    assert source.require_checkpoint_output_directory(tmp_path / "checkpoints") == (
        tmp_path / "checkpoints"
    )
    with pytest.raises(ValueError, match="must be Git-ignored"):
        source.require_checkpoint_output_directory(tmp_path / "results")

    untracked = tmp_path / "untracked.txt"
    untracked.write_text("not committed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        source.capture_repository_commit()
    with pytest.raises(ValueError, match="must be tracked in HEAD"):
        source.require_tracked_repository_path(untracked)

    untracked.unlink()
    (tmp_path / "deps/bello-stack/description.txt").write_text(
        "dirty submodule\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        source.capture_repository_commit()
    run_git(tmp_path / "deps/bello-stack", "restore", "description.txt")
    tracked.write_text("second\n", encoding="utf-8")
    run_git(tmp_path, "add", "tracked.txt")
    run_git(tmp_path, "commit", "-m", "second")
    with pytest.raises(ValueError, match="source commit does not match"):
        source.require_repository_commit(commit)
    with pytest.warns(RuntimeWarning, match="mismatch override enabled"):
        source.require_repository_commit(commit, allow_mismatch=True)
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        source.require_repository_commit(commit, allow_mismatch=True)


def test_checkpoint_failure_never_publishes_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeEnv()
    config = ppo_config()
    final_path = tmp_path / "failed"
    save_calls = 0
    closed = False

    class FailingCheckpointer:
        def save(self, path: Path, item: Any, *, force: bool) -> None:
            nonlocal save_calls
            del item, force
            save_calls += 1
            path.mkdir(parents=True)
            if save_calls == 2:
                (path / "partial").write_text("partial", encoding="utf-8")
                raise RuntimeError("simulated continuation failure")

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FailingCheckpointer()
    )

    with pytest.raises(RuntimeError, match="simulated continuation failure"):
        save_checkpoint(
            final_path, runner_state(config, env), metadata(config, tmp_path)
        )

    assert save_calls == 2
    assert closed
    assert not final_path.exists()
    assert list(tmp_path.glob(".failed.tmp-*")) == []


def test_checkpoint_refuses_existing_path(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    path.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(
            path,
            runner_state(ppo_config(), FakeEnv()),
            metadata(ppo_config(), tmp_path),
        )


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    [
        (
            lambda document: document.pop("git_commit"),
            "predates embedded Git provenance",
        ),
        (
            lambda document: document.update(git_commit="not-a-commit"),
            "git_commit must contain exactly 40",
        ),
        (
            lambda document: document.update(unexpected=True),
            "unknown field.*unexpected",
        ),
        (
            lambda document: document.update(metadata_sha256="not-a-hash"),
            "metadata_sha256 must have the form",
        ),
        (
            lambda document: document["metadata"]["env"].update(decimation=7),
            "metadata fingerprint does not match",
        ),
    ],
)
def test_metadata_envelope_and_fingerprint_are_strict(
    tmp_path: Path, mutation, error_match: str
) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    mutation(document)
    path = tmp_path / "invalid"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match=error_match):
        load_checkpoint_metadata(path)


def test_metadata_rejects_unknown_resolved_config_key(tmp_path: Path) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    document["metadata"]["env"]["not_a_field"] = True
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    path = tmp_path / "unknown_config"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match=r"unknown config key.*metadata\.env"):
        load_checkpoint_metadata(path)


def test_evaluation_mismatch_override_migrates_retired_sphere_asset_field(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    expected_metadata = metadata(config, tmp_path)
    document = checkpoint_document(expected_metadata, runner_state(config, FakeEnv()))
    document["git_commit"] = "2" * 40
    document["metadata"]["asset"]["end_effector_sphere_mode"] = "both"
    document["metadata"]["asset"]["joint_availability"].update(
        waist_roll=False,
        waist_pitch=False,
        neck_yaw=False,
        head_pitch=False,
    )
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    path = tmp_path / "retired_sphere_asset_field"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match="source commit does not match"):
        load_checkpoint_metadata(path)

    restored = load_checkpoint_metadata(path, allow_git_commit_mismatch=True)

    assert restored.asset == AssetPrepConfig()
    with pytest.raises(ValueError, match=r"unknown config key.*metadata\.asset"):
        restore_runner_checkpoint(
            path,
            FakeEnv(),
            expected_metadata,
            allow_git_commit_mismatch=True,
        )


def test_evaluation_mismatch_override_validates_retired_sphere_asset_field(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    document["git_commit"] = "2" * 40
    document["metadata"]["asset"]["end_effector_sphere_mode"] = "left"
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    path = tmp_path / "invalid_retired_sphere_asset_field"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match="end_effector_sphere_mode must be"):
        load_checkpoint_metadata(path, allow_git_commit_mismatch=True)


def test_evaluation_mismatch_override_rejects_enabled_retired_joint(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    document["git_commit"] = "2" * 40
    document["metadata"]["asset"]["joint_availability"]["waist_roll"] = True
    document["metadata_sha256"] = checkpoint._fingerprint(document["metadata"])
    path = tmp_path / "enabled_retired_joint"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match="waist_roll must be false"):
        load_checkpoint_metadata(path, allow_git_commit_mismatch=True)


def test_checkpoint_checks_git_commit_before_revision_private_fields(
    tmp_path: Path,
    exact_source_checkout,
) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    document["git_commit"] = "2" * 40
    document["future_revision_field"] = True
    path = tmp_path / "source_mismatch"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match="source commit does not match"):
        load_checkpoint_metadata(path)

    with pytest.raises(ValueError, match="unknown field.*future_revision_field"):
        load_checkpoint_metadata(path, allow_git_commit_mismatch=True)


def test_checkpoint_rechecks_source_before_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ppo_config()
    checkpoint_metadata = metadata(config, tmp_path)
    state = runner_state(config, FakeEnv())
    commits = iter((GIT_COMMIT, "2" * 40))
    monkeypatch.setattr(checkpoint, "capture_repository_commit", lambda: next(commits))

    with pytest.raises(RuntimeError, match="commit changed after training"):
        save_checkpoint(tmp_path / "source_changed", state, checkpoint_metadata)
    assert not (tmp_path / "source_changed").exists()


def test_resume_allows_budget_extension_and_rejects_semantic_changes(
    tmp_path: Path,
) -> None:
    env = FakeEnv()
    config = ppo_config()
    state = runner_state(config, env)
    recorded_metadata = metadata(config, tmp_path)
    path = tmp_path / "resume"
    save_checkpoint(path, state, recorded_metadata)
    extended = replace(
        recorded_metadata,
        ppo=replace(config, total_timesteps=config.total_timesteps * 2),
    )

    restored, _, _, effective_metadata = restore_runner_checkpoint(path, env, extended)

    assert int(restored.update) == int(state.update)
    assert effective_metadata.ppo == extended.ppo
    changed_reward = replace(
        extended,
        reward=replace(
            extended.reward,
            weights=replace(
                extended.reward.weights,
                track_lin_vel_xy=3.0,
            ),
        ),
    )
    with pytest.raises(ValueError, match="resume checkpoint contract.*reward"):
        restore_runner_checkpoint(path, env, changed_reward)
    too_short = replace(
        recorded_metadata,
        ppo=replace(config, total_timesteps=config.num_envs * config.num_steps * 2),
    )
    with pytest.raises(ValueError, match="budget ends before.*2 < 3"):
        restore_runner_checkpoint(path, env, too_short)


def test_checkpoint_rejects_operational_asset_output_root(tmp_path: Path) -> None:
    config = ppo_config()
    noncanonical = replace(
        metadata(config, tmp_path),
        asset=AssetPrepConfig(output_root=tmp_path / "cache"),
    )

    with pytest.raises(ValueError, match="output_root.*must be normalized"):
        checkpoint_document(noncanonical, runner_state(config, FakeEnv()))


def test_fixed_tuple_coercion_rejects_truncation() -> None:
    with pytest.raises(ValueError, match="pair must contain exactly 2 values, got 3"):
        checkpoint.coerce_value(tuple[int, int], [1, 2, 3], path="pair")


@pytest.mark.parametrize(
    ("fingerprint_field", "restore_runner", "error_match"),
    [
        ("train_layout_sha256", False, "evaluation state layout does not match"),
        (
            "continuation_layout_sha256",
            True,
            "continuation state layout does not match",
        ),
    ],
)
def test_restore_rejects_layout_fingerprint_before_affected_orbax_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fingerprint_field: str,
    restore_runner: bool,
    error_match: str,
) -> None:
    env = FakeEnv()
    config = ppo_config()
    state = runner_state(config, env)
    document = checkpoint_document(metadata(config, tmp_path), state)
    document[fingerprint_field] = "sha256:" + "0" * 64
    path = tmp_path / "layout_mismatch"
    write_checkpoint_document(path, document)
    restore_calls = 0
    closed = False

    class FakeCheckpointer:
        def restore(self, path: Path, *, item: Any, restore_args: Any) -> Any:
            nonlocal restore_calls
            del path, restore_args
            restore_calls += 1
            return item

        def close(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FakeCheckpointer()
    )
    monkeypatch.setattr(
        checkpoint.ocp.checkpoint_utils,
        "construct_restore_args",
        lambda item: item,
    )

    with pytest.raises(ValueError, match=error_match):
        if restore_runner:
            restore_runner_checkpoint(path, env, metadata(config, tmp_path))
        else:
            restore_evaluation_checkpoint(path)

    assert restore_calls == (1 if restore_runner else 0)
    assert closed == restore_runner


def test_save_and_load_reject_updates_outside_training_range(
    tmp_path: Path,
) -> None:
    env = FakeEnv()
    config = ppo_config()
    checkpoint_metadata = metadata(config, tmp_path)
    state = runner_state(config, env)

    with pytest.raises(ValueError, match="outside the valid range"):
        save_checkpoint(
            tmp_path / "invalid_save",
            state._replace(update=jnp.asarray(config.num_updates + 1)),
            checkpoint_metadata,
        )
    assert not (tmp_path / "invalid_save").exists()

    document = checkpoint_document(checkpoint_metadata, state)
    document["update"] = config.num_updates + 1
    invalid_load = tmp_path / "invalid_load"
    write_checkpoint_document(invalid_load, document)
    with pytest.raises(ValueError, match="outside the valid range"):
        load_checkpoint_metadata(invalid_load)


def test_restore_rejects_state_update_that_disagrees_with_metadata(
    tmp_path: Path,
) -> None:
    env = FakeEnv()
    config = ppo_config()
    path = tmp_path / "mismatch"
    save_checkpoint(path, runner_state(config, env), metadata(config, tmp_path))
    metadata_path = path / checkpoint.METADATA_FILE
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    document["update"] = 2
    metadata_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match recorded update"):
        restore_evaluation_checkpoint(path)


def test_runner_restore_rejects_environment_contract_mismatch(
    tmp_path: Path,
) -> None:
    env = FakeEnv()
    config = ppo_config()
    document = checkpoint_document(
        replace(
            metadata(config, tmp_path),
            action_layout=ActionLayout(("other_0", "other_1")),
        ),
        runner_state(config, env),
    )
    path = tmp_path / "environment_mismatch"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match="action layout"):
        restore_runner_checkpoint(
            path,
            env,
            replace(
                metadata(config, tmp_path),
                action_layout=ActionLayout(("other_0", "other_1")),
            ),
        )
