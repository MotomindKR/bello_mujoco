from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import bello_mujoco.checkpoint as checkpoint
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.checkpoint import (
    CheckpointConfig,
    CheckpointMetadata,
    load_checkpoint_metadata,
    restore_runner_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
    timestamped_checkpoint_name,
)
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    BelloState,
    DoneConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.policy import (
    ActorCriticConfig,
    apply_policy,
    policy_observation_size,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.train.ppo import (
    PPOConfig,
    RunnerState,
    init_runner_state,
)

ACTOR_LAYOUT = ObservationLayout((("actor_observation", 4),))
CRITIC_LAYOUT = ObservationLayout((("critic_observation", 5),))
ACTION_LAYOUT = ActionLayout(("action_0", "action_1"))


class FakeEnv:
    config = BelloEnvConfig()
    action_clip = 1.0
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
            foot_air_time=foot_zeros,
            foot_contact_time=foot_zeros,
            foot_contact_debounce_time=foot_zeros,
            last_foot_contact=jnp.zeros((batch_size, 2), dtype=bool),
            command=jnp.zeros((batch_size, 3), dtype=jnp.float32),
            gait_frequency=jnp.ones((batch_size,), dtype=jnp.float32),
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


def ppo_config() -> PPOConfig:
    return PPOConfig(
        seed=11,
        policy=ActorCriticConfig(
            hidden_size=8,
            hidden_depth=1,
            obs_stack_size=2,
        ),
        total_timesteps=24,
        num_envs=2,
        num_steps=2,
        num_minibatches=1,
    )


def metadata(config: PPOConfig, tmp_path: Path) -> CheckpointMetadata:
    return CheckpointMetadata(
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
    )


def checkpoint_document(
    metadata: CheckpointMetadata, state: RunnerState
) -> dict[str, Any]:
    return checkpoint._checkpoint_document(
        metadata,
        checkpoint._train_checkpoint_item(state),
        checkpoint._continuation_checkpoint_item(state),
        checkpoint.capture_runtime_fingerprints(metadata),
    )


def save_checkpoint(
    path: Path, state: RunnerState, metadata: CheckpointMetadata
) -> None:
    save_training_checkpoint(
        path,
        state,
        metadata,
        checkpoint.capture_runtime_fingerprints(metadata),
    )


def write_checkpoint_document(path: Path, document: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / checkpoint.METADATA_FILE).write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )


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
    train_state, eval_metadata = restore_training_checkpoint(path)
    restored, static_policy, optimizer, resume_metadata = restore_runner_checkpoint(
        path, env
    )
    document = json.loads((path / checkpoint.METADATA_FILE).read_text(encoding="utf-8"))

    assert_tree_equal(restored, state)
    assert eval_metadata == resume_metadata == checkpoint_metadata
    assert load_checkpoint_metadata(path) == checkpoint_metadata
    assert callable(optimizer.init)
    observation = jnp.arange(checkpoint_metadata.obs_size, dtype=jnp.float32)
    critic_observation = jnp.arange(
        checkpoint_metadata.critic_obs_size, dtype=jnp.float32
    )
    restored_policy = eqx.combine(restored.params, static_policy)
    expected_policy = eqx.combine(state.params, static_policy)
    assert_tree_equal(
        apply_policy(train_state.policy, observation, critic_observation),
        apply_policy(expected_policy, observation, critic_observation),
    )
    assert_tree_equal(
        apply_policy(restored_policy, observation, critic_observation),
        apply_policy(expected_policy, observation, critic_observation),
    )
    assert int(train_state.update) == 3
    assert (path / checkpoint.STATE_DIR).is_dir()
    assert (path / checkpoint.CONTINUATION_DIR).is_dir()
    assert document["schema_version"] == 5
    assert document["kind"] == checkpoint.CHECKPOINT_KIND
    assert document["update"] == 3
    assert document["config_fingerprint"].startswith("sha256:")
    assert document["train_layout_fingerprint"].startswith("sha256:")
    assert document["continuation_layout_fingerprint"].startswith("sha256:")
    assert set(document["runtime_fingerprints"]) == {
        "source_assets",
        "prepared_assets",
        "simulation_scene",
        "viewer_scene",
        "implementation",
        "dependency_lock",
        "active_distributions",
    }


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
    checkpoint_metadata = CheckpointMetadata(
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
    restored, _, _, restored_metadata = restore_runner_checkpoint(path, env)

    assert_tree_equal(restored, state)
    assert restored_metadata == checkpoint_metadata
    assert restored.env_state.data.act.shape == (config.num_envs, 0)


def test_checkpoint_config_and_timestamped_name() -> None:
    config = CheckpointConfig(
        name="run", every_updates=25, resume_path=Path("previous/update_000025")
    )

    assert config.every_updates == 25
    assert config.resume_path == Path("previous/update_000025")
    assert (
        timestamped_checkpoint_name(
            config,
            PPOConfig(seed=7),
            timestamp=datetime(2026, 7, 12, 1, 30, 45, 123456),
        )
        == "run_20260712_013045_123456"
    )


def test_checkpoint_failure_never_publishes_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = FakeEnv()
    config = ppo_config()
    final_path = tmp_path / "failed"
    save_calls = 0

    class FailingCheckpointer:
        def save(self, path: Path, item: Any, *, force: bool) -> None:
            nonlocal save_calls
            del item, force
            save_calls += 1
            path.mkdir(parents=True)
            if save_calls == 2:
                (path / "partial").write_text("partial", encoding="utf-8")
                raise RuntimeError("simulated continuation failure")

    monkeypatch.setattr(
        checkpoint.ocp, "PyTreeCheckpointer", lambda: FailingCheckpointer()
    )

    with pytest.raises(RuntimeError, match="simulated continuation failure"):
        save_checkpoint(
            final_path, runner_state(config, env), metadata(config, tmp_path)
        )

    assert save_calls == 2
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
        (lambda document: document.update(schema_version=2), "unsupported.*version 2"),
        (
            lambda document: document.update(kind="evaluation"),
            "unsupported checkpoint kind",
        ),
        (
            lambda document: document.update(unexpected=True),
            "unknown field.*unexpected",
        ),
        (
            lambda document: document.update(config_fingerprint="not-a-hash"),
            "config_fingerprint must have the form",
        ),
        (
            lambda document: document["metadata"]["env"].update(decimation=7),
            "config fingerprint does not match",
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
    document["config_fingerprint"] = checkpoint._fingerprint(document["metadata"])
    path = tmp_path / "unknown_config"
    write_checkpoint_document(path, document)

    with pytest.raises(ValueError, match=r"unknown config key.*metadata\.env"):
        load_checkpoint_metadata(path)


def test_checkpoint_rejects_runtime_fingerprint_mismatch(tmp_path: Path) -> None:
    config = ppo_config()
    document = checkpoint_document(
        metadata(config, tmp_path), runner_state(config, FakeEnv())
    )
    document["runtime_fingerprints"]["implementation"] = "sha256:" + "0" * 64
    path = tmp_path / "runtime_mismatch"
    write_checkpoint_document(path, document)

    with pytest.raises(
        ValueError, match="runtime fingerprint mismatch.*implementation"
    ):
        load_checkpoint_metadata(path)


def test_checkpoint_rejects_runtime_inputs_changed_after_training_start(
    tmp_path: Path,
) -> None:
    config = ppo_config()
    checkpoint_metadata = metadata(config, tmp_path)
    state = runner_state(config, FakeEnv())
    startup_runtime = checkpoint.capture_runtime_fingerprints(checkpoint_metadata)
    stale_runtime = replace(
        startup_runtime,
        implementation="sha256:" + "0" * 64,
    )

    with pytest.raises(RuntimeError, match="runtime inputs changed.*implementation"):
        checkpoint._checkpoint_document(
            checkpoint_metadata,
            checkpoint._train_checkpoint_item(state),
            checkpoint._continuation_checkpoint_item(state),
            stale_runtime,
        )


def test_active_runtime_distributions_must_match_lock(monkeypatch) -> None:
    monkeypatch.setattr(
        checkpoint,
        "_active_runtime_distribution_pairs",
        lambda: ("bello-mujoco==9.9.9",),
    )
    monkeypatch.setattr(
        checkpoint,
        "_locked_distribution_versions",
        lambda: {"bello-mujoco": "0.1.0"},
    )

    with pytest.raises(RuntimeError, match=r"bello-mujoco==9\.9\.9 \(lock: 0\.1\.0\)"):
        checkpoint._active_distributions_fingerprint()


def test_prepared_asset_fingerprint_covers_referenced_meshes(tmp_path: Path) -> None:
    asset_config = AssetPrepConfig(output_root=tmp_path / "prepared")
    initial = checkpoint.capture_asset_runtime_fingerprints(asset_config)
    prepared = checkpoint.prepare_asset(asset_config)
    mesh_path = next((prepared.urdf_path.parents[1] / "meshes").glob("*.stl"))

    mesh_path.write_bytes(mesh_path.read_bytes() + b"changed")
    changed = checkpoint.capture_asset_runtime_fingerprints(asset_config)

    assert changed.prepared_assets != initial.prepared_assets
    assert changed.simulation_scene == initial.simulation_scene


def test_runtime_scene_fingerprints_cover_resolved_terrain() -> None:
    flat = checkpoint.capture_asset_runtime_fingerprints(
        AssetPrepConfig(), checkpoint.TerrainConfig(mode="plane")
    )
    tilted = checkpoint.capture_asset_runtime_fingerprints(
        AssetPrepConfig(),
        checkpoint.TerrainConfig(
            mode="plane",
            seed=7,
            slope_angle_range=(-0.02, 0.02),
        ),
    )

    assert tilted.prepared_assets == flat.prepared_assets
    assert tilted.simulation_scene != flat.simulation_scene
    assert tilted.viewer_scene != flat.viewer_scene


@pytest.mark.integration
def test_built_wheel_contains_checkpoint_runtime_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_fingerprints = checkpoint.capture_asset_runtime_fingerprints(
        AssetPrepConfig()
    )
    dist_dir = tmp_path / "dist"
    build_env = os.environ.copy()
    build_env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        env=build_env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    wheel_path = next(dist_dir.glob("*.whl"))
    installed = tmp_path / "installed"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            str(installed),
            "--no-deps",
            str(wheel_path),
        ],
        cwd=tmp_path,
        env=build_env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "from pathlib import Path",
                    "from dataclasses import asdict",
                    "import json",
                    "import bello_mujoco.checkpoint as checkpoint",
                    f"root = Path({str(installed)!r}).resolve()",
                    "assert Path(checkpoint.__file__).resolve().is_relative_to(root)",
                    "assert checkpoint._runtime_input_bytes('pyproject.toml')",
                    "assert checkpoint._runtime_input_bytes('uv.lock')",
                    "assert checkpoint._implementation_fingerprint().startswith('sha256:')",
                    "captured = checkpoint.capture_asset_runtime_fingerprints(",
                    "    checkpoint.AssetPrepConfig()",
                    ")",
                    "print(json.dumps(asdict(captured), sort_keys=True))",
                )
            ),
        ],
        cwd=tmp_path,
        env=os.environ | {"PYTHONPATH": str(installed)},
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert json.loads(result.stdout) == checkpoint.to_builtin(source_fingerprints)


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
        ("train_layout_fingerprint", False, "training state layout does not match"),
        (
            "continuation_layout_fingerprint",
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

    class FakeCheckpointer:
        def restore(self, path: Path, *, item: Any, restore_args: Any) -> Any:
            nonlocal restore_calls
            del path, restore_args
            restore_calls += 1
            return item

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
            restore_runner_checkpoint(path, env)
        else:
            restore_training_checkpoint(path)

    assert restore_calls == (1 if restore_runner else 0)


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
        restore_training_checkpoint(path)


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
        restore_runner_checkpoint(path, env)
