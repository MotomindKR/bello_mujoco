import jax
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import jax.numpy as jnp
import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from bello_mujoco import deployment
from bello_mujoco.deployment import actor_onnx
from bello_mujoco.envs.actions import bounded_motor_target
from bello_mujoco.policy import ActorCritic, ActorCriticConfig
from bello_mujoco.training.ppo import (
    PPOConfig,
    ObservationNormState,
    normalize_observation,
)


@pytest.mark.parametrize("activation", ["tanh", "relu", "elu", "swish"])
@pytest.mark.parametrize("distribution", ["gaussian", "tanh_gaussian"])
@pytest.mark.parametrize(
    "normalization_mode", ["disabled", "variance_floor", "std_offset"]
)
def test_onnx_matches_actor_normalization_and_motor_mapping(
    activation, distribution, normalization_mode
):
    config = PPOConfig(
        policy=ActorCriticConfig(
            hidden_dims=(8, 6), activation=activation, action_distribution=distribution
        ),
        normalize_observations=normalization_mode != "disabled",
        observation_norm_epsilon_mode="std_offset"
        if normalization_mode == "std_offset"
        else "variance_floor",
    )
    actor = ActorCritic(5, 2, config.policy, key=jax.random.PRNGKey(1))
    norm = ObservationNormState(
        mean=jnp.arange(5) * 0.1, var=jnp.arange(5) * 0.2, count=jnp.asarray(10.0)
    )
    default = jnp.array([0.2, -0.3, 0.4])
    limits = jnp.array([[-1.0, 2.0], [-3.0, 1.0], [-1.0, 1.0]])
    indices = jnp.array([0, 1])
    metadata = {"git_commit": "a" * 40}
    graph = actor_onnx(
        actor,
        norm,
        config,
        default_ctrl=default,
        ctrl_limits=limits,
        policy_indices=indices,
        action_clip=1.0,
        metadata=metadata,
    )
    runtime = ReferenceEvaluator(graph)
    for observation in np.random.default_rng(5).normal(size=(3, 5)).astype(np.float32):
        action, motors = runtime.run(None, {"observation": observation[None]})
        expected_action = actor.action_mean(
            normalize_observation(jnp.asarray(observation), norm, config)
        )
        expected_motor = bounded_motor_target(
            expected_action, default, limits, indices, 1.0
        )
        np.testing.assert_allclose(action[0], expected_action, atol=1e-6)
        np.testing.assert_allclose(motors[0], expected_motor, atol=1e-6)


def test_onnx_effort_mapping_clips_only_motor_targets():
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(4,)))
    actor = ActorCritic(3, 2, config.policy, key=jax.random.PRNGKey(5))
    norm = ObservationNormState(jnp.zeros(3), jnp.ones(3), jnp.asarray(1.0))
    graph = actor_onnx(
        actor,
        norm,
        config,
        default_ctrl=[0.2, 0.3, 0.4],
        ctrl_limits=[[-1, 1]] * 3,
        policy_indices=[0, 2],
        action_clip=1.0,
        metadata={},
        effort_scale=[4.0, 6.0],
    )
    obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    action, target = ReferenceEvaluator(graph).run(None, {"observation": obs})
    expected = np.asarray([0.2, 0.3, 0.4])
    expected[[0, 2]] += action[0] * [4.0, 6.0]
    np.testing.assert_allclose(target[0], np.clip(expected, -1.0, 1.0), atol=1e-6)


def test_policy_publication_does_not_overwrite_or_leave_staging_files(tmp_path):
    output = tmp_path / "policy.onnx"
    deployment._publish_policy(output, b"complete model")
    with pytest.raises(FileExistsError):
        deployment._publish_policy(output, b"replacement")
    assert output.read_bytes() == b"complete model"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("with_motion", [False, True])
def test_bundle_contains_policy_and_only_required_reference(with_motion):
    motion = SimpleNamespace(
        fps=50.0,
        joint_names=("hinge",),
        body_names=("body",),
        qpos=np.zeros((2, 8)),
        qvel=np.zeros((2, 7)),
        body_pos=np.zeros((2, 1, 3)),
        body_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1)),
        body_lin_vel=np.zeros((2, 1, 3)),
        body_ang_vel=np.zeros((2, 1, 3)),
        foot_contacts=np.zeros((2, 2)),
    )
    content = deployment.deployment_bundle(
        b"actor", motion=motion if with_motion else None
    )
    with ZipFile(BytesIO(content)) as archive:
        assert archive.namelist() == ["policy.onnx"] + (
            ["motion.npz"] if with_motion else []
        )
        assert archive.read("policy.onnx") == b"actor"
        if with_motion:
            with np.load(
                BytesIO(archive.read("motion.npz")), allow_pickle=False
            ) as clip:
                np.testing.assert_array_equal(clip["qpos"], motion.qpos)
                assert float(clip["fps"]) == motion.fps


@pytest.mark.parametrize("failure_call", [1, 2])
def test_policy_publication_reports_file_and_directory_sync_failures(
    tmp_path, monkeypatch, failure_call
):
    calls = 0

    def fail_sync(_descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("durability failed")

    monkeypatch.setattr(deployment.os, "fsync", fail_sync)
    output = tmp_path / "policy.onnx"
    with pytest.raises(OSError, match="durability failed"):
        deployment._publish_policy(output, b"complete model")
    if failure_call == 1:
        assert not output.exists()
    else:
        assert output.read_bytes() == b"complete model"
    assert not list(tmp_path.glob(".policy.onnx.*"))
