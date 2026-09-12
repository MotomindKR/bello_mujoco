"""Opt-in parity against the compiled Rust consumer, without a robot or GPU."""

import json
from io import BytesIO
import os
from pathlib import Path
import subprocess
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.deployment import actor_onnx, deployment_bundle
from bello_mujoco.envs.actions import bounded_motor_target
from bello_mujoco.policy import ActorCritic, ActorCriticConfig
from bello_mujoco.training.ppo import (
    ObservationNormState,
    PPOConfig,
    normalize_observation,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def rust_policy(tmp_path):
    executable = os.environ.get("BELLO_POLICY_BIN")
    if not executable:
        pytest.skip("set BELLO_POLICY_BIN and ORT_DYLIB_PATH for Rust parity")
    executable = str(Path(executable).resolve(strict=True))

    def git(path, *args):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(path),
                "-c",
                "user.name=Runtime Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "protocol.file.allow=always",
                *args,
            ],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    source = tmp_path / "source"
    dependency = tmp_path / "dependency"
    for path in (source, dependency):
        path.mkdir()
        git(path, "init")
        git(path, "commit", "--allow-empty", "-m", "Synthetic fixture")
    git(source, "submodule", "add", str(dependency), "deps/bello-stack")
    git(source, "commit", "-am", "Pin synthetic dependency")
    commit = git(source, "rev-parse", "HEAD")
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(8,), activation="swish"))
    actor = ActorCritic(3, 2, config.policy, key=jax.random.PRNGKey(7))
    norm = ObservationNormState(jnp.array([0.1, 0.2, 0.3]), jnp.ones(3), jnp.array(3.0))
    default = jnp.array([0.2, -0.3, 0.4])
    limits = jnp.array([[-1.0, 2.0], [-3.0, 1.0], [-1.0, 1.0]])
    indices = jnp.array([0, 2])
    metadata = {
        "git_commit": commit,
        "task": "locomotion",
        "robot": "synthetic",
        "observation_size": 3,
        "observation_frame": [["synthetic", 3]],
        "observation_stack_size": 1,
        "observation_max_abs": 100.0,
        "coordinates": ["first", "held", "last"],
        "policy_indices": [0, 2],
        "default_position": default.tolist(),
        "kp": [2.0] * 3,
        "kd": [0.1] * 3,
        "control_timestep": 0.02,
    }

    def run(
        observations=(),
        *,
        metadata_updates=None,
        content=None,
        check=False,
        omit_physics=False,
    ):
        if content is None:
            graph = actor_onnx(
                actor,
                norm,
                config,
                default_ctrl=default,
                ctrl_limits=limits,
                policy_indices=indices,
                action_clip=1.0,
                metadata=metadata | (metadata_updates or {}),
            )
            content = deployment_bundle(graph.SerializeToString(), physics=b"<mujoco/>")
        if omit_physics:
            buffer = BytesIO()
            with ZipFile(BytesIO(content)) as original, ZipFile(buffer, "w") as archive:
                for name in original.namelist():
                    if name != "physics.xml":
                        archive.writestr(name, original.read(name))
            content = buffer.getvalue()
        bundle = tmp_path / "deployment.zip"
        bundle.write_bytes(content)
        return subprocess.run(
            [executable, "--policy", str(bundle), "--source", str(source)]
            + (["--check"] if check else []),
            input="".join(json.dumps(row) + "\n" for row in observations),
            text=True,
            capture_output=True,
            timeout=30,
        )

    def expected(observation):
        action = actor.action_mean(
            normalize_observation(jnp.asarray(observation), norm, config)
        )
        target = bounded_motor_target(action, default, limits, indices, 1.0)
        return action, target

    return run, expected, source


def test_rust_rejects_incomplete_bundle(rust_policy):
    run, _, _ = rust_policy
    result = run(check=True, omit_physics=True)
    assert result.returncode != 0
    assert "missing physics.xml" in result.stderr


def test_rust_bundle_inference_matches_training(rust_policy):
    run, expected, _ = rust_policy
    observations = np.random.default_rng(2).normal(size=(5, 3)).astype(np.float32)
    result = run(observations.tolist())
    assert result.returncode == 0, result.stderr
    outputs = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(outputs) == len(observations)
    for observation, output in zip(observations, outputs, strict=True):
        action, target = expected(observation)
        np.testing.assert_allclose(output["action"], action, atol=1e-6)
        np.testing.assert_allclose(output["motor_target"], target, atol=1e-6)


@pytest.mark.parametrize(
    "observation, message",
    [
        ([0.0], "observation shape mismatch"),
        ([101.0, 0.0, 0.0], "implausible observation"),
    ],
)
def test_rust_rejects_bad_live_observations(rust_policy, observation, message):
    run, _, _ = rust_policy
    result = run([observation])
    assert result.returncode != 0
    assert message in result.stderr
    assert not result.stdout


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"git_commit": "0" * 40}, "requires bello_mujoco commit"),
        ({"task": "motion_tracking"}, "missing motion.npz"),
        ({"observation_size": 4}, "invalid stacked observation dimensions"),
        (
            {"observation_size": 4, "observation_frame": [["synthetic", 4]]},
            "ONNX observation shape/type disagrees",
        ),
        ({"policy_indices": [0]}, "ONNX action shape/type disagrees"),
    ],
)
def test_rust_rejects_incomplete_or_incompatible_bundle(rust_policy, updates, message):
    run, _, _ = rust_policy
    result = run(metadata_updates=updates, check=True)
    assert result.returncode != 0
    assert message in result.stderr


def test_rust_rejects_dirty_source(rust_policy):
    run, _, source = rust_policy
    (source / "uncommitted").touch()
    result = run(check=True)
    assert result.returncode != 0
    assert "source checkout is dirty" in result.stderr


def test_rust_rejects_corrupt_bundle(rust_policy):
    run, _, _ = rust_policy
    result = run(content=b"incomplete download", check=True)
    assert result.returncode != 0
    assert not result.stdout
