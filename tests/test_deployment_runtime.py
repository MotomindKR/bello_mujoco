"""Opt-in parity against the compiled Rust consumer, without a robot or GPU."""

import json

import numpy as np
import pytest


pytestmark = pytest.mark.integration


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
