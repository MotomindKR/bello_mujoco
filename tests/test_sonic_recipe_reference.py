"""Optional pure-tensor parity against pinned reference functions, without Isaac."""

import ast
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.sonic_transfer import BelloSonicTransferEnv


def reference_function(filename, name, namespace):
    path = Path(__file__).resolve().parents[1] / "deps/gear_sonic" / filename
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def test_wrist_anti_shake_matches_pinned_torch_function():
    torch = pytest.importorskip("torch")
    velocities = np.random.default_rng(3).normal(size=(2, 3)).astype(np.float32) * 4
    command = SimpleNamespace(robot_body_ang_vel_w=torch.tensor(velocities[None]))
    source_env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command)
    )
    function = reference_function(
        "envs/manager_env/mdp/rewards.py",
        "anti_shake_ang_vel_l2",
        {"torch": torch, "_get_body_indexes": lambda command, names: [0, 1]},
    )
    env = object.__new__(BelloSonicTransferEnv)
    object.__setattr__(
        env, "motion", SimpleNamespace(body_names=("left_wrist", "right_wrist"))
    )
    object.__setattr__(
        env, "config", SimpleNamespace(reward=SimpleNamespace(anti_shake_threshold=1.5))
    )
    # Name resolution is the Mini-specific part; compare the actual tensor math.
    from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            BelloMotionTrackingEnv,
            "end_effector_names",
            property(lambda _: env.motion.body_names),
        )
        result = env._anti_shake_cost(
            jnp.pad(jnp.asarray(velocities), ((0, 0), (0, 3)))
        )
    expected = function(source_env, "motion", threshold=1.5, body_names=None)
    np.testing.assert_allclose(result, expected.numpy()[0], rtol=1e-6)


def test_full_orientation_matches_pinned_termination_function():
    torch = pytest.importorskip("torch")
    angles = np.array([0.0, 0.9, 1.1, 2.5], dtype=np.float32)
    quat = np.stack(
        [np.cos(angles / 2), np.zeros(4), np.zeros(4), np.sin(angles / 2)], axis=1
    )
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (4, 1))
    command = SimpleNamespace(
        anchor_quat_w=torch.tensor(identity), robot_anchor_quat_w=torch.tensor(quat)
    )
    function = reference_function(
        "envs/manager_env/mdp/terminations.py",
        "exceeded_anchor_ori",
        {
            "quat_error_magnitude": lambda a, b: (
                2 * torch.acos(torch.abs((a * b).sum(-1)).clamp(0, 1))
            )
        },
    )
    source_env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda _: command)
    )
    expected = function(source_env, None, "motion", threshold=1.0)
    env = object.__new__(BelloSonicTransferEnv)
    object.__setattr__(
        env, "config", SimpleNamespace(done=SimpleNamespace(root_orientation_error=1.0))
    )
    actual = [
        env._anchor_orientation_terminated(jnp.asarray(a), jnp.asarray(b))
        for a, b in zip(identity, quat, strict=True)
    ]
    np.testing.assert_array_equal(actual, expected.numpy())
