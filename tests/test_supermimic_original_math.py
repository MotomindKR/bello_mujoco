"""Numerical behavior of the pinned original learner, independent of YAML values."""

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.supermimic.model import MLP, integrate_velocity
from bello_mujoco.supermimic.optimization import apply_optimizer, optimizers
from bello_mujoco.supermimic.config import LearningConfig
from bello_mujoco.supermimic.state import PhysicalState


def test_integrated_dynamics_matches_world_frame_rigid_body_update_and_jacobian():
    orientation = Rotation.from_rotvec([0.3, -0.2, 0.7])
    state = PhysicalState(
        jnp.array([1.0, 2.0, 0.8]),
        jnp.asarray(orientation.as_quat(scalar_first=True)),
        jnp.array([0.4, -0.3, 0.1]),
        jnp.array([0.2, 0.1, -0.4]),
        jnp.array([0.3, -0.2]),
        jnp.array([0.5, -0.1]),
    )
    delta = np.array([0.1, -0.2, 0.3, -0.15, 0.3, 0.2, 0.4, -0.3], dtype=np.float32)
    dt, coefficient = 0.02, 0.625
    result = jax.jit(integrate_velocity)(state, delta, dt, coefficient)
    world_dv, world_dw = orientation.apply(delta[:3]), orientation.apply(delta[3:6])
    np.testing.assert_allclose(
        result.position,
        np.asarray(state.position)
        + (np.asarray(state.linear_velocity) + coefficient * world_dv) * dt,
        atol=2e-7,
    )
    expected_rotation = (
        Rotation.from_rotvec(
            (np.asarray(state.angular_velocity) + coefficient * world_dw) * dt
        )
        * orientation
    )
    np.testing.assert_allclose(
        result.quaternion, expected_rotation.as_quat(scalar_first=True), atol=2e-7
    )
    np.testing.assert_allclose(
        result.linear_velocity, np.asarray(state.linear_velocity) + world_dv, atol=1e-7
    )
    np.testing.assert_allclose(
        result.angular_velocity,
        np.asarray(state.angular_velocity) + world_dw,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.joint_position,
        np.asarray(state.joint_position)
        + (np.asarray(state.joint_velocity) + coefficient * delta[6:]) * dt,
        atol=1e-7,
    )
    jacobian = jax.jacfwd(
        lambda d: integrate_velocity(state, d, dt, coefficient).flatten()
    )(jnp.zeros(8))
    assert np.isfinite(jacobian).all()
    np.testing.assert_allclose(
        jacobian[:3, :3], orientation.as_matrix() * dt * coefficient, atol=1e-7
    )


def test_soft_termination_uses_maximum_normalized_error(toy_runner):
    env, _, _ = toy_runner
    reference = env.reference_state(0.0, jnp.zeros(2), 0.0)
    for fraction in (0.0, 0.5, 1.0, 1.5):
        state = replace(
            reference,
            position=reference.position.at[2].add(
                env.mdp.config.height_error * fraction
            ),
        )
        expected = 1 / (
            1 + np.exp(-(fraction - 1) / env.mdp.config.termination_softness)
        )
        assert float(env.mdp.soft_termination(state, reference)) == pytest.approx(
            expected, rel=2e-5, abs=1e-8
        )


def test_muon_partition_and_updates_match_torch():
    torch = pytest.importorskip("torch")
    config = LearningConfig()
    network = MLP((4, 5, 3, 2), key=jax.random.key(16))
    leaves, structure = jax.tree.flatten(eqx.filter(network, eqx.is_inexact_array))
    parameters = [
        torch.nn.Parameter(torch.tensor(np.asarray(x).copy())) for x in leaves
    ]
    torch_network = jax.tree.unflatten(structure, parameters)
    middle = [layer.weight for layer in torch_network.layers[1:-1]]
    middle_ids = {id(p) for p in middle}
    adam = torch.optim.Adam(
        [p for p in parameters if id(p) not in middle_ids],
        lr=config.learning_rate,
        betas=(0.9, 0.95),
    )
    muon = torch.optim.Muon(
        middle, lr=config.learning_rate, adjust_lr_fn="match_rms_adamw"
    )
    optimizer = optimizers(config)[0]
    state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    random = np.random.default_rng(9)
    for _ in range(5):
        gradients = [random.normal(size=x.shape).astype(np.float32) for x in leaves]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = torch.tensor(gradient)
        torch.nn.utils.clip_grad_norm_(parameters, config.max_gradient_norm)
        torch.nn.utils.clip_grad_value_(parameters, config.actor_gradient_clip)
        adam.step()
        muon.step()
        network, state, valid = eqx.filter_jit(apply_optimizer)(
            network,
            jax.tree.unflatten(structure, [jnp.asarray(g) for g in gradients]),
            state,
            optimizer,
            config.learning_rate,
        )
        assert valid
        for actual, expected in zip(
            jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array)),
            parameters,
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, expected.detach().numpy(), atol=3e-6, rtol=2e-5
            )


@pytest.fixture
def toy_runner():
    from test_supermimic_runner import toy_runner as factory

    return factory.__wrapped__()
