import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.supermimic.model import MLP, NetworkConfig


@pytest.mark.parametrize(
    "name,precision",
    [
        ("float32", jax.lax.Precision.HIGHEST),
        ("tensorfloat32", jax.lax.Precision.HIGH),
        ("bfloat16", jax.lax.Precision.HIGH),
    ],
)
def test_network_precision_reaches_matrix_multiplications_and_has_finite_gradients(
    name, precision
):
    network = MLP((4, 8, 2), key=jax.random.PRNGKey(0), matmul_precision=name)
    value = jnp.ones((3, 4))
    graph = jax.make_jaxpr(network)(value).jaxpr
    dots = [eqn for eqn in graph.eqns if eqn.primitive.name == "dot_general"]
    assert len(dots) == 2
    assert all(eqn.params["precision"] == (precision, precision) for eqn in dots)
    dtype = jnp.bfloat16 if name == "bfloat16" else jnp.float32
    assert all(eqn.invars[0].aval.dtype == dtype for eqn in dots)
    assert network(value).dtype == jnp.float32
    assert all(layer.weight.dtype == jnp.float32 for layer in network.layers)
    gradient = jax.grad(lambda x: network(x).sum())(value)
    assert np.isfinite(gradient).all()


def test_bfloat16_layer_norm_and_optimizer_keep_float32_parameters():
    import equinox as eqx
    import optax

    network = MLP(
        (4, 8, 2), key=jax.random.key(4), matmul_precision="bfloat16", layer_norm=True
    )
    value = jnp.arange(12, dtype=jnp.float32).reshape((3, 4)) / 10
    gradient = eqx.filter_grad(lambda net: jnp.mean(net(value) ** 2))(network)
    optimizer = optax.adam(1e-3)
    parameters = eqx.filter(network, eqx.is_inexact_array)
    updates, state = optimizer.update(gradient, optimizer.init(parameters))
    trained = eqx.apply_updates(network, updates)
    assert all(
        leaf.dtype == jnp.float32
        for leaf in jax.tree.leaves(eqx.filter(trained, eqx.is_inexact_array))
    )
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(state))
    assert np.isfinite(trained(value)).all()


def test_invalid_network_precision_is_rejected():
    with pytest.raises(ValueError, match="matmul_precision"):
        NetworkConfig(matmul_precision="automatic")


def test_rejected_nonfinite_optimizer_step_preserves_parameters_and_moments():
    import equinox as eqx
    import optax
    from bello_mujoco.supermimic.optimization import apply_optimizer

    network = MLP((4, 8, 2), key=jax.random.key(0))
    gradient = jax.tree.map(jnp.ones_like, eqx.filter(network, eqx.is_inexact_array))
    gradient = eqx.tree_at(
        lambda net: net.layers[0].bias, gradient, jnp.full(8, jnp.nan)
    )
    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    updated, next_state, valid = eqx.filter_jit(apply_optimizer)(
        network, gradient, state, optimizer, 1.0
    )
    assert not valid
    for actual, expected in zip(
        jax.tree.leaves((updated, next_state)),
        jax.tree.leaves((network, state)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("optimizer_index", [0, 1, 2])
def test_adam_and_gradient_clipping_match_torch_for_identical_gradients(
    optimizer_index,
):
    import equinox as eqx
    from bello_mujoco.supermimic.config import LearningConfig
    from bello_mujoco.supermimic.optimization import (
        apply_optimizer,
        learning_rate,
        optimizers,
    )

    torch = pytest.importorskip("torch")
    cfg = LearningConfig()
    network = MLP((4, 8, 2), key=jax.random.key(0), layer_norm=True)
    leaves, structure = jax.tree.flatten(eqx.filter(network, eqx.is_inexact_array))
    parameters = [
        torch.nn.Parameter(torch.tensor(np.asarray(x).copy())) for x in leaves
    ]
    reference = torch.optim.Adam(
        parameters,
        lr=cfg.learning_rate,
        betas=(0.9, 0.95 if optimizer_index == 0 else 0.999),
    )
    optimizer = optimizers(cfg)[optimizer_index]
    state = optimizer.init(eqx.filter(network, eqx.is_inexact_array))
    for step in range(4):
        gradients = []
        for x in leaves:
            indices = np.arange(x.size).reshape(x.shape)
            values = np.sin(indices + step).astype(np.float32)
            values[indices % 7 == 0] = (-1.0) ** step * 1e-10
            values[indices % 11 == 0] = 0.0
            gradients.append(jnp.asarray(values))
        for parameter, grad in zip(parameters, gradients, strict=True):
            parameter.grad = torch.tensor(np.asarray(grad).copy())
        torch.nn.utils.clip_grad_norm_(parameters, cfg.max_gradient_norm)
        if optimizer_index == 0:
            torch.nn.utils.clip_grad_value_(parameters, cfg.actor_gradient_clip)
        rate = float(learning_rate(cfg, step))
        reference.param_groups[0]["lr"] = rate
        reference.step()
        network, state, valid = apply_optimizer(
            network, jax.tree.unflatten(structure, gradients), state, optimizer, rate
        )
        assert valid
        for actual, expected in zip(
            jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array)),
            parameters,
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, expected.detach().numpy(), atol=2e-7, rtol=1e-6
            )
        adam = next(s for s in state if hasattr(s, "mu"))
        for mu, nu, parameter in zip(
            jax.tree.leaves(adam.mu), jax.tree.leaves(adam.nu), parameters, strict=True
        ):
            np.testing.assert_allclose(
                mu, reference.state[parameter]["exp_avg"].numpy(), atol=5e-8, rtol=1e-6
            )
            np.testing.assert_allclose(
                nu,
                reference.state[parameter]["exp_avg_sq"].numpy(),
                atol=5e-8,
                rtol=1e-6,
            )
