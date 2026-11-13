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

    network = MLP((4, 8, 2), key=jax.random.key(4), matmul_precision="bfloat16")
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


def test_bfloat16_backward_uses_matching_dot_operand_types():
    import equinox as eqx

    network = MLP((4, 8, 2), key=jax.random.key(0), matmul_precision="bfloat16")
    graph = eqx.filter_make_jaxpr(eqx.filter_grad(lambda net, x: jnp.sum(net(x) ** 2)))
    closed, _, _ = graph(network, jnp.ones((3, 4)))
    dots = [e for e in closed.jaxpr.eqns if e.primitive.name == "dot_general"]
    assert len(dots) > len(network.layers)
    assert all(
        e.invars[0].aval.dtype == e.invars[1].aval.dtype == jnp.bfloat16 for e in dots
    )


def test_masked_nonfinite_samples_do_not_enter_running_moments():
    from bello_mujoco.supermimic.model import Moments

    moments = jax.jit(
        lambda values: Moments.empty(2).update(values, jnp.array([True, False, True]))
    )(jnp.array([[1.0, 2.0], [jnp.nan, jnp.inf], [3.0, 6.0]]))
    np.testing.assert_allclose(moments.mean, [2, 4])
    np.testing.assert_allclose(moments.variance, [1, 4])
    assert moments.count == 2


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_network_outputs_and_gradients_match_torch_with_identical_weights(precision):
    import equinox as eqx

    torch = pytest.importorskip("torch")
    functional = torch.nn.functional
    network = MLP(
        (16, 32, 24, 5),
        key=jax.random.key(12),
        matmul_precision=precision,
    )
    leaves, structure = jax.tree.flatten(eqx.filter(network, eqx.is_inexact_array))
    parameters = [
        torch.tensor(np.asarray(x).copy(), requires_grad=True) for x in leaves
    ]
    reference = jax.tree.unflatten(structure, parameters)
    value = np.random.default_rng(7).normal(size=(24, 16)).astype(np.float32)
    expected = torch.tensor(value, requires_grad=True)
    dtype = torch.bfloat16 if precision == "bfloat16" else torch.float32
    result = expected
    for i, layer in enumerate(reference.layers):
        result = functional.linear(
            result.to(dtype), layer.weight.to(dtype), layer.bias.to(dtype)
        )
        if i < len(reference.layers) - 1:
            if reference.norms:
                norm = reference.norms[i]
                result = functional.layer_norm(
                    result.float(),
                    (result.shape[-1],),
                    norm.weight,
                    norm.bias,
                    norm.eps,
                )
            result = functional.gelu(result, approximate="none")
    result = result.float()
    result.square().mean().backward()
    _, (gradient, input_gradient) = eqx.filter_jit(
        eqx.filter_value_and_grad(lambda pair: jnp.square(pair[0](pair[1])).mean())
    )((network, jnp.asarray(value)))
    pairs = [
        (network(value), result.detach().numpy()),
        (input_gradient, expected.grad.numpy()),
    ]
    pairs.extend(
        zip(
            jax.tree.leaves(eqx.filter(gradient, eqx.is_inexact_array)),
            (p.grad.numpy() for p in parameters),
            strict=True,
        )
    )
    for actual, expected in pairs:
        error = np.linalg.norm(np.asarray(actual) - expected) / max(
            np.linalg.norm(expected), 1e-12
        )
        assert error < (0.025 if precision == "bfloat16" else 1e-5)


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
    network = MLP((4, 8, 2), key=jax.random.key(0))
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
        adam = next(
            s
            for s in jax.tree.leaves(
                state, is_leaf=lambda s: hasattr(s, "mu") and hasattr(s, "nu")
            )
            if hasattr(s, "mu") and hasattr(s, "nu")
        )
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
