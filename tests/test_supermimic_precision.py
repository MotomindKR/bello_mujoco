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
    gradient = jax.grad(lambda x: network(x).sum())(value)
    assert np.isfinite(gradient).all()


def test_invalid_network_precision_is_rejected():
    with pytest.raises(ValueError, match="matmul_precision"):
        NetworkConfig(matmul_precision="automatic")
