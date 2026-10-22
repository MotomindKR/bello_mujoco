"""Native JAX deterministic dynamics, goal-conditioned actor, and value model."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.state import PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Moments:
    mean: jax.Array
    variance: jax.Array
    count: jax.Array

    @classmethod
    def empty(cls, size):
        return cls(jnp.zeros(size), jnp.ones(size), jnp.asarray(0.0))

    def update(self, values, mask=None):
        values = values.reshape((-1, self.mean.shape[0]))
        weights = jnp.ones(values.shape[0]) if mask is None else mask.reshape(-1)
        batch_count = weights.sum()
        batch_mean = (values * weights[:, None]).sum(axis=0) / jnp.maximum(
            batch_count, 1
        )
        batch_var = (((values - batch_mean) ** 2) * weights[:, None]).sum(
            axis=0
        ) / jnp.maximum(batch_count, 1)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        mean = self.mean + delta * batch_count / jnp.maximum(total, 1)
        var = (
            self.variance * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / jnp.maximum(total, 1)
        ) / jnp.maximum(total, 1)
        return Moments(
            jnp.where(batch_count > 0, mean, self.mean),
            jnp.where(batch_count > 0, var, self.variance),
            total,
        )

    def std(self, floor=1e-2):
        return jnp.sqrt(jnp.maximum(self.variance, floor**2))

    def normalize(self, values, floor=1e-2):
        return (values - self.mean) / self.std(floor)


@dataclass(frozen=True)
class NetworkConfig:
    dynamics_widths: tuple[int, ...] = (2048, 2048, 2048)
    actor_widths: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_widths: tuple[int, ...] = (256, 128)
    latent_size: int = 32

    def __post_init__(self):
        for widths in (self.dynamics_widths, self.actor_widths, self.critic_widths):
            if not widths or any(type(n) is not int or n <= 0 for n in widths):
                raise ValueError("network widths must be positive integer tuples")
        if type(self.latent_size) is not int or self.latent_size <= 0:
            raise ValueError("latent_size must be a positive integer")


class MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    activation: str = eqx.field(static=True)

    def __init__(
        self, sizes, *, key, activation="elu", layer_norm=False, output_scale=1.0
    ):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            eqx.nn.Linear(a, b, key=k)
            for a, b, k in zip(sizes[:-1], sizes[1:], keys, strict=True)
        )
        self.norms = (
            tuple(eqx.nn.LayerNorm(n) for n in sizes[1:-1]) if layer_norm else ()
        )
        self.activation = activation
        last = self.layers[-1]
        self.layers = self.layers[:-1] + (
            eqx.tree_at(
                lambda layer: (layer.weight, layer.bias),
                last,
                (last.weight * output_scale, jnp.zeros_like(last.bias)),
            ),
        )

    def __call__(self, value):
        for i, layer in enumerate(self.layers):
            value = (
                jnp.matmul(value, layer.weight.T, precision=jax.lax.Precision.HIGHEST)
                + layer.bias
            )
            if i < len(self.layers) - 1:
                if self.norms:
                    shape = value.shape
                    value = jax.vmap(self.norms[i])(
                        value.reshape((-1, shape[-1]))
                    ).reshape(shape)
                value = (
                    jax.nn.silu(value)
                    if self.activation == "silu"
                    else jax.nn.elu(value)
                )
        return value


def invariant_features(state):
    inverse = rot.conjugate(state.quaternion)
    tilt = rot.multiply(rot.conjugate(rot.heading(state.quaternion)), state.quaternion)
    return jnp.concatenate(
        (
            state.position[..., 2:3],
            rot.rotation_6d(tilt),
            rot.rotate(inverse, state.linear_velocity),
            rot.rotate(inverse, state.angular_velocity),
            state.joint_position,
            state.joint_velocity,
        ),
        axis=-1,
    )


def state_delta(start, end):
    inverse = rot.conjugate(start.quaternion)
    return jnp.concatenate(
        (
            rot.rotate(
                rot.conjugate(rot.heading(start.quaternion)),
                end.position - start.position,
            ),
            rot.log(rot.multiply(inverse, end.quaternion)),
            rot.rotate(inverse, end.linear_velocity - start.linear_velocity),
            rot.rotate(inverse, end.angular_velocity - start.angular_velocity),
            end.joint_position - start.joint_position,
            end.joint_velocity - start.joint_velocity,
        ),
        axis=-1,
    )


def apply_delta(state, delta):
    n = state.joint_position.shape[-1]
    dp, dr, dv, dw, dq, dqd = jnp.split(delta, (3, 6, 9, 12, 12 + n), axis=-1)
    return PhysicalState(
        state.position + rot.rotate(rot.heading(state.quaternion), dp),
        rot.normalize(rot.multiply(state.quaternion, rot.exp(dr))),
        state.linear_velocity + rot.rotate(state.quaternion, dv),
        state.angular_velocity + rot.rotate(state.quaternion, dw),
        state.joint_position + dq,
        state.joint_velocity + dqd,
    )


class WorldModel(eqx.Module):
    network: MLP
    joint_count: int = eqx.field(static=True)

    def __init__(self, joint_count, context_size, *, config=NetworkConfig(), key):
        self.joint_count = joint_count
        self.network = MLP(
            (13 + 4 * joint_count + context_size,)
            + config.dynamics_widths
            + (12 + 2 * joint_count,),
            key=key,
            activation="silu",
            layer_norm=True,
            output_scale=0.0,
        )

    @staticmethod
    def features(state, previous_motor, motor, context):
        return jnp.concatenate(
            (invariant_features(state), previous_motor, motor, context), axis=-1
        )

    def step(self, state, previous_motor, motor, context, input_norm, output_norm):
        features = self.features(state, previous_motor, motor, context)
        normalized = self.network(input_norm.normalize(features))
        delta = normalized * output_norm.std(1e-4) + output_norm.mean
        return apply_delta(state, delta)


class Actor(eqx.Module):
    encoder: MLP
    decoder: MLP

    def __init__(
        self,
        sample_size,
        history_frames,
        goal_size,
        joint_count,
        *,
        config=NetworkConfig(),
        key,
    ):
        enc, dec = jax.random.split(key)
        self.encoder = MLP(
            (goal_size,) + config.actor_widths + (config.latent_size,), key=enc
        )
        self.decoder = MLP(
            (sample_size * history_frames + config.latent_size,)
            + config.actor_widths
            + (joint_count,),
            key=dec,
            output_scale=0.01,
        )

    def __call__(self, history, goal, observation_norm, goal_norm):
        latent = jnp.tanh(self.encoder(goal_norm.normalize(goal)))
        values = observation_norm.normalize(history)
        values = values.reshape(values.shape[:-2] + (-1,))
        return self.decoder(jnp.concatenate((latent, values), axis=-1))


class Critic(eqx.Module):
    network: MLP

    def __init__(self, size, *, config=NetworkConfig(), key):
        self.network = MLP((size,) + config.critic_widths + (1,), key=key)

    def __call__(self, features, normalizer):
        return self.network(normalizer.normalize(features))[..., 0]


def world_model_loss(model, window, input_norm, output_norm, *, error_floor):
    tr = window.transitions
    initial = jax.tree.map(lambda x: x[:, 0], tr.state)
    controls = (tr.previous_motor, tr.motor, tr.context)
    controls = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), controls)

    def step(state, inputs):
        prediction = model.step(state, *inputs, input_norm, output_norm)
        return prediction, prediction

    _, predicted = jax.lax.scan(step, initial, controls)
    predicted = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), predicted)
    error = state_delta(tr.successor, predicted) / output_norm.std(error_floor)
    squared = jnp.mean(error**2, axis=-1)
    return (squared * window.mask).sum() / jnp.maximum(window.mask.sum(), 1)
