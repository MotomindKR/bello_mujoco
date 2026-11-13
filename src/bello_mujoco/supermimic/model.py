"""Native JAX deterministic dynamics, goal-conditioned actor, and value model."""

from dataclasses import dataclass
from typing import Literal

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

    def update(self, values, mask=None, *, axis_name=None):
        values = values.reshape((-1, self.mean.shape[0]))
        weights = jnp.ones(values.shape[0]) if mask is None else mask.reshape(-1)
        values = jnp.where(weights[:, None] > 0, values, 0.0)
        batch_count = weights.sum()
        batch_sum = (values * weights[:, None]).sum(axis=0)
        if axis_name is not None:
            batch_count, batch_sum = jax.lax.psum((batch_count, batch_sum), axis_name)
        batch_mean = batch_sum / jnp.maximum(batch_count, 1)
        squared = (((values - batch_mean) ** 2) * weights[:, None]).sum(axis=0)
        if axis_name is not None:
            squared = jax.lax.psum(squared, axis_name)
        batch_var = squared / jnp.maximum(batch_count, 1)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        mean = self.mean + delta * batch_count / jnp.maximum(total, 1)
        var = (
            self.variance * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / jnp.maximum(total, 1)
        ) / jnp.maximum(total, 1)
        updated = Moments(
            jnp.where(batch_count > 0, mean, self.mean),
            jnp.where(batch_count > 0, var, self.variance),
            total,
        )
        return updated

    def std(self, floor=1e-2):
        return jnp.sqrt(jnp.maximum(self.variance, 0.0)) + floor

    def normalize(self, values, floor=1e-2):
        return (values - self.mean) / self.std(floor)


@dataclass(frozen=True)
class NetworkConfig:
    dynamics_widths: tuple[int, ...] = (2048, 2048, 2048)
    actor_widths: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_widths: tuple[int, ...] = (256, 128)
    latent_size: int = 32
    matmul_precision: Literal["float32", "tensorfloat32", "bfloat16"] = "float32"

    def __post_init__(self):
        for widths in (self.dynamics_widths, self.actor_widths, self.critic_widths):
            if not widths or any(type(n) is not int or n <= 0 for n in widths):
                raise ValueError("network widths must be positive integer tuples")
        if type(self.latent_size) is not int or self.latent_size <= 0:
            raise ValueError("latent_size must be a positive integer")
        if self.matmul_precision not in ("float32", "tensorfloat32", "bfloat16"):
            raise ValueError(
                "matmul_precision must be float32, tensorfloat32, or bfloat16"
            )


class MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    norms: tuple[eqx.nn.LayerNorm, ...]
    matmul_precision: str = eqx.field(static=True)

    def __init__(
        self,
        sizes,
        *,
        key,
        matmul_precision="float32",
    ):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = tuple(
            eqx.nn.Linear(a, b, key=k)
            for a, b, k in zip(sizes[:-1], sizes[1:], keys, strict=True)
        )
        self.norms = tuple(eqx.nn.LayerNorm(n) for n in sizes[1:-1])
        if matmul_precision not in ("float32", "tensorfloat32", "bfloat16"):
            raise ValueError("invalid MLP matmul precision")
        self.matmul_precision = matmul_precision

    def __call__(self, value):
        mixed = self.matmul_precision == "bfloat16"
        for i, layer in enumerate(self.layers):
            dtype = jnp.bfloat16 if mixed else jnp.float32
            value = jnp.matmul(
                value.astype(dtype),
                layer.weight.T.astype(dtype),
                precision=(
                    jax.lax.Precision.HIGHEST
                    if self.matmul_precision == "float32"
                    else jax.lax.Precision.HIGH
                ),
            ) + layer.bias.astype(dtype)
            if i < len(self.layers) - 1:
                shape = value.shape
                value = jax.vmap(self.norms[i])(
                    value.astype(jnp.float32).reshape((-1, shape[-1]))
                ).reshape(shape)
                value = jax.nn.gelu(value, approximate=False)
        return value.astype(jnp.float32)


def invariant_features(state):
    inverse = rot.conjugate(state.quaternion)
    return jnp.concatenate(
        (
            state.position[..., 2:3],
            rot.rotate(inverse, jnp.asarray([0.0, 0.0, -1.0])),
            rot.rotate(inverse, state.linear_velocity),
            rot.rotate(inverse, state.angular_velocity),
            state.joint_position,
            state.joint_velocity,
        ),
        axis=-1,
    )


def context_statistics(environment, joint_count):
    cfg = environment.model_randomization
    bias = environment.reset.joint_default_offset_noise
    friction = cfg.friction_range if cfg.enabled else (1.0, 1.0)
    com = cfg.torso_com_offset_range if cfg.enabled else (0.0, 0.0)
    offset = cfg.joint_angle_offset_range if cfg.enabled else (0.0, 0.0)
    ranges = (
        [(offset[0] - bias, offset[1] + bias)] * joint_count + [friction] + [com] * 3
    )
    center = tuple((lo + hi) / 2 for lo, hi in ranges)
    scale = tuple((hi - lo) / 2 if (hi - lo) / 2 > 1e-3 else 1.0 for lo, hi in ranges)
    return center, scale


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


def integrate_velocity(state, delta, timestep, substep_coefficient):
    dv, dw, dqd = jnp.split(delta, (3, 6), axis=-1)
    c = substep_coefficient
    inverse = rot.conjugate(state.quaternion)
    local_v = rot.rotate(inverse, state.linear_velocity)
    local_w = rot.rotate(inverse, state.angular_velocity)
    pose_v = rot.rotate(state.quaternion, local_v + c * dv)
    pose_w = rot.rotate(state.quaternion, local_w + c * dw)
    return PhysicalState(
        state.position + pose_v * timestep,
        rot.normalize(rot.multiply(rot.exp(pose_w * timestep), state.quaternion)),
        state.linear_velocity + rot.rotate(state.quaternion, dv),
        state.angular_velocity + rot.rotate(state.quaternion, dw),
        state.joint_position + (state.joint_velocity + c * dqd) * timestep,
        state.joint_velocity + dqd,
    )


class WorldModel(eqx.Module):
    network: MLP
    joint_count: int = eqx.field(static=True)
    timestep: float = eqx.field(static=True)
    substep_coefficient: float = eqx.field(static=True)
    action_scale: tuple[float, ...] = eqx.field(static=True)
    action_offset: tuple[float, ...] = eqx.field(static=True)
    context_center: tuple[float, ...] = eqx.field(static=True)
    context_scale: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, env, *, config=NetworkConfig(), key):
        self.joint_count = env.action_size
        self.timestep = env.control_timestep
        decimation = env.config.decimation
        self.substep_coefficient = (decimation + 1) / (2 * decimation)
        self.action_scale = tuple(float(x) for x in env.mapping.scale)
        self.action_offset = tuple(float(x) for x in env.mapping.neutral_joint)
        self.context_center, self.context_scale = context_statistics(
            env.config, self.joint_count
        )
        self.network = MLP(
            (self.input_size + self.joint_count + 4,)
            + config.dynamics_widths
            + (6 + self.joint_count,),
            key=key,
            matmul_precision=config.matmul_precision,
        )

    @property
    def input_size(self):
        return 10 + 3 * self.joint_count

    def features(self, state, action):
        return jnp.concatenate(
            (
                invariant_features(state),
                action * jnp.asarray(self.action_scale)
                + jnp.asarray(self.action_offset),
            ),
            axis=-1,
        )

    def context_features(self, context):
        return (context[..., :-1] - jnp.asarray(self.context_center)) / jnp.asarray(
            self.context_scale
        )

    def step(self, state, action, context, input_norm):
        features = jnp.concatenate(
            (
                input_norm.normalize(self.features(state, action)),
                self.context_features(context),
            ),
            axis=-1,
        )
        return integrate_velocity(
            state, self.network(features), self.timestep, self.substep_coefficient
        )


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
            (goal_size,) + config.actor_widths + (config.latent_size,),
            key=enc,
            matmul_precision=config.matmul_precision,
        )
        self.decoder = MLP(
            (sample_size * history_frames + config.latent_size,)
            + config.actor_widths
            + (joint_count,),
            key=dec,
            matmul_precision=config.matmul_precision,
        )

    def __call__(self, history, goal, observation_norm, goal_norm):
        return self.decode(history, self.encode_goal(goal, goal_norm), observation_norm)

    def encode_goal(self, goal, goal_norm):
        return jnp.tanh(self.encoder(goal_norm.normalize(goal)))

    def decode(self, history, latent, observation_norm):
        values = observation_norm.normalize(history)
        values = values.reshape(values.shape[:-2] + (-1,))
        return self.decoder(jnp.concatenate((latent, values), axis=-1))


class Critic(eqx.Module):
    network: MLP

    def __init__(self, size, *, config=NetworkConfig(), key):
        self.network = MLP(
            (size,) + config.critic_widths + (1,),
            key=key,
            matmul_precision=config.matmul_precision,
        )

    @property
    def input_size(self):
        return self.network.layers[0].weight.shape[1]

    def __call__(self, features):
        return self.network(features)[..., 0]


def world_model_loss(model, window, input_norm, nominal_errors):
    tr = window.transitions
    initial = jax.tree.map(lambda x: x[:, 0], tr.state)
    controls = (tr.action, tr.context, window.mask)
    controls = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), controls)

    def step(state, inputs):
        action, context, valid = inputs
        safe_state = jax.tree.map(
            lambda value, first: jnp.where(valid[:, None] > 0, value, first),
            state,
            initial,
        )
        prediction = model.step(safe_state, action, context, input_norm)
        prediction = jax.tree.map(
            lambda predicted, previous: jnp.where(
                valid[:, None] > 0, predicted, previous
            ),
            prediction,
            state,
        )
        return prediction, prediction

    _, predicted = jax.lax.scan(
        step, initial, controls, unroll=min(8, controls[0].shape[0])
    )
    predicted = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), predicted)

    def loss_channels(state):
        rotation = jnp.stack(
            tuple(rot.rotate(state.quaternion, axis) for axis in jnp.eye(3)), axis=-1
        )
        return jnp.concatenate(
            (
                state.position / nominal_errors[0],
                rotation.reshape(rotation.shape[:-2] + (9,))
                * (0.5**0.5 / nominal_errors[1]),
                state.joint_position / nominal_errors[2],
                state.joint_velocity / nominal_errors[3],
            ),
            axis=-1,
        )

    error = loss_channels(predicted) - loss_channels(tr.successor)
    error = jnp.where(window.mask[..., None] > 0, error, 0.0)
    squared = jnp.mean(error**2, axis=-1)
    return (squared * window.mask).sum() / jnp.maximum(window.mask.sum(), 1)
