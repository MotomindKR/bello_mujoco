"""JAX implementation of the released SONIC encoder/FSQ/decoder with LoRA."""

from __future__ import annotations

from dataclasses import dataclass
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.sonic.embodiment import MiniEmbodimentAdapter


@dataclass(frozen=True)
class SonicNetworkConfig:
    future_frames: int = 10
    reference_features_per_frame: int = 64
    history_steps: int = 10
    source_action_size: int = 29
    token_count: int = 2
    token_size: int = 32
    fsq_levels: int = 32
    decoder_lora_rank: int = 64
    decoder_lora_alpha: float = 128.0
    critic_lora_rank: int = 32
    critic_lora_alpha: float = 32.0

    @property
    def reference_size(self) -> int:
        return self.future_frames * self.reference_features_per_frame

    @property
    def source_proprio_size(self) -> int:
        return 6 * self.history_steps + 3 * self.source_action_size * self.history_steps


class LoRALinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    lora_a: jax.Array
    lora_b: jax.Array
    alpha: float = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        weight,
        bias,
        *,
        rank: int,
        alpha: float,
        key: jax.Array,
    ):
        weight = jnp.asarray(weight, dtype=jnp.float32)
        bias = jnp.asarray(bias, dtype=jnp.float32)
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(
                f"linear tensors have incompatible shapes {weight.shape} and {bias.shape}"
            )
        if rank < 0 or (rank == 0 and alpha != 0.0):
            raise ValueError(
                "LoRA rank must be nonnegative and rank zero requires alpha zero"
            )
        self.weight = weight
        self.bias = bias
        self.rank = rank
        self.alpha = float(alpha)
        if rank:
            bound = 1.0 / math.sqrt(weight.shape[1])
            self.lora_a = jax.random.uniform(
                key, (rank, weight.shape[1]), minval=-bound, maxval=bound
            )
            self.lora_b = jnp.zeros((weight.shape[0], rank), dtype=jnp.float32)
        else:
            self.lora_a = jnp.zeros((0, weight.shape[1]), dtype=jnp.float32)
            self.lora_b = jnp.zeros((weight.shape[0], 0), dtype=jnp.float32)

    def __call__(self, value: jax.Array) -> jax.Array:
        # The imported backbone is part of the checkpoint but never trainable.
        base = value @ jax.lax.stop_gradient(self.weight).T + jax.lax.stop_gradient(
            self.bias
        )
        if self.rank == 0:
            return base
        delta = (value @ self.lora_a.T) @ self.lora_b.T
        return base + delta * (self.alpha / self.rank)

    def merged_weight(self) -> jax.Array:
        if self.rank == 0:
            return self.weight
        return self.weight + (self.lora_b @ self.lora_a) * (self.alpha / self.rank)


class SonicMLP(eqx.Module):
    layers: tuple[LoRALinear, ...]

    def __init__(
        self,
        tensors: tuple[tuple[np.ndarray, np.ndarray], ...],
        *,
        rank: int,
        alpha: float,
        key: jax.Array,
    ):
        keys = jax.random.split(key, len(tensors))
        self.layers = tuple(
            LoRALinear(weight, bias, rank=rank, alpha=alpha, key=layer_key)
            for (weight, bias), layer_key in zip(tensors, keys, strict=True)
        )
        for previous, following in zip(self.layers, self.layers[1:], strict=False):
            if previous.weight.shape[0] != following.weight.shape[1]:
                raise ValueError("SONIC MLP layer dimensions are not contiguous")

    @property
    def input_size(self) -> int:
        return int(self.layers[0].weight.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.layers[-1].weight.shape[0])

    def __call__(self, value: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            value = jax.nn.silu(layer(value))
        return self.layers[-1](value)


def finite_scalar_quantize(value: jax.Array, levels: int = 32) -> jax.Array:
    """Finite Scalar Quantization matching vector-quantize-pytorch inference."""

    if levels < 2:
        raise ValueError("FSQ levels must be at least two")
    half_l = (levels - 1) * (1.0 - 1.0e-3) / 2.0
    offset = 0.5 if levels % 2 == 0 else 0.0
    shift = math.atanh(offset / half_l) if offset else 0.0
    bounded = jnp.tanh(value + shift) * half_l - offset
    quantized = bounded + jax.lax.stop_gradient(jnp.round(bounded) - bounded)
    return quantized / (levels // 2)


class SonicActor(eqx.Module):
    encoder: SonicMLP
    decoder: SonicMLP
    adapter: MiniEmbodimentAdapter = eqx.field(static=True)
    config: SonicNetworkConfig = eqx.field(static=True)

    def __init__(
        self,
        encoder: SonicMLP,
        decoder: SonicMLP,
        adapter: MiniEmbodimentAdapter,
        config: SonicNetworkConfig,
    ):
        if encoder.input_size != config.reference_size:
            raise ValueError(
                f"SONIC encoder expects {encoder.input_size} inputs; required "
                f"release layout has {config.reference_size}"
            )
        if encoder.output_size != config.token_count * config.token_size:
            raise ValueError("SONIC encoder output does not match the FSQ token layout")
        expected_decoder = (
            config.token_count * config.token_size + config.source_proprio_size
        )
        if (
            decoder.input_size != expected_decoder
            or decoder.output_size != config.source_action_size
        ):
            raise ValueError(
                "SONIC decoder tensor shapes do not match release config: "
                f"got {decoder.input_size}->{decoder.output_size}, expected "
                f"{expected_decoder}->{config.source_action_size}"
            )
        self.encoder = encoder
        self.decoder = decoder
        self.adapter = adapter
        self.config = config

    @property
    def per_step_observation_size(self) -> int:
        return self.config.reference_size + 6 + 3 * self.adapter.target_size

    @property
    def observation_size(self) -> int:
        return self.per_step_observation_size * self.config.history_steps

    def __call__(self, observation: jax.Array) -> jax.Array:
        if observation.shape[-1] != self.observation_size:
            raise ValueError(
                f"SONIC actor observation must end in {self.observation_size} values, "
                f"got {observation.shape}"
            )
        steps = observation.reshape(
            observation.shape[:-1]
            + (self.config.history_steps, self.per_step_observation_size)
        )
        reference = steps[..., -1, : self.config.reference_size]
        cursor = self.config.reference_size
        gravity = steps[..., :, cursor : cursor + 3]
        cursor += 3
        angular_velocity = steps[..., :, cursor : cursor + 3]
        cursor += 3
        size = self.adapter.target_size
        joint_position = steps[..., :, cursor : cursor + size]
        cursor += size
        joint_velocity = steps[..., :, cursor : cursor + size]
        cursor += size
        action_history = steps[..., :, cursor : cursor + size]

        source_position = self.adapter.target_vector_to_source(joint_position)
        source_velocity = self.adapter.target_vector_to_source(joint_velocity)
        source_actions = self.adapter.target_vector_to_source(action_history)
        proprioception = jnp.concatenate(
            (
                angular_velocity.reshape(angular_velocity.shape[:-2] + (-1,)),
                source_position.reshape(source_position.shape[:-2] + (-1,)),
                source_velocity.reshape(source_velocity.shape[:-2] + (-1,)),
                source_actions.reshape(source_actions.shape[:-2] + (-1,)),
                gravity.reshape(gravity.shape[:-2] + (-1,)),
            ),
            axis=-1,
        )
        latent = self.encoder(reference).reshape(
            reference.shape[:-1] + (self.config.token_count, self.config.token_size)
        )
        tokens = finite_scalar_quantize(latent, self.config.fsq_levels).reshape(
            reference.shape[:-1] + (-1,)
        )
        source_action = self.decoder(jnp.concatenate((tokens, proprioception), axis=-1))
        return self.adapter.source_action_to_target(source_action)


class SonicCritic(eqx.Module):
    network: SonicMLP
    adapter: MiniEmbodimentAdapter = eqx.field(static=True)
    body_count: int = eqx.field(static=True)
    history_steps: int = eqx.field(static=True)

    def __init__(
        self,
        network: SonicMLP,
        adapter: MiniEmbodimentAdapter,
        *,
        body_count: int,
        history_steps: int = 10,
    ):
        expected = 589 + 9 * body_count + (6 + 3 * 29) * history_steps
        if network.input_size != expected:
            raise ValueError(
                f"SONIC critic expects {network.input_size} inputs; assembled "
                f"privileged layout has {expected}"
            )
        if network.output_size != 1:
            raise ValueError("SONIC critic must produce one scalar value")
        self.network = network
        self.adapter = adapter
        self.body_count = body_count
        self.history_steps = history_steps

    @property
    def per_step_observation_size(self) -> int:
        # ref joints (580), anchor pos/orientation, body pos/orientation,
        # root velocities, target joints, and previous target action.
        return 589 + 9 * self.body_count + 6 + 3 * self.adapter.target_size

    @property
    def observation_size(self) -> int:
        return self.per_step_observation_size * self.history_steps

    def __call__(self, observation: jax.Array) -> jax.Array:
        if observation.shape[-1] != self.observation_size:
            raise ValueError(
                f"SONIC critic observation must end in {self.observation_size} values, "
                f"got {observation.shape}"
            )
        steps = observation.reshape(
            observation.shape[:-1]
            + (self.history_steps, self.per_step_observation_size)
        )
        static_size = 589 + 9 * self.body_count
        latest = steps[..., -1, :static_size]
        cursor = static_size
        base_linear = steps[..., :, cursor : cursor + 3]
        cursor += 3
        base_angular = steps[..., :, cursor : cursor + 3]
        cursor += 3
        size = self.adapter.target_size
        position = steps[..., :, cursor : cursor + size]
        cursor += size
        velocity = steps[..., :, cursor : cursor + size]
        cursor += size
        actions = steps[..., :, cursor : cursor + size]
        position = self.adapter.target_vector_to_source(position)
        velocity = self.adapter.target_vector_to_source(velocity)
        actions = self.adapter.target_vector_to_source(actions)
        assembled = jnp.concatenate(
            (
                latest,
                base_linear.reshape(base_linear.shape[:-2] + (-1,)),
                base_angular.reshape(base_angular.shape[:-2] + (-1,)),
                position.reshape(position.shape[:-2] + (-1,)),
                velocity.reshape(velocity.shape[:-2] + (-1,)),
                actions.reshape(actions.shape[:-2] + (-1,)),
            ),
            axis=-1,
        )
        return self.network(assembled)


class SonicActorCritic(eqx.Module):
    actor: SonicActor
    critic: SonicCritic
    log_std: jax.Array
    min_log_std: float = eqx.field(static=True)
    max_log_std: float = eqx.field(static=True)
    action_distribution: str = eqx.field(static=True)
    action_limit: float = eqx.field(static=True)

    def __init__(
        self,
        actor: SonicActor,
        critic: SonicCritic,
        source_log_std: jax.Array,
    ):
        source_log_std = jnp.asarray(source_log_std, dtype=jnp.float32)
        if source_log_std.shape != (actor.config.source_action_size,):
            raise ValueError(
                "SONIC checkpoint log_std must contain one value per source action"
            )
        self.actor = actor
        self.critic = critic
        self.log_std = actor.adapter.source_scale_to_target(source_log_std)
        self.min_log_std = math.log(0.001)
        self.max_log_std = math.log(0.6)
        self.action_distribution = "gaussian"
        self.action_limit = 100.0

    def __call__(
        self, actor_observation: jax.Array, critic_observation: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if critic_observation is None:
            raise ValueError("SONIC requires a privileged critic observation")
        mean = self.actor(actor_observation)
        value = jnp.squeeze(self.critic(critic_observation), axis=-1)
        return mean, jnp.clip(self.log_std, self.min_log_std, self.max_log_std), value

    def action_mean(self, observation: jax.Array) -> jax.Array:
        return self.actor(observation)

    def trainable_filter(self):
        """Select only adapters and exploration scale for PPO optimization."""

        def selected(path, value):
            if not eqx.is_array(value):
                return False
            names = tuple(
                getattr(entry, "name", None)
                for entry in path
                if getattr(entry, "name", None) is not None
            )
            return names == ("log_std",) or (
                value.size > 0
                and names[-1:]
                in {
                    ("lora_a",),
                    ("lora_b",),
                }
            )

        return jax.tree_util.tree_map_with_path(selected, self)


def sonic_policy_template(
    adapter: MiniEmbodimentAdapter,
    *,
    key: jax.Array,
    config: SonicNetworkConfig = SonicNetworkConfig(),
) -> SonicActorCritic:
    """Build the fixed release topology for restoring a self-contained policy."""

    def network(dims, *, rank, alpha, network_key):
        tensors = tuple(
            (
                np.zeros((output_size, input_size), dtype=np.float32),
                np.zeros(output_size, dtype=np.float32),
            )
            for input_size, output_size in zip(dims[:-1], dims[1:], strict=True)
        )
        return SonicMLP(
            tensors,
            rank=rank,
            alpha=alpha,
            key=network_key,
        )

    encoder_key, decoder_key, critic_key = jax.random.split(key, 3)
    encoder = network(
        (640, 2048, 1024, 512, 512, 64),
        rank=0,
        alpha=0.0,
        network_key=encoder_key,
    )
    decoder = network(
        (994, 2048, 2048, 1024, 1024, 512, 512, 29),
        rank=config.decoder_lora_rank,
        alpha=config.decoder_lora_alpha,
        network_key=decoder_key,
    )
    critic = network(
        (1645, 2048, 2048, 1024, 1024, 512, 512, 1),
        rank=config.critic_lora_rank,
        alpha=config.critic_lora_alpha,
        network_key=critic_key,
    )
    return SonicActorCritic(
        SonicActor(encoder, decoder, adapter, config),
        SonicCritic(critic, adapter, body_count=14, history_steps=config.history_steps),
        jnp.zeros(config.source_action_size, dtype=jnp.float32),
    )
