"""Feed-forward Gaussian actor-critic policy utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import math
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from bello_mujoco.training.ppo import ObservationNormState, PPOConfig

from bello_mujoco.sonic_policy import (
    SONIC_ACTOR_OBSERVATION_SIZE,
    SONIC_BELLO_PROPRIO_SIZE,
    SONIC_SMPL_INPUT_SIZE,
    SONIC_TOKEN_FLAT_SIZE,
    SonicDecoderInitialization,
    SonicReleaseWeights,
    sonic_fsq,
)

LOG_2PI = math.log(2.0 * math.pi)


def identity_activation(x: jax.Array) -> jax.Array:
    return x


@dataclass(frozen=True)
class ActorCriticConfig:
    hidden_dims: tuple[int, ...] = (256, 256)
    activation: Literal["tanh", "relu", "elu", "swish"] = "tanh"
    init_log_std: float = 0.0
    min_log_std: float = -5.0
    max_log_std: float = 1.0
    obs_stack_size: int = 10
    action_distribution: Literal["gaussian", "tanh_gaussian"] = "tanh_gaussian"
    action_limit: float = 1.0
    architecture: Literal[
        "mlp", "twist_teacher", "twist_student", "twist2", "sonic"
    ] = "mlp"
    layer_norm_final_hidden: bool = False


class LayeredMLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    final_activation: Callable[[jax.Array], jax.Array] = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_dims: tuple[int, ...],
        *,
        activation: Callable[[jax.Array], jax.Array],
        final_activation: Callable[[jax.Array], jax.Array],
        key: jax.Array,
    ):
        dims = (in_size, *hidden_dims, out_size)
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = tuple(
            eqx.nn.Linear(input_size, output_size, key=layer_key)
            for input_size, output_size, layer_key in zip(
                dims[:-1], dims[1:], keys, strict=True
            )
        )
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.final_activation(self.layers[-1](x))


class ActorCritic(eqx.Module):
    actor: eqx.Module
    critic: eqx.Module
    log_std: jax.Array
    min_log_std: float = eqx.field(static=True)
    max_log_std: float = eqx.field(static=True)
    action_distribution: str = eqx.field(static=True)
    action_limit: float = eqx.field(static=True)
    stochastic_actor: bool = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: ActorCriticConfig = ActorCriticConfig(),
        *,
        critic_obs_size: int | None = None,
        sonic_release: SonicReleaseWeights | None = None,
        sonic_initialization: SonicDecoderInitialization | None = None,
        key: jax.Array,
    ):
        if config.action_distribution not in {"gaussian", "tanh_gaussian"}:
            raise ValueError(
                "action_distribution must be 'gaussian' or 'tanh_gaussian', got "
                f"{config.action_distribution!r}"
            )
        actor_key, critic_key = jax.random.split(key)
        critic_obs_size = obs_size if critic_obs_size is None else critic_obs_size
        activation_fn = activation_function(config.activation)
        if config.architecture == "mlp":
            self.actor = make_mlp(
                obs_size=obs_size,
                output_size=action_size,
                config=config,
                activation=activation_fn,
                key=actor_key,
            )
            self.critic = make_mlp(
                obs_size=critic_obs_size,
                output_size=1,
                config=config,
                activation=activation_fn,
                key=critic_key,
            )
        elif config.architecture in {"twist_teacher", "twist_student"}:
            self.actor = TwistActor(
                obs_size,
                action_size,
                stage=config.architecture.removeprefix("twist_"),
                hidden_dims=config.hidden_dims,
                key=actor_key,
            )
            self.critic = TwistCritic(
                critic_obs_size,
                action_size,
                config.hidden_dims,
                key=critic_key,
            )
        elif config.architecture == "twist2":
            self.actor = Twist2Actor(
                obs_size,
                action_size,
                hidden_dims=config.hidden_dims,
                key=actor_key,
            )
            self.critic = Twist2Critic(
                critic_obs_size,
                action_size,
                config.hidden_dims,
                key=critic_key,
            )
        elif config.architecture == "sonic":
            if sonic_release is None or sonic_initialization is None:
                raise ValueError(
                    "SONIC policy construction requires exported release weights "
                    "and a Bello decoder initialization"
                )
            if obs_size != SONIC_ACTOR_OBSERVATION_SIZE:
                raise ValueError(
                    f"SONIC actor expected {SONIC_ACTOR_OBSERVATION_SIZE} "
                    f"observations, got {obs_size}"
                )
            if action_size != 25:
                raise ValueError(
                    f"SONIC Bello actor requires 25 actions, got {action_size}"
                )
            self.actor = SonicActor(sonic_release, sonic_initialization)
            self.critic = make_mlp(
                obs_size=critic_obs_size,
                output_size=1,
                config=config,
                activation=activation_fn,
                key=critic_key,
            )
        else:
            raise ValueError(f"unsupported policy architecture {config.architecture!r}")
        if config.architecture == "sonic":
            assert sonic_initialization is not None
            self.log_std = jnp.clip(
                jnp.log(jnp.asarray(sonic_initialization.output_std)),
                config.min_log_std,
                config.max_log_std,
            )
        else:
            self.log_std = jnp.full(
                (action_size,), config.init_log_std, dtype=jnp.float32
            )
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std
        self.action_distribution = config.action_distribution
        self.action_limit = config.action_limit
        self.stochastic_actor = config.architecture == "twist2"

    def __call__(
        self,
        obs: jax.Array,
        critic_obs: jax.Array | None = None,
        *,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        critic_obs = obs if critic_obs is None else critic_obs
        mean = self.actor(obs, key=key) if self.stochastic_actor else self.actor(obs)
        value = jnp.squeeze(self.critic(critic_obs), axis=-1)
        return mean, policy_log_std(self), value

    def action_mean(self, obs: jax.Array) -> jax.Array:
        return transform_action(self, self.actor(obs))


def init_actor_critic(
    rng: jax.Array,
    obs_size: int,
    action_size: int,
    config: ActorCriticConfig = ActorCriticConfig(),
    *,
    critic_obs_size: int | None = None,
    sonic_release: SonicReleaseWeights | None = None,
    sonic_initialization: SonicDecoderInitialization | None = None,
) -> ActorCritic:
    return ActorCritic(
        obs_size,
        action_size,
        config,
        critic_obs_size=critic_obs_size,
        sonic_release=sonic_release,
        sonic_initialization=sonic_initialization,
        key=rng,
    )


def sonic_frozen_parameter_hashes(policy: ActorCritic) -> dict[str, str]:
    """Return content hashes for every array outside SONIC's trainable edges."""

    if not isinstance(policy.actor, SonicActor):
        raise TypeError("frozen SONIC hashes require a SONIC actor")
    arrays = {
        **{
            f"smpl_encoder_weight_{index}": value
            for index, value in enumerate(policy.actor.smpl_encoder.weights)
        },
        **{
            f"smpl_encoder_bias_{index}": value
            for index, value in enumerate(policy.actor.smpl_encoder.biases)
        },
        "decoder_token_projection": policy.actor.decoder.token_projection_weight,
        **{
            f"decoder_middle_weight_{index}": value
            for index, value in enumerate(policy.actor.decoder.middle_weights)
        },
        **{
            f"decoder_middle_bias_{index}": value
            for index, value in enumerate(policy.actor.decoder.middle_biases)
        },
    }
    result = {}
    for name, array in arrays.items():
        value = np.ascontiguousarray(jax.device_get(array))
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
        result[name] = digest.hexdigest()
    return result


def policy_trainable_filter(policy: ActorCritic):
    """Return the optimizer filter, structurally excluding SONIC's frozen core."""

    specification = jax.tree_util.tree_map(eqx.is_array, policy)
    if not isinstance(policy.actor, SonicActor):
        return specification
    specification = eqx.tree_at(
        lambda value: value.actor.smpl_encoder.weights,
        specification,
        tuple(False for _ in policy.actor.smpl_encoder.weights),
    )
    specification = eqx.tree_at(
        lambda value: value.actor.smpl_encoder.biases,
        specification,
        tuple(False for _ in policy.actor.smpl_encoder.biases),
    )
    specification = eqx.tree_at(
        lambda value: value.actor.decoder.token_projection_weight,
        specification,
        False,
    )
    specification = eqx.tree_at(
        lambda value: value.actor.decoder.middle_weights,
        specification,
        tuple(False for _ in policy.actor.decoder.middle_weights),
    )
    return eqx.tree_at(
        lambda value: value.actor.decoder.middle_biases,
        specification,
        tuple(False for _ in policy.actor.decoder.middle_biases),
    )


def partition_policy_parameters(policy: ActorCritic):
    """Partition an actor-critic using its architecture-specific freeze contract."""

    return eqx.partition(policy, policy_trainable_filter(policy))


def make_mlp(
    *,
    obs_size: int,
    output_size: int,
    config: ActorCriticConfig,
    activation: Callable[[jax.Array], jax.Array],
    key: jax.Array,
) -> eqx.Module:
    return LayeredMLP(
        obs_size,
        output_size,
        config.hidden_dims,
        activation=activation,
        final_activation=identity_activation,
        key=key,
    )


def _xavier_linear(
    in_size: int,
    out_size: int,
    *,
    gain: float,
    key: jax.Array,
) -> eqx.nn.Linear:
    layer = eqx.nn.Linear(in_size, out_size, key=key)
    bound = gain * math.sqrt(6.0 / float(in_size + out_size))
    weight = jax.random.uniform(
        key,
        layer.weight.shape,
        dtype=layer.weight.dtype,
        minval=-bound,
        maxval=bound,
    )
    layer = eqx.tree_at(lambda value: value.weight, layer, weight)
    if layer.bias is not None:
        layer = eqx.tree_at(lambda value: value.bias, layer, jnp.zeros_like(layer.bias))
    return layer


class SonicFrozenSmplEncoder(eqx.Module):
    """The released SMPL encoder; all arrays are explicit stop-gradient leaves."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __init__(self, release: SonicReleaseWeights):
        self.weights = tuple(
            jnp.asarray(value) for value in release.smpl_encoder_weights
        )
        self.biases = tuple(jnp.asarray(value) for value in release.smpl_encoder_biases)

    def __call__(self, value: jax.Array) -> jax.Array:
        for index, (weight, bias) in enumerate(
            zip(self.weights, self.biases, strict=True)
        ):
            value = jnp.matmul(
                value,
                jax.lax.stop_gradient(weight).T,
                precision=jax.lax.Precision.HIGHEST,
            ) + jax.lax.stop_gradient(bias)
            if index != len(self.weights) - 1:
                value = jax.nn.silu(value)
        return value


class SonicEdgeDecoder(eqx.Module):
    """Frozen SONIC core with trainable Bello input/output edge projections."""

    token_projection_weight: jax.Array
    proprioception_projection_weight: jax.Array
    first_bias: jax.Array
    middle_weights: tuple[jax.Array, ...]
    middle_biases: tuple[jax.Array, ...]
    output_weight: jax.Array
    output_bias: jax.Array

    def __init__(self, initialization: SonicDecoderInitialization):
        self.token_projection_weight = jnp.asarray(
            initialization.token_projection_weight
        )
        self.proprioception_projection_weight = jnp.asarray(
            initialization.proprioception_projection_weight
        )
        self.first_bias = jnp.asarray(initialization.first_bias)
        self.middle_weights = tuple(
            jnp.asarray(value) for value in initialization.middle_weights
        )
        self.middle_biases = tuple(
            jnp.asarray(value) for value in initialization.middle_biases
        )
        self.output_weight = jnp.asarray(initialization.output_weight)
        self.output_bias = jnp.asarray(initialization.output_bias)

    def __call__(self, tokens: jax.Array, proprioception: jax.Array) -> jax.Array:
        value = (
            jnp.matmul(
                tokens,
                jax.lax.stop_gradient(self.token_projection_weight).T,
                precision=jax.lax.Precision.HIGHEST,
            )
            + jnp.matmul(
                proprioception,
                self.proprioception_projection_weight.T,
                precision=jax.lax.Precision.HIGHEST,
            )
            + self.first_bias
        )
        value = jax.nn.silu(value)
        for weight, bias in zip(self.middle_weights, self.middle_biases, strict=True):
            value = jax.nn.silu(
                jnp.matmul(
                    value,
                    jax.lax.stop_gradient(weight).T,
                    precision=jax.lax.Precision.HIGHEST,
                )
                + jax.lax.stop_gradient(bias)
            )
        return (
            jnp.matmul(
                value,
                self.output_weight.T,
                precision=jax.lax.Precision.HIGHEST,
            )
            + self.output_bias
        )


class SonicActor(eqx.Module):
    """Online SMPL encoder and strict edge-only Bello SONIC controller."""

    smpl_encoder: SonicFrozenSmplEncoder
    decoder: SonicEdgeDecoder

    def __init__(
        self,
        release: SonicReleaseWeights,
        initialization: SonicDecoderInitialization,
    ):
        self.smpl_encoder = SonicFrozenSmplEncoder(release)
        self.decoder = SonicEdgeDecoder(initialization)

    def __call__(self, observation: jax.Array) -> jax.Array:
        if observation.shape[-1] != SONIC_ACTOR_OBSERVATION_SIZE:
            raise ValueError(
                f"SONIC observation must have {SONIC_ACTOR_OBSERVATION_SIZE} "
                f"features, got {observation.shape[-1]}"
            )
        smpl_input = observation[:SONIC_SMPL_INPUT_SIZE]
        proprioception = observation[SONIC_SMPL_INPUT_SIZE:]
        if proprioception.shape[-1] != SONIC_BELLO_PROPRIO_SIZE:
            raise RuntimeError("SONIC Bello proprioception slice has drifted")
        latent = self.smpl_encoder(smpl_input)
        tokens = sonic_fsq(latent.reshape((2, 32))).reshape((SONIC_TOKEN_FLAT_SIZE,))
        return self.decoder(tokens, proprioception)


class TwistBackbone(eqx.Module):
    """Released TWIST SiLU backbone with final-hidden LayerNorm."""

    hidden: tuple[eqx.nn.Linear, ...]
    final_norm: eqx.nn.LayerNorm
    output: eqx.nn.Linear

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_dims: tuple[int, ...],
        *,
        initialization: Literal["default", "twist2_actor"] = "default",
        key: jax.Array,
    ):
        if hidden_dims != (512, 512, 256, 128):
            raise ValueError(
                "TWIST actor/critic hidden dimensions must be (512, 512, 256, 128)"
            )
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = (in_size, *hidden_dims)
        if initialization == "twist2_actor":
            gains = (0.5, *(1.0 for _ in hidden_dims[1:]))
            self.hidden = tuple(
                _xavier_linear(
                    input_size,
                    output_size,
                    gain=gain,
                    key=layer_key,
                )
                for input_size, output_size, gain, layer_key in zip(
                    dims[:-1], dims[1:], gains, keys[:-1], strict=True
                )
            )
            self.output = _xavier_linear(
                hidden_dims[-1], out_size, gain=0.1, key=keys[-1]
            )
        else:
            self.hidden = tuple(
                eqx.nn.Linear(input_size, output_size, key=layer_key)
                for input_size, output_size, layer_key in zip(
                    dims[:-1], dims[1:], keys[:-1], strict=True
                )
            )
            self.output = eqx.nn.Linear(hidden_dims[-1], out_size, key=keys[-1])
        self.final_norm = eqx.nn.LayerNorm(hidden_dims[-1])

    def __call__(self, value: jax.Array) -> jax.Array:
        for index, layer in enumerate(self.hidden):
            value = layer(value)
            if index == len(self.hidden) - 1:
                value = self.final_norm(value)
            value = jax.nn.silu(value)
        return self.output(value)


class TwistTeacherReferenceEncoder(eqx.Module):
    """20-frame reference encoder from the released TWIST architecture."""

    frame_projection: eqx.nn.Linear
    convolution1: eqx.nn.Conv1d
    convolution2: eqx.nn.Conv1d
    latent_projection: eqx.nn.Linear
    frame_count: int = eqx.field(static=True)
    frame_size: int = eqx.field(static=True)

    def __init__(
        self,
        frame_size: int,
        *,
        frame_count: int = 20,
        key: jax.Array,
    ):
        if frame_count != 20:
            raise ValueError("TWIST teacher encoder requires 20 future frames")
        keys = jax.random.split(key, 4)
        self.frame_projection = eqx.nn.Linear(frame_size, 60, key=keys[0])
        self.convolution1 = eqx.nn.Conv1d(60, 40, kernel_size=6, stride=2, key=keys[1])
        self.convolution2 = eqx.nn.Conv1d(40, 20, kernel_size=4, stride=2, key=keys[2])
        self.latent_projection = eqx.nn.Linear(60, 128, key=keys[3])
        self.frame_count = frame_count
        self.frame_size = frame_size

    def __call__(self, reference: jax.Array) -> jax.Array:
        frames = reference.reshape((self.frame_count, self.frame_size))
        projected = jax.vmap(self.frame_projection)(frames).T
        hidden = jax.nn.silu(self.convolution1(projected))
        hidden = jax.nn.silu(self.convolution2(hidden))
        if hidden.shape != (20, 3):
            raise RuntimeError(
                f"TWIST teacher convolution shape drifted: {hidden.shape}"
            )
        return self.latent_projection(hidden.reshape(-1))


class TwistStudentReferenceEncoder(eqx.Module):
    """Separate current-frame encoder used only by the deployable student."""

    projection: eqx.nn.Linear
    latent_projection: eqx.nn.Linear

    def __init__(self, frame_size: int, *, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.projection = eqx.nn.Linear(frame_size, 60, key=key1)
        self.latent_projection = eqx.nn.Linear(60, 128, key=key2)

    def __call__(self, reference: jax.Array) -> jax.Array:
        return self.latent_projection(jax.nn.silu(self.projection(reference)))


class TwistActor(eqx.Module):
    reference_encoder: eqx.Module
    backbone: TwistBackbone
    stage: str = eqx.field(static=True)
    reference_frame_size: int = eqx.field(static=True)
    proprioception_frame_size: int = eqx.field(static=True)
    reference_input_size: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        *,
        stage: str,
        hidden_dims: tuple[int, ...],
        key: jax.Array,
    ):
        if stage not in {"teacher", "student"}:
            raise ValueError(f"invalid TWIST actor stage {stage!r}")
        encoder_key, backbone_key = jax.random.split(key)
        teacher_reference_frame_size = 35 + action_size
        student_reference_frame_size = 8 + action_size
        proprioception_frame_size = 5 + 3 * action_size
        if stage == "teacher":
            reference_frame_size = teacher_reference_frame_size
            reference_input_size = 20 * reference_frame_size
            privileged_size = 3 + 1 + 27 + 2 + 5 + 2 * action_size
            expected_obs_size = (
                reference_input_size + proprioception_frame_size + privileged_size
            )
            self.reference_encoder = TwistTeacherReferenceEncoder(
                reference_frame_size, key=encoder_key
            )
            backbone_input_size = (
                128 + reference_frame_size + (obs_size - reference_input_size)
            )
        else:
            reference_frame_size = student_reference_frame_size
            reference_input_size = reference_frame_size
            student_frame_size = reference_frame_size + proprioception_frame_size
            expected_obs_size = 11 * student_frame_size
            self.reference_encoder = TwistStudentReferenceEncoder(
                reference_frame_size, key=encoder_key
            )
            backbone_input_size = 128 + obs_size
        if obs_size != expected_obs_size:
            raise ValueError(
                f"TWIST {stage} actor expected {expected_obs_size} observations, "
                f"got {obs_size}"
            )
        self.backbone = TwistBackbone(
            backbone_input_size,
            action_size,
            hidden_dims,
            key=backbone_key,
        )
        self.stage = stage
        self.reference_frame_size = reference_frame_size
        self.proprioception_frame_size = proprioception_frame_size
        self.reference_input_size = reference_input_size

    def __call__(self, observation: jax.Array) -> jax.Array:
        reference = self._reference_observation(observation)
        latent = self.reference_encoder(reference)
        if self.stage == "teacher":
            remaining = observation[self.reference_input_size :]
            current_reference = reference[: self.reference_frame_size]
            features = jnp.concatenate((remaining, current_reference, latent))
        else:
            features = jnp.concatenate((observation, latent))
        return self.backbone(features)

    def _reference_observation(self, observation: jax.Array) -> jax.Array:
        if self.stage == "teacher":
            return observation[: self.reference_input_size]
        frame_size = self.reference_frame_size + self.proprioception_frame_size
        current_start = 10 * frame_size
        return observation[current_start : current_start + self.reference_frame_size]


class TwistCritic(eqx.Module):
    """Released privileged critic with its own temporal reference encoder."""

    reference_encoder: TwistTeacherReferenceEncoder
    backbone: TwistBackbone
    reference_frame_size: int = eqx.field(static=True)
    reference_input_size: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_dims: tuple[int, ...],
        *,
        key: jax.Array,
    ):
        encoder_key, backbone_key = jax.random.split(key)
        self.reference_frame_size = 35 + action_size
        self.reference_input_size = 20 * self.reference_frame_size
        if obs_size <= self.reference_input_size:
            raise ValueError("TWIST critic requires a privileged observation suffix")
        self.reference_encoder = TwistTeacherReferenceEncoder(
            self.reference_frame_size, key=encoder_key
        )
        self.backbone = TwistBackbone(
            128 + self.reference_frame_size + obs_size - self.reference_input_size,
            1,
            hidden_dims,
            key=backbone_key,
        )

    def __call__(self, observation: jax.Array) -> jax.Array:
        reference = observation[: self.reference_input_size]
        latent = self.reference_encoder(reference)
        remaining = observation[self.reference_input_size :]
        current_reference = reference[: self.reference_frame_size]
        return self.backbone(jnp.concatenate((remaining, current_reference, latent)))


class Twist2CurrentCommandEncoder(eqx.Module):
    """Released offset-zero motion encoder used by the TWIST2 actor."""

    projection: eqx.nn.Linear
    latent_projection: eqx.nn.Linear

    def __init__(self, frame_size: int, *, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.projection = eqx.nn.Linear(frame_size, 60, key=key1)
        self.latent_projection = eqx.nn.Linear(60, 128, key=key2)

    def __call__(self, command: jax.Array) -> jax.Array:
        return self.latent_projection(jax.nn.silu(self.projection(command)))


class Twist2HistoryEncoder(eqx.Module):
    """Ten-frame actor history encoder from the pinned TWIST2 release."""

    frame_projection: eqx.nn.Linear
    convolution1: eqx.nn.Conv1d
    convolution2: eqx.nn.Conv1d
    latent_projection: eqx.nn.Linear
    frame_size: int = eqx.field(static=True)

    def __init__(self, frame_size: int, *, key: jax.Array):
        keys = jax.random.split(key, 4)
        self.frame_projection = eqx.nn.Linear(frame_size, 60, key=keys[0])
        self.convolution1 = eqx.nn.Conv1d(60, 40, kernel_size=4, stride=2, key=keys[1])
        self.convolution2 = eqx.nn.Conv1d(40, 20, kernel_size=2, stride=1, key=keys[2])
        self.latent_projection = eqx.nn.Linear(60, 128, key=keys[3])
        self.frame_size = frame_size

    def __call__(self, history: jax.Array) -> jax.Array:
        frames = history.reshape((10, self.frame_size))
        projected = jax.vmap(self.frame_projection)(frames).T
        hidden = jax.nn.silu(self.convolution1(projected))
        hidden = jax.nn.silu(self.convolution2(hidden))
        if hidden.shape != (20, 3):
            raise RuntimeError(
                f"TWIST2 history convolution shape drifted: {hidden.shape}"
            )
        return self.latent_projection(hidden.reshape(-1))


class Twist2OffsetZeroEncoder(eqx.Module):
    """Corrected TWIST2 future branch retaining the complete current command."""

    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear
    dropout_probability: float = eqx.field(static=True)

    def __init__(
        self,
        frame_size: int,
        *,
        dropout_probability: float = 0.1,
        key: jax.Array,
    ):
        keys = jax.random.split(key, 3)
        self.layer1 = _xavier_linear(frame_size, 256, gain=0.5, key=keys[0])
        self.layer2 = _xavier_linear(256, 128, gain=0.5, key=keys[1])
        self.layer3 = _xavier_linear(128, 128, gain=0.5, key=keys[2])
        self.dropout_probability = dropout_probability

    def __call__(
        self, command: jax.Array, *, key: jax.Array | None = None
    ) -> jax.Array:
        key1 = key2 = None
        if key is not None:
            key1, key2 = jax.random.split(key)
        hidden = jax.nn.silu(self.layer1(command))
        hidden = _dropout(hidden, self.dropout_probability, key1)
        hidden = jax.nn.silu(self.layer2(hidden))
        hidden = _dropout(hidden, self.dropout_probability, key2)
        return self.layer3(hidden)


class Twist2Actor(eqx.Module):
    current_encoder: Twist2CurrentCommandEncoder
    history_encoder: Twist2HistoryEncoder
    offset_zero_encoder: Twist2OffsetZeroEncoder
    backbone: TwistBackbone
    command_size: int = eqx.field(static=True)
    proprioception_size: int = eqx.field(static=True)
    frame_size: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        *,
        hidden_dims: tuple[int, ...],
        key: jax.Array,
    ):
        keys = jax.random.split(key, 4)
        self.command_size = 6 + action_size
        self.proprioception_size = 5 + 3 * action_size
        self.frame_size = self.command_size + self.proprioception_size
        expected_obs_size = 11 * self.frame_size + self.command_size
        if obs_size != expected_obs_size:
            raise ValueError(
                f"TWIST2 actor expected {expected_obs_size} observations, got "
                f"{obs_size}"
            )
        self.current_encoder = Twist2CurrentCommandEncoder(
            self.command_size, key=keys[0]
        )
        self.history_encoder = Twist2HistoryEncoder(self.frame_size, key=keys[1])
        self.offset_zero_encoder = Twist2OffsetZeroEncoder(
            self.command_size, key=keys[2]
        )
        self.backbone = TwistBackbone(
            self.frame_size + 3 * 128,
            action_size,
            hidden_dims,
            initialization="twist2_actor",
            key=keys[3],
        )

    def __call__(
        self, observation: jax.Array, *, key: jax.Array | None = None
    ) -> jax.Array:
        current = observation[: self.frame_size]
        current_command = current[: self.command_size]
        history_start = self.frame_size
        history_end = history_start + 10 * self.frame_size
        history = observation[history_start:history_end]
        offset_zero = observation[history_end : history_end + self.command_size]
        features = jnp.concatenate(
            (
                current,
                self.current_encoder(current_command),
                self.history_encoder(history),
                self.offset_zero_encoder(offset_zero, key=key),
            )
        )
        return self.backbone(features)


class Twist2Critic(eqx.Module):
    reference_encoder: TwistTeacherReferenceEncoder
    backbone: TwistBackbone
    reference_frame_size: int = eqx.field(static=True)
    reference_input_size: int = eqx.field(static=True)

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        hidden_dims: tuple[int, ...],
        *,
        key: jax.Array,
    ):
        encoder_key, backbone_key = jax.random.split(key)
        self.reference_frame_size = 48 + action_size
        self.reference_input_size = 20 * self.reference_frame_size
        if obs_size <= self.reference_input_size:
            raise ValueError("TWIST2 critic requires a private observation suffix")
        self.reference_encoder = TwistTeacherReferenceEncoder(
            self.reference_frame_size, key=encoder_key
        )
        self.backbone = TwistBackbone(
            128 + self.reference_frame_size + obs_size - self.reference_input_size,
            1,
            hidden_dims,
            key=backbone_key,
        )

    def __call__(self, observation: jax.Array) -> jax.Array:
        reference = observation[: self.reference_input_size]
        latent = self.reference_encoder(reference)
        remaining = observation[self.reference_input_size :]
        current_reference = reference[: self.reference_frame_size]
        return self.backbone(jnp.concatenate((remaining, current_reference, latent)))


def _dropout(value: jax.Array, probability: float, key: jax.Array | None) -> jax.Array:
    if key is None or probability == 0.0:
        return value
    keep_probability = 1.0 - probability
    mask = jax.random.bernoulli(key, keep_probability, value.shape)
    return jnp.where(mask, value / keep_probability, 0.0)


def policy_log_std(policy: ActorCritic) -> jax.Array:
    return jnp.clip(policy.log_std, policy.min_log_std, policy.max_log_std)


def clamp_policy_log_std(policy: ActorCritic) -> ActorCritic:
    return eqx.tree_at(lambda model: model.log_std, policy, policy_log_std(policy))


def policy_observation_size(obs_size: int, config: ActorCriticConfig) -> int:
    return obs_size * config.obs_stack_size


def init_observation_stack(obs: jax.Array, stack_size: int) -> jax.Array:
    obs = jnp.asarray(obs)
    return jnp.repeat(obs[..., None, :], stack_size, axis=-2)


def stacked_observation(obs_stack: jax.Array) -> jax.Array:
    return obs_stack.reshape(obs_stack.shape[:-2] + (-1,))


def update_observation_stack(
    obs_stack: jax.Array,
    obs: jax.Array,
    done: jax.Array,
) -> jax.Array:
    next_stack = jnp.concatenate((obs_stack[..., 1:, :], obs[..., None, :]), axis=-2)
    reset_stack = init_observation_stack(obs, obs_stack.shape[-2])
    done = done.astype(bool)
    while done.ndim < obs_stack.ndim:
        done = done[..., None]
    return jnp.where(done, reset_stack, next_stack)


def apply_policy(
    policy: ActorCritic,
    obs: jax.Array,
    critic_obs: jax.Array | None = None,
    *,
    network_rng: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if critic_obs is None:
        critic_obs = obs
    if obs.ndim == 1:
        return policy(obs, critic_obs, key=network_rng)
    if network_rng is None:
        return jax.vmap(lambda actor_obs, value_obs: policy(actor_obs, value_obs))(
            obs, critic_obs
        )
    keys = jax.random.split(network_rng, obs.shape[0])
    return jax.vmap(
        lambda actor_obs, value_obs, key: policy(actor_obs, value_obs, key=key)
    )(obs, critic_obs, keys)


def apply_actor(policy: ActorCritic, obs: jax.Array) -> jax.Array:
    if obs.ndim == 1:
        return policy.action_mean(obs)
    return jax.vmap(policy.action_mean)(obs)


def sample_actor_action(
    policy: ActorCritic,
    obs: jax.Array,
    rng: jax.Array,
) -> jax.Array:
    """Sample an actor action without evaluating the critic."""

    def sample_one(observation: jax.Array, key: jax.Array) -> jax.Array:
        latent_mean = policy.actor(observation)
        latent_action = latent_mean + jnp.exp(
            policy_log_std(policy)
        ) * jax.random.normal(key, latent_mean.shape)
        return transform_action(policy, latent_action)

    if obs.ndim == 1:
        return sample_one(obs, rng)
    return jax.vmap(sample_one)(obs, jax.random.split(rng, obs.shape[0]))


def sample_action_with_latent_and_mean(
    policy: ActorCritic,
    obs: jax.Array,
    rng: jax.Array,
    critic_obs: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    network_rng = None
    action_rng = rng
    if policy.stochastic_actor:
        network_rng, action_rng = jax.random.split(rng)
    latent_mean, log_std, value = apply_policy(
        policy, obs, critic_obs, network_rng=network_rng
    )
    noise = jax.random.normal(action_rng, latent_mean.shape)
    latent_action = latent_mean + jnp.exp(log_std) * noise
    action = transform_action(policy, latent_action)
    log_prob = policy_log_prob(
        policy,
        latent_mean,
        log_std,
        action,
        latent_action,
    )
    return action, latent_action, log_prob, value, transform_action(policy, latent_mean)


def evaluate_actions(
    policy: ActorCritic,
    obs: jax.Array,
    action: jax.Array,
    critic_obs: jax.Array | None = None,
    *,
    latent_action: jax.Array | None = None,
    entropy_rng: jax.Array | None = None,
    network_rng: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    latent_mean, log_std, value = apply_policy(
        policy, obs, critic_obs, network_rng=network_rng
    )
    log_prob = policy_log_prob(
        policy, latent_mean, log_std, action, latent_action=latent_action
    )
    entropy = gaussian_entropy(log_std)
    if policy.action_distribution == "tanh_gaussian" and entropy_rng is not None:
        entropy = reparameterized_policy_entropy(
            policy, latent_mean, log_std, entropy_rng
        )
    return log_prob, entropy, value


def transform_action(policy: ActorCritic, latent_action: jax.Array) -> jax.Array:
    if policy.action_distribution == "gaussian":
        return latent_action
    return policy.action_limit * jnp.tanh(latent_action)


def policy_log_prob(
    policy: ActorCritic,
    latent_mean: jax.Array,
    log_std: jax.Array,
    action: jax.Array,
    latent_action: jax.Array | None = None,
) -> jax.Array:
    if policy.action_distribution == "gaussian":
        return gaussian_log_prob(latent_mean, log_std, action)
    limit = jnp.asarray(policy.action_limit, dtype=action.dtype)
    normalized_action = jnp.clip(action / limit, -1.0 + 1.0e-6, 1.0 - 1.0e-6)
    if latent_action is None:
        latent_action = jnp.arctanh(normalized_action)
    base_log_prob = gaussian_log_prob(latent_mean, log_std, latent_action)
    log_det = jnp.log(limit) + 2.0 * (
        jnp.log(2.0) - latent_action - jax.nn.softplus(-2.0 * latent_action)
    )
    return base_log_prob - jnp.sum(log_det, axis=-1)


def gaussian_log_prob(
    mean: jax.Array, log_std: jax.Array, action: jax.Array
) -> jax.Array:
    inv_std = jnp.exp(-log_std)
    normalized = (action - mean) * inv_std
    log_probs = -0.5 * (normalized**2 + 2.0 * log_std + LOG_2PI)
    return jnp.sum(log_probs, axis=-1)


def gaussian_entropy(log_std: jax.Array) -> jax.Array:
    entropy = 0.5 * (1.0 + LOG_2PI) + log_std
    return jnp.sum(entropy, axis=-1)


def reparameterized_policy_entropy(
    policy: ActorCritic,
    latent_mean: jax.Array,
    log_std: jax.Array,
    rng: jax.Array,
) -> jax.Array:
    """One-sample pathwise entropy estimate for the current squashed policy."""

    latent_action = latent_mean + jnp.exp(log_std) * jax.random.normal(
        rng, latent_mean.shape
    )
    action = transform_action(policy, latent_action)
    return -policy_log_prob(
        policy,
        latent_mean,
        log_std,
        action,
        latent_action=latent_action,
    )


def activation_function(name: str):
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    if name == "elu":
        return jax.nn.elu
    if name == "swish":
        return jax.nn.swish
    raise ValueError(f"unsupported activation: {name!r}")


class PolicyInferenceStep(NamedTuple):
    action_mean: jax.Array
    requested_action: jax.Array
    applied_action: jax.Array


def infer_policy_action(
    policy: Any,
    observation: jax.Array,
    normalization: ObservationNormState,
    config: PPOConfig,
    action_clip: float,
    *,
    rng: jax.Array | None = None,
) -> PolicyInferenceStep:
    """Evaluate the actor with the same input and action contract used by PPO."""
    from bello_mujoco.training.ppo import normalize_observation

    policy_observation = normalize_observation(observation, normalization, config)
    action_mean = apply_actor(policy, policy_observation)
    requested_action = (
        action_mean
        if rng is None
        else sample_actor_action(policy, policy_observation, rng)
    )
    applied_action = jnp.clip(requested_action, -action_clip, action_clip)
    return PolicyInferenceStep(action_mean, requested_action, applied_action)
