"""Feed-forward Gaussian actor-critic policy utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from bello_mujoco.training.ppo import ObservationNormState, PPOConfig

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
    architecture: Literal["mlp", "twist_teacher", "twist_student"] = "mlp"
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

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        config: ActorCriticConfig = ActorCriticConfig(),
        *,
        critic_obs_size: int | None = None,
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
        else:
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
        self.log_std = jnp.full((action_size,), config.init_log_std, dtype=jnp.float32)
        self.min_log_std = config.min_log_std
        self.max_log_std = config.max_log_std
        self.action_distribution = config.action_distribution
        self.action_limit = config.action_limit

    def __call__(
        self, obs: jax.Array, critic_obs: jax.Array | None = None
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        critic_obs = obs if critic_obs is None else critic_obs
        mean = self.actor(obs)
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
) -> ActorCritic:
    return ActorCritic(
        obs_size, action_size, config, critic_obs_size=critic_obs_size, key=rng
    )


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
        key: jax.Array,
    ):
        if hidden_dims != (512, 512, 256, 128):
            raise ValueError(
                "TWIST actor/critic hidden dimensions must be (512, 512, 256, 128)"
            )
        keys = jax.random.split(key, len(hidden_dims) + 1)
        dims = (in_size, *hidden_dims)
        self.hidden = tuple(
            eqx.nn.Linear(input_size, output_size, key=layer_key)
            for input_size, output_size, layer_key in zip(
                dims[:-1], dims[1:], keys[:-1], strict=True
            )
        )
        self.final_norm = eqx.nn.LayerNorm(hidden_dims[-1])
        self.output = eqx.nn.Linear(hidden_dims[-1], out_size, key=keys[-1])

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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if critic_obs is None:
        critic_obs = obs
    if obs.ndim == 1:
        return policy(obs, critic_obs)
    return jax.vmap(policy)(obs, critic_obs)


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
    latent_mean, log_std, value = apply_policy(policy, obs, critic_obs)
    noise = jax.random.normal(rng, latent_mean.shape)
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    latent_mean, log_std, value = apply_policy(policy, obs, critic_obs)
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
