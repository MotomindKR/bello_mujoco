"""Separate dynamics, analytic-gradient actor, and imagined-return critic updates."""

from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bello_mujoco.supermimic.imagination import actor_loss
from bello_mujoco.supermimic.mdp import lambda_returns
from bello_mujoco.supermimic.model import (
    Actor,
    Critic,
    Moments,
    WorldModel,
    world_model_loss,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LearnerState:
    actor: Actor
    model: WorldModel
    critic: Critic
    actor_optimizer: object
    model_optimizer: object
    critic_optimizer: object
    input_norm: Moments
    output_norm: Moments
    observation_norm: Moments
    goal_norm: Moments
    critic_norm: Moments


def optimizers(config):
    def make(beta2, element_clip=None):
        transforms = [optax.clip_by_global_norm(config.max_gradient_norm)]
        if element_clip is not None:
            transforms.append(optax.clip(element_clip))
        return optax.chain(
            *transforms, optax.scale_by_adam(b1=0.9, b2=beta2), optax.scale(-1.0)
        )

    return make(0.95, config.actor_gradient_clip), make(0.999), make(0.999)


def initialize_learner(env, config, networks, context_size, key):
    actor_key, model_key, critic_key = jax.random.split(key, 3)
    n, codec = env.action_size, env.codec
    actor = Actor(
        codec.sample_size,
        codec.config.history_frames,
        codec.goal_size,
        n,
        config=networks,
        key=actor_key,
    )
    model = WorldModel(n, context_size, config=networks, key=model_key)
    critic_size = (
        22 + 3 * n + 3 * len(env.kinematics.body_names) + context_size + codec.goal_size
    )
    critic = Critic(critic_size, config=networks, key=critic_key)
    actor_opt, model_opt, critic_opt = optimizers(config)
    return LearnerState(
        actor,
        model,
        critic,
        actor_opt.init(eqx.filter(actor, eqx.is_inexact_array)),
        model_opt.init(eqx.filter(model, eqx.is_inexact_array)),
        critic_opt.init(eqx.filter(critic, eqx.is_inexact_array)),
        Moments.empty(13 + 4 * n + context_size),
        Moments.empty(12 + 2 * n),
        Moments.empty(codec.sample_size),
        Moments.empty(codec.goal_size),
        Moments.empty(critic_size),
    )


def learning_rate(config, update):
    progress = jnp.minimum(update / config.learning_rate_decay_updates, 1.0)
    return config.learning_rate * (
        config.learning_rate_final_fraction
        + (1.0 - config.learning_rate_final_fraction)
        * 0.5
        * (1.0 + jnp.cos(jnp.pi * progress))
    )


def finite_tree(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


def apply_optimizer(network, gradient, state, optimizer, rate):
    updates, next_state = optimizer.update(
        gradient, state, eqx.filter(network, eqx.is_inexact_array)
    )
    result = eqx.apply_updates(network, jax.tree.map(lambda x: rate * x, updates))
    valid = (
        finite_tree(gradient)
        & finite_tree(eqx.filter(result, eqx.is_inexact_array))
        & finite_tree(next_state)
    )
    result, next_state = jax.lax.cond(
        valid, lambda: (result, next_state), lambda: (network, state)
    )
    return result, next_state, valid


def update_world_model(learner, window, config, update):
    input_norm, output_norm = learner.input_norm, learner.output_norm
    n = learner.model.joint_count
    floor = jnp.concatenate(
        (
            jnp.full(3, 0.003),
            jnp.full(3, 0.005),
            jnp.full(3, 0.02),
            jnp.full(3, 0.05),
            jnp.full(n, 0.003),
            jnp.full(n, 0.05),
        )
    )
    loss, gradient = eqx.filter_value_and_grad(world_model_loss)(
        learner.model, window, input_norm, output_norm, error_floor=floor
    )
    model, state, valid = apply_optimizer(
        learner.model,
        gradient,
        learner.model_optimizer,
        optimizers(config)[1],
        learning_rate(config, update),
    )
    valid &= jnp.isfinite(loss) & finite_tree((input_norm, output_norm))
    candidate = replace(
        learner,
        model=model,
        model_optimizer=state,
        input_norm=input_norm,
        output_norm=output_norm,
    )
    result = jax.lax.cond(valid, lambda: candidate, lambda: learner)
    return result, {
        "model/loss": loss / config.window_length,
        "model/gradient_norm": optax.global_norm(gradient),
        "invalid/model_update": (~valid).astype(jnp.float32),
    }


def update_actor_critic(learner, starts, env, config, update, key):
    actor_key, critic_key = jax.random.split(key)
    norms = (
        learner.input_norm,
        learner.output_norm,
        learner.observation_norm,
        learner.goal_norm,
    )
    (loss, rollout), gradient = eqx.filter_value_and_grad(actor_loss, has_aux=True)(
        learner.actor,
        learner.model,
        learner.critic,
        starts,
        env,
        norms,
        learner.critic_norm,
        key=actor_key,
        horizon=config.imagination_horizon,
        gamma=config.gamma,
    )
    actor_opt, _, critic_opt = optimizers(config)
    actor, actor_state, actor_valid = apply_optimizer(
        learner.actor,
        gradient,
        learner.actor_optimizer,
        actor_opt,
        learning_rate(config, update),
    )
    actor_valid &= jnp.isfinite(loss) & finite_tree(rollout)
    detached = jax.tree.map(jax.lax.stop_gradient, rollout)
    norm = learner.critic_norm.update(detached.critic_inputs)
    values = learner.critic(detached.critic_inputs, norm)
    targets = jax.lax.stop_gradient(
        lambda_returns(
            detached.rewards,
            detached.terminations,
            values,
            config.gamma,
            config.gae_lambda,
        )
    )
    features = detached.critic_inputs[:-1].reshape((-1, norm.mean.size))
    targets = targets.reshape(-1)
    batches = features.shape[0] // config.critic_minibatches

    def epoch(carry, rng):
        network, state = carry
        indices = jax.random.permutation(rng, features.shape[0]).reshape(
            (config.critic_minibatches, batches)
        )

        def minibatch(carry, index):
            network, state = carry

            def objective(net):
                return jnp.mean((net(features[index], norm) - targets[index]) ** 2)

            value, grad = eqx.filter_value_and_grad(objective)(network)
            network, state, valid = apply_optimizer(
                network, grad, state, critic_opt, learning_rate(config, update)
            )
            return (network, state), (value, valid & jnp.isfinite(value))

        return jax.lax.scan(minibatch, (network, state), indices)

    (critic, critic_state), (critic_losses, critic_valid) = jax.lax.scan(
        epoch,
        (learner.critic, learner.critic_optimizer),
        jax.random.split(critic_key, config.critic_epochs),
    )
    valid = actor_valid & jnp.all(critic_valid) & starts.valid & finite_tree(norm)
    candidate = replace(
        learner,
        actor=actor,
        actor_optimizer=actor_state,
        critic=critic,
        critic_optimizer=critic_state,
        critic_norm=norm,
    )
    result = jax.lax.cond(valid, lambda: candidate, lambda: learner)
    return result, {
        "actor/loss": loss,
        "critic/loss": critic_losses.mean(),
        "actor/gradient_norm": optax.global_norm(gradient),
        "imagination/reward": rollout.rewards.mean(),
        "invalid/actor_critic_update": (~valid).astype(jnp.float32),
    }
