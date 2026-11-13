"""Separate dynamics, analytic-gradient actor, and imagined-return critic updates."""

from dataclasses import dataclass, replace

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from bello_mujoco.supermimic.imagination import actor_loss
from bello_mujoco.supermimic.mdp import (
    imagined_return,
    lambda_returns,
    survival_weights,
)
from bello_mujoco.supermimic.model import (
    Actor,
    Critic,
    Moments,
    WorldModel,
    MLP,
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
    observation_norm: Moments
    goal_norm: Moments


def optimizers(config):
    def labels(params):
        hidden = {
            id(layer.weight)
            for module in jax.tree.leaves(params, is_leaf=lambda x: isinstance(x, MLP))
            if isinstance(module, MLP)
            for layer in module.layers[1:-1]
        }
        return jax.tree.map(
            lambda value: "muon" if id(value) in hidden else "adam", params
        )

    def make(beta2, element_clip=None):
        transforms = [optax.clip_by_global_norm(config.max_gradient_norm)]
        if element_clip is not None:
            transforms.append(optax.clip(element_clip))
        return optax.chain(
            *transforms,
            optax.partition(
                {
                    "muon": scale_by_muon(config.muon_weight_decay),
                    "adam": optax.scale_by_adam(b1=0.9, b2=beta2),
                },
                labels,
            ),
            optax.scale(-1.0),
        )

    return make(0.95, config.actor_gradient_clip), make(0.999), make(0.999)


def scale_by_muon(weight_decay):
    """Released PyTorch Muon momentum and match_rms_adamw scaling."""

    def init(params):
        return jax.tree.map(jnp.zeros_like, params)

    def update(updates, momentum, params=None):
        if params is None:
            raise ValueError("Muon requires parameters for decoupled weight decay")
        momentum = jax.tree.map(
            lambda old, grad: old + 0.05 * (grad - old), momentum, updates
        )

        def orthogonalize(grad, moment):
            value = (grad + 0.95 * (moment - grad)).astype(jnp.bfloat16)
            transpose = value.shape[0] > value.shape[1]
            value = value.T if transpose else value
            norm = jnp.sqrt(jnp.sum(value.astype(jnp.float32) ** 2)).astype(
                jnp.bfloat16
            )
            value = value / jnp.maximum(norm, jnp.asarray(1e-7, jnp.bfloat16))

            def step(_, value):
                def dot(a, b):
                    return jnp.matmul(a, b, preferred_element_type=jnp.float32)

                gram = dot(value, value.T).astype(jnp.bfloat16)
                correction = (
                    -4.775 * gram.astype(jnp.float32) + 2.0315 * dot(gram, gram)
                ).astype(jnp.bfloat16)
                return (
                    3.4445 * value.astype(jnp.float32) + dot(correction, value)
                ).astype(jnp.bfloat16)

            value = jax.lax.fori_loop(0, 5, step, value)
            value = value.T if transpose else value
            return value.astype(jnp.float32) * (0.2 * max(grad.shape) ** 0.5)

        updates = jax.tree.map(orthogonalize, updates, momentum)
        return jax.tree.map(
            lambda update, parameter: update + weight_decay * parameter, updates, params
        ), momentum

    return optax.GradientTransformation(init, update)


def initialize_learner(env, config, networks, context_size, key):
    actor_key, model_key, critic_key = jax.random.split(key, 3)
    n, codec = env.action_size, env.codec
    actor = Actor(
        codec.sample_size,
        codec.config.history_frames,
        codec.actor_goal_size,
        n,
        config=networks,
        key=actor_key,
    )
    if context_size != n + 5:
        raise ValueError("world-model context must match controlled joints")
    model = WorldModel(env, config=networks, key=model_key)
    critic_size = 14 + 3 * n + codec.goal_size
    critic = Critic(critic_size, config=networks, key=critic_key)
    actor_opt, model_opt, critic_opt = optimizers(config)
    return LearnerState(
        actor,
        model,
        critic,
        actor_opt.init(eqx.filter(actor, eqx.is_inexact_array)),
        model_opt.init(eqx.filter(model, eqx.is_inexact_array)),
        critic_opt.init(eqx.filter(critic, eqx.is_inexact_array)),
        Moments.empty(model.input_size),
        Moments.empty(codec.sample_size),
        Moments.empty(codec.actor_goal_size),
    )


def learning_rate(config, update):
    del update
    return config.learning_rate


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
    input_norm = learner.input_norm
    errors = (
        config.model_root_position_error,
        config.model_root_orientation_error,
        config.model_joint_position_error,
        config.model_joint_velocity_error,
    )
    loss, gradient = eqx.filter_value_and_grad(world_model_loss)(
        learner.model, window, input_norm, errors
    )
    model, state, valid = apply_optimizer(
        learner.model,
        gradient,
        learner.model_optimizer,
        optimizers(config)[1],
        learning_rate(config, update),
    )
    valid &= jnp.isfinite(loss) & finite_tree(input_norm) & jnp.any(window.mask)
    candidate = replace(
        learner,
        model=model,
        model_optimizer=state,
        input_norm=input_norm,
    )
    result = jax.lax.cond(valid, lambda: candidate, lambda: learner)
    return result, {
        "model/loss": loss,
        "model/gradient_norm": optax.global_norm(gradient),
        "invalid/model_update": (~valid).astype(jnp.float32),
    }


def update_actor_critic(learner, starts, env, config, update, key):
    # Unroll the differentiable tree separately from MJX collection compilation.
    kinematics = replace(env.kinematics, unroll=True)
    env = replace(
        env, kinematics=kinematics, mdp=replace(env.mdp, kinematics=kinematics)
    )
    actor_key, critic_key = jax.random.split(key)
    norms = (
        learner.input_norm,
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
    critic_inputs = detached.critic_inputs
    return_bound = env.mdp.maximum_return(config.gamma, env.max_episode_steps)
    raw_values = learner.critic(critic_inputs)
    targets = jax.lax.stop_gradient(
        jnp.minimum(
            lambda_returns(
                detached.rewards,
                detached.terminations,
                raw_values,
                config.gamma,
                config.gae_lambda,
            ),
            return_bound,
        )
    )
    features = critic_inputs[:-1].reshape((-1, learner.critic.input_size))
    targets = targets.reshape(-1)
    batches = features.shape[0] // config.critic_minibatches

    indices = jax.random.permutation(critic_key, features.shape[0]).reshape(
        (config.critic_minibatches, batches)
    )

    def epoch(carry, _):
        network, state = carry

        def minibatch(carry, index):
            network, state = carry

            def objective(net):
                return jnp.mean((net(features[index]) - targets[index]) ** 2)

            value, grad = eqx.filter_value_and_grad(objective)(network)
            updated, next_state, valid = apply_optimizer(
                network, grad, state, critic_opt, learning_rate(config, update)
            )
            return (updated, next_state), (value, valid & jnp.isfinite(value))

        return jax.lax.scan(minibatch, (network, state), indices)

    (critic, critic_state), (critic_losses, critic_valid) = jax.lax.scan(
        epoch,
        (learner.critic, learner.critic_optimizer),
        None,
        length=config.critic_epochs,
    )
    raw_bootstrap = learner.critic(detached.critic_inputs[-1])
    bootstrap = jnp.minimum(raw_bootstrap, return_bound)
    returns = imagined_return(
        detached.rewards, detached.terminations, bootstrap, config.gamma
    )

    def exceeds_bound(values, bound):
        return jnp.any(values > bound + 1e-5 * max(1.0, bound))

    invalid_bound = (
        exceeds_bound(detached.rewards, env.mdp.maximum_step_reward)
        | exceeds_bound(targets, return_bound)
        | exceeds_bound(returns, return_bound)
        | (-loss > return_bound + 1e-5 * max(1.0, return_bound))
    )
    invalid_loss = jnp.any(critic_losses > config.max_critic_loss)
    valid = (
        actor_valid
        & jnp.all(critic_valid)
        & starts.valid
        & finite_tree((raw_values, raw_bootstrap, targets, returns))
        & ~invalid_bound
        & ~invalid_loss
    )
    candidate = replace(
        learner,
        actor=actor,
        actor_optimizer=actor_state,
        critic=critic,
        critic_optimizer=critic_state,
    )
    result = jax.lax.cond(valid, lambda: candidate, lambda: learner)
    return result, {
        "actor/loss": loss,
        "critic/loss": critic_losses.mean(),
        "actor/gradient_norm": optax.global_norm(gradient),
        "imagination/step_reward": (
            rollout.rewards * survival_weights(rollout.terminations)[:-1]
        ).mean(),
        "diagnostic/imagined_reward_abs_max": jnp.abs(rollout.rewards).max(),
        "diagnostic/critic_target_abs_max": jnp.abs(targets).max(),
        "diagnostic/critic_value_abs_max": jnp.abs(raw_values).max(),
        "diagnostic/critic_target_max": targets.max(),
        "diagnostic/actor_return_max": returns.max(),
        "diagnostic/return_upper_bound": jnp.asarray(return_bound),
        "diagnostic/critic_loss_max": critic_losses.max(),
        "diagnostic/imagined_survival": survival_weights(rollout.terminations)[
            -1
        ].mean(),
        "invalid/actor_critic_update": (~valid).astype(jnp.float32),
        "invalid/return_bound": invalid_bound.astype(jnp.float32),
        "invalid/critic_loss": invalid_loss.astype(jnp.float32),
    }
