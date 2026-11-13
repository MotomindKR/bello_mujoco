"""Short on-policy rollouts through learned dynamics and the actual command path."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic.feedback import (
    actor_sample,
    add_actor_noise,
    advance_feedback,
)
from bello_mujoco.supermimic.mdp import (
    critic_features,
    imagined_return,
    survival_weights,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ImaginedRollout:
    rewards: jax.Array
    terminations: jax.Array
    critic_inputs: jax.Array


def imagine(actor, model, starts, env, normalizers, *, key, horizon):
    input_norm, observation_norm, goal_norm = normalizers
    tr = starts.transition
    count = tr.action.shape[-1]

    def single(start, history, feedback_history, noise_key):
        context, xy, yaw = start.context, start.alignment_xy, start.alignment_yaw
        delay = context[-1].astype(jnp.int32)
        noise_key, history_key = jax.random.split(noise_key)
        history = add_actor_noise(
            history,
            history_key,
            joint_count=count,
            noise=env.config.observation_noise,
            filtering=env.config.observation_low_pass_filter,
        )

        def advance(frame, _):
            frame = env.advance_reference_time(frame)
            return frame, frame

        _, frames = jax.lax.scan(advance, start.reference_frame, None, length=horizon)
        frames = jnp.concatenate((start.reference_frame[None], frames))
        references = jax.vmap(lambda f: env.reference_state(f, xy, yaw))(frames)

        def step(carry, inputs):
            state, observations, feedback, rng, survival = carry
            frame, reference = inputs
            active = survival > 0
            model_state = jax.tree.map(
                lambda value, first: jnp.where(active, value, first), state, start.state
            )
            observations = jnp.where(active, observations, history)
            feedback = jax.tree.map(
                lambda value, first: jnp.where(active, value, first),
                feedback,
                feedback_history,
            )
            rng, feedback_key, observation_key = jax.random.split(rng, 3)
            goal = env.codec.actor_goal(env.goal(frame, model_state, xy, yaw))
            action = jnp.where(
                active, actor(observations, goal, observation_norm, goal_norm), 0.0
            )
            predicted = model.step(model_state, action, context, input_norm)
            successor = jax.tree.map(
                lambda value, previous: jnp.where(active, value, previous),
                predicted,
                state,
            )
            termination = jnp.where(
                active, env.mdp.soft_termination(predicted, reference), 1.0
            )
            feedback = advance_feedback(
                feedback,
                successor,
                feedback_key,
                noise=env.config.observation_noise,
                filtering=env.config.observation_low_pass_filter,
                timestep=env.control_timestep,
            )
            measured = jax.tree.map(lambda x: x[delay], feedback)
            sample = actor_sample(
                env.codec,
                measured,
                action,
                context[:count],
                observation_key,
                noise=env.config.observation_noise,
                filtering=env.config.observation_low_pass_filter,
            )
            observations = env.codec.advance_history(observations, sample)
            return (
                successor,
                observations,
                feedback,
                rng,
                survival * (1.0 - termination),
            ), (successor, action, termination)

        initial = (start.state, history, feedback_history, noise_key, jnp.asarray(1.0))
        _, (states, actions, terminations) = jax.lax.scan(
            step, initial, (frames[:-1], jax.tree.map(lambda x: x[:-1], references))
        )
        states = jax.tree.map(
            lambda first, rest: jnp.concatenate((first[None], rest)),
            start.state,
            states,
        )
        previous = jnp.concatenate((start.previous_action[None], actions[:-1]))
        goals = jax.vmap(lambda frame, state: env.goal(frame, state, xy, yaw))(
            frames, states
        )
        features = jax.vmap(
            lambda state, goal: critic_features(
                state, goal, model.context_features(context)
            )
        )(states, goals)
        return states, actions, previous, frames, references, features, terminations

    trajectories = jax.vmap(single)(
        tr,
        starts.history,
        starts.feedback_history,
        jax.random.split(key, tr.action.shape[0]),
    )
    states, actions, previous, frames, references, critic_inputs, terminals = (
        jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectories)
    )

    def flatten(value):
        return value.reshape((-1,) + value.shape[2:])

    def restore(value):
        return value.reshape((horizon + 1, tr.action.shape[0]) + value.shape[1:])

    bodies = jax.tree.map(
        restore, jax.vmap(env.kinematics.forward)(jax.tree.map(flatten, states))
    )
    xy = jnp.broadcast_to(tr.alignment_xy, (horizon + 1,) + tr.alignment_xy.shape)
    yaw = jnp.broadcast_to(tr.alignment_yaw, frames.shape)
    targets = jax.tree.map(
        restore,
        jax.vmap(env.reference_bodies)(
            frames.reshape(-1), flatten(xy), yaw.reshape(-1)
        ),
    )

    def reward(state, reference, action, previous_action, body, target):
        return env.mdp.reward(
            state, reference, action, previous_action, bodies=body, targets=target
        )[0]

    successors, next_bodies = jax.tree.map(lambda x: x[1:], (states, bodies))
    scored_references, scored_targets = jax.tree.map(
        lambda x: x[:-1], (references, targets)
    )
    rewards = jax.vmap(reward)(
        *jax.tree.map(
            flatten,
            (
                successors,
                scored_references,
                actions,
                previous,
                next_bodies,
                scored_targets,
            ),
        )
    ).reshape((horizon, tr.action.shape[0]))
    rewards = jnp.where(survival_weights(terminals)[:-1] > 0, rewards, 0.0)
    return ImaginedRollout(rewards, terminals, critic_inputs)


def actor_loss(actor, model, critic, starts, env, normalizers, *, key, horizon, gamma):
    rollout = imagine(actor, model, starts, env, normalizers, key=key, horizon=horizon)
    bootstrap = jnp.minimum(
        critic(rollout.critic_inputs[-1]),
        env.mdp.maximum_return(gamma, env.max_episode_steps),
    )
    objective = imagined_return(rollout.rewards, rollout.terminations, bootstrap, gamma)
    return -objective.mean(), rollout
