"""Short on-policy rollouts through learned dynamics and the actual command path."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic.feedback import (
    actor_sample,
    add_actor_noise,
    advance_feedback,
)
from bello_mujoco.supermimic.mdp import critic_features, imagined_return


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ImaginedRollout:
    rewards: jax.Array
    terminations: jax.Array
    critic_inputs: jax.Array


def imagine(actor, model, starts, env, normalizers, *, key, horizon):
    """Return time-major batches. Parameters are constant, state Jacobians remain live."""
    input_norm, output_norm, observation_norm, goal_norm = normalizers
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
            reference_orientation=env.codec.config.reference_orientation,
            noise=env.config.observation_noise,
            filtering=env.config.observation_low_pass_filter,
        )

        def advance(frame, _):
            frame = env.advance_reference_time(frame)
            return frame, frame

        _, frames = jax.lax.scan(advance, start.reference_frame, None, length=horizon)
        frames = jnp.concatenate((start.reference_frame[None], frames))
        references = jax.vmap(lambda f: env.reference_state(f, xy, yaw))(frames)
        goals = jax.vmap(env.goal)(frames)
        latents = actor.encode_goal(goals[:-1], goal_norm)

        def step(carry, inputs):
            state, observations, feedback, rng = carry
            latent, reference_quaternion = inputs
            rng, feedback_key, observation_key = jax.random.split(rng, 3)
            action = actor.decode(observations, latent, observation_norm)
            successor = model.step(state, action, context, input_norm, output_norm)
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
                reference_quaternion,
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
            ), (successor, action)

        initial = (
            start.state,
            history,
            feedback_history,
            noise_key,
        )
        _, (states, actions) = jax.lax.scan(
            step, initial, (latents, references.quaternion[1:])
        )
        states = jax.tree.map(
            lambda first, rest: jnp.concatenate((first[None], rest)),
            start.state,
            states,
        )
        previous = jnp.concatenate((start.previous_action[None], actions[:-1]))
        return states, actions, previous, frames, references, goals

    trajectories = jax.vmap(single)(
        tr,
        starts.history,
        starts.feedback_history,
        jax.random.split(key, tr.action.shape[0]),
    )
    states, actions, previous, frames, references, goals = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectories
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
    contexts = jnp.broadcast_to(tr.context, (horizon + 1,) + tr.context.shape)

    def features(state, reference, goal, context, body, target):
        return critic_features(
            state, reference, goal, context, env.kinematics, bodies=body, targets=target
        )

    critic_inputs = restore(
        jax.vmap(features)(
            *jax.tree.map(
                flatten, (states, references, goals, contexts, bodies, targets)
            )
        )
    )

    def reward(state, reference, action, previous_action, body, target):
        return env.mdp.reward(
            state, reference, action, previous_action, bodies=body, targets=target
        )[0]

    successors, next_references, next_bodies, next_targets = jax.tree.map(
        lambda x: x[1:], (states, references, bodies, targets)
    )
    rewards = jax.vmap(reward)(
        *jax.tree.map(
            flatten,
            (successors, next_references, actions, previous, next_bodies, next_targets),
        )
    ).reshape((horizon, tr.action.shape[0]))
    terminals = jax.vmap(env.mdp.soft_termination)(
        *jax.tree.map(flatten, (successors, next_references))
    ).reshape(rewards.shape)
    clips = jax.vmap(env.clip_id)(frames[1:].reshape(-1)).reshape(rewards.shape)
    end = env.reference_bank.time_offsets[clips] + env.reference_bank.lengths[clips] - 1
    terminals = jnp.where(frames[1:] >= end, 1.0, terminals)
    return ImaginedRollout(rewards, terminals, critic_inputs)


def actor_loss(
    actor, model, critic, starts, env, normalizers, critic_norm, *, key, horizon, gamma
):
    rollout = imagine(actor, model, starts, env, normalizers, key=key, horizon=horizon)
    bootstrap = critic(rollout.critic_inputs[-1], critic_norm)
    objective = imagined_return(rollout.rewards, rollout.terminations, bootstrap, gamma)
    return -objective.mean(), rollout
