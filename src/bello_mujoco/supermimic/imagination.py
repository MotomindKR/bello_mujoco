"""Short on-policy rollouts through learned dynamics and the actual command path."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic.feedback import actor_sample, advance_feedback
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

        def value_input(state, motor, observations, frame):
            reference = env.reference_state(frame, xy, yaw)
            return critic_features(
                state, reference, env.goal(frame), context, motor, observations
            )

        def step(carry, _):
            state, motor, previous_action, observations, feedback, frame, rng = carry
            rng, feedback_key, observation_key = jax.random.split(rng, 3)
            features = value_input(state, motor, observations, frame)
            action = actor(observations, env.goal(frame), observation_norm, goal_norm)
            command = env.mapping.command(action, motor, context[:count])
            successor = model.step(
                state, motor, command.accepted, context, input_norm, output_norm
            )
            next_frame = env.advance_reference_time(frame)
            reference = env.reference_state(next_frame, xy, yaw)
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
                command.action,
                reference.quaternion,
                context[:count],
                observation_key,
                noise=env.config.observation_noise,
                filtering=env.config.observation_low_pass_filter,
            )
            observations = env.codec.advance_history(observations, sample)
            reward, _ = env.mdp.reward(
                successor, reference, command.action, previous_action
            )
            terminal = env.mdp.soft_termination(successor, reference)
            clip = env.clip_id(next_frame)
            end = (
                env.reference_bank.time_offsets[clip]
                + env.reference_bank.lengths[clip]
                - 1
            )
            terminal = jnp.where(next_frame >= end, 1.0, terminal)
            return (
                successor,
                command.accepted,
                command.action,
                observations,
                feedback,
                next_frame,
                rng,
            ), (reward, terminal, features)

        initial = (
            start.state,
            start.previous_motor,
            start.previous_action,
            history,
            feedback_history,
            start.reference_frame,
            noise_key,
        )
        final, (rewards, terminals, features) = jax.lax.scan(
            step, initial, None, length=horizon
        )
        final_features = value_input(final[0], final[1], final[3], final[5])
        return ImaginedRollout(
            rewards, terminals, jnp.concatenate((features, final_features[None]))
        )

    result = jax.vmap(single)(
        tr,
        starts.history,
        starts.feedback_history,
        jax.random.split(key, tr.action.shape[0]),
    )
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), result)


def actor_loss(
    actor, model, critic, starts, env, normalizers, critic_norm, *, key, horizon, gamma
):
    rollout = imagine(actor, model, starts, env, normalizers, key=key, horizon=horizon)
    bootstrap = critic(rollout.critic_inputs[-1], critic_norm)
    objective = imagined_return(rollout.rewards, rollout.terminations, bootstrap, gamma)
    return -objective.mean(), rollout
