"""MJX-Warp collection preserving terminal successors and exact controller history."""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp

from bello_mujoco.envs.geometry import tree_where
from bello_mujoco.supermimic.control import motor_limit_flags
from bello_mujoco.supermimic.replay import ReplayBuffer, Transition
from bello_mujoco.supermimic.model import state_delta
from bello_mujoco.supermimic.sampling import completed_return
from bello_mujoco.supermimic.state import PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectorState:
    environment: object
    observation: jax.Array
    history: jax.Array
    episode_ids: jax.Array
    start_bins: jax.Array
    discounted_returns: jax.Array
    vector_steps: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RunnerState:
    learner: object
    collector: CollectorState
    replay: ReplayBuffer
    sampling: object
    key: jax.Array
    update: jax.Array
    actor_updates: jax.Array


def maximum_return(env, config):
    if config.gamma < 1:
        return 5.0 * env.control_timestep / (1.0 - config.gamma)
    return 5.0 * env.control_timestep * episode_horizon(env, config)


def episode_horizon(env, config):
    return (
        env.max_episode_steps
        if config.max_episode_steps is None
        else min(env.max_episode_steps, config.max_episode_steps)
    )


def collection_termination(episode_length, done, info, config):
    if config.max_episode_steps is None:
        return done, info
    timeout = info["done/timeout"] | (episode_length >= config.max_episode_steps)
    done = done | timeout
    return done, {**info, "done/timeout": timeout, "done/any": done}


def make_transition(env, before, after, observation, action, episode_ids, reward, info):
    physical = jax.vmap(env.physical_state)(before)
    successor = jax.vmap(env.physical_state)(after)
    clip = env.clip_id(before.reference_time)
    end = env.reference_bank.time_offsets[clip] + env.reference_bank.lengths[clip] - 1
    remaining = jnp.ceil(
        (end - before.reference_time) / (env.reference_bank.fps * env.control_timestep)
    ).astype(jnp.int32)
    return Transition(
        physical,
        successor,
        PhysicalState.unflatten(before.actor_feedback_history[:, 0], env.action_size),
        before.data.ctrl,
        after.data.ctrl,
        action,
        before.last_action,
        observation,
        jax.vmap(env.context)(before, jnp.arange(action.shape[0])),
        before.alignment_xy,
        before.alignment_yaw,
        before.reference_time,
        clip,
        episode_ids,
        before.episode_length,
        remaining,
        reward,
        info["done/terminal"],
        info["done/timeout"],
    )


def initialize_collection(env, config, learner, key):
    key, reset_key = jax.random.split(key)
    state, observation = env.reset_batch(config.num_envs, reset_key)
    history = jax.vmap(env.codec.initial_history)(observation)
    bins = jnp.searchsorted(env.sampler.starts, state.reference_time, side="right") - 1
    collector = CollectorState(
        state,
        observation,
        history,
        jnp.zeros(config.num_envs, jnp.int32),
        bins,
        jnp.zeros(config.num_envs),
        jnp.asarray(0, jnp.int32),
    )
    flags = {
        "done/terminal": jnp.zeros(config.num_envs, bool),
        "done/timeout": jnp.zeros(config.num_envs, bool),
    }
    example = make_transition(
        env,
        state,
        state,
        observation,
        jnp.zeros((config.num_envs, env.action_size)),
        collector.episode_ids,
        jnp.zeros(config.num_envs),
        flags,
    )
    replay_bytes = (
        sum(
            math.prod(value.shape) * value.dtype.itemsize
            for value in jax.tree.leaves(example)
        )
        * config.replay_steps
    )
    if replay_bytes > config.max_replay_bytes:
        raise ValueError(
            f"replay needs {replay_bytes} bytes; config.max_replay_bytes={config.max_replay_bytes}"
        )
    replay = ReplayBuffer.empty(example, config.replay_steps)
    return RunnerState(
        learner,
        collector,
        replay,
        env.sampler.initial_state(),
        key,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def collect_step(runner, env, config):
    key, action_key, physics_key, reset_key = jax.random.split(runner.key, 4)
    learner, collector = runner.learner, runner.collector
    before = collector.environment
    goals = jax.vmap(env.goal)(before.reference_time)
    learner = replace(
        learner,
        observation_norm=learner.observation_norm.update(collector.observation),
        goal_norm=learner.goal_norm.update(goals),
    )
    action = learner.actor(
        collector.history, goals, learner.observation_norm, learner.goal_norm
    )
    action = jnp.clip(
        action + config.action_noise * jax.random.normal(action_key, action.shape),
        -env.action_clip,
        env.action_clip,
    )
    after, observation, reward, done, info = env.step_batch(
        before, action, physics_key, reset=False
    )
    done, info = collection_termination(after.episode_length, done, info, config)
    transition = make_transition(
        env,
        before,
        after,
        jax.vmap(env.observation)(before),
        action,
        collector.episode_ids,
        reward,
        info,
    )
    discounted = (
        collector.discounted_returns + config.gamma**before.episode_length * reward
    )
    sampling = env.sampler.update(
        runner.sampling,
        collector.start_bins,
        completed_return(
            discounted,
            after.episode_return,
            after.episode_length,
            info["done/terminal"],
            info["done/timeout"],
            gamma=config.gamma,
            max_episode_steps=episode_horizon(env, config),
        ),
        done,
        maximum_return(env, config),
    )
    dynamic_env = replace(
        env,
        reset_probabilities=env.sampler.probabilities(
            sampling, maximum_return(env, config)
        ),
    )
    reset_state, reset_observation = dynamic_env.reset_batch(config.num_envs, reset_key)
    next_history = jax.vmap(env.codec.advance_history)(collector.history, observation)
    reset_history = jax.vmap(env.codec.initial_history)(reset_observation)
    bins = (
        jnp.searchsorted(env.sampler.starts, reset_state.reference_time, side="right")
        - 1
    )
    next_collector = CollectorState(
        tree_where(done, reset_state, after),
        jnp.where(done[:, None], reset_observation, observation),
        jnp.where(done[:, None, None], reset_history, next_history),
        collector.episode_ids + done.astype(jnp.int32),
        jnp.where(done, bins, collector.start_bins),
        jnp.where(done, 0.0, discounted),
        collector.vector_steps + 1,
    )
    learner = replace(
        learner,
        input_norm=learner.input_norm.update(
            learner.model.features(
                transition.state,
                transition.previous_motor,
                transition.motor,
                transition.context,
            )
        ),
        output_norm=learner.output_norm.update(
            state_delta(transition.state, transition.successor)
        ),
    )
    result = replace(
        runner,
        learner=learner,
        collector=next_collector,
        sampling=sampling,
        replay=runner.replay.insert(transition),
        key=key,
    )
    commands = jax.vmap(env.mapping.command)(
        action, before.data.ctrl, before.joint_default_offset
    )
    limit_flags = motor_limit_flags(
        commands, after.data.actuator_force, env.model.actuator_peak_force
    )
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(value))
                for value in jax.tree.leaves(
                    (
                        transition,
                        observation,
                        after.actor_feedback_history,
                        learner.input_norm,
                        learner.output_norm,
                        learner.observation_norm,
                        learner.goal_norm,
                        after.data.actuator_force,
                    )
                )
            ]
        )
    )
    return result, {
        **env.physics_metrics(after.data),
        "collection/reward": reward.mean(),
        "collection/completed": done.sum(),
        "collection/failures": info["done/terminal"].sum(),
        "collection/limited_fraction": jnp.mean(
            jnp.any(jnp.abs(commands.requested - commands.accepted) > 1e-6, axis=-1)
        ),
        **{
            f"collection/{name}_fraction": jnp.mean(jnp.any(flags, axis=-1))
            for name, flags in limit_flags.items()
        },
        **{
            f"motor/{name}_fraction": jnp.mean(flags, axis=0)
            for name, flags in limit_flags.items()
        },
        "invalid/collection": (jnp.any(info["done/nonfinite"]) | ~finite).astype(
            jnp.float32
        ),
    }
