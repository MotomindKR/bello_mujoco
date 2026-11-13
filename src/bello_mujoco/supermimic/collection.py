"""MJX-Warp collection preserving terminal successors and exact controller history."""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp

from bello_mujoco.envs.geometry import mujoco_values_valid, tree_where
from bello_mujoco.supermimic.control import motor_limit_flags
from bello_mujoco.supermimic.replay import ReplayBuffer, Transition
from bello_mujoco.supermimic.model import state_delta
from bello_mujoco.supermimic.sampling import SamplingWindow
from bello_mujoco.supermimic.state import PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectorState:
    environment: object
    observation: jax.Array
    history: jax.Array
    episode_ids: jax.Array
    sampling_window: SamplingWindow
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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectionPolicy:
    key: jax.Array
    physics_key: jax.Array
    reset_key: jax.Array
    goals: jax.Array
    inputs_valid: jax.Array
    policy_output: jax.Array
    action: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CollectionUpdate:
    runner: RunnerState
    done: jax.Array
    reset_key: jax.Array
    reset_probabilities: jax.Array
    metrics: dict
    valid_count: jax.Array


def maximum_return(env, config):
    return env.mdp.maximum_return(config.gamma, episode_horizon(env, config))


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


def initialize_collection(env, config, learner, key, *, axis_name=None):
    key, reset_key = jax.random.split(key)
    if axis_name is not None:
        reset_key = jax.random.fold_in(reset_key, jax.lax.axis_index(axis_name))
    state, observation = env.reset_batch(config.num_envs, reset_key)
    history = jax.vmap(env.codec.initial_history)(observation)
    collector = CollectorState(
        state,
        observation,
        history,
        jnp.zeros(config.num_envs, jnp.int32),
        SamplingWindow.empty(config.num_envs, env.sampler.config.horizon),
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
        + config.num_envs * jnp.dtype(bool).itemsize
    ) * config.replay_steps
    if replay_bytes > config.max_replay_bytes:
        raise ValueError(
            f"replay needs {replay_bytes} bytes; config.max_replay_bytes={config.max_replay_bytes}"
        )
    replay = ReplayBuffer.empty(example, config.replay_steps)
    return RunnerState(
        learner,
        collector,
        replay,
        env.sampler.initial_state(maximum_return(env, config)),
        key,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def prepare_collection(runner, env, config, *, axis_name=None):
    key, action_key, physics_key, reset_key = jax.random.split(runner.key, 4)
    if axis_name is not None:
        rank = jax.lax.axis_index(axis_name)
        action_key = jax.random.fold_in(action_key, rank)
    learner, collector = runner.learner, runner.collector
    before = collector.environment
    full_goals = jax.vmap(env.goal)(
        before.reference_time,
        jax.vmap(env.physical_state)(before),
        before.alignment_xy,
        before.alignment_yaw,
    )
    goals = env.codec.actor_goal(full_goals)
    inputs_valid = valid_environments(
        (collector.observation, collector.history, goals)
    ) & valid_joint_speeds(
        jax.vmap(env.physical_state)(before), config.max_joint_velocity
    )
    action = learner.actor(
        collector.history,
        goals,
        learner.observation_norm,
        learner.goal_norm,
    )
    policy_output = action + config.action_noise * jax.random.normal(
        action_key, action.shape
    )
    action = jnp.clip(
        policy_output,
        -env.action_clip,
        env.action_clip,
    )
    return CollectionPolicy(
        key, physics_key, reset_key, goals, inputs_valid, policy_output, action
    )


def step_collection(collector, policy, env, *, axis_name=None):
    physics_key = policy.physics_key
    if axis_name is not None:
        physics_key = jax.random.fold_in(physics_key, jax.lax.axis_index(axis_name))
    return env.step_batch(
        collector.environment,
        jnp.where(
            valid_environments(policy.policy_output)[:, None], policy.action, 0.0
        ),
        physics_key,
        reset=False,
    )


def update_collection(runner, policy, outcome, env, config, *, axis_name=None):
    learner, collector = runner.learner, runner.collector
    before = collector.environment
    key, goals, inputs_valid = policy.key, policy.goals, policy.inputs_valid
    action, policy_output = policy.action, policy.policy_output
    after, observation, reward, done, info = outcome
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
    features = learner.model.features(transition.state, transition.action)
    delta = state_delta(transition.state, transition.successor)
    fields = {
        "transition": transition,
        "policy_output": policy_output,
        "physics": (after.data.qpos, after.data.qvel, after.data.qacc),
        "observation": observation,
        "history": collector.history,
        "goal": goals,
        "feedback": after.actor_feedback_history,
        "actuator_force": after.data.actuator_force,
        "episode_return": after.episode_return,
        "model_input": features,
        "model_target": delta,
    }
    rejected = {name: ~valid_environments(value) for name, value in fields.items()}
    rejected["joint_velocity"] = ~(
        valid_joint_speeds(transition.state, config.max_joint_velocity)
        & valid_joint_speeds(transition.successor, config.max_joint_velocity)
    )
    rejected["physics_state"] = info["done/nonfinite"]
    rejected["reward"] = info["reward/invalid"]
    valid = (
        inputs_valid
        & ~info["done/nonfinite"]
        & ~jnp.any(jnp.stack(tuple(rejected.values())), axis=0)
    )
    done = done | ~valid
    horizon = env.sampler.config.horizon
    window = collector.sampling_window.record(
        collector.vector_steps % horizon,
        env.sampler.bin_index(before.reference_time),
        env.sampler.bin_index(after.reference_time),
        jnp.where(valid, reward, 0.0),
        info["done/terminal"] | ~valid,
        transition.remaining_steps <= 1,
        info["done/timeout"],
        valid,
    )
    sampling = jax.lax.cond(
        (collector.vector_steps + 1) % horizon == 0,
        lambda: env.sampler.update(
            runner.sampling,
            window,
            maximum_return(env, config),
            config.gamma,
            axis_name=axis_name,
        ),
        lambda: runner.sampling,
    )
    next_history = jax.vmap(env.codec.advance_history)(collector.history, observation)
    next_collector = CollectorState(
        after,
        observation,
        next_history,
        collector.episode_ids + done.astype(jnp.int32),
        window,
        collector.vector_steps + 1,
    )
    learner = replace(
        learner,
        observation_norm=learner.observation_norm.update(
            collector.observation, valid, axis_name=axis_name
        ),
        goal_norm=learner.goal_norm.update(goals, valid, axis_name=axis_name),
        input_norm=learner.input_norm.update(features, valid, axis_name=axis_name),
    )
    result = replace(
        runner,
        learner=learner,
        collector=next_collector,
        sampling=sampling,
        replay=runner.replay.insert(transition, valid),
        key=key,
    )
    commands = jax.vmap(env.mapping.command)(action, before.joint_default_offset)
    limit_flags = motor_limit_flags(
        commands, after.data.actuator_force, env.model.actuator_peak_force
    )
    first = jnp.argmax(~valid)
    reward_sum, valid_count = jnp.where(valid, reward, 0.0).sum(), valid.sum()
    if axis_name is not None:
        reward_sum, valid_count = jax.lax.psum((reward_sum, valid_count), axis_name)
    metrics = {
        **env.physics_metrics(after.data),
        "collection/step_reward": reward_sum / jnp.maximum(valid_count, 1),
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
        "collection/rejected": (~valid).sum(),
        "sampling/max_reset_probability": sampling.probabilities.max(),
        "sampling/effective_reset_bins": 1.0 / jnp.sum(sampling.probabilities**2),
        "sampling/visited_fraction": jnp.sum(
            (sampling.visits > 0) & env.sampler.eligible
        )
        / env.sampler.eligible.sum(),
        "diagnostic/environment": jnp.where(jnp.any(~valid), first, -1),
        "diagnostic/clip": transition.clip_id[first],
        "diagnostic/frame": transition.reference_frame[first]
        - env.reference_bank.time_offsets[transition.clip_id[first]],
        "diagnostic/vector_step": collector.vector_steps,
        **{f"diagnostic/invalid/{name}": mask.sum() for name, mask in rejected.items()},
    }

    return CollectionUpdate(
        result,
        done,
        policy.reset_key,
        sampling.probabilities,
        metrics,
        valid_count,
    )


def reset_collection(update, env, config, *, axis_name=None):
    reset_key = update.reset_key
    if axis_name is not None:
        reset_key = jax.random.fold_in(reset_key, jax.lax.axis_index(axis_name))
    return replace(env, reset_probabilities=update.reset_probabilities).reset_batch(
        config.num_envs, reset_key
    )


def finish_collection(update, reset, env, *, axis_name=None):
    runner, done = update.runner, update.done
    collector = runner.collector
    reset_state, reset_observation = reset
    reset_history = jax.vmap(env.codec.initial_history)(reset_observation)
    next_collector = replace(
        collector,
        environment=tree_where(done, reset_state, collector.environment),
        observation=jnp.where(done[:, None], reset_observation, collector.observation),
        history=jnp.where(done[:, None, None], reset_history, collector.history),
    )
    finite = jnp.all(
        jnp.stack(
            [
                jnp.all(jnp.isfinite(value))
                for value in jax.tree.leaves(
                    (
                        next_collector.observation,
                        next_collector.history,
                        jax.vmap(env.physical_state)(next_collector.environment),
                        next_collector.environment.actor_feedback_history,
                        runner.learner.input_norm,
                        runner.learner.observation_norm,
                        runner.learner.goal_norm,
                        runner.sampling,
                    )
                )
            ]
        )
    )
    if axis_name is not None:
        finite = jax.lax.pmin(finite.astype(jnp.int32), axis_name).astype(bool)
    return replace(runner, collector=next_collector), {
        **update.metrics,
        "invalid/collection": ((update.valid_count == 0) | ~finite).astype(jnp.float32),
    }


def collect_step(runner, env, config, *, axis_name=None):
    policy = prepare_collection(runner, env, config, axis_name=axis_name)
    outcome = step_collection(runner.collector, policy, env, axis_name=axis_name)
    update = update_collection(
        runner, policy, outcome, env, config, axis_name=axis_name
    )
    reset = reset_collection(update, env, config, axis_name=axis_name)
    return finish_collection(update, reset, env, axis_name=axis_name)


def valid_joint_speeds(state, limit):
    return jnp.all(jnp.abs(state.joint_velocity) <= limit, axis=-1)


def valid_environments(tree):
    return jnp.all(
        jnp.stack(
            [
                jnp.all(
                    mujoco_values_valid(value).reshape((value.shape[0], -1)), axis=1
                )
                for value in jax.tree.leaves(tree)
            ]
        ),
        axis=0,
    )
