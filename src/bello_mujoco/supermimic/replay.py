"""Fixed-capacity device replay with episode-safe windows and actor histories."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from bello_mujoco.supermimic.state import PhysicalState


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transition:
    state: PhysicalState
    successor: PhysicalState
    filtered_feedback: PhysicalState
    previous_motor: jax.Array
    motor: jax.Array
    action: jax.Array
    previous_action: jax.Array
    observation: jax.Array
    context: jax.Array
    alignment_xy: jax.Array
    alignment_yaw: jax.Array
    reference_frame: jax.Array
    clip_id: jax.Array
    episode_id: jax.Array
    episode_step: jax.Array
    remaining_steps: jax.Array
    reward: jax.Array
    terminal: jax.Array
    timeout: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayWindow:
    transitions: Transition
    mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayStart:
    transition: Transition
    history: jax.Array
    feedback_history: PhysicalState
    valid: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ReplayBuffer:
    data: Transition
    cursor: jax.Array
    size: jax.Array

    @classmethod
    def empty(cls, example, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("replay capacity must be a positive integer")
        leaves = jax.tree.leaves(example)
        if any(
            value.ndim == 0 or value.shape[0] != leaves[0].shape[0] for value in leaves
        ):
            raise ValueError("replay example fields require the same environment batch")
        return cls(
            jax.tree.map(lambda x: jnp.zeros((capacity,) + x.shape, x.dtype), example),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )

    @property
    def capacity(self):
        return self.data.reward.shape[0]

    @property
    def num_envs(self):
        return self.data.reward.shape[1]

    def insert(self, transition):
        values = jax.tree.map(
            lambda x, y: x.at[self.cursor].set(y), self.data, transition
        )
        return ReplayBuffer(
            values,
            (self.cursor + 1) % self.capacity,
            jnp.minimum(self.size + 1, self.capacity),
        )

    def rows(self, age):
        return (self.cursor - self.size + age) % self.capacity

    def sample_windows(self, key, batch_size, length):
        if not 1 <= length <= self.capacity or batch_size <= 0:
            raise ValueError("window length and batch size exceed replay dimensions")
        time_key, env_key = jax.random.split(key)
        age = jax.random.randint(
            time_key, (batch_size,), 0, jnp.maximum(1, self.size - length + 1)
        )
        env = jax.random.randint(env_key, (batch_size,), 0, self.num_envs)
        ages = age[:, None] + jnp.arange(length)
        rows = self.rows(ages)
        transitions = jax.tree.map(lambda x: x[rows, env[:, None]], self.data)
        ended = transitions.terminal | transitions.timeout
        before = jnp.concatenate(
            (jnp.ones((batch_size, 1), bool), ~ended[:, :-1]), axis=1
        )
        valid = (
            (ages < self.size)
            & (transitions.episode_id == transitions.episode_id[:, :1])
            & (transitions.clip_id == transitions.clip_id[:, :1])
            & jnp.all(transitions.context == transitions.context[:, :1], axis=-1)
            & before
        )
        return ReplayWindow(transitions, jnp.cumprod(valid.astype(jnp.float32), axis=1))

    def sample_starts(
        self, key, batch_size, *, history_frames, feedback_frames, horizon
    ):
        if min(batch_size, history_frames, feedback_frames, horizon) <= 0:
            raise ValueError("start sampling dimensions must be positive")
        if max(history_frames, feedback_frames) > self.capacity:
            raise ValueError("replay cannot reconstruct the requested history")
        ages = jnp.arange(self.capacity)[:, None]
        rows = self.rows(ages[:, 0])
        steps = self.data.episode_step[rows]
        needed = jnp.minimum(steps, max(history_frames, feedback_frames) - 1)
        eligible = (
            (ages < self.size)
            & (ages >= needed)
            & (self.data.remaining_steps[rows] >= horizon)
        )
        counts = jnp.cumsum(eligible.reshape(-1), dtype=jnp.int32)
        count = counts[-1]
        draw = jax.random.randint(key, (batch_size,), 0, jnp.maximum(1, count))
        flat = jnp.searchsorted(counts, draw, side="right")
        flat = jnp.minimum(flat, counts.size - 1)
        age, env = flat // self.num_envs, flat % self.num_envs
        row = self.rows(age)
        transition = jax.tree.map(lambda x: x[row, env], self.data)

        def history_indices(length, newest_first=False):
            offsets = (
                jnp.arange(length) if newest_first else jnp.arange(length - 1, -1, -1)
            )
            back = jnp.minimum(offsets[None], transition.episode_step[:, None])
            return self.rows(age[:, None] - back)

        history = self.data.observation[history_indices(history_frames), env[:, None]]
        feedback = jax.tree.map(
            lambda x: x[history_indices(feedback_frames, True), env[:, None]],
            self.data.filtered_feedback,
        )
        return ReplayStart(transition, history, feedback, count > 0)
