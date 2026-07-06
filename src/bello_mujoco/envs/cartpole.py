"""Small continuous-action CartPole for PPO smoke tests."""

from __future__ import annotations

from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class CartPoleState:
    obs: jax.Array
    step_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    last_episode_return: jax.Array
    last_episode_length: jax.Array


@dataclass(frozen=True)
class ContinuousCartPoleEnv:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    x_threshold: float = 2.4
    theta_threshold_radians: float = 12.0 * 2.0 * jnp.pi / 360.0
    max_episode_steps: int = 500

    @property
    def total_mass(self) -> float:
        return self.masspole + self.masscart

    @property
    def polemass_length(self) -> float:
        return self.masspole * self.length

    @property
    def action_size(self) -> int:
        return 1

    @property
    def observation_size(self) -> int:
        return 4

    def reset(self, command: jax.Array | None = None) -> tuple[CartPoleState, jax.Array]:
        del command
        obs = jnp.asarray([0.0, 0.0, 0.05, 0.0], dtype=jnp.float32)
        state = CartPoleState(
            obs=obs,
            step_count=jnp.asarray(0, dtype=jnp.int32),
            episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            episode_length=jnp.asarray(0, dtype=jnp.int32),
            last_episode_return=jnp.asarray(0.0, dtype=jnp.float32),
            last_episode_length=jnp.asarray(0, dtype=jnp.int32),
        )
        return state, obs

    def reset_batch(self, batch_size: int, command: jax.Array | None = None) -> tuple[CartPoleState, jax.Array]:
        del command
        offsets = jnp.linspace(-0.04, 0.04, batch_size, dtype=jnp.float32)
        obs = jnp.stack(
            [
                jnp.zeros(batch_size, dtype=jnp.float32),
                jnp.zeros(batch_size, dtype=jnp.float32),
                0.05 + offsets,
                jnp.zeros(batch_size, dtype=jnp.float32),
            ],
            axis=-1,
        )
        state = CartPoleState(
            obs=obs,
            step_count=jnp.zeros(batch_size, dtype=jnp.int32),
            episode_return=jnp.zeros(batch_size, dtype=jnp.float32),
            episode_length=jnp.zeros(batch_size, dtype=jnp.int32),
            last_episode_return=jnp.zeros(batch_size, dtype=jnp.float32),
            last_episode_length=jnp.zeros(batch_size, dtype=jnp.int32),
        )
        return state, obs

    def step(
        self, state: CartPoleState, action: jax.Array
    ) -> tuple[CartPoleState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        x, x_dot, theta, theta_dot = state.obs
        force = self.force_mag * jnp.clip(action[0], -1.0, 1.0)
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        next_obs = jnp.asarray([x, x_dot, theta, theta_dot], dtype=jnp.float32)
        next_step = state.step_count + 1

        failed = (
            (x < -self.x_threshold)
            | (x > self.x_threshold)
            | (theta < -self.theta_threshold_radians)
            | (theta > self.theta_threshold_radians)
        )
        timeout = next_step >= self.max_episode_steps
        done = failed | timeout
        reward = jnp.where(failed, 0.0, 1.0).astype(jnp.float32)
        episode_return = state.episode_return + reward
        episode_length = state.episode_length + 1
        reset_state, reset_obs = self.reset()
        next_state = CartPoleState(
            obs=jnp.where(done, reset_obs, next_obs),
            step_count=jnp.where(done, reset_state.step_count, next_step),
            episode_return=jnp.where(done, reset_state.episode_return, episode_return),
            episode_length=jnp.where(done, reset_state.episode_length, episode_length),
            last_episode_return=jnp.where(done, episode_return, state.last_episode_return),
            last_episode_length=jnp.where(done, episode_length, state.last_episode_length),
        )
        info = {
            "episode_return": jnp.where(done, episode_return, 0.0),
            "episode_length": jnp.where(done, episode_length, 0),
            "returned_episode": done,
        }
        return next_state, next_state.obs, reward, done, info

    def step_batch(
        self, state: CartPoleState, action: jax.Array
    ) -> tuple[CartPoleState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
        return jax.vmap(self.step)(state, action)
