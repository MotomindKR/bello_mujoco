from __future__ import annotations
from typing import Any, NamedTuple, Protocol
import chex
import equinox as eqx
import jax
import optax
from bello_mujoco.policy import ActorCritic


class PPOEnvironment(Protocol):
    """Explicit environment surface consumed by the shared PPO runner."""

    @property
    def observation_size(self) -> int: ...

    @property
    def critic_observation_size(self) -> int: ...

    @property
    def action_size(self) -> int: ...

    @property
    def action_clip(self) -> float: ...

    @property
    def control_timestep(self) -> float: ...

    @property
    def ppo_stagger_initial_step_count(self) -> bool: ...

    def reset_batch(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> tuple[Any, jax.Array]: ...

    def initial_step_counts(
        self, batch_size: int, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def observation_batch(
        self, state: Any, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def critic_observation_batch(
        self, state: Any, rng: jax.Array | None = None
    ) -> jax.Array: ...

    def step_batch(
        self, state: Any, action: jax.Array, rng: jax.Array | None = None
    ) -> tuple[Any, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]: ...

    def ppo_action_diagnostics(self, env_action: jax.Array) -> dict[str, jax.Array]: ...

    def initial_task_state(self) -> Any: ...

    def with_task_state(self, task_state: Any) -> PPOEnvironment: ...

    def update_task_state(self, task_state: Any, info: dict[str, jax.Array]) -> Any: ...

    def task_state_metrics(self, task_state: Any) -> dict[str, jax.Array]: ...

    def with_initial_step_counts(self, state: Any, step_count: jax.Array) -> Any: ...


@chex.dataclass(frozen=True)
class TrainState:
    policy: ActorCritic
    opt_state: optax.OptState
    update: jax.Array


class Transition(NamedTuple):
    done: jax.Array
    terminal: jax.Array
    bootstrap_value: jax.Array
    # Action sampled from the policy distribution for PPO log-prob/ratio calculations.
    action: jax.Array
    # Exact pre-transform Gaussian sample; tanh saturation is not invertible in float32.
    latent_action: jax.Array
    # Action actually sent to the env after optional env-level clipping.
    env_action: jax.Array
    action_mean: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    critic_obs: jax.Array
    info: dict[str, jax.Array]


class ValueNormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


class ObservationNormState(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


class RunnerState(NamedTuple):
    params: eqx.Module
    opt_state: optax.OptState
    update: jax.Array
    env_state: Any
    obs_stack: jax.Array
    critic_obs_stack: jax.Array
    rng: jax.Array
    learning_rate: jax.Array
    value_norm_state: ValueNormState
    observation_norm_state: ObservationNormState
    critic_observation_norm_state: ObservationNormState
    task_state: Any
