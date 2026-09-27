from __future__ import annotations
from dataclasses import dataclass, field, fields
import math
from typing import Any, Callable, Literal, get_args
from bello_mujoco.policy import ActorCriticConfig

LearningRateSchedule = Literal["constant", "linear", "adaptive"]
AdaptiveKLStatistic = Literal["mean", "max"]
ObservationNormUpdateMode = Literal["online", "rollout"]
ObservationNormEpsilonMode = Literal["variance_floor", "std_offset"]
_VALIDATOR_METADATA_KEY = "validator"


FieldValidator = Callable[[str, Any], None]


def _validated(default: Any, validator: FieldValidator):
    return field(
        default=default,
        metadata={_VALIDATOR_METADATA_KEY: validator},
    )


def _no_scalar_constraint(_name: str, _value: Any) -> None:
    pass


def _positive_integer(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _positive_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value}")


def _optional_positive_finite(name: str, value: float | None) -> None:
    if value is not None:
        _positive_finite(name, value)


def _optional_positive_integer(name: str, value: int | None) -> None:
    if value is not None:
        _positive_integer(name, value)


def _nonnegative_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative and finite, got {value}")


def _unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _greater_than_one_finite(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 1:
        raise ValueError(f"{name} must be finite and > 1, got {value}")


def _one_of(*choices: str) -> FieldValidator:
    allowed = frozenset(choices)

    def validate(name: str, value: str) -> None:
        if value not in allowed:
            options = ", ".join(repr(choice) for choice in choices)
            raise ValueError(f"{name} must be one of {options}, got {value!r}")

    return validate


@dataclass(frozen=True)
class PPOConfig:
    seed: int = _validated(0, _no_scalar_constraint)
    total_timesteps: int = _validated(1_000_000, _positive_integer)
    num_envs: int = _validated(64, _positive_integer)
    num_steps: int = _validated(24, _positive_integer)
    update_epochs: int = _validated(5, _positive_integer)
    num_minibatches: int = _validated(4, _positive_integer)
    learning_rate: float = _validated(1.0e-3, _positive_finite)
    critic_learning_rate: float | None = _validated(None, _optional_positive_finite)
    schedule: LearningRateSchedule = _validated(
        "adaptive",
        _one_of(*get_args(LearningRateSchedule)),
    )
    desired_kl: float = _validated(0.01, _positive_finite)
    target_kl: float | None = _validated(None, _optional_positive_finite)
    min_learning_rate: float = _validated(1.0e-5, _positive_finite)
    max_learning_rate: float = _validated(1.0e-2, _positive_finite)
    adaptive_lr_factor: float = _validated(1.5, _greater_than_one_finite)
    adaptive_kl_statistic: AdaptiveKLStatistic = _validated(
        "max",
        _one_of(*get_args(AdaptiveKLStatistic)),
    )
    gamma: float = _validated(0.99, _unit_interval)
    gae_lambda: float = _validated(0.95, _unit_interval)
    clip_eps: float = _validated(0.2, _positive_finite)
    ent_coef: float = _validated(0.01, _nonnegative_finite)
    vf_coef: float = _validated(1.0, _nonnegative_finite)
    max_grad_norm: float = _validated(1.0, _positive_finite)
    adam_epsilon: float = _validated(1.0e-5, _positive_finite)
    value_loss_scale: float = _validated(0.5, _positive_finite)
    normalize_value: bool = _validated(False, _no_scalar_constraint)
    value_norm_eps: float = _validated(1.0e-8, _positive_finite)
    normalize_observations: bool = _validated(False, _no_scalar_constraint)
    normalize_critic_observations: bool = _validated(True, _no_scalar_constraint)
    observation_norm_update_mode: ObservationNormUpdateMode = _validated(
        "online", _one_of("online", "rollout")
    )
    observation_norm_update_updates: int | None = _validated(
        None, _optional_positive_integer
    )
    observation_norm_epsilon_mode: ObservationNormEpsilonMode = _validated(
        "variance_floor", _one_of("variance_floor", "std_offset")
    )
    observation_norm_eps: float = _validated(1.0e-5, _positive_finite)
    observation_norm_clip: float = _validated(10.0, _positive_finite)
    rollout_observation_max_abs: float = _validated(1.0e4, _positive_finite)
    rollout_reward_max_abs: float = _validated(1.0e3, _positive_finite)
    rollout_value_max_abs: float = _validated(1.0e4, _positive_finite)
    target_max_abs: float = _validated(1.0e4, _positive_finite)
    gradient_norm_max: float = _validated(1.0e4, _positive_finite)
    policy: ActorCriticConfig = _validated(
        ActorCriticConfig(),
        _no_scalar_constraint,
    )

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // (self.num_envs * self.num_steps)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


def validate_action_limit_contract(config: PPOConfig, env_action_clip: float) -> None:
    if config.policy.action_distribution == "gaussian":
        return
    if config.policy.action_limit != env_action_clip:
        raise ValueError(
            "config.policy.action_limit must equal env.action_clip so PPO "
            "log-probabilities describe the actions applied by the environment, got "
            f"{config.policy.action_limit} != {env_action_clip}"
        )


def _validate_declared_fields(config: PPOConfig) -> None:
    for config_field in fields(config):
        if _VALIDATOR_METADATA_KEY not in config_field.metadata:
            raise RuntimeError(
                f"PPOConfig.{config_field.name} has no declared validation"
            )
        validator = config_field.metadata[_VALIDATOR_METADATA_KEY]
        if not callable(validator):
            raise RuntimeError(
                f"PPOConfig.{config_field.name} has an invalid validator"
            )
        validator(config_field.name, getattr(config, config_field.name))


def validate_ppo_config(config: PPOConfig) -> None:
    _validate_declared_fields(config)
    if config.num_updates <= 0:
        raise ValueError(
            "total_timesteps must cover at least one rollout batch, got "
            f"total_timesteps={config.total_timesteps}, "
            f"num_envs={config.num_envs}, num_steps={config.num_steps}"
        )
    if config.batch_size % config.num_minibatches != 0:
        raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
    if config.min_learning_rate > config.max_learning_rate:
        raise ValueError(
            "min_learning_rate must be <= max_learning_rate, got "
            f"{config.min_learning_rate} > {config.max_learning_rate}"
        )
    if config.schedule == "adaptive" and not (
        config.min_learning_rate <= config.learning_rate <= config.max_learning_rate
    ):
        raise ValueError(
            "adaptive learning_rate must be within "
            "[min_learning_rate, max_learning_rate], got "
            f"{config.learning_rate} outside "
            f"[{config.min_learning_rate}, {config.max_learning_rate}]"
        )
    if any(hidden_dim <= 0 for hidden_dim in config.policy.hidden_dims):
        raise ValueError(
            "policy.hidden_dims entries must be positive, got "
            f"{config.policy.hidden_dims}"
        )
    if config.policy.obs_stack_size <= 0:
        raise ValueError(
            "policy.obs_stack_size must be positive, got "
            f"{config.policy.obs_stack_size}"
        )
    if config.policy.activation not in {"tanh", "relu", "elu", "swish"}:
        raise ValueError(
            "policy.activation must be 'tanh', 'relu', 'elu', or 'swish', got "
            f"{config.policy.activation!r}"
        )
    if config.policy.action_distribution not in {"gaussian", "tanh_gaussian"}:
        raise ValueError(
            "policy.action_distribution must be 'gaussian' or 'tanh_gaussian', got "
            f"{config.policy.action_distribution!r}"
        )
    if not math.isfinite(config.policy.action_limit) or config.policy.action_limit <= 0:
        raise ValueError(
            "policy.action_limit must be positive and finite, got "
            f"{config.policy.action_limit}"
        )
    if not math.isfinite(config.policy.init_log_std):
        raise ValueError(
            f"policy.init_log_std must be finite, got {config.policy.init_log_std}"
        )
    if not math.isfinite(config.policy.min_log_std):
        raise ValueError(
            f"policy.min_log_std must be finite, got {config.policy.min_log_std}"
        )
    if not math.isfinite(config.policy.max_log_std):
        raise ValueError(
            f"policy.max_log_std must be finite, got {config.policy.max_log_std}"
        )
    if config.policy.min_log_std > config.policy.max_log_std:
        raise ValueError(
            "policy.min_log_std must be <= policy.max_log_std, got "
            f"{config.policy.min_log_std} > {config.policy.max_log_std}"
        )
    if not (
        config.policy.min_log_std
        <= config.policy.init_log_std
        <= config.policy.max_log_std
    ):
        raise ValueError(
            "policy.init_log_std must be within "
            "[policy.min_log_std, policy.max_log_std], got "
            f"{config.policy.init_log_std} outside "
            f"[{config.policy.min_log_std}, {config.policy.max_log_std}]"
        )


def parameter_learning_rate(
    config: PPOConfig,
    actor_learning_rate,
    path,
):
    """Resolve the update rate without creating a second PPO implementation."""

    if config.critic_learning_rate is None:
        return actor_learning_rate
    first = path[0] if path else None
    if getattr(first, "name", None) == "critic":
        return config.critic_learning_rate
    return actor_learning_rate
