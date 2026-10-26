"""Validated immutable contracts for Mini-native motion learning."""

from dataclasses import dataclass, field, replace
import math

from bello_mujoco.envs.feedback import validate_command_delay
from bello_mujoco.envs.feedback import ObservationLowPassFilterConfig
from bello_mujoco.envs.motion_tracking import (
    MotionObservationLatencyConfig,
    MotionObservationNoiseConfig,
    MotionResetConfig,
    MotionTrackingEnvConfig,
)
from bello_mujoco.sim import ModelRandomizationConfig, SimulationPhysicsConfig


@dataclass(frozen=True)
class ControlConfig:
    """Simulation timing and the action bound applied only during collection."""

    physics_timestep: float = 0.005
    decimation: int = 4
    command_delay_seconds: float = 0.0
    action_clip: float = 20.0

    def __post_init__(self):
        if not math.isfinite(self.physics_timestep) or self.physics_timestep <= 0:
            raise ValueError("physics_timestep must be positive and finite")
        if type(self.decimation) is not int or self.decimation <= 0:
            raise ValueError("decimation must be a positive integer")
        if not math.isfinite(self.action_clip) or self.action_clip <= 0:
            raise ValueError("action_clip must be positive and finite")
        validate_command_delay(
            self.command_delay_seconds, self.physics_timestep, self.decimation
        )

    @property
    def timestep(self):
        return self.physics_timestep * self.decimation


@dataclass(frozen=True)
class ObservationConfig:
    history_frames: int = 10
    future_frames: int = 10
    reference_orientation: bool = True
    goal_joint_velocity_scale: float = 0.1

    def __post_init__(self):
        for name in ("history_frames", "future_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.reference_orientation) is not bool:
            raise ValueError("reference_orientation must be a boolean")
        if (
            not math.isfinite(self.goal_joint_velocity_scale)
            or self.goal_joint_velocity_scale <= 0
        ):
            raise ValueError("goal_joint_velocity_scale must be positive and finite")


@dataclass(frozen=True)
class LearningConfig:
    seed: int = 0
    total_timesteps: int = 3276800000
    num_envs: int = 4096
    max_episode_steps: int | None = None
    replay_steps: int = 1000
    max_replay_bytes: int = 16 * 1024**3
    warmup_steps: int = 64
    model_warmup_updates: int = 500
    window_length: int = 8
    model_batches: int = 10
    model_batch_size: int = 2048
    imagined_batch_size: int = 4096
    imagination_horizon: int = 32
    critic_epochs: int = 5
    critic_minibatches: int = 4
    update_to_data_ratio: float = 0.4
    action_noise: float = 0.1
    learning_rate: float = 0.0002
    learning_rate_decay_updates: int = 250000
    learning_rate_final_fraction: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_gradient_norm: float = 1.0
    actor_gradient_clip: float = 0.1

    def __post_init__(self):
        if self.max_episode_steps is not None and (
            type(self.max_episode_steps) is not int or self.max_episode_steps <= 0
        ):
            raise ValueError("max_episode_steps must be a positive integer or null")
        for name in (
            "total_timesteps",
            "num_envs",
            "replay_steps",
            "max_replay_bytes",
            "warmup_steps",
            "window_length",
            "model_batches",
            "model_batch_size",
            "imagined_batch_size",
            "imagination_horizon",
            "critic_epochs",
            "critic_minibatches",
            "learning_rate_decay_updates",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seed", "model_warmup_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "update_to_data_ratio",
            "learning_rate",
            "max_gradient_norm",
            "actor_gradient_clip",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.update_to_data_ratio > 1:
            raise ValueError("update_to_data_ratio must be at most one")
        for name in ("gamma", "gae_lambda", "learning_rate_final_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError("action_noise must be finite and nonnegative")
        if self.replay_steps < max(self.window_length, self.warmup_steps):
            raise ValueError(
                "replay_steps must hold warmup and a complete model window"
            )
        if self.warmup_steps < self.window_length:
            raise ValueError("warmup_steps must collect a complete model window")
        if (
            self.imagined_batch_size
            * self.imagination_horizon
            % self.critic_minibatches
        ):
            raise ValueError("imagined transitions must divide into critic_minibatches")
        if (
            self.total_timesteps % self.num_envs
            or self.total_timesteps <= self.warmup_steps * self.num_envs
        ):
            raise ValueError(
                "total_timesteps must be a whole environment batch beyond warmup"
            )


@dataclass(frozen=True)
class EnvironmentConfig:
    control: ControlConfig = field(default_factory=ControlConfig)
    reset: MotionResetConfig = field(default_factory=MotionResetConfig)
    noise: MotionObservationNoiseConfig = field(
        default_factory=MotionObservationNoiseConfig
    )
    filtering: ObservationLowPassFilterConfig = field(
        default_factory=ObservationLowPassFilterConfig
    )
    latency: MotionObservationLatencyConfig = field(
        default_factory=MotionObservationLatencyConfig
    )
    randomization: ModelRandomizationConfig = field(
        default_factory=ModelRandomizationConfig
    )

    def __post_init__(self):
        if self.control.command_delay_seconds != 0:
            raise ValueError(
                "SuperMimic raw-action dynamics requires zero command delay"
            )

    def motion_config(self):
        return MotionTrackingEnvConfig(
            decimation=self.control.decimation,
            action_clip=self.control.action_clip,
            anchor_body_name="pelvis_link",
            observation_mode="proprioceptive",
            simulation_physics=replace(
                SimulationPhysicsConfig(), timestep=self.control.physics_timestep
            ),
            command_delay_seconds=self.control.command_delay_seconds,
            reset=self.reset,
            observation_noise=self.noise,
            observation_low_pass_filter=self.filtering,
            observation_latency=self.latency,
            model_randomization=self.randomization,
        )
