from __future__ import annotations
import jax
import jax.numpy as jnp
from bello_mujoco.policy import ActorCritic, policy_log_std
from bello_mujoco.training.ppo_types import Transition, PPOEnvironment


def safe_returned_mean(
    info: dict[str, jax.Array], key: str = "episode_return"
) -> jax.Array:
    value = info.get(key)
    returned_episode = info.get("returned_episode")
    if value is None or returned_episode is None:
        return jnp.asarray(0.0, dtype=jnp.float32)
    mask = returned_episode.astype(jnp.float32)
    return jnp.sum(value.astype(jnp.float32) * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def safe_returned_max(
    info: dict[str, jax.Array], key: str = "episode_return"
) -> jax.Array:
    value = info.get(key)
    returned_episode = info.get("returned_episode")
    if value is None or returned_episode is None:
        return jnp.asarray(jnp.nan, dtype=jnp.float32)
    mask = returned_episode.astype(bool)
    masked_value = jnp.where(mask, value.astype(jnp.float32), -jnp.inf)
    return jnp.where(jnp.any(mask), jnp.max(masked_value), jnp.nan)


def command_conditioned_metrics(
    info: dict[str, jax.Array], reward: jax.Array
) -> dict[str, jax.Array]:
    """Report reward and completed-episode outcomes for each command mode."""

    standing = info.get("command/standing")
    moving = info.get("command/moving")
    if standing is None or moving is None:
        return {}

    metrics = {
        "command/standing_mean_reward": masked_mean_or_zero(reward, standing),
        "command/standing_positive_reward_fraction": masked_mean_or_zero(
            (reward > 0.0).astype(jnp.float32), standing
        ),
        "command/moving_mean_reward": masked_mean_or_zero(reward, moving),
    }
    track_lin_vel_xy = info.get("reward/track_lin_vel_xy")
    if track_lin_vel_xy is not None:
        metrics["command/standing_track_lin_vel_xy"] = masked_mean_or_zero(
            track_lin_vel_xy, standing
        )
        metrics["command/moving_track_lin_vel_xy"] = masked_mean_or_zero(
            track_lin_vel_xy, moving
        )
    track_ang_vel_z = info.get("reward/track_ang_vel_z")
    if track_ang_vel_z is not None:
        metrics["command/standing_track_ang_vel_z"] = masked_mean_or_zero(
            track_ang_vel_z, standing
        )
        metrics["command/moving_track_ang_vel_z"] = masked_mean_or_zero(
            track_ang_vel_z, moving
        )
    moving_lin_vel_xy_error = info.get("reward/moving_lin_vel_xy_error")
    if moving_lin_vel_xy_error is not None:
        metrics["command/moving_lin_vel_xy_error"] = masked_mean_or_zero(
            moving_lin_vel_xy_error, moving
        )
    moving_ang_vel_z_error = info.get("reward/moving_ang_vel_z_error")
    if moving_ang_vel_z_error is not None:
        metrics["command/moving_ang_vel_z_error"] = masked_mean_or_zero(
            moving_ang_vel_z_error, moving
        )
    returned = info.get("returned_episode")
    episode_return = info.get("episode_return")
    episode_length = info.get("episode_length")
    if returned is None or episode_return is None or episode_length is None:
        return metrics

    for name, mode in (("standing", standing), ("moving", moving)):
        completed = returned.astype(bool) & mode.astype(bool)
        metrics[f"episode/{name}_count"] = jnp.sum(completed.astype(jnp.float32))
        metrics[f"episode/{name}_return"] = masked_mean_or_zero(
            episode_return, completed
        )
        metrics[f"episode/{name}_length"] = masked_mean_or_zero(
            episode_length, completed
        )
    return metrics


def physics_diagnostics(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    diagnostics = {}
    for name, value in info.items():
        if not name.startswith("physics/"):
            continue
        value = value.astype(jnp.float32)
        if name.endswith("_overflow"):
            diagnostics[f"mean_{name}"] = value.mean()
        else:
            diagnostics[f"max_{name}"] = value.max()
    return diagnostics


def gait_diagnostics(
    info: dict[str, jax.Array], control_timestep: float
) -> dict[str, jax.Array]:
    metrics = {
        f"mean_{name}": value.astype(jnp.float32).mean()
        for name, value in info.items()
        if name.startswith("gait/")
    }
    standing = info.get("command/standing")
    moving = info.get("command/moving")
    if standing is not None and moving is not None:
        for name, value in info.items():
            if not name.startswith("gait/"):
                continue
            suffix = name.removeprefix("gait/")
            metrics[f"gait/standing_{suffix}"] = masked_mean_or_zero(value, standing)
            metrics[f"gait/moving_{suffix}"] = masked_mean_or_zero(value, moving)

    touchdown = info.get("gait/touchdown")
    liftoff = info.get("gait/liftoff")
    if touchdown is not None:
        metrics["gait/footfalls_per_second"] = (
            touchdown.astype(jnp.float32).mean() / control_timestep
        )
        if standing is not None and moving is not None:
            metrics["gait/standing_footfalls_per_second"] = (
                masked_mean_or_zero(touchdown, standing) / control_timestep
            )
            metrics["gait/moving_footfalls_per_second"] = (
                masked_mean_or_zero(touchdown, moving) / control_timestep
            )
    if liftoff is not None:
        metrics["gait/liftoffs_per_second"] = (
            liftoff.astype(jnp.float32).mean() / control_timestep
        )
    for force_name in ("joint_force", "contact_force", "contact_impulse"):
        rollout_max = info.get(f"gait/{force_name}_max_body_weight")
        if rollout_max is None:
            continue
        rollout_max = rollout_max.astype(jnp.float32).reshape(-1)
        metrics[f"gait/{force_name}_p95_body_weight"] = jnp.quantile(rollout_max, 0.95)
        metrics[f"gait/{force_name}_p99_body_weight"] = jnp.quantile(rollout_max, 0.99)
        metrics[f"gait/{force_name}_worst_body_weight"] = jnp.max(rollout_max)
    return metrics


def masked_mean_or_zero(value: jax.Array, mask: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    return jnp.sum(value * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def action_diagnostics(
    transitions: Transition,
    env: PPOEnvironment,
    policy: ActorCritic,
) -> dict[str, jax.Array]:
    action = transitions.action
    env_action = transitions.env_action
    action_mean = transitions.action_mean
    log_std = policy_log_std(policy)
    metrics = {
        "action_rms": jnp.sqrt(jnp.mean(action**2)),
        "action_abs_max": jnp.max(jnp.abs(action)),
        "action_mean_rms": jnp.sqrt(jnp.mean(action_mean**2)),
        "action_mean_abs_max": jnp.max(jnp.abs(action_mean)),
        "action_std_mean": jnp.mean(jnp.exp(log_std)),
        "action_std_min": jnp.min(jnp.exp(log_std)),
        "action_std_max": jnp.max(jnp.exp(log_std)),
        "action_clip_fraction": jnp.mean(
            (jnp.abs(action - env_action) > 1.0e-6).astype(jnp.float32)
        ),
    }
    if policy.action_distribution == "tanh_gaussian":
        action_limit = jnp.asarray(policy.action_limit, dtype=action_mean.dtype)
        metrics["action_bound_saturation_fraction"] = jnp.mean(
            (jnp.abs(action_mean) >= 0.98 * action_limit).astype(jnp.float32)
        )
    return {
        **metrics,
        "env_action_abs_max": jnp.max(jnp.abs(env_action)),
        **env.ppo_action_diagnostics(env_action),
    }


def environment_metric_means(info: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Reduce per-transition diagnostics using their public metric namespaces."""
    metrics = {}
    for name, value in info.items():
        group = name.partition("/")[0]
        if group in {"reward", "done", "random_push", "foot_contact", "target"}:
            key = f"mean_{name}"
        elif group == "sampling":
            key = name
        elif group == "command":
            key = f"{name}_fraction"
        elif group == "barbell":
            metrics[f"max_{name}"] = value.astype(jnp.float32).max()
            key = f"mean_{name}"
        else:
            continue
        metrics[key] = value.astype(jnp.float32).mean()
    return metrics
