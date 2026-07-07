import jax.numpy as jnp

from bello_mujoco.envs.rewards import RewardConfig, compute_velocity_reward


def test_velocity_reward_prefers_matching_command() -> None:
    config = RewardConfig()
    common = {
        "config": config,
        "command": jnp.asarray([0.1, 0.0, 0.05], dtype=jnp.float32),
        "base_ang_vel": jnp.asarray([0.0, 0.0, 0.05], dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height_target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(4, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(4, dtype=jnp.float32),
        "joint_vel": jnp.zeros(4, dtype=jnp.float32),
        "action": jnp.zeros(4, dtype=jnp.float32),
        "last_action": jnp.zeros(4, dtype=jnp.float32),
        "actuator_force": jnp.zeros(4, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(4, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(4, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([2, 3], dtype=jnp.int32),
        "done": jnp.asarray(False),
    }

    matching_reward, matching_terms = compute_velocity_reward(
        base_lin_vel=jnp.asarray([0.1, 0.0, 0.0], dtype=jnp.float32), **common
    )
    mismatched_reward, _ = compute_velocity_reward(
        base_lin_vel=jnp.asarray([-0.5, 0.5, 0.0], dtype=jnp.float32), **common
    )

    assert matching_reward > mismatched_reward
    assert matching_terms["reward/track_lin_vel_xy"].shape == ()
    assert jnp.all(jnp.isfinite(matching_reward))


def test_velocity_reward_penalizes_termination() -> None:
    config = RewardConfig(termination_weight=-3.0)
    common = {
        "config": config,
        "command": jnp.zeros(3, dtype=jnp.float32),
        "base_lin_vel": jnp.zeros(3, dtype=jnp.float32),
        "base_ang_vel": jnp.zeros(3, dtype=jnp.float32),
        "projected_gravity": jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32),
        "base_height": jnp.asarray(config.base_height_target, dtype=jnp.float32),
        "joint_pos": jnp.zeros(2, dtype=jnp.float32),
        "joint_pos_rel": jnp.zeros(2, dtype=jnp.float32),
        "joint_vel": jnp.zeros(2, dtype=jnp.float32),
        "action": jnp.zeros(2, dtype=jnp.float32),
        "last_action": jnp.zeros(2, dtype=jnp.float32),
        "actuator_force": jnp.zeros(2, dtype=jnp.float32),
        "joint_limit_low": -jnp.ones(2, dtype=jnp.float32),
        "joint_limit_high": jnp.ones(2, dtype=jnp.float32),
        "hold_joint_indices": jnp.asarray([1], dtype=jnp.int32),
    }

    alive_reward, _ = compute_velocity_reward(done=jnp.asarray(False), **common)
    done_reward, terms = compute_velocity_reward(done=jnp.asarray(True), **common)

    assert done_reward < alive_reward
    assert terms["reward/termination"] == -3.0
