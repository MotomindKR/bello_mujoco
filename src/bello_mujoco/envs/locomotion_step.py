"""Shared scalar and batched locomotion physics and transition bookkeeping."""

from __future__ import annotations

from bello_mujoco.envs.actions import limit_motor_step
from bello_mujoco.envs.barbell import bar_metrics
from functools import partial
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
from mujoco import mjx
from bello_mujoco.envs.locomotion_state import BelloState
from bello_mujoco.envs.geometry import tree_where
from bello_mujoco.envs.torso import slew_torso_command
from bello_mujoco.envs.gait import (
    advance_gait_phase,
    advance_standing_blend,
    next_start_preference,
    resolve_gait_command_transition,
)
from bello_mujoco.envs.locomotion_diagnostics import (
    gait_diagnostics,
    command_diagnostics,
)

if TYPE_CHECKING:
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def step_physics_with_impacts(
    env: BelloLocomotionEnv,
    state: BelloState,
    data: mjx.Data,
    batch_size: int | None = None,
) -> tuple[mjx.Data, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    foot_count = int(env.model.foot_body_ids.size)
    shape = () if batch_size is None else (batch_size,)
    if batch_size is None:
        foot_velocity = env._foot_world_linear_velocity
        contact_terms = partial(env._contact_terms, world_id=0)
    else:
        world_ids = jnp.arange(batch_size, dtype=jnp.int32)
        foot_velocity = jax.vmap(env._foot_world_linear_velocity)

        def contact_terms(current):
            return jax.vmap(env._contact_terms)(current, world_ids)

    touchdown_eligible = (
        (state.step_count > 0)[..., None]
        & ~state.last_foot_contact.astype(bool)
        & (state.foot_contact_debounce_time <= 0.0)
    )
    zeros = jnp.zeros((*shape, foot_count), dtype=jnp.float32)
    initial = (
        data,
        jnp.zeros((*shape, foot_count), dtype=bool),
        zeros,
        zeros,
        jnp.zeros((*shape, foot_count, 3), dtype=jnp.float32),
        jnp.zeros((*shape, env.model.mj_model.nu), dtype=jnp.float32),
    )
    physics_timestep = jnp.asarray(env.model.mj_model.opt.timestep, dtype=jnp.float32)

    def substep(index, carry):
        (
            current_data,
            touchdown,
            normal_impulse,
            peak_normal_force,
            touchdown_velocity,
            requested_force_peak,
        ) = carry
        current_data = current_data.replace(
            ctrl=env.delayed_ctrl(state.data.ctrl, data.ctrl, index)
        )
        current_state = state.replace(data=current_data)
        preimpact_velocity = foot_velocity(current_state)
        next_data = mjx.step(env.model.mjx_model, current_data)
        next_state = state.replace(data=next_data)
        raw_contact, normal_force, *_ = contact_terms(next_state)
        new_touchdown = touchdown_eligible & ~touchdown & raw_contact.astype(bool)
        touchdown_velocity = jnp.where(
            new_touchdown[..., None], preimpact_velocity, touchdown_velocity
        )
        return (
            next_data,
            touchdown | new_touchdown,
            normal_impulse + normal_force * physics_timestep,
            jnp.maximum(peak_normal_force, normal_force),
            touchdown_velocity,
            jnp.maximum(
                requested_force_peak,
                jnp.abs(env.requested_actuator_force(next_data)),
            ),
        )

    return jax.lax.fori_loop(0, env.decimation, substep, initial)


def advance_step_state(
    env: BelloLocomotionEnv,
    state: BelloState,
    next_data_state: BelloState,
    action: jax.Array,
    actor_feedback_history: jax.Array,
    raw_tracking_ang_vel_z: jax.Array,
) -> BelloState:
    next_gait_phase = advance_gait_phase(
        state.gait_phase,
        state.command,
        env.command_config,
        env.config.gait,
        env.control_timestep,
        state.standing_transition_pending,
    )
    return state.replace(
        data=next_data_state.data,
        last_action=action,
        last_action_delta=action - state.last_action,
        tracking_ang_vel_z=env.filter_tracking_ang_vel_z(
            state.tracking_ang_vel_z,
            raw_tracking_ang_vel_z,
        ),
        actor_feedback_history=actor_feedback_history,
        gait_phase=next_gait_phase,
        standing_reward_blend=advance_standing_blend(
            state.standing_reward_blend,
            state.command,
            state.standing_transition_pending,
            env.command_config,
            env.config.gait,
            env.control_timestep,
        ),
        command_resample_step_count=state.command_resample_step_count + 1,
        step_count=state.step_count + 1,
    )


def finish_command_step(
    env: BelloLocomotionEnv, previous: BelloState, current: BelloState
) -> BelloState:
    return current.replace(
        next_start_left=next_start_preference(
            previous.command,
            current.command,
            previous.standing_transition_pending,
            previous.next_start_left,
            env.command_config,
            env.config.gait,
        )
    )


def step_outcome(
    env: BelloLocomotionEnv,
    state: BelloState,
    reward_state: BelloState,
    impacts: tuple,
    motor_saturation: jax.Array,
    world_id: jax.Array | int = 0,
):
    """Shared post-physics transition, vmapped by the batched Warp path."""
    (
        foot_touchdown,
        foot_normal_impulse,
        foot_peak_normal_force,
        touchdown_foot_linear_velocity,
        requested_actuator_force,
    ) = impacts
    action = reward_state.last_action
    done_flags = env.done_flags(reward_state, world_id)
    done = done_flags["done/any"]
    terminal = done_flags["done/terminal"]
    (
        raw_foot_contact,
        foot_normal_force,
        foot_linear_velocity,
        foot_angular_speed,
        undesired_contact,
        self_collision,
    ) = env._contact_terms(reward_state, world_id)
    foot_contact, foot_contact_debounce_time = env.debounce_foot_contact(
        reward_state, raw_foot_contact, done
    )
    reward, reward_info = env.reward(
        state,
        reward_state,
        action,
        terminal,
        foot_contact,
        foot_touchdown,
        foot_normal_impulse,
        foot_peak_normal_force,
        touchdown_foot_linear_velocity,
        requested_actuator_force,
        foot_linear_velocity,
        foot_angular_speed,
        world_id=world_id,
    )
    reward, reward_info = env.sanitize_nonfinite_reward(reward, reward_info, done_flags)
    episode_return = state.episode_return + reward
    episode_track_lin_vel_xy = (
        state.episode_track_lin_vel_xy + reward_info["reward/track_lin_vel_xy"]
    )
    episode_length = state.episode_length + 1
    pending_command = jnp.where(
        reward_state.standing_transition_pending[..., None],
        jnp.zeros_like(reward_state.command),
        reward_state.command_target,
    )
    next_command, next_gait_phase, standing_transition_pending = (
        resolve_gait_command_transition(
            reward_state.command,
            pending_command,
            state.gait_phase,
            reward_state.gait_phase,
            reward_state.standing_transition_pending,
            env.command_config,
            env.config.gait,
            state.next_start_left,
            -env._locomotion_right_axis()[:2],
            env.control_timestep,
        )
    )
    next_state = reward_state.replace(
        torso_command=slew_torso_command(
            state.torso_command,
            state.torso_command_target,
            env.command_config.torso.slew_rate,
            env.control_timestep,
        ),
        foot_contact_debounce_time=foot_contact_debounce_time,
        last_foot_contact=foot_contact.astype(bool),
        command=next_command,
        command_target=pending_command,
        gait_phase=next_gait_phase,
        standing_transition_pending=standing_transition_pending,
        episode_return=episode_return,
        episode_track_lin_vel_xy=episode_track_lin_vel_xy,
        episode_length=episode_length,
        last_episode_return=jnp.where(done, episode_return, state.last_episode_return),
        last_episode_track_lin_vel_xy=jnp.where(
            done, episode_track_lin_vel_xy, state.last_episode_track_lin_vel_xy
        ),
        last_episode_length=jnp.where(done, episode_length, state.last_episode_length),
    )
    info = {
        "episode_return": jnp.where(done, episode_return, 0.0),
        "episode_reward/track_lin_vel_xy": jnp.where(
            done, episode_track_lin_vel_xy, 0.0
        ),
        "episode_length": jnp.where(done, episode_length, 0),
        "foot_contact/raw": raw_foot_contact,
        "foot_contact/filtered": foot_contact,
        "foot_contact/debounce_time": foot_contact_debounce_time,
        "target/motor_saturation_fraction": motor_saturation,
        "command/torso_roll": reward_state.torso_command[0],
        "command/torso_pitch": reward_state.torso_command[1],
        "command/torso_yaw": reward_state.torso_command[2],
        "latency/observation_steps": (
            next_state.actor_feedback_delay_steps.astype(jnp.float32)
        ),
        **command_diagnostics(
            env, reward_state.command, reward_state.standing_transition_pending
        ),
        "gait/self_collision_contacts": self_collision,
        "gait/undesired_contacts": undesired_contact,
        **gait_diagnostics(
            env,
            state,
            reward_state,
            foot_contact,
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            robot_weight=env._robot_weight(world_id)
            if env.config.arm_pose_path is not None
            else None,
        ),
        "returned_episode": done,
        **done_flags,
        **reward_info,
    }
    if env.config.arm_pose_path is not None and env.model.bar_body_id >= 0:
        info.update(bar_metrics(env.model, reward_state.data, world_id))
    if env.reward_config.enabled("track_torso_orientation"):
        info["tracking/torso_orientation_error"] = env._torso_orientation_error(
            reward_state
        )
    return next_state, reward, done, info


def step(
    env: BelloLocomotionEnv, state: BelloState, action: jax.Array
) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    requested_action = jnp.asarray(action, dtype=jnp.float32)
    action = jnp.clip(
        requested_action,
        -env.action_clip,
        env.action_clip,
    )
    ctrl, motor_saturation = env._action_target_details(action)
    if env.config.arm_pose_path is not None:
        ctrl = env.apply_arm_target(ctrl, state.step_count)
    ctrl = limit_motor_step(ctrl, state.data.ctrl, env.model.target_limits)
    data = state.data.replace(ctrl=ctrl)
    (
        data,
        foot_touchdown,
        foot_normal_impulse,
        foot_peak_normal_force,
        touchdown_foot_linear_velocity,
        requested_actuator_force,
    ) = step_physics_with_impacts(env, state, data)
    next_data_state = state.replace(data=data)
    reward_state = advance_step_state(
        env,
        state,
        next_data_state,
        action,
        env._advance_actor_feedback_history(
            state.actor_feedback_history, next_data_state
        ),
        env._raw_tracking_ang_vel_z(next_data_state),
    )
    next_state, reward, done, info = step_outcome(
        env,
        state,
        reward_state,
        (
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
        ),
        motor_saturation,
    )
    next_state = finish_command_step(env, state, next_state)
    observation = env.observation(next_state)
    info.update(
        base_height=env._base_height(next_state),
        bootstrap_observation=observation,
        bootstrap_critic_observation=env.critic_observation(next_state),
    )
    return next_state, observation, reward, done, info


def step_batch(
    env: BelloLocomotionEnv,
    state: BelloState,
    action: jax.Array,
    rng: jax.Array | None = None,
) -> tuple[BelloState, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    batch_size = int(action.shape[0])
    resample_rng = reset_obs_rng = None
    push_rng = feedback_rng = obs_rng = None
    if rng is not None:
        (
            resample_rng,
            _,
            reset_obs_rng,
            push_rng,
            obs_rng,
        ) = jax.random.split(rng, 5)
        feedback_rng = jax.random.fold_in(obs_rng, 1)
        obs_rng = jax.random.fold_in(obs_rng, 2)
    requested_action = jnp.asarray(action, dtype=jnp.float32)
    action = jnp.clip(
        requested_action,
        -env.action_clip,
        env.action_clip,
    )
    joint_angle_offset = env._joint_angle_offset_batch(batch_size)
    ctrl, motor_saturation = jax.vmap(env._action_target_details)(
        action, joint_angle_offset
    )
    if env.config.arm_pose_path is not None:
        ctrl = jax.vmap(env.apply_arm_target)(
            ctrl, state.step_count, jnp.arange(batch_size)
        )
    ctrl = jax.vmap(
        lambda target, previous: limit_motor_step(
            target, previous, env.model.target_limits
        )
    )(ctrl, state.data.ctrl)
    data = state.data.replace(
        ctrl=ctrl,
        _impl=state.data._impl.replace(nworld=batch_size),
    )
    data, push_steps_remaining, push_applied, push_velocity = (
        env._maybe_apply_random_push_batch(data, state.push_steps_remaining, push_rng)
    )
    state = state.replace(push_steps_remaining=push_steps_remaining)
    (
        data,
        foot_touchdown,
        foot_normal_impulse,
        foot_peak_normal_force,
        touchdown_foot_linear_velocity,
        requested_actuator_force,
    ) = step_physics_with_impacts(env, state, data, batch_size)
    next_data_state = state.replace(data=data)
    if feedback_rng is None:
        actor_feedback_history = jax.vmap(env._advance_actor_feedback_history)(
            state.actor_feedback_history, next_data_state
        )
    else:
        feedback_keys = jax.random.split(feedback_rng, batch_size)
        actor_feedback_history = jax.vmap(env._advance_actor_feedback_history)(
            state.actor_feedback_history, next_data_state, feedback_keys
        )
    reward_state = advance_step_state(
        env,
        state,
        next_data_state,
        action,
        actor_feedback_history,
        jax.vmap(env._raw_tracking_ang_vel_z)(next_data_state),
    )
    world_ids = jnp.arange(batch_size, dtype=jnp.int32)
    transitioned_state, reward, done, info = jax.vmap(partial(step_outcome, env))(
        state,
        reward_state,
        (
            foot_touchdown,
            foot_normal_impulse,
            foot_peak_normal_force,
            touchdown_foot_linear_velocity,
            requested_actuator_force,
        ),
        motor_saturation,
        world_ids,
    )
    (
        next_command,
        next_gait_phase,
        standing_transition_pending,
        command_resample_step_count,
        command_resampled,
        next_command_target,
    ) = env._maybe_resample_batch_command_and_phase(transitioned_state, resample_rng)
    stepped_state = transitioned_state.replace(
        command=next_command,
        command_target=next_command_target,
        gait_phase=next_gait_phase,
        standing_transition_pending=standing_transition_pending,
        command_resample_step_count=command_resample_step_count,
        torso_command_target=(
            jnp.where(
                command_resampled[:, None],
                env.command_config.torso.sample(
                    jax.random.fold_in(resample_rng, 4), (batch_size,)
                ),
                transitioned_state.torso_command_target,
            )
            if resample_rng is not None and env.command_config.enabled
            else transitioned_state.torso_command_target
        ),
    )
    stepped_state = finish_command_step(env, state, stepped_state)
    stepped_obs = env.observation_batch(stepped_state, obs_rng)
    stepped_critic_obs = env.critic_observation_batch(stepped_state)
    info.update(
        {
            "base_height": jax.vmap(env._base_height)(stepped_state, world_ids),
            "bootstrap_observation": stepped_obs,
            "bootstrap_critic_observation": stepped_critic_obs,
            "random_push/applied": push_applied,
            "random_push/lin_vel_x": push_velocity[:, 0],
            "random_push/lin_vel_y": push_velocity[:, 1],
            "command/resampled": command_resampled.astype(jnp.float32),
        }
    )

    def reset_terminated_worlds(_):
        reset_source_command = jnp.where(
            stepped_state.standing_transition_pending[:, None],
            jnp.zeros_like(stepped_state.command),
            stepped_state.command_target,
        )
        reset_state, reset_obs = env.reset_batch(
            batch_size, command=reset_source_command, rng=reset_obs_rng
        )
        reset_state = reset_state.replace(
            command_resample_step_count=(stepped_state.command_resample_step_count),
            last_episode_return=stepped_state.last_episode_return,
            last_episode_track_lin_vel_xy=(stepped_state.last_episode_track_lin_vel_xy),
            last_episode_length=stepped_state.last_episode_length,
        )
        return (
            tree_where(done, reset_state, stepped_state),
            jnp.where(done[:, None], reset_obs, stepped_obs),
        )

    next_state, next_obs = jax.lax.cond(
        jnp.any(done),
        reset_terminated_worlds,
        lambda _: (stepped_state, stepped_obs),
        operand=None,
    )
    return next_state, next_obs, reward, done, info
