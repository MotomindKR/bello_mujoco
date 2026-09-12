from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest


from bello_mujoco.models.compiler import (
    CompiledRobot,
    compile_robot,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionAdaptiveSamplingConfig,
    MotionDoneConfig,
    MotionObservationLatencyConfig,
    MotionObservationNoiseConfig,
    MotionPushConfig,
    MotionRewardConfig,
    MotionResetConfig,
    MotionTrackingEnvConfig,
    quat_multiply,
    yaw_quat_from_angle,
)
from bello_mujoco.motion.clip import (
    load_motion_clip,
    motion_tracking_asset_config,
    sample_motion,
)
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import load_compiled_model

pytestmark = pytest.mark.integration


requires_warp_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="compiled MJX-Warp integration requires a GPU backend",
)
MOTION_TASK = motion_task_definition("bello")


@pytest.fixture(scope="module")
def motion_asset(tmp_path_factory) -> CompiledRobot:
    return compile_robot(
        motion_tracking_asset_config(
            asset="bello",
        )
    )


@pytest.fixture(scope="module")
def reference_motion(motion_asset: CompiledRobot, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("reference-motion") / "test_motion.npz"
    model = load_compiled_model(
        motion_asset,
        actuator_config=motion_tracking_asset_config(asset="bello").actuator_config,
        load_mjx=False,
    )
    mj_model = model.mj_model
    frame_count = 4
    qpos = np.repeat(model.default_qpos[None, :], frame_count, axis=0)
    qpos[:, 1] += np.linspace(0.0, 0.06, frame_count)
    qvel = np.zeros((frame_count, mj_model.nv), dtype=np.float64)
    qvel[:, 1] = 0.6
    body_ids = np.asarray(
        [mj_model.body(name).id for name in MOTION_TASK.default_tracked_body_names]
    )
    body_pos = np.empty((frame_count, len(body_ids), 3))
    body_quat = np.empty((frame_count, len(body_ids), 4))
    body_lin_vel = np.empty((frame_count, len(body_ids), 3))
    body_ang_vel = np.empty((frame_count, len(body_ids), 3))
    data = mujoco.MjData(mj_model)
    velocity = np.empty(6)
    for frame in range(frame_count):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(mj_model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                mj_model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            body_ang_vel[frame, body_index] = velocity[:3]
            body_lin_vel[frame, body_index] = velocity[3:]
    np.savez_compressed(
        path,
        fps=np.asarray(10.0),
        joint_names=np.asarray(model.actuator_joint_names),
        body_names=np.asarray(MOTION_TASK.default_tracked_body_names),
        qpos=qpos,
        qvel=qvel,
        body_pos=body_pos,
        body_quat_wxyz=body_quat,
        body_lin_vel=body_lin_vel,
        body_ang_vel=body_ang_vel,
        foot_contacts=np.zeros((frame_count, 2)),
    )
    return path


def tracking_config(path: Path, *, random_start: bool = False):
    return MotionTrackingEnvConfig(
        motion_path=path,
        reset=MotionResetConfig(
            random_start=random_start,
            initial_xy_range=(0.0, 0.0),
            initial_yaw_range=(0.0, 0.0),
            joint_pos_noise=(0.0, 0.0),
            joint_vel_noise=(0.0, 0.0),
        ),
    )


def beyondmimic_config(path: Path) -> MotionTrackingEnvConfig:
    return MotionTrackingEnvConfig(
        motion_path=path,
        formulation="beyondmimic",
        anchor_body_name="torso_link",
        reset=MotionResetConfig(
            random_start=False,
            initial_xy_range=(0.0, 0.0),
            initial_yaw_range=(0.0, 0.0),
            joint_pos_noise=(0.0, 0.0),
            joint_vel_noise=(0.0, 0.0),
        ),
        reward=MotionRewardConfig(
            root_position_weight=0.5,
            root_orientation_weight=0.5,
            body_position_weight=1.0,
            body_orientation_weight=1.0,
            body_linear_velocity_weight=1.0,
            body_angular_velocity_weight=1.0,
            action_rate_weight=-0.1,
            joint_limit_weight=-10.0,
            undesired_contact_weight=0.0,
            root_position_sigma=0.3,
            root_orientation_sigma=0.4,
            body_position_sigma=0.3,
            body_orientation_sigma=0.4,
            body_linear_velocity_sigma=1.0,
            body_angular_velocity_sigma=np.pi,
        ),
        done=MotionDoneConfig(
            root_height_error=0.25,
            root_orientation_error=0.8,
            end_effector_position_error=0.25,
        ),
    )


def expected_physical_ctrl(
    env: BelloMotionTrackingEnv, logical_target: np.ndarray
) -> np.ndarray:
    requested = env.model.actuator_coordinate_transform @ logical_target
    ctrlrange = np.asarray(env.model.mj_model.actuator_ctrlrange)
    return np.clip(requested, ctrlrange[:, 0], ctrlrange[:, 1])


def observation_component_slice(
    env: BelloMotionTrackingEnv, component_name: str
) -> slice:
    offset = 0
    for name, size in env.observation_layout.components:
        if name == component_name:
            return slice(offset, offset + size)
        offset += size
    raise KeyError(component_name)


def test_motion_loader_and_shortest_arc_interpolation(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    model = load_compiled_model(
        motion_asset,
        actuator_config=motion_tracking_asset_config(asset="bello").actuator_config,
        load_mjx=False,
    )
    clip = load_motion_clip(reference_motion, model)
    clip = clip.__class__(
        **{**clip.__dict__, "qpos": clip.qpos.at[1, 3:7].multiply(-1)}
    )

    sample = jax.jit(lambda: sample_motion(clip, jnp.asarray(0.05)))()

    assert sample.qpos.shape == (model.mj_model.nq,)
    assert float(jnp.linalg.norm(sample.qpos[3:7])) == pytest.approx(1.0)
    assert np.allclose(sample.qpos[3:7], np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_native_motion_rollout_does_not_load_mjx(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    from bello_mujoco.evaluation.motion import (
        native_motion_step,
        reset_native_motion_rollout,
    )

    config = MotionTrackingEnvConfig(motion_path=reference_motion)
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config, load_mjx=False)
    data = mujoco.MjData(env.model.mj_model)

    state, observation = reset_native_motion_rollout(env, data)
    stepped, next_observation, reward, done, info = native_motion_step(
        env, data, state, jnp.zeros(env.action_size)
    )

    assert env.model.mjx_model is None
    assert observation.shape == (env.observation_size,)
    assert next_observation.shape == observation.shape
    assert int(stepped.step_count) == 1
    assert np.isfinite(float(reward))
    assert bool(done) == bool(info["done/any"])
    with pytest.raises(RuntimeError, match="native MuJoCo only"):
        env.reset()


def test_tracking_reset_matches_reference_and_is_nonterminal(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion)
    )
    state, observation = env.reset(reference_time=jnp.asarray(0.1))
    reward, _ = env._reward(state, state)
    done = env.termination_flags(state)

    assert (env.model.mj_model.nq, env.model.mj_model.nv, env.model.mj_model.nu) == (
        32,
        31,
        25,
    )
    assert env.action_size == 25
    assert observation.shape == (env.observation_size,)
    assert env.critic_observation(state).shape == (env.critic_observation_size,)
    assert np.isfinite(float(reward)) and float(reward) > 0.0
    assert not bool(done["done/any"])


def test_aligned_reference_preserves_body_local_root_angular_velocity(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion)
    )
    angular_velocity = jnp.asarray((0.4, -0.2, 0.1))
    motion = replace(
        env.motion,
        qvel=env.motion.qvel.at[:, 3:6].set(angular_velocity),
    )
    env = replace(env, motion=motion)

    reference = env.aligned_reference(
        jnp.asarray(0.1),
        jnp.zeros(2),
        jnp.asarray(np.pi / 2.0),
    )

    np.testing.assert_allclose(
        np.asarray(reference.qvel[3:6]),
        np.asarray(angular_velocity),
        atol=1.0e-7,
    )


def test_beyondmimic_observes_body_local_root_gyro_at_nonzero_yaw(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, beyondmimic_config(reference_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    root_quat = yaw_quat_from_angle(jnp.asarray(np.pi / 2.0))
    angular_velocity = jnp.asarray((0.4, -0.2, 0.1))
    data = state.data.replace(
        qpos=state.data.qpos.at[3:7].set(root_quat),
        qvel=state.data.qvel.at[3:6].set(angular_velocity),
    )
    state = state.replace(data=mjx.forward(env.model.mjx_model, data))

    observation = env.observation(state)

    np.testing.assert_allclose(
        np.asarray(
            observation[observation_component_slice(env, "base_angular_velocity")]
        ),
        np.asarray(angular_velocity),
        atol=1.0e-6,
    )


def test_beyondmimic_zero_action_targets_nominal_pose(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset,
        beyondmimic_config(reference_motion),
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    ctrl = env.action_target(state, jnp.zeros(env.action_size))

    np.testing.assert_allclose(
        np.asarray(ctrl),
        np.asarray(env.model.default_ctrl),
        atol=1.0e-6,
    )
    assert env.observation_size == 5 * env.action_size + 15
    assert env.critic_observation_size == env.observation_size + 9 * len(
        env.motion.body_names
    )


def test_beyondmimic_scales_each_physical_motor_by_effort_and_stiffness(
    motion_asset,
    reference_motion,
):
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, beyondmimic_config(reference_motion), load_mjx=False
    )
    model = env.model.mj_model
    policy = env.model.policy_ctrl_indices
    expected_scale = (
        0.25 * env.model.actuator_peak_force[policy] / model.actuator_gainprm[policy, 0]
    )
    np.testing.assert_allclose(env.beyondmimic_action_scale, expected_scale, atol=1e-6)
    action = (
        jnp.zeros(env.action_size)
        .at[env.action_layout.names.index("left_ankle_motor_1")]
        .set(1.0)
    )
    expected = env.model.default_ctrl.copy()
    expected[policy] += expected_scale * action
    limits = model.actuator_ctrlrange
    np.testing.assert_allclose(
        env.action_to_ctrl(action),
        np.clip(expected, limits[:, 0], limits[:, 1]),
        atol=1e-6,
    )


def test_beyondmimic_bounded_actions_use_motor_limits(motion_asset, reference_motion):
    config = replace(
        beyondmimic_config(reference_motion), formulation="beyondmimic_joint_limits"
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config, load_mjx=False)
    policy = env.model.policy_ctrl_indices
    limits = env.model.mj_model.actuator_ctrlrange
    offset = jnp.linspace(-0.005, 0.005, env.action_size)
    full_offset = np.zeros(env.model.mj_model.nu)
    full_offset[policy] = offset
    center = np.clip(
        env.model.default_ctrl + env.model.actuator_coordinate_transform @ full_offset,
        limits[:, 0],
        limits[:, 1],
    )
    for sign, column in ((-1.0, 0), (1.0, 1)):
        ctrl = env.action_to_ctrl(jnp.full(env.action_size, sign), offset)
        np.testing.assert_allclose(ctrl[policy], limits[policy, column], atol=1e-6)
    np.testing.assert_allclose(
        env.action_to_ctrl(jnp.zeros(env.action_size), offset), center, atol=1e-6
    )


def test_beyondmimic_body_targets_follow_anchor_xy_and_yaw(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, beyondmimic_config(reference_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    qpos = state.data.qpos.at[:2].add(jnp.asarray([0.2, -0.1]))
    qpos = qpos.at[3:7].set(
        quat_multiply(yaw_quat_from_angle(jnp.asarray(0.4)), qpos[3:7])
    )
    state = state.replace(
        data=mjx.forward(env.model.mjx_model, state.data.replace(qpos=qpos))
    )
    reference = env.aligned_reference(
        state.reference_time, state.alignment_xy, state.alignment_yaw
    )
    target_position, _ = env._relative_body_targets(state, reference)
    anchor_index = env.anchor_body_index

    np.testing.assert_allclose(
        np.asarray(target_position[anchor_index, :2]),
        np.asarray(state.data.xpos[env.anchor_body_id, :2]),
        atol=1.0e-5,
    )
    assert float(target_position[anchor_index, 2]) == pytest.approx(
        float(reference.body_pos[anchor_index, 2]), abs=1.0e-5
    )
    assert not bool(env.termination_flags(state)["done/orientation"])


def test_adaptive_sampling_spreads_failure_weight_backward(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = beyondmimic_config(reference_motion)
    config = config.__class__(
        **{
            **config.__dict__,
            "adaptive_sampling": MotionAdaptiveSamplingConfig(
                enabled=True,
                bin_seconds=0.1,
                ema_alpha=0.1,
                kernel_size=3,
                kernel_decay=0.5,
                uniform_ratio=0.1,
            ),
        }
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config)
    failure_bin = env.sampling_bin_count - 1
    failure = jnp.zeros(env.sampling_bin_count).at[failure_bin].set(1.0)
    probabilities = env._sampling_probabilities(failure)
    scores = np.asarray(failure) + 0.1 / env.sampling_bin_count
    kernel = np.asarray([1.0, 0.5, 0.25])
    kernel /= kernel.sum()
    padded = np.pad(scores, (0, len(kernel) - 1), mode="edge")
    expected = sum(
        padded[offset : offset + env.sampling_bin_count] * weight
        for offset, weight in enumerate(kernel)
    )
    expected /= expected.sum()

    np.testing.assert_allclose(probabilities, expected, rtol=1.0e-6)
    assert float(probabilities[failure_bin]) > float(probabilities[failure_bin - 1])
    assert float(probabilities[failure_bin - 1]) > float(probabilities[failure_bin - 2])


def test_adaptive_sampling_uses_full_width_final_bin_without_endpoint_clipping(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = beyondmimic_config(reference_motion)
    config = config.__class__(
        **{
            **config.__dict__,
            "reset": MotionResetConfig(random_start=True),
            "adaptive_sampling": MotionAdaptiveSamplingConfig(
                enabled=True,
                bin_seconds=0.1,
                ema_alpha=0.1,
                kernel_size=1,
                kernel_decay=0.8,
                uniform_ratio=0.0,
            ),
        }
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config)
    failure = jnp.zeros(env.sampling_bin_count).at[-1].set(1.0)
    keys = jax.random.split(jax.random.PRNGKey(7), 256)
    sampled = jax.vmap(lambda key: env._sample_reference_time(key, failure))(keys)
    max_time = env.motion.duration - env.control_timestep
    final_bin_start = max_time * (env.sampling_bin_count - 1) / env.sampling_bin_count

    assert np.all(np.asarray(sampled) >= final_bin_start)
    assert np.all(np.asarray(sampled) < max_time)
    assert np.unique(np.asarray(sampled)).size > 200


def test_adaptive_sampling_ema_uses_terminal_counts_per_phase_bin(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    from types import SimpleNamespace

    config = beyondmimic_config(reference_motion)
    config = config.__class__(
        **{
            **config.__dict__,
            "adaptive_sampling": MotionAdaptiveSamplingConfig(
                enabled=True,
                bin_seconds=0.1,
                ema_alpha=0.1,
                kernel_size=1,
                kernel_decay=0.8,
                uniform_ratio=0.1,
            ),
        }
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config)
    max_time = env.motion.duration - env.control_timestep
    reference_time = jnp.asarray([0.0, 0.01, 0.99 * max_time, 0.99 * max_time])
    state = SimpleNamespace(
        reference_time=reference_time,
        sampling_failure_ema=jnp.zeros((4, env.sampling_bin_count)),
    )

    updated = env._update_sampling_failure_ema(
        state, jnp.asarray([True, False, True, True])
    )

    expected = np.zeros(env.sampling_bin_count)
    expected[0] = 0.1
    expected[-1] = 0.2
    np.testing.assert_allclose(updated, expected, atol=1.0e-7)


def test_beyondmimic_observation_noise_is_actor_only(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = beyondmimic_config(reference_motion)
    config = config.__class__(
        **{
            **config.__dict__,
            "observation_noise": MotionObservationNoiseConfig(
                enabled=True,
                anchor_position=0.25,
                anchor_orientation=0.05,
                base_linear_velocity=0.5,
                base_angular_velocity=0.2,
                joint_position=0.01,
                joint_velocity=0.5,
            ),
        }
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config)
    state, clean = env.reset(reference_time=jnp.asarray(0.1))
    noisy = env.observation(state, jax.random.PRNGKey(0))

    assert not np.allclose(np.asarray(clean), np.asarray(noisy))
    np.testing.assert_allclose(
        np.asarray(env.critic_observation(state)[: env.observation_size]),
        np.asarray(clean),
    )


def test_beyondmimic_observation_latency_delays_only_actor_feedback(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = replace(
        beyondmimic_config(reference_motion),
        observation_latency=MotionObservationLatencyConfig(
            enabled=True, delay_steps=(2, 2)
        ),
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config)
    state, initial_observation = env.reset(
        jax.random.PRNGKey(0), reference_time=jnp.asarray(0.1)
    )
    qpos_indices = jnp.asarray(env.model.policy_qpos_indices)
    data = state.data.replace(
        qpos=state.data.qpos.at[qpos_indices].add(0.1),
    )
    data = mjx.forward(env.model.mjx_model, data)
    state = state.replace(
        data=data,
        last_action=jnp.full(env.action_size, 0.4),
        actor_feedback_history=env._advance_actor_feedback_history(
            state.actor_feedback_history, data
        ),
    )

    delayed = env.observation(state)
    current = env.critic_observation(state)[: env.observation_size]
    joint_slice = observation_component_slice(env, "joint_position")
    action_slice = observation_component_slice(env, "last_action")

    assert int(state.actor_feedback_delay_steps) == 2
    np.testing.assert_allclose(
        np.asarray(delayed[joint_slice]),
        np.asarray(initial_observation[joint_slice]),
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(current[joint_slice]),
        np.asarray(initial_observation[joint_slice]) + 0.1,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(np.asarray(delayed[action_slice]), 0.4, atol=1.0e-6)


def test_observation_latency_rejects_invalid_step_range(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = replace(
        beyondmimic_config(reference_motion),
        observation_latency=MotionObservationLatencyConfig(
            enabled=True, delay_steps=(2, 1)
        ),
    )

    with pytest.raises(ValueError, match="lower bound exceeds upper bound"):
        BelloMotionTrackingEnv.from_compiled(motion_asset, config)


def test_tracking_nonfinite_state_is_terminal_and_reward_is_sanitized(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    state = state.replace(
        data=state.data.replace(cvel=state.data.cvel.at[0, 0].set(jnp.nan))
    )
    done_flags = env.termination_flags(state)
    reward, reward_info = env._reward(state, state)
    reward, reward_info = env._sanitize_nonfinite_reward(
        reward, reward_info, done_flags
    )

    assert bool(done_flags["done/nonfinite"])
    assert bool(done_flags["done/terminal"])
    assert float(reward) == 0.0
    assert all(float(value) == 0.0 for value in reward_info.values())


def test_tracking_joint_limit_cost_is_bounded(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    indices = jnp.asarray(env.model.policy_qpos_indices)
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[indices].set(1.0e6))
    )

    assert float(env._joint_limit_cost(state)) == 1.0


def test_tracking_torque_costs_are_normalized_by_peak_torque(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    peak = jnp.asarray(env.model.actuator_peak_force)
    state = state.replace(data=state.data.replace(actuator_force=peak * 1.5))

    torque_l1, torque_peak = env._torque_costs(state)

    assert float(torque_l1) == pytest.approx(1.5)
    assert float(torque_peak) == pytest.approx(0.25)


@pytest.mark.integration
@requires_warp_gpu
def test_tracking_batch_reset_and_step_are_jittable(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.from_compiled(
        motion_asset, tracking_config(reference_motion), batch_size=2
    )
    state, observation = jax.jit(lambda key: env.reset_batch(2, key))(
        jax.random.PRNGKey(0)
    )
    next_state, next_observation, reward, done, info = jax.jit(env.step_batch)(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        jax.random.PRNGKey(1),
    )
    jax.block_until_ready((next_state, next_observation, reward, done, info))

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == (2, env.observation_size)
    assert next_state.last_action.shape == (2, env.action_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["done/terminal"].shape == (2,)


@pytest.mark.integration
@requires_warp_gpu
def test_beyondmimic_batch_step_with_sampling_noise_and_push_is_jittable(
    motion_asset: CompiledRobot, reference_motion: Path
) -> None:
    config = beyondmimic_config(reference_motion)
    config = config.__class__(
        **{
            **config.__dict__,
            "observation_noise": MotionObservationNoiseConfig(
                enabled=True,
                anchor_position=0.25,
                anchor_orientation=0.05,
                base_linear_velocity=0.5,
                base_angular_velocity=0.2,
                joint_position=0.01,
                joint_velocity=0.5,
            ),
            "observation_latency": MotionObservationLatencyConfig(
                enabled=True, delay_steps=(0, 2)
            ),
            "adaptive_sampling": MotionAdaptiveSamplingConfig(
                enabled=True,
                bin_seconds=0.1,
                ema_alpha=0.1,
                kernel_size=2,
                kernel_decay=0.8,
                uniform_ratio=0.1,
            ),
            "push": MotionPushConfig(
                enabled=True,
                interval_seconds=(0.02, 0.02),
            ),
        }
    )
    env = BelloMotionTrackingEnv.from_compiled(motion_asset, config, batch_size=2)
    state, observation = jax.jit(lambda key: env.reset_batch(2, key))(
        jax.random.PRNGKey(0)
    )
    next_state, next_observation, reward, done, info = jax.jit(env.step_batch)(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        jax.random.PRNGKey(1),
    )
    jax.block_until_ready((next_state, next_observation, reward, done, info))

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == observation.shape
    assert next_state.actor_feedback_history.shape[1] == 3
    assert np.all(np.asarray(next_state.actor_feedback_delay_steps) <= 2)
    assert next_state.sampling_failure_ema.shape == (2, env.sampling_bin_count)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["reward/relative_body_position"].shape == (2,)
    assert info["sampling/entropy"].shape == (2,)
    assert info["push/applied"].shape == (2,)
