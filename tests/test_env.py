from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BadOrientationDoneConfig,
    BaseHeightDoneConfig,
    BelloEnvConfig,
    BelloLocomotionEnv,
    ContactConfig,
    DoneConfig,
    GaitConfig,
    ObservationConfig,
    ObservationLatencyConfig,
    ObservationLowPassFilterConfig,
    ObservationNoiseConfig,
    RandomPushConfig,
    SpawnConfig,
    TorsoComCurriculumConfig,
    advance_gait_phase,
    desired_gait_foot_contact,
    gait_phase_features,
    resolve_gait_command_transition,
)
from bello_mujoco.envs.rewards import (
    CommandConfig,
    RewardConfig,
    RewardLinks,
    command_is_moving,
    command_is_standing,
)
from bello_mujoco.evaluation.render import (
    mujoco_done_flags,
    mujoco_eval_info,
    mujoco_eval_state,
    mujoco_foot_height,
    mujoco_link_ang_vel,
    mujoco_link_lin_vel,
    mujoco_policy_observation,
    reset_mujoco_data,
)
from bello_mujoco.sim import ModelRandomizationConfig
from bello_mujoco.terrain import TerrainConfig, terrain_surface_height


def observation_component_slice(
    env: BelloLocomotionEnv, component_name: str, *, critic: bool = False
) -> slice:
    offset = 0
    layout = env.critic_observation_layout if critic else env.observation_layout
    for name, size in layout.components:
        if name == component_name:
            return slice(offset, offset + size)
        offset += size
    raise AssertionError(f"observation component not found: {component_name}")


def test_reset_exposes_explicit_actor_critic_and_action_contracts(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, observation = env.reset()

    assert env.action_layout.size == env.action_size == 13
    assert set(env.action_layout.names) == {
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
    }
    assert env.observation_layout.components == (
        ("imu_linear_acceleration", 3),
        ("imu_angular_velocity", 3),
        ("projected_gravity", 3),
        ("command", 3),
        ("gait_phase", 3),
        ("joint_position", 13),
        ("joint_velocity", 13),
        ("last_action", 13),
    )
    assert env.critic_observation_layout.components[-3:] == (
        ("base_linear_velocity", 3),
        ("base_height", 1),
        ("actuator_force", 13),
    )
    assert observation.shape == (env.observation_size,)
    assert env.critic_observation(state).shape == (env.critic_observation_size,)
    assert state.last_action.shape == (env.action_size,)


def test_point_nine_hz_gait_clock_is_alternating_and_zero_while_standing() -> None:
    command_config = CommandConfig()
    gait_config = GaitConfig(frequency_hz=0.9, stance_fraction=0.6)
    moving = jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32)
    standing = jnp.zeros(3, dtype=jnp.float32)

    phase = advance_gait_phase(
        jnp.asarray(0.0), moving, command_config, gait_config, 0.02
    )

    assert phase == pytest.approx(0.018)
    assert gait_phase_features(
        jnp.asarray(0.0), moving, command_config
    ) == pytest.approx(jnp.asarray([0.0, 1.0, 1.0]))
    assert gait_phase_features(phase, standing, command_config) == pytest.approx(
        jnp.zeros(3)
    )
    assert advance_gait_phase(
        phase, standing, command_config, gait_config, 0.02
    ) == pytest.approx(0.0)
    assert desired_gait_foot_contact(jnp.asarray(0.2), gait_config) == pytest.approx(
        jnp.asarray([1.0, 0.0])
    )
    assert desired_gait_foot_contact(jnp.asarray(0.6), gait_config) == pytest.approx(
        jnp.asarray([0.0, 1.0])
    )


def test_gait_clock_requires_multiple_control_steps_per_half_cycle(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    invalid_frequency = 0.5 / env.control_timestep

    with pytest.raises(ValueError, match="less than half a cycle"):
        replace(
            env,
            config=replace(
                env.config,
                gait=GaitConfig(frequency_hz=invalid_frequency),
            ),
        )


def test_bilateral_mirrors_are_explicit_and_involutive(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)

    action = jnp.linspace(-0.9, 0.9, env.action_size)
    assert env.mirror_action(env.mirror_action(action)) == pytest.approx(action)
    names = env.action_layout.names
    assert env.mirror_action(action)[names.index("right_hip_pitch_joint")] == (
        pytest.approx(-action[names.index("left_hip_pitch_joint")])
    )
    assert env.mirror_action(action)[names.index("right_hip_roll_joint")] == (
        pytest.approx(action[names.index("left_hip_roll_joint")])
    )

    observation = jnp.linspace(-1.0, 1.0, 2 * env.observation_size)
    mirrored = env.mirror_actor_observation(observation)
    assert env.mirror_actor_observation(mirrored) == pytest.approx(observation)


def test_robot_specific_locomotion_action_and_command_frame_contracts(
    bello_asset: PreparedAsset,
    bello_mini_asset: PreparedAsset,
) -> None:
    bello = BelloLocomotionEnv.load_prepared(bello_asset, load_mjx=False)
    mini = BelloLocomotionEnv.load_prepared(bello_mini_asset, load_mjx=False)

    assert bello.reward_config.links.track_lin_vel_xy_frame is None
    assert mini.reward_config.links.track_lin_vel_xy_frame is None
    assert bello._reward_frame_link("track_lin_vel_xy") == "pelvis_link"
    assert mini._reward_frame_link("track_lin_vel_xy") == "imu_link"
    assert mini.action_size == 14
    assert "waist_roll_joint" in mini.action_layout.names

    action = jnp.linspace(-0.9, 0.9, mini.action_size)
    mirrored = mini.mirror_action(action)
    names = mini.action_layout.names
    waist_roll = names.index("waist_roll_joint")
    assert mirrored[names.index("right_hip_pitch_joint")] == pytest.approx(
        action[names.index("left_hip_pitch_joint")]
    )
    assert mirrored[names.index("right_hip_roll_joint")] == pytest.approx(
        -action[names.index("left_hip_roll_joint")]
    )
    assert mirrored[waist_roll] == pytest.approx(-action[waist_roll])
    assert mini.mirror_action(mirrored) == pytest.approx(action)


def test_foot_heading_uses_each_mirrored_foot_forward_axis(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env._reset(
        jnp.zeros(3),
        None,
        0,
        forward=False,
        align_clearance=False,
    )

    heading_error_degrees = np.degrees(np.asarray(env._foot_heading_error(state)))

    assert np.all(np.abs(heading_error_degrees) < 5.0)
    assert heading_error_degrees[0] == pytest.approx(-heading_error_degrees[1], abs=0.1)


def test_bello_mini_foot_heading_uses_its_rotated_command_frame(
    bello_mini_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_mini_asset, load_mjx=False)
    mj_data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, mj_data)
    zeros = jnp.zeros(2, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        mj_data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=0,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(0),
    )

    heading_error_degrees = np.degrees(np.asarray(env._foot_heading_error(state)))

    assert np.all(np.abs(heading_error_degrees) < 5.0)


def test_native_data_adapter_uses_shared_observation_function(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mujoco.mj_forward(env.model.mj_model, mj_data)
    command = np.asarray([0.3, -0.1, 0.2], dtype=np.float32)
    last_action = np.linspace(-0.2, 0.2, env.action_size, dtype=np.float32)
    zeros = jnp.zeros(2, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=7,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(7),
    )

    direct = env.observation(state)
    adapted = mujoco_policy_observation(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        step_count=7,
    )
    assert direct == pytest.approx(adapted, abs=1.0e-6)


def test_native_data_adapter_preserves_constraint_forces(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, load_mjx=False)
    mj_data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, mj_data)
    mj_data.qpos[env.model.root_qpos_adr + 2] -= 0.02
    mujoco.mj_forward(env.model.mj_model, mj_data)
    assert mj_data.efc_force.size > 0

    expected = np.asarray(mj_data.efc_force, dtype=np.float32).copy()
    state = mujoco_eval_state(
        env,
        mj_data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_contact_debounce_time=jnp.zeros(2, dtype=jnp.float32),
        last_foot_contact=jnp.zeros(2, dtype=jnp.float32),
        step_count=0,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(0),
    )

    assert state.data._impl.efc__force == pytest.approx(expected)

    mj_data.efc_force[0] = 2.0e6
    assert mujoco_done_flags(env, mj_data, 0)["done/unstable"]


def test_mjx_link_kinematics_match_native_mujoco(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = np.linspace(-0.2, 0.3, env.model.mj_model.nv)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    state, _ = env.reset()
    state = state.replace(
        data=state.data.replace(
            qpos=jnp.asarray(mj_data.qpos),
            qvel=jnp.asarray(mj_data.qvel),
            xpos=jnp.asarray(mj_data.xpos),
            xquat=jnp.asarray(mj_data.xquat),
            cvel=jnp.asarray(mj_data.cvel),
            subtree_com=jnp.asarray(mj_data.subtree_com),
            _impl=state.data._impl.replace(cacc=jnp.asarray(mj_data.cacc)),
        )
    )

    for link in ("bello_root", "pelvis_link", "torso_link"):
        assert env._link_body_lin_vel(state, link) == pytest.approx(
            mujoco_link_lin_vel(env, mj_data, link), abs=1.0e-5
        )
        assert env._link_body_ang_vel(state, link) == pytest.approx(
            mujoco_link_ang_vel(env, mj_data, link), abs=1.0e-5
        )


def test_tracking_yaw_velocity_uses_recursive_low_pass_filter(
    bello_asset: PreparedAsset,
) -> None:
    reward_config = RewardConfig(
        tracking=replace(
            RewardConfig().tracking,
            ang_vel_z_lpf_cutoff_hz=5.0,
        )
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        reward_config=reward_config,
    )
    previous = jnp.asarray(0.25, dtype=jnp.float32)
    measured = jnp.asarray(1.25, dtype=jnp.float32)

    filtered = env.filter_tracking_ang_vel_z(previous, measured)
    alpha = 1.0 - np.exp(-2.0 * np.pi * 5.0 * env.control_timestep)

    assert filtered == pytest.approx(0.25 + alpha, abs=1.0e-6)
    assert env.filter_tracking_ang_vel_z(filtered, measured) > filtered


def test_reset_batch_and_step_have_finite_fixed_shapes(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, batch_size=2)
    state, observation = env.reset_batch(2, rng=jax.random.PRNGKey(0))
    assert state.tracking_ang_vel_z == pytest.approx(
        jax.vmap(env._raw_tracking_ang_vel_z)(state),
        abs=1.0e-6,
    )
    next_state, next_observation, reward, done, info = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(1),
    )

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == observation.shape
    assert reward.shape == done.shape == (2,)
    assert jnp.all(jnp.isfinite(next_observation))
    assert jnp.all(jnp.isfinite(reward))
    assert next_state.data.ctrl.shape == (2, env.model.mj_model.nu)
    assert "reward/track_lin_vel_xy" in info
    assert jnp.all(jnp.isfinite(info["gait/foot_heading_error_mean_deg"]))
    assert jnp.all(jnp.isfinite(info["gait/foot_heading_error_max_deg"]))
    assert "done/nonfinite" in info
    assert "done/unstable" in info


def test_step_batch_skips_full_reset_when_no_world_is_done(
    bello_asset: PreparedAsset, monkeypatch
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, batch_size=2)
    state, _ = env.reset_batch(2)
    reset_callbacks = []
    original_reset_batch = BelloLocomotionEnv.reset_batch

    def instrumented_reset_batch(self, *args, **kwargs):
        jax.debug.callback(lambda _: reset_callbacks.append(True), jnp.asarray(0))
        return original_reset_batch(self, *args, **kwargs)

    monkeypatch.setattr(
        BelloLocomotionEnv,
        "reset_batch",
        instrumented_reset_batch,
    )
    step = jax.jit(
        lambda item: env.step_batch(
            item,
            jnp.zeros((2, env.action_size), dtype=jnp.float32),
        )
    )

    _, observation, _, done, _ = step(state)
    observation.block_until_ready()
    assert not jnp.any(done)
    assert reset_callbacks == []

    timed_out = state.replace(
        step_count=jnp.full(2, env.max_episode_steps, dtype=jnp.int32)
    )
    _, observation, _, done, _ = step(timed_out)
    observation.block_until_ready()
    assert jnp.all(done)
    assert reset_callbacks == [True]


def test_reset_forward_populates_velocity_derived_imu_observation(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            spawn=SpawnConfig(
                joint_vel_range=(0.1, 0.1),
                root_vel_range=(0.25, 0.25),
            )
        ),
    )

    state, observation = env.reset(rng=jax.random.PRNGKey(3))
    pelvis_id = env._link_body_id(env.model.imu_body_name)
    imu_ang_vel = observation[observation_component_slice(env, "imu_angular_velocity")]

    assert jnp.allclose(state.data.qvel[:6], 0.25)
    assert not jnp.allclose(state.data.cvel[pelvis_id], 0.0)
    assert imu_ang_vel == pytest.approx(env._imu_angular_velocity(state))
    assert not jnp.allclose(imu_ang_vel, 0.0)


def test_timeout_auto_reset_preserves_terminal_diagnostics(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=1,
        env_config=BelloEnvConfig(max_episode_steps=1),
    )
    state, _ = env.reset_batch(1)
    next_state, _, _, done, info = env.step_batch(
        state, jnp.zeros((1, env.action_size), dtype=jnp.float32)
    )

    assert bool(done[0])
    assert bool(info["done/timeout"][0])
    assert not bool(info["done/terminal"][0])
    assert int(next_state.step_count[0]) == 0
    assert int(info["episode_length"][0]) == 1


def test_action_target_reports_joint_and_motor_saturation_separately(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    ctrl, joint_saturation, motor_saturation = env._action_target_details(
        jnp.full(env.action_size, env.action_clip)
    )
    ctrl_range = env.model.mj_model.actuator_ctrlrange

    assert ctrl.shape == (env.model.mj_model.nu,)
    assert jnp.all(ctrl >= ctrl_range[:, 0])
    assert jnp.all(ctrl <= ctrl_range[:, 1])
    assert 0.0 <= joint_saturation <= 1.0
    assert 0.0 <= motor_saturation <= 1.0


def test_contact_debounce_has_explicit_transitions(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            contact=ContactConfig(
                foot_contact_on_seconds=0.04,
                foot_contact_off_seconds=0.04,
            )
        ),
    )
    state, _ = env.reset()
    raw = jnp.asarray([True, False])
    first, elapsed = env.debounce_foot_contact(state, raw, jnp.asarray(False))
    second, _ = env.debounce_foot_contact(
        state.replace(foot_contact_debounce_time=elapsed), raw, jnp.asarray(False)
    )
    assert jnp.array_equal(first.astype(bool), jnp.asarray([False, False]))
    assert jnp.array_equal(second.astype(bool), jnp.asarray([True, False]))


def test_touchdown_is_emitted_on_first_raw_contact_sample_only(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env.reset()
    raw = jnp.asarray([True, False])
    initialized = state.replace(step_count=jnp.asarray(1))

    touchdown = env._foot_touchdown(initialized, raw)
    debouncing = env._foot_touchdown(
        initialized.replace(
            foot_contact_debounce_time=jnp.asarray([0.02, 0.0], dtype=jnp.float32)
        ),
        raw,
    )
    filtered = env._foot_touchdown(
        initialized.replace(last_foot_contact=jnp.asarray([True, False])), raw
    )

    assert jnp.array_equal(touchdown, jnp.asarray([True, False]))
    assert not jnp.any(debouncing)
    assert not jnp.any(filtered)


def test_observation_noise_changes_only_noisy_evaluation(
    bello_asset: PreparedAsset,
) -> None:
    noise = ObservationNoiseConfig(enabled=True, joint_pos=(-0.1, 0.1))
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(observation=ObservationConfig(noise=noise)),
    )
    state, clean = env.reset()
    noisy = env.observation(state, jax.random.PRNGKey(0))

    assert clean.shape == noisy.shape
    assert not jnp.allclose(clean, noisy)
    assert env.observation(state) == pytest.approx(clean)


def test_observation_latency_delays_only_actor_feedback(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            observation_latency=ObservationLatencyConfig(
                enabled=True,
                delay_steps=(2, 2),
            )
        ),
    )
    state, _ = env.reset(rng=jax.random.PRNGKey(0))
    assert int(state.actor_feedback_delay_steps) == 2
    assert state.actor_feedback_history.shape == (3, env.actor_feedback_size)
    current_feedback = env._actor_feedback(state)
    delayed_feedback = current_feedback + 0.25
    state = state.replace(
        actor_feedback_history=state.actor_feedback_history.at[2].set(delayed_feedback),
        actor_feedback_delay_steps=jnp.asarray(2, dtype=jnp.int32),
        command=jnp.asarray((0.1, 0.2, 0.3), dtype=jnp.float32),
        last_action=jnp.full(env.action_size, 0.4, dtype=jnp.float32),
    )

    observation = env.observation(state)
    critic_observation = env.critic_observation(state)
    delayed = env._unpack_actor_feedback(delayed_feedback)
    current = env._unpack_actor_feedback(current_feedback)

    for component, expected in zip(
        (
            "imu_linear_acceleration",
            "imu_angular_velocity",
            "projected_gravity",
            "joint_position",
            "joint_velocity",
        ),
        delayed,
        strict=True,
    ):
        assert observation[observation_component_slice(env, component)] == (
            pytest.approx(expected)
        )
    assert observation[observation_component_slice(env, "command")] == (
        pytest.approx(state.command)
    )
    assert observation[observation_component_slice(env, "last_action")] == (
        pytest.approx(state.last_action)
    )
    assert critic_observation[
        observation_component_slice(env, "joint_position", critic=True)
    ] == pytest.approx(current[3])


def test_actor_feedback_low_pass_matches_hardware_first_order_response(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            observation_low_pass_filter=ObservationLowPassFilterConfig(
                enabled=True,
                cutoff_hz=10.0,
            )
        ),
    )
    previous = jnp.concatenate(
        (
            jnp.zeros(3),
            jnp.zeros(3),
            jnp.asarray((0.0, 0.0, -1.0)),
            jnp.zeros(env.action_size),
            jnp.zeros(env.action_size),
        )
    )
    measured = jnp.concatenate(
        (
            jnp.ones(3),
            jnp.ones(3),
            jnp.asarray((0.0, -1.0, 0.0)),
            jnp.ones(env.action_size),
            jnp.ones(env.action_size),
        )
    )
    timestep = env.control_timestep
    filtered = env._low_pass_actor_feedback(previous, measured, timestep)
    acceleration, angular_velocity, gravity, position, velocity = (
        env._unpack_actor_feedback(filtered)
    )
    alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * timestep)
    expected_gravity = np.asarray((0.0, -alpha, -(1.0 - alpha)))
    expected_gravity /= np.linalg.norm(expected_gravity)

    assert acceleration == pytest.approx(np.full(3, alpha))
    assert angular_velocity == pytest.approx(np.full(3, alpha))
    assert gravity == pytest.approx(expected_gravity)
    assert position == pytest.approx(np.ones(env.action_size))
    assert velocity == pytest.approx(np.full(env.action_size, alpha))


def test_actor_feedback_noise_is_filtered_before_history(
    bello_asset: PreparedAsset,
) -> None:
    noise = ObservationNoiseConfig(
        enabled=True,
        base_ang_vel=(1.0, 1.0),
        joint_pos=(0.5, 0.5),
        joint_vel=(2.0, 2.0),
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(noise=noise),
            observation_low_pass_filter=ObservationLowPassFilterConfig(
                enabled=True,
                cutoff_hz=10.0,
                noise_before_filter=True,
                residual_noise_scale=0.0,
            ),
        ),
    )
    state, _ = env.reset()
    measured = env._actor_feedback(state)
    advanced = env._advance_actor_feedback_history(
        state.actor_feedback_history, state, jax.random.PRNGKey(4)
    )[0]
    previous = env._unpack_actor_feedback(state.actor_feedback_history[0])
    filtered = env._unpack_actor_feedback(advanced)
    alpha = 1.0 - np.exp(-2.0 * np.pi * 10.0 * env.control_timestep)

    assert filtered[1] == pytest.approx(previous[1] + alpha)
    assert filtered[3] == pytest.approx(env._unpack_actor_feedback(measured)[3] + 0.5)
    assert filtered[4] == pytest.approx(previous[4] + 2.0 * alpha)


def test_torso_com_curriculum_scales_fixed_world_offsets(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=4,
        env_config=BelloEnvConfig(
            model_randomization=ModelRandomizationConfig(
                enabled=True,
                seed=31,
                torso_com_offset_range=(-0.1, 0.1),
            ),
            torso_com_curriculum=TorsoComCurriculumConfig(
                enabled=True,
                ramp_steps=100,
            ),
        ),
    )
    torso_id = mujoco.mj_name2id(
        env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    full = np.asarray(env.model.mjx_model.body_ipos)[:, torso_id]
    offset = env.model.torso_com_offset
    zero = np.asarray(
        env.with_task_state(jnp.asarray([0.0])).model.mjx_model.body_ipos
    )[:, torso_id]
    half = np.asarray(
        env.with_task_state(jnp.asarray([50.0])).model.mjx_model.body_ipos
    )[:, torso_id]
    complete = np.asarray(
        env.with_task_state(jnp.asarray([100.0])).model.mjx_model.body_ipos
    )[:, torso_id]

    np.testing.assert_allclose(zero, full - offset, atol=1.0e-7)
    np.testing.assert_allclose(half, full - 0.5 * offset, atol=1.0e-7)
    np.testing.assert_allclose(complete, full, atol=1.0e-7)
    assert env.initial_task_state() == pytest.approx(jnp.asarray([0.0]))
    assert env.update_task_state(jnp.asarray([99.5]), {}) == pytest.approx(
        jnp.asarray([100.0])
    )


@pytest.mark.parametrize(
    "delay_steps",
    ((-1, 1), (2, 1), (0, 1, 2)),
)
def test_observation_latency_rejects_invalid_step_range(
    bello_asset: PreparedAsset,
    delay_steps: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="observation_latency.delay_steps"):
        BelloLocomotionEnv.load_prepared(
            bello_asset,
            env_config=BelloEnvConfig(
                observation_latency=ObservationLatencyConfig(
                    enabled=True,
                    delay_steps=delay_steps,
                )
            ),
        )


@pytest.mark.parametrize("cutoff_hz", (0.0, -1.0, float("nan")))
def test_observation_low_pass_filter_rejects_invalid_cutoff(
    bello_asset: PreparedAsset,
    cutoff_hz: float,
) -> None:
    with pytest.raises(ValueError, match="observation_low_pass_filter.cutoff_hz"):
        BelloLocomotionEnv.load_prepared(
            bello_asset,
            env_config=BelloEnvConfig(
                observation_low_pass_filter=ObservationLowPassFilterConfig(
                    enabled=True,
                    cutoff_hz=cutoff_hz,
                )
            ),
        )


@pytest.mark.parametrize("residual_noise_scale", (-0.1, 1.1, float("nan")))
def test_observation_low_pass_filter_rejects_invalid_residual_noise_scale(
    bello_asset: PreparedAsset,
    residual_noise_scale: float,
) -> None:
    with pytest.raises(ValueError, match="residual_noise_scale"):
        BelloLocomotionEnv.load_prepared(
            bello_asset,
            env_config=BelloEnvConfig(
                observation_low_pass_filter=ObservationLowPassFilterConfig(
                    enabled=True,
                    residual_noise_scale=residual_noise_scale,
                )
            ),
        )


def test_torso_com_curriculum_requires_model_randomization(
    bello_asset: PreparedAsset,
) -> None:
    with pytest.raises(ValueError, match="requires model_randomization.enabled"):
        BelloLocomotionEnv.load_prepared(
            bello_asset,
            env_config=BelloEnvConfig(
                torso_com_curriculum=TorsoComCurriculumConfig(enabled=True)
            ),
        )


def test_command_sampling_and_periodic_resampling_are_bounded(
    bello_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(
        lin_vel_x=(0.2, 0.4),
        lin_vel_y=(-0.1, 0.1),
        ang_vel_yaw=(-0.2, 0.2),
        standing_probability=0.0,
        resample_seconds=0.02,
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset, batch_size=4, command_config=command_config
    )
    state, _ = env.reset_batch(4, rng=jax.random.PRNGKey(0))
    next_state, _, _, _, info = env.step_batch(
        state,
        jnp.zeros((4, env.action_size)),
        rng=jax.random.PRNGKey(1),
    )

    assert jnp.all((state.command[:, 0] >= 0.2) & (state.command[:, 0] <= 0.4))
    assert info["command/standing"] == pytest.approx(jnp.zeros(4))
    assert info["command/moving"] == pytest.approx(jnp.ones(4))
    assert not jnp.allclose(next_state.command, state.command)


def test_batch_command_mode_cohort_survives_resampling(
    bello_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(
        standing_probability=0.25,
        resample_seconds=0.02,
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset, batch_size=8, command_config=command_config
    )
    state, _ = env.reset_batch(8, rng=jax.random.PRNGKey(10))
    standing = command_is_standing(state.command, command_config)
    reset_command = env._reset_batch_command(state.command, 8, jax.random.PRNGKey(12))
    next_state, _, _, _, _ = env.step_batch(
        state,
        jnp.zeros((8, env.action_size)),
        rng=jax.random.PRNGKey(11),
    )

    assert jnp.sum(standing) == 2
    assert jnp.array_equal(command_is_standing(reset_command, command_config), standing)
    assert jnp.array_equal(
        command_is_standing(next_state.command, command_config), standing
    )
    assert jnp.all(command_is_moving(next_state.command[~standing], command_config))


def test_training_reset_starts_moving_commands_at_natural_gait_boundaries(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, batch_size=64)
    moving = jnp.asarray([0.5, -0.1, 0.2], dtype=jnp.float32)

    state, observation = env.reset_batch(
        64,
        command=moving,
        rng=jax.random.PRNGKey(20),
    )

    assert jnp.unique(state.gait_phase) == pytest.approx(jnp.asarray([0.0, 0.5]))
    assert desired_gait_foot_contact(state.gait_phase, env.config.gait) == (
        pytest.approx(jnp.ones((64, 2)))
    )
    expected_phase_features = gait_phase_features(
        state.gait_phase,
        state.command,
        env.command_config,
    )
    assert observation[:, observation_component_slice(env, "gait_phase")] == (
        pytest.approx(expected_phase_features)
    )


@pytest.mark.parametrize(
    ("request_phase", "boundary_previous", "boundary_next"),
    (
        (0.05, 0.49, 0.51),
        (0.3, 0.49, 0.51),
        (0.55, 0.99, 0.01),
        (0.8, 0.99, 0.01),
    ),
)
def test_standing_transition_waits_for_the_next_gait_boundary(
    request_phase: float,
    boundary_previous: float,
    boundary_next: float,
) -> None:
    command_config = CommandConfig(
        standing_transitions_enabled=True,
    )
    moving = jnp.asarray([0.5, -0.1, 0.2], dtype=jnp.float32)
    standing = jnp.zeros(3, dtype=jnp.float32)
    resolve_transition = jax.jit(
        lambda command, requested, previous, next_, pending: (
            resolve_gait_command_transition(
                command,
                requested,
                previous,
                next_,
                pending,
                command_config,
            )
        )
    )

    command, phase, pending = resolve_transition(
        moving,
        standing,
        jnp.asarray(request_phase),
        jnp.asarray(request_phase),
        jnp.asarray(False),
    )

    assert command == pytest.approx(moving)
    assert phase == pytest.approx(request_phase)
    assert bool(pending)
    assert command_is_moving(command, command_config)

    command, phase, pending = resolve_transition(
        command,
        standing,
        jnp.asarray(boundary_previous),
        jnp.asarray(boundary_next),
        pending,
    )

    assert command == pytest.approx(standing)
    assert phase == pytest.approx(0.0)
    assert not bool(pending)
    assert command_is_standing(command, command_config)


def test_moving_request_cancels_pending_standing_transition() -> None:
    command_config = CommandConfig(standing_transitions_enabled=True)
    moving = jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32)
    replacement = jnp.asarray([-0.3, 0.2, -0.1], dtype=jnp.float32)

    command, phase, pending = resolve_gait_command_transition(
        moving,
        replacement,
        jnp.asarray(0.3),
        jnp.asarray(0.35),
        jnp.asarray(True),
        command_config,
    )

    assert command == pytest.approx(replacement)
    assert phase == pytest.approx(0.35)
    assert not bool(pending)


def test_standing_transition_resolver_vectorizes_over_worlds() -> None:
    command_config = CommandConfig(standing_transitions_enabled=True)
    moving = jnp.asarray([[0.5, 0.0, 0.0], [0.4, 0.1, 0.0]], dtype=jnp.float32)

    command, phase, pending = resolve_gait_command_transition(
        moving,
        jnp.zeros_like(moving),
        jnp.asarray([0.49, 0.30]),
        jnp.asarray([0.51, 0.32]),
        jnp.asarray([True, True]),
        command_config,
    )

    assert command[0] == pytest.approx(jnp.zeros(3))
    assert phase[0] == pytest.approx(0.0)
    assert not bool(pending[0])
    assert command[1] == pytest.approx(moving[1])
    assert phase[1] == pytest.approx(0.32)
    assert bool(pending[1])


def test_boundary_step_keeps_moving_reward_then_returns_standing_observation(
    bello_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(standing_transitions_enabled=True)
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        command_config=command_config,
    )
    moving = jnp.asarray([0.5, 0.0, 0.0], dtype=jnp.float32)
    state, _ = env.reset(command=moving)
    state = state.replace(
        gait_phase=jnp.asarray(0.49, dtype=jnp.float32),
        standing_transition_pending=jnp.asarray(True),
    )

    next_state, observation, _, done, info = env.step(
        state, jnp.zeros(env.action_size, dtype=jnp.float32)
    )

    assert not bool(done)
    assert info["command/moving"] == pytest.approx(1.0)
    assert info["gait/active"] == pytest.approx(1.0)
    for name in (
        "stand_still",
        "stand_foot_motion",
        "stand_foot_heading",
        "stand_contact",
        "stand_upright",
    ):
        assert info[f"reward/{name}"] == pytest.approx(0.0)
    assert command_is_standing(next_state.command, command_config)
    assert next_state.gait_phase == pytest.approx(0.0)
    assert not bool(next_state.standing_transition_pending)
    assert observation[observation_component_slice(env, "command")] == pytest.approx(
        jnp.zeros(3)
    )
    assert observation[observation_component_slice(env, "gait_phase")] == pytest.approx(
        jnp.zeros(3)
    )

    following_state, _, _, following_done, following_info = env.step(
        next_state, jnp.zeros(env.action_size, dtype=jnp.float32)
    )

    assert not bool(following_done)
    assert following_info["command/standing"] == pytest.approx(1.0)
    assert following_info["gait/active"] == pytest.approx(0.0)
    for name in ("gait_contact", "gait_stance_slip", "gait_swing_clearance"):
        assert following_info[f"reward/{name}"] == pytest.approx(0.0)
    assert command_is_standing(following_state.command, command_config)


def test_moving_resample_cancels_pending_stop_at_boundary(
    bello_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(
        standing_probability=0.0,
        resample_seconds=0.02,
        standing_transitions_enabled=True,
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=1,
        command_config=command_config,
    )
    moving = jnp.asarray([[0.5, 0.0, 0.0]], dtype=jnp.float32)
    state, _ = env.reset_batch(1, command=moving)
    state = state.replace(
        gait_phase=jnp.asarray([0.49], dtype=jnp.float32),
        standing_transition_pending=jnp.asarray([True]),
    )

    next_state, observation, _, done, info = env.step_batch(
        state,
        jnp.zeros((1, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(21),
    )

    assert not bool(done[0])
    assert info["command/moving"][0] == pytest.approx(1.0)
    assert command_is_moving(next_state.command[0], command_config)
    assert next_state.gait_phase[0] == pytest.approx(0.0)
    assert not bool(next_state.standing_transition_pending[0])
    assert observation[0, observation_component_slice(env, "command")] == (
        pytest.approx(next_state.command[0])
    )
    assert observation[0, observation_component_slice(env, "gait_phase")] == (
        pytest.approx(jnp.asarray([0.0, 1.0, 1.0]))
    )


def test_episode_reset_honors_a_pending_standing_request(
    bello_asset: PreparedAsset,
) -> None:
    command_config = CommandConfig(
        standing_probability=0.5,
        standing_transitions_enabled=True,
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=2,
        command_config=command_config,
    )
    moving = jnp.asarray([[0.5, 0.0, 0.0], [0.4, 0.1, 0.0]], dtype=jnp.float32)
    state, _ = env.reset_batch(2, command=moving)
    state = state.replace(
        gait_phase=jnp.full(2, 0.3, dtype=jnp.float32),
        standing_transition_pending=jnp.asarray([True, False]),
        step_count=jnp.full(2, env.max_episode_steps, dtype=jnp.int32),
    )

    next_state, _, _, done, _ = env.step_batch(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        rng=jax.random.PRNGKey(32),
    )

    assert jnp.all(done)
    assert command_is_standing(next_state.command[0], command_config)
    assert command_is_moving(next_state.command[1], command_config)
    assert not jnp.any(next_state.standing_transition_pending)


def test_random_push_overwrites_root_xy_velocity_on_interval(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=2,
        env_config=BelloEnvConfig(
            random_push=RandomPushConfig(
                enabled=True,
                interval_seconds=0.02,
                lin_vel_x=(0.5, 0.5),
                lin_vel_y=(-0.25, -0.25),
            )
        ),
    )
    state, _ = env.reset_batch(2)
    data, applied, velocity = env._maybe_apply_random_push_batch(
        state.data,
        jnp.ones(2, dtype=jnp.int32),
        jax.random.PRNGKey(0),
    )
    root_qvel = env.model.root_qvel_adr

    assert applied.tolist() == [True, True]
    assert velocity == pytest.approx(jnp.asarray([[0.5, -0.25]] * 2))
    assert data.qvel[:, root_qvel : root_qvel + 2] == pytest.approx(velocity)


def test_rough_terrain_foot_height_queries_local_elevation(
    bello_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="rough",
        size=(0.5, 0.5, 0.05),
        hfield_resolution=17,
        hfield_amplitude=0.03,
        hfield_frequency=8.0,
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(terrain=terrain, spawn=SpawnConfig(clearance=0.01)),
    )
    box_geom_ids = [
        next(
            int(geom_id)
            for geom_id in env.model.foot_geom_ids
            if env.model.geom_body_ids[geom_id] == foot_body_id
            and env.model.mj_model.geom_type[geom_id] == int(mujoco.mjtGeom.mjGEOM_BOX)
        )
        for foot_body_id in env.model.foot_body_ids
    ]
    env = replace(
        env,
        model=replace(
            env.model,
            foot_geom_ids=np.asarray(box_geom_ids, dtype=np.int32),
        ),
    )
    state, _ = env.reset()
    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    expected_height = []
    for geom_id in box_geom_ids:
        signs = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float32,
        )
        corners = signs * np.asarray(env.model.mj_model.geom_size[geom_id])
        rotation = np.asarray(state.data.geom_xmat[geom_id])
        world_corners = np.asarray(state.data.geom_xpos[geom_id]) + corners @ rotation.T
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            terrain_geom_id,
            world_corners[:, :2],
            geom_quat=env.model.terrain_geom_quat[0],
        )
        expected_height.append(np.min(world_corners[:, 2] - terrain_height))

    assert env._foot_height(state, 0) == pytest.approx(expected_height, abs=1.0e-5)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    assert mujoco_foot_height(env, mj_data) == pytest.approx(
        expected_height, abs=1.0e-5
    )


def test_foot_height_uses_physical_sole_instead_of_ankle_origin(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env.reset()

    sole_clearance = env._foot_height(state)
    ankle_height = state.data.xpos[jnp.asarray(env.model.foot_body_ids), 2]

    assert jnp.all(sole_clearance >= 0.0)
    assert jnp.all(ankle_height - sole_clearance > 0.05)


def test_sloped_plane_foot_height_uses_minimum_terrain_clearance(
    bello_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        slope_angle_range=(np.deg2rad(10.0), np.deg2rad(10.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(terrain=terrain),
    )
    box_geom_ids = []
    for foot_body_id in env.model.foot_body_ids:
        box_geom_ids.append(
            next(
                int(geom_id)
                for geom_id in env.model.foot_geom_ids
                if env.model.geom_body_ids[geom_id] == foot_body_id
                and env.model.mj_model.geom_type[geom_id]
                == int(mujoco.mjtGeom.mjGEOM_BOX)
            )
        )
    env = replace(
        env,
        model=replace(
            env.model,
            foot_geom_ids=np.asarray(box_geom_ids, dtype=np.int32),
        ),
    )
    state, _ = env.reset()
    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    expected = []
    for geom_id in box_geom_ids:
        signs = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
            dtype=np.float32,
        )
        corners = signs * np.asarray(env.model.mj_model.geom_size[geom_id])
        rotation = np.asarray(state.data.geom_xmat[geom_id])
        world_corners = np.asarray(state.data.geom_xpos[geom_id]) + corners @ rotation.T
        terrain_height = terrain_surface_height(
            env.model.mj_model,
            terrain_geom_id,
            world_corners[:, :2],
            geom_quat=env.model.terrain_geom_quat[0],
        )
        expected.append(np.min(world_corners[:, 2] - terrain_height))

    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = np.asarray(state.data.qpos)
    mj_data.qvel[:] = np.asarray(state.data.qvel)
    mujoco.mj_forward(env.model.mj_model, mj_data)

    assert env._foot_height(state) == pytest.approx(expected, abs=1.0e-5)
    assert mujoco_foot_height(env, mj_data) == pytest.approx(expected, abs=1.0e-5)


def test_base_height_and_critic_observation_are_terrain_relative(
    bello_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        slope_angle_range=(np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(terrain=terrain, root_link="pelvis_link"),
    )
    state, _ = env.reset()
    root_qpos = env.model.root_qpos_adr
    original_xy = state.data.qpos[root_qpos : root_qpos + 2]
    translated_xy = original_xy + jnp.asarray([0.5, -0.25])
    terrain_delta = env._terrain_height(translated_xy) - env._terrain_height(
        original_xy
    )
    translated_data = state.data.replace(
        qpos=(
            state.data.qpos.at[root_qpos : root_qpos + 2]
            .set(translated_xy)
            .at[root_qpos + 2]
            .add(terrain_delta)
        )
    )
    translated = state.replace(
        data=mjx.kinematics(env.model.mjx_model, translated_data)
    )
    base_height_slice = observation_component_slice(env, "base_height", critic=True)

    assert env._base_height(translated) == pytest.approx(
        env._base_height(state), abs=1.0e-5
    )
    assert env.critic_observation(translated)[base_height_slice] == pytest.approx(
        env.critic_observation(state)[base_height_slice], abs=1.0e-5
    )
    assert bool(env.done_flags(translated)["done/base_height"]) == bool(
        env.done_flags(state)["done/base_height"]
    )
    for sample in (state, translated):
        mj_data = mujoco.MjData(env.model.mj_model)
        mj_data.qpos[:] = np.asarray(sample.data.qpos)
        mj_data.qvel[:] = np.asarray(sample.data.qvel)
        mujoco.mj_forward(env.model.mj_model, mj_data)
        assert mujoco_eval_info(env, mj_data)["base_height"] == pytest.approx(
            env._base_height(sample), abs=1.0e-5
        )


def test_batched_slope_resets_to_identical_sole_clearance_without_rng(
    bello_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        pos=(1.5, -0.75, 0.2),
        seed=19,
        slope_angle_range=(-np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        batch_size=8,
        env_config=BelloEnvConfig(
            terrain=terrain,
            spawn=SpawnConfig(clearance=0.01),
            model_randomization=ModelRandomizationConfig(enabled=True),
        ),
    )

    state, _ = env.reset_batch(8, rng=None)
    minimum_clearance = jax.vmap(lambda item: jnp.min(env._foot_height(item)))(state)

    assert minimum_clearance == pytest.approx(jnp.full(8, 0.01), abs=2.0e-5)
    assert int(np.ravel(np.asarray(state.data._impl.nacon))[0]) == 0


def test_native_and_mjx_reset_share_slope_relative_sole_clearance(
    bello_asset: PreparedAsset,
) -> None:
    terrain = TerrainConfig(
        mode="plane",
        pos=(1.5, -0.75, 0.2),
        slope_angle_range=(np.deg2rad(2.0), np.deg2rad(2.0)),
    )
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        env_config=BelloEnvConfig(
            terrain=terrain,
            spawn=SpawnConfig(clearance=0.01),
        ),
    )
    state, _ = env.reset()
    mj_data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, mj_data)

    assert np.min(mujoco_foot_height(env, mj_data)) == pytest.approx(0.01, abs=1.0e-5)
    assert np.min(np.asarray(env._foot_height(state))) == pytest.approx(
        0.01, abs=1.0e-5
    )
    assert mj_data.qpos[env.model.root_qpos_adr + 2] == pytest.approx(
        np.asarray(state.data.qpos)[env.model.root_qpos_adr + 2], abs=1.0e-5
    )


def test_done_flags_share_height_orientation_timeout_and_nonfinite_contract(
    bello_asset: PreparedAsset,
) -> None:
    done_config = DoneConfig(
        base_height=BaseHeightDoneConfig(min_height=0.8, link="bello_root"),
        bad_orientation=BadOrientationDoneConfig(limit_angle=0.5, link="bello_root"),
    )
    env = BelloLocomotionEnv.load_prepared(bello_asset, done_config=done_config)
    state, _ = env.reset()
    root = env.model.root_qpos_adr
    low = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[root + 2].set(0.7))
    )
    nonfinite = state.replace(
        data=state.data.replace(qvel=state.data.qvel.at[0].set(jnp.nan))
    )

    assert bool(env.done_flags(low)["done/base_height"])
    assert bool(env.done_flags(nonfinite)["done/nonfinite"])
    assert not bool(env.done_flags(state)["done/any"])


def test_large_finite_physics_state_terminates_and_sanitizes_reward(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env.reset()
    exploded = state.replace(
        data=state.data.replace(qvel=state.data.qvel.at[0].set(1.0e5))
    )
    flags = env.done_flags(exploded)
    reward, terms = env.sanitize_nonfinite_reward(
        jnp.asarray(-1.0e20),
        {
            "reward/joint_pose": jnp.asarray(-1.0e20),
            "reward/termination": jnp.asarray(-1.0),
        },
        flags,
    )

    assert bool(flags["done/unstable"])
    assert not bool(flags["done/nonfinite"])
    assert terms["reward/joint_pose"] == pytest.approx(0.0)
    assert terms["reward/termination"] == pytest.approx(-1.0)
    assert reward == pytest.approx(-1.0)


def test_large_or_nonfinite_constraint_force_terminates_episode(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env.reset()
    large_force = state.replace(
        data=state.data.replace(
            qfrc_constraint=state.data.qfrc_constraint.at[0].set(2.0e6)
        )
    )
    nonfinite_force = state.replace(
        data=state.data.replace(
            qfrc_constraint=state.data.qfrc_constraint.at[0].set(jnp.nan)
        )
    )
    large_contact_force = state.replace(
        data=state.data.replace(
            _impl=state.data._impl.replace(
                efc__force=state.data._impl.efc__force.at[0].set(2.0e6)
            )
        )
    )
    nonfinite_contact_force = state.replace(
        data=state.data.replace(
            _impl=state.data._impl.replace(
                efc__force=state.data._impl.efc__force.at[0].set(jnp.nan)
            )
        )
    )

    assert bool(env.done_flags(large_force)["done/unstable"])
    assert bool(env.done_flags(nonfinite_force)["done/nonfinite"])
    assert bool(env.done_flags(large_contact_force)["done/unstable"])
    assert bool(env.done_flags(nonfinite_contact_force)["done/nonfinite"])


def test_stance_width_uses_lateral_frame_axis(bello_asset: PreparedAsset) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset)
    state, _ = env.reset()
    left, right = env.model.foot_body_ids
    xpos = (
        state.data.xpos.at[left]
        .set(jnp.asarray([0.2, 5.0, 0.0]))
        .at[right]
        .set(jnp.asarray([0.0, -5.0, 0.0]))
    )
    state = state.replace(data=state.data.replace(xpos=xpos))

    command_frame = env._reward_frame_link("track_lin_vel_xy")
    assert command_frame == env.model.robot.locomotion_command_frame_body_name
    assert env._stance_width(state, command_frame) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "env_config, message",
    [
        (BelloEnvConfig(action_clip=0.0), "action_clip"),
        (BelloEnvConfig(root_link=""), "root_link"),
        (
            BelloEnvConfig(contact=ContactConfig(foot_contact_on_seconds=-0.1)),
            "foot_contact_on_seconds",
        ),
        (
            BelloEnvConfig(observation=ObservationConfig(privileged=True)),
            "actor observation must remain deployable",
        ),
        (
            BelloEnvConfig(
                critic_observation=ObservationConfig(
                    privileged=True,
                    noise=ObservationNoiseConfig(enabled=True),
                )
            ),
            "critic_observation.noise.enabled must be false",
        ),
        (
            BelloEnvConfig(
                numerical_safety=replace(
                    BelloEnvConfig().numerical_safety, max_abs_velocity=0.0
                )
            ),
            "numerical_safety.max_abs_velocity",
        ),
        (
            BelloEnvConfig(
                numerical_safety=replace(
                    BelloEnvConfig().numerical_safety,
                    max_abs_constraint_force=0.0,
                )
            ),
            "numerical_safety.max_abs_constraint_force",
        ),
    ],
)
def test_environment_rejects_invalid_static_contracts(
    bello_asset: PreparedAsset, env_config: BelloEnvConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BelloLocomotionEnv.load_prepared(bello_asset, env_config=env_config)


def test_environment_rejects_missing_reward_link(
    bello_asset: PreparedAsset,
) -> None:
    reward = RewardConfig(links=RewardLinks(track_lin_vel_xy="missing_link"))
    with pytest.raises(ValueError, match="missing_link"):
        BelloLocomotionEnv.load_prepared(bello_asset, reward_config=reward)


def test_environment_rejects_missing_robot_command_frame(
    bello_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(bello_asset, load_mjx=False)
    model = replace(
        env.model,
        robot=replace(
            env.model.robot,
            locomotion_command_frame_body_name="missing_command_frame",
        ),
    )

    with pytest.raises(ValueError, match="missing_command_frame"):
        replace(env, model=model)


def test_observation_and_action_layouts_reject_ambiguous_contracts() -> None:
    with pytest.raises(ValueError, match="component names must be unique"):
        ObservationLayout((("joint", 1), ("joint", 2)))
    with pytest.raises(ValueError, match="sizes must be positive"):
        ObservationLayout((("joint", 0),))
    with pytest.raises(ValueError, match="names must be unique"):
        ActionLayout(("left", "left"))
