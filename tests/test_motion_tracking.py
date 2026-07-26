from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import (
    PreparedAsset,
    prepare_asset,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionAdaptiveSamplingConfig,
    MotionDoneConfig,
    MotionObservationNoiseConfig,
    MotionPushConfig,
    MotionRewardConfig,
    MotionResetConfig,
    MotionTrackingEnvConfig,
    quat_multiply,
    yaw_quat_from_angle,
)
from bello_mujoco.experiment_profiles import motion_training_profile
from bello_mujoco.motion import (
    DEFAULT_TRACKED_BODY_NAMES,
    FIXED_TRACKING_JOINT_AVAILABILITY,
    FIXED_TRACKING_JOINT_NAMES,
    GMR_BELLO_JOINT_NAMES,
    MOTION_FORMAT_VERSION,
    load_motion_clip,
    motion_tracking_asset_config,
    sample_motion,
)
from bello_mujoco.sim import load_prepared_bello_model
from bello_mujoco.tools.import_gmr_motion import (
    _build_qpos,
    _validate_legacy_motion,
    _validate_self_collisions,
)


@pytest.fixture(scope="module")
def fixed_waist_asset(tmp_path_factory) -> PreparedAsset:
    return prepare_asset(
        motion_tracking_asset_config(
            output_root=tmp_path_factory.mktemp("fixed-waist-asset"),
        )
    )


@pytest.fixture(scope="module")
def fixed_waist_motion(fixed_waist_asset: PreparedAsset, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fixed-waist-motion") / "test_motion.npz"
    model = load_prepared_bello_model(
        fixed_waist_asset,
        actuator_config=motion_tracking_asset_config().actuator_config,
        load_mjx=False,
    )
    mj_model = model.mj_model
    frame_count = 4
    qpos = np.repeat(model.default_qpos[None, :], frame_count, axis=0)
    qpos[:, 1] += np.linspace(0.0, 0.06, frame_count)
    qvel = np.zeros((frame_count, mj_model.nv), dtype=np.float64)
    qvel[:, 1] = 0.6
    body_ids = np.asarray(
        [mj_model.body(name).id for name in DEFAULT_TRACKED_BODY_NAMES]
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
        format_version=np.asarray(MOTION_FORMAT_VERSION),
        fps=np.asarray(10.0),
        joint_names=np.asarray(model.actuator_joint_names),
        body_names=np.asarray(DEFAULT_TRACKED_BODY_NAMES),
        source_sha256=np.asarray("synthetic"),
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
            joint_position_weight=0.0,
            joint_velocity_weight=0.0,
            body_position_weight=1.0,
            body_orientation_weight=1.0,
            body_velocity_weight=0.0,
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


def test_legacy_gmr_import_defaults_missing_foot_contacts() -> None:
    frame_count = 2
    arrays = _validate_legacy_motion(
        {
            "fps": 30.0,
            "root_pos": np.zeros((frame_count, 3)),
            "root_rot": np.tile((0.0, 0.0, 0.0, 1.0), (frame_count, 1)),
            "dof_pos": np.zeros((frame_count, len(GMR_BELLO_JOINT_NAMES))),
        }
    )

    np.testing.assert_array_equal(
        arrays["foot_contacts"], np.zeros((frame_count, 2), dtype=np.float32)
    )


def test_fixed_head_gmr_import_accepts_25_joint_schema() -> None:
    frame_count = 2
    fixed_head_joint_count = sum(
        name not in FIXED_TRACKING_JOINT_NAMES for name in GMR_BELLO_JOINT_NAMES
    )

    arrays = _validate_legacy_motion(
        {
            "fps": 30.0,
            "root_pos": np.zeros((frame_count, 3)),
            "root_rot": np.tile((0.0, 0.0, 0.0, 1.0), (frame_count, 1)),
            "dof_pos": np.zeros((frame_count, fixed_head_joint_count)),
        }
    )

    assert arrays["dof_pos"].shape == (frame_count, 25)


def test_default_motion_training_config_has_fixed_waist_and_head_contract() -> None:
    config = motion_training_profile()

    assert config.asset_config.joint_availability == FIXED_TRACKING_JOINT_AVAILABILITY
    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.asset_config.collision_mode == "box"
    assert len(config.asset_config.actuator_config.policy_joint_names) == 25
    assert config.motion.directory == Path("motions/local/lafan1_fixed_head")
    assert config.motion.sequence == "walk1_subject1"
    assert config.checkpoint_config.name == "walk1_subject1_bello_motion_seed0"
    assert config.video_config.enabled
    assert config.video_config.render_final
    assert config.video_config.log_wandb
    assert config.video_config.wandb_name == "eval/motion_policy"
    assert config.config.num_envs == 4096
    assert config.config.learning_rate == pytest.approx(1.0e-3)
    assert config.config.policy.init_log_std == pytest.approx(0.0)
    assert config.config.policy.action_distribution == "tanh_gaussian"
    assert config.env_config.formulation == "beyondmimic_joint_limits"
    assert config.warp.njmax == 2048


def test_gmr_joint_columns_are_remapped_by_name(
    fixed_waist_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        fixed_waist_asset,
        actuator_config=motion_tracking_asset_config().actuator_config,
        load_mjx=False,
    )
    source_joint_pos = np.arange(len(GMR_BELLO_JOINT_NAMES), dtype=np.float64)[None, :]
    qpos = _build_qpos(
        model,
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        source_joint_pos,
    )

    source_index = {name: index for index, name in enumerate(GMR_BELLO_JOINT_NAMES)}
    for name, address in zip(
        model.actuator_joint_names, model.actuator_qpos_indices, strict=True
    ):
        assert qpos[0, address] == source_index[name]


def test_gmr_import_rejects_training_box_intersections(
    fixed_waist_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(
        fixed_waist_asset,
        actuator_config=motion_tracking_asset_config().actuator_config,
        load_mjx=False,
    )
    qpos = np.repeat(model.default_qpos[None, :], 2, axis=0)
    for name in ("left_hip_roll_joint", "right_hip_roll_joint"):
        joint_id = model.mj_model.joint(name).id
        qpos[:, model.mj_model.jnt_qposadr[joint_id]] = 0.2

    with pytest.raises(ValueError, match="training collision model"):
        _validate_self_collisions(model, qpos)


def test_motion_loader_and_shortest_arc_interpolation(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    model = load_prepared_bello_model(
        fixed_waist_asset,
        actuator_config=motion_tracking_asset_config().actuator_config,
        load_mjx=False,
    )
    clip = load_motion_clip(fixed_waist_motion, model)
    clip = clip.__class__(
        **{**clip.__dict__, "qpos": clip.qpos.at[1, 3:7].multiply(-1)}
    )

    sample = jax.jit(lambda: sample_motion(clip, jnp.asarray(0.05)))()

    assert sample.qpos.shape == (model.mj_model.nq,)
    assert float(jnp.linalg.norm(sample.qpos[3:7])) == pytest.approx(1.0)
    assert np.allclose(sample.qpos[3:7], np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_native_motion_rollout_does_not_load_mjx(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    from bello_mujoco.tools.eval_motion_policy import (
        native_motion_step,
        reset_native_motion_rollout,
    )

    config = MotionTrackingEnvConfig(motion_path=fixed_waist_motion)
    env = BelloMotionTrackingEnv.load_prepared_native(fixed_waist_asset, config)
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
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, observation = env.reset(reference_time=jnp.asarray(0.1))
    reward, _ = env._reward(state, state)
    done = env._done_flags(state)

    assert (env.model.mj_model.nq, env.model.mj_model.nv, env.model.mj_model.nu) == (
        32,
        31,
        25,
    )
    assert env.action_size == 25
    assert observation.shape == (env.observation_size,)
    assert env.critic_observation(state).shape == (env.critic_observation_size,)
    assert float(reward) > 0.8
    assert not bool(done["done/any"])


def test_zero_action_targets_next_reference_pose(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    ctrl = env._action_to_ctrl(state, jnp.zeros(env.action_size))
    reference = sample_motion(env.motion, jnp.asarray(0.1 + env.control_timestep))
    logical_target = np.asarray(reference.qpos)[env.model.actuator_qpos_indices]
    logical_target = np.clip(
        logical_target,
        env.model.joint_target_low,
        env.model.joint_target_high,
    )

    np.testing.assert_allclose(
        np.asarray(ctrl),
        expected_physical_ctrl(env, logical_target),
        atol=1.0e-6,
    )
    index = {name: i for i, name in enumerate(env.model.actuator_joint_names)}
    pitch = index["left_ankle_pitch_joint"]
    roll = index["left_ankle_roll_joint"]
    assert float(ctrl[pitch]) == pytest.approx(
        -float(logical_target[pitch]) + float(logical_target[roll]), abs=1.0e-6
    )
    assert float(ctrl[roll]) == pytest.approx(
        float(logical_target[pitch]) + float(logical_target[roll]), abs=1.0e-6
    )
    stacked_observation = jnp.tile(env.observation(state), 4)
    diagnostic_ctrl = env._diagnostic_action_to_ctrl(
        stacked_observation, jnp.zeros(env.action_size)
    )
    np.testing.assert_allclose(
        np.asarray(diagnostic_ctrl), np.asarray(ctrl), atol=1.0e-6
    )


def test_reference_residual_uses_quarter_logical_range_and_clips_targets(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, observation = env.reset(reference_time=jnp.asarray(0.1))
    expected_scale = 0.25 * (env.model.joint_target_high - env.model.joint_target_low)

    np.testing.assert_allclose(
        np.asarray(env.reference_residual_action_scale),
        expected_scale,
        atol=1.0e-7,
    )

    diagnostic_observation = np.asarray(observation).copy()
    diagnostic_observation[: env.action_size] = (
        env.model.joint_target_high - env.model.default_joint_target
    )
    diagnostic_ctrl = env._diagnostic_action_to_ctrl(
        jnp.tile(jnp.asarray(diagnostic_observation), 4),
        jnp.ones(env.action_size),
    )
    np.testing.assert_allclose(
        np.asarray(diagnostic_ctrl),
        expected_physical_ctrl(env, env.model.joint_target_high),
        atol=1.0e-6,
    )


def test_beyondmimic_zero_action_targets_nominal_pose(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset,
        beyondmimic_config(fixed_waist_motion),
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    ctrl = env._action_to_ctrl(state, jnp.zeros(env.action_size))

    np.testing.assert_allclose(
        np.asarray(ctrl),
        np.asarray(env.model.default_ctrl),
        atol=1.0e-6,
    )
    assert env.observation_size == 5 * env.action_size + 15
    assert env.critic_observation_size == env.observation_size + 9 * len(
        env.motion.body_names
    )


def test_beyondmimic_uses_effort_stiffness_scale_without_logical_clipping(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset,
        beyondmimic_config(fixed_waist_motion),
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    configured = env.actuator_config.by_name
    expected_scale = np.asarray(
        [
            0.25 * configured[name].effort_limit / configured[name].gains[0]
            for name in env.action_layout.names
        ]
    )
    np.testing.assert_allclose(
        np.asarray(env.beyondmimic_action_scale), expected_scale, atol=1.0e-7
    )

    roll = env.action_layout.names.index("left_ankle_roll_joint")
    action = np.zeros(env.action_size, dtype=np.float32)
    action[roll] = 1.0
    logical_target = env.model.default_joint_target + expected_scale * action
    assert logical_target[roll] > env.model.joint_target_high[roll]

    ctrl = env._action_to_ctrl(state, jnp.asarray(action))
    expected_ctrl = expected_physical_ctrl(env, logical_target)
    clipped_ctrl = expected_physical_ctrl(
        env,
        np.clip(
            logical_target,
            env.model.joint_target_low,
            env.model.joint_target_high,
        ),
    )
    np.testing.assert_allclose(np.asarray(ctrl), expected_ctrl, atol=1.0e-6)
    assert not np.allclose(expected_ctrl, clipped_ctrl)


def test_beyondmimic_joint_limit_actions_use_asymmetric_bounded_mapping(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    config = replace(
        beyondmimic_config(fixed_waist_motion),
        formulation="beyondmimic_joint_limits",
    )
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config)
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    policy = env.model.policy_ctrl_indices
    offset = np.linspace(-0.005, 0.005, env.action_size, dtype=np.float32)
    center = np.clip(
        env.model.default_joint_target[policy] + offset,
        env.model.joint_target_low[policy],
        env.model.joint_target_high[policy],
    )

    low = env.beyondmimic_action_to_joint_target(
        -jnp.ones(env.action_size), jnp.asarray(offset)
    )
    zero = env.beyondmimic_action_to_joint_target(
        jnp.zeros(env.action_size), jnp.asarray(offset)
    )
    high = env.beyondmimic_action_to_joint_target(
        jnp.ones(env.action_size), jnp.asarray(offset)
    )

    np.testing.assert_allclose(
        np.asarray(low)[policy], env.model.joint_target_low[policy], atol=1.0e-6
    )
    np.testing.assert_allclose(np.asarray(zero)[policy], center, atol=1.0e-6)
    np.testing.assert_allclose(
        np.asarray(high)[policy], env.model.joint_target_high[policy], atol=1.0e-6
    )

    state = state.replace(joint_default_offset=jnp.asarray(offset))
    ctrl = env._action_to_ctrl(state, jnp.ones(env.action_size))
    np.testing.assert_allclose(
        np.asarray(ctrl),
        expected_physical_ctrl(env, env.model.joint_target_high),
        atol=1.0e-6,
    )


def test_motion_layouts_and_ppo_protocol_are_explicit(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    reference_env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    beyondmimic_env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, beyondmimic_config(fixed_waist_motion)
    )

    assert reference_env.task == "motion_tracking"
    assert not reference_env.ppo_stagger_initial_step_count
    assert (
        reference_env.ppo_action_diagnostics(
            jnp.zeros((1, 1, reference_env.action_size))
        )
        == {}
    )
    assert reference_env.action_layout.names == reference_env.motion.joint_names
    assert reference_env.observation_layout.components == (
        ("reference_joint_position", 25),
        ("reference_joint_velocity", 25),
        ("joint_position", 25),
        ("joint_velocity", 25),
        ("last_action", 25),
        ("root_position_error", 3),
        ("root_orientation_error", 3),
        ("root_linear_velocity_error", 3),
        ("root_angular_velocity_error", 3),
        ("reference_phase", 2),
    )
    assert reference_env.critic_observation_layout.components[-2:] == (
        ("tracked_body_position_error", 30),
        ("tracked_body_orientation_error", 30),
    )
    assert beyondmimic_env.observation_layout.components[2:6] == (
        ("anchor_position_error", 3),
        ("anchor_orientation_error_6d", 6),
        ("base_linear_velocity", 3),
        ("base_angular_velocity", 3),
    )
    assert beyondmimic_env.critic_observation_layout.components[-2:] == (
        ("tracked_body_position_relative_to_anchor", 30),
        ("tracked_body_orientation_relative_to_anchor_6d", 60),
    )


def test_beyondmimic_body_targets_follow_anchor_xy_and_yaw(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, beyondmimic_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    qpos = state.data.qpos.at[:2].add(jnp.asarray([0.2, -0.1]))
    qpos = qpos.at[3:7].set(
        quat_multiply(yaw_quat_from_angle(jnp.asarray(0.4)), qpos[3:7])
    )
    state = state.replace(
        data=mjx.forward(env.model.mjx_model, state.data.replace(qpos=qpos))
    )
    reference = env._aligned_reference(
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
    assert not bool(env._done_flags(state)["done/orientation"])


def test_adaptive_sampling_spreads_failure_weight_backward(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    config = beyondmimic_config(fixed_waist_motion)
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
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config)
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
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    config = beyondmimic_config(fixed_waist_motion)
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
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config)
    failure = jnp.zeros(env.sampling_bin_count).at[-1].set(1.0)
    keys = jax.random.split(jax.random.PRNGKey(7), 256)
    sampled = jax.vmap(lambda key: env._sample_reference_time(key, failure))(keys)
    max_time = env.motion.duration - env.control_timestep
    final_bin_start = max_time * (env.sampling_bin_count - 1) / env.sampling_bin_count

    assert np.all(np.asarray(sampled) >= final_bin_start)
    assert np.all(np.asarray(sampled) < max_time)
    assert np.unique(np.asarray(sampled)).size > 200


def test_adaptive_sampling_ema_uses_terminal_counts_per_phase_bin(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    from types import SimpleNamespace

    config = beyondmimic_config(fixed_waist_motion)
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
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config)
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
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    config = beyondmimic_config(fixed_waist_motion)
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
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config)
    state, clean = env.reset(reference_time=jnp.asarray(0.1))
    noisy = env.observation(state, jax.random.PRNGKey(0))

    assert not np.allclose(np.asarray(clean), np.asarray(noisy))
    np.testing.assert_allclose(
        np.asarray(env.critic_observation(state)[: env.observation_size]),
        np.asarray(clean),
    )


def test_tracking_nonfinite_state_is_terminal_and_reward_is_sanitized(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    state = state.replace(
        data=state.data.replace(cvel=state.data.cvel.at[0, 0].set(jnp.nan))
    )
    done_flags = env._done_flags(state)
    reward, reward_info = env._reward(state, state)
    reward, reward_info = env._sanitize_nonfinite_reward(
        reward, reward_info, done_flags
    )

    assert bool(done_flags["done/nonfinite"])
    assert bool(done_flags["done/terminal"])
    assert float(reward) == 0.0
    assert all(float(value) == 0.0 for value in reward_info.values())


def test_tracking_joint_limit_cost_is_bounded(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    indices = jnp.asarray(env.model.policy_qpos_indices)
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[indices].set(1.0e6))
    )

    assert float(env._joint_limit_cost(state)) == 1.0


@pytest.mark.integration
def test_tracking_batch_reset_and_step_are_jittable(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion), batch_size=2
    )
    state, observation = jax.jit(lambda key: env.reset_batch(2, key))(
        jax.random.PRNGKey(0)
    )
    next_state, next_observation, reward, done, info = jax.jit(env.step_batch)(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        jax.random.PRNGKey(1),
    )

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == (2, env.observation_size)
    assert next_state.last_action.shape == (2, env.action_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["done/terminal"].shape == (2,)


@pytest.mark.integration
def test_beyondmimic_batch_step_with_sampling_noise_and_push_is_jittable(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    config = beyondmimic_config(fixed_waist_motion)
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
    env = BelloMotionTrackingEnv.load_prepared(fixed_waist_asset, config, batch_size=2)
    state, observation = jax.jit(lambda key: env.reset_batch(2, key))(
        jax.random.PRNGKey(0)
    )
    next_state, next_observation, reward, done, info = jax.jit(env.step_batch)(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        jax.random.PRNGKey(1),
    )

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == observation.shape
    assert next_state.sampling_failure_ema.shape == (2, env.sampling_bin_count)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["reward/relative_body_position"].shape == (2,)
    assert info["sampling/entropy"].shape == (2,)
    assert info["push/applied"].shape == (2,)
