from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
)
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.configuration import (
    CONFIG_ROOT,
    motion_training_profile,
    structured_config,
)
from bello_mujoco.hardware.runtime import HardwareFrame, require_nominal_pose
from bello_mujoco.hardware.motion import (
    HardwareMotionConfig,
    HardwareMotionPolicyAdapter,
    PoseFrame,
    validate_hardware_motion_checkpoint,
    validate_hardware_motion_config,
    wrap_angle,
    xyzw_to_wxyz,
)
from bello_mujoco.policy import init_actor_critic, policy_observation_size
from bello_mujoco.cli import HardwareCommandConfig
from bello_mujoco.training.motion import (
    MotionTrainingRunConfig,
    motion_checkpoint_metadata,
    resolve_motion_selection,
)
from bello_mujoco.training.ppo import (
    critic_observation_size,
    init_observation_norm_state,
    init_value_norm_state,
)


def load_level4_recipe() -> MotionTrainingRunConfig:
    return resolve_motion_selection(
        motion_training_profile(
            Path(
                "motions/asap_fixed_head/"
                "video_single_foot_balance_level4_filter_amass.npz"
            )
        )
    )


def load_runner_config() -> HardwareCommandConfig:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hardware",
            overrides=[
                "task=motion",
                "checkpoint_path=checkpoints/motion",
                "standing_checkpoint_path=checkpoints/standing",
            ],
        )
    return structured_config(HardwareCommandConfig, config)


@pytest.fixture(scope="module")
def level4_adapter() -> HardwareMotionPolicyAdapter:
    run_config = load_level4_recipe()
    run_config = replace(
        run_config,
        env_config=replace(
            run_config.env_config,
            formulation="beyondmimic_joint_limits",
        ),
        config=replace(
            run_config.config,
            policy=replace(
                run_config.config.policy,
                action_distribution="tanh_gaussian",
            ),
        ),
    )
    env = BelloMotionTrackingEnv.load_native(
        run_config.asset_config, run_config.env_config
    )
    actor_size = policy_observation_size(env.observation_size, run_config.config.policy)
    critic_size = policy_observation_size(
        critic_observation_size(env), run_config.config.policy
    )
    metadata = motion_checkpoint_metadata(run_config, env, actor_size, critic_size)
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        actor_size,
        env.action_size,
        run_config.config.policy,
        critic_obs_size=critic_size,
    )
    normalization = CheckpointNormalizationState(
        value=init_value_norm_state(),
        observation=init_observation_norm_state(actor_size),
        critic_observation=init_observation_norm_state(critic_size),
    )
    return HardwareMotionPolicyAdapter(
        env,
        policy,
        metadata,
        normalization,
        HardwareMotionConfig(),
    )


def reference_frame(
    adapter: HardwareMotionPolicyAdapter,
) -> tuple[HardwareFrame, PoseFrame]:
    reference = adapter.env.aligned_reference(
        jax.numpy.asarray(adapter.start_time),
        jax.numpy.zeros(2),
        jax.numpy.zeros(()),
    )
    orientation_wxyz = np.asarray(reference.qpos[3:7])
    orientation_xyzw = orientation_wxyz[[1, 2, 3, 0]]
    return (
        HardwareFrame(
            positions=np.asarray(adapter.env._reference_ctrl(reference)),
            velocities=np.zeros(adapter.env.action_size),
            orientation_xyzw=orientation_xyzw,
            angular_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            max_motor_age_ms=0,
            imu_age_ms=0,
            max_group_age_ms=0,
            sample_span_ms=0.0,
            state_frame_round_trip_ms=0.0,
            group_states=("IDLE", "IDLE"),
            group_causes=("UNSPECIFIED", "UNSPECIFIED"),
            group_details=("", ""),
        ),
        PoseFrame(
            position=np.asarray(reference.qpos[:3]),
            orientation_xyzw=orientation_xyzw,
            linear_velocity=np.asarray(reference.qvel[:3]),
            angular_velocity=np.asarray(reference.qvel[3:6]),
            sequence=1,
            source_age_ms=0,
            received_at=0.0,
        ),
    )


def test_level4_adapter_uses_motion_schema_and_physical_targets(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    frame, pose = reference_frame(adapter)
    adapter.initialize_alignment(pose)

    result = adapter.step(frame, pose, reference_time=adapter.start_time)

    assert adapter.metadata.task == "motion_tracking"
    assert adapter.metadata.motion is not None
    assert adapter.metadata.ppo.policy.action_distribution == "tanh_gaussian"
    assert adapter.env.config.formulation == "beyondmimic_joint_limits"
    assert adapter.env.model.mj_model.nu == adapter.env.action_size == 25
    assert len(adapter.physical_coordinate_names) == 25
    assert result.observation.shape == (adapter.env.observation_size,)
    assert result.action.shape == (25,)
    assert result.physical_target.shape == (25,)
    assert np.all(np.isfinite(result.physical_target))
    assert result.anchor_position_error == pytest.approx(0.0, abs=1.0e-6)
    assert result.anchor_orientation_error == pytest.approx(0.0, abs=1.0e-6)


def test_motion_runner_nominal_target_uses_model_default_pose(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    expected = adapter.transform @ np.asarray(adapter.env.model.default_joint_target)
    limits = np.asarray(adapter.env.model.mj_model.actuator_ctrlrange)

    assert adapter.nominal_physical_target == pytest.approx(
        np.clip(expected, limits[:, 0], limits[:, 1])
    )


def test_motion_runner_checks_nominal_pose_after_ramp(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    frame, _ = reference_frame(adapter)
    frame = replace(frame, positions=adapter.nominal_physical_target.copy())
    require_nominal_pose(frame, adapter)

    index = adapter.physical_coordinate_names.index("left_shoulder_yaw_joint")
    positions = frame.positions.copy()
    positions[index] += 0.26
    with pytest.raises(RuntimeError, match="left_shoulder_yaw_joint"):
        require_nominal_pose(replace(frame, positions=positions), adapter)

    ankle_index = adapter.physical_coordinate_names.index("right_ankle_motor_2")
    positions = frame.positions.copy()
    positions[ankle_index] += 0.99
    require_nominal_pose(replace(frame, positions=positions), adapter)

    positions[ankle_index] += 0.02
    with pytest.raises(RuntimeError, match="right_ankle_motor_2"):
        require_nominal_pose(replace(frame, positions=positions), adapter)

    positions[index] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        require_nominal_pose(replace(frame, positions=positions), adapter)


def test_motion_hardware_uses_estimator_body_local_gyro_at_nonzero_yaw(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    frame, pose = reference_frame(adapter)
    half_sqrt_two = np.sqrt(0.5)
    orientation_xyzw = np.asarray((0.0, 0.0, half_sqrt_two, half_sqrt_two))
    state_frame_angular_velocity = np.asarray((4.0, -2.0, 1.0))
    estimator_angular_velocity = np.asarray((0.4, -0.2, 0.1))
    frame = replace(
        frame,
        orientation_xyzw=orientation_xyzw,
        angular_velocity=state_frame_angular_velocity,
    )
    pose = replace(
        pose,
        orientation_xyzw=orientation_xyzw,
        angular_velocity=estimator_angular_velocity,
    )
    adapter.initialize_alignment(pose)

    hardware_state = adapter._state(frame, pose, adapter.start_time)
    qvel = hardware_state[adapter._inference_layout.qvel]
    root_qvel = adapter.env.model.root_qvel_adr

    np.testing.assert_allclose(
        qvel[root_qvel + 3 : root_qvel + 6],
        estimator_angular_velocity,
        atol=1.0e-7,
    )


def test_motion_runner_defaults_to_read_only() -> None:
    config = load_runner_config()

    assert config.checkpoint_path == Path("checkpoints/motion")
    assert config.standing_checkpoint_path == Path("checkpoints/standing")
    assert config.task == "motion"
    assert not config.arm
    assert config.motion_path is None
    assert config.max_linear_acceleration is None
    assert HardwareMotionConfig().policy_transition_seconds == 0.2
    assert HardwareMotionConfig().max_motion_tilt_radians == pytest.approx(
        math.radians(70.0)
    )
    validate_hardware_motion_config(HardwareMotionConfig())
    validate_hardware_motion_config(HardwareMotionConfig(arm=True))


def test_motion_checkpoint_validation_rejects_task_and_formulation_mismatch(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    with pytest.raises(ValueError, match="motion-tracking checkpoint"):
        validate_hardware_motion_checkpoint(
            adapter.env, replace(adapter.metadata, task="locomotion")
        )
    with pytest.raises(ValueError, match="bounded BeyondMimic"):
        validate_hardware_motion_checkpoint(
            adapter.env,
            replace(
                adapter.metadata,
                motion_env=replace(
                    adapter.metadata.motion_env,
                    formulation="beyondmimic",
                ),
            ),
        )
    with pytest.raises(ValueError, match="tanh_gaussian"):
        validate_hardware_motion_checkpoint(
            adapter.env,
            replace(
                adapter.metadata,
                ppo=replace(
                    adapter.metadata.ppo,
                    policy=replace(
                        adapter.metadata.ppo.policy,
                        action_distribution="gaussian",
                    ),
                ),
            ),
        )


def test_motion_hardware_quaternion_helpers() -> None:
    half = np.sqrt(0.5)
    xyzw = np.asarray((0.0, 0.0, half, half))

    assert xyzw_to_wxyz(xyzw) == pytest.approx((half, 0.0, 0.0, half))
    assert wrap_angle(3.0 * np.pi) == pytest.approx(-np.pi)
