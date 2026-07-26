from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from bello_mujoco.checkpoint import (
    CheckpointNormalizationState,
)
from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv
from bello_mujoco.hardware_locomotion import HardwareFrame
from bello_mujoco.hardware_motion import (
    HardwareMotionConfig,
    HardwareMotionPolicyAdapter,
    require_nominal_pose,
    rotate_vector_xyzw,
    validate_hardware_motion_checkpoint,
    validate_hardware_motion_config,
    wrap_angle,
    yaw_from_wxyz,
    yaw_from_xyzw,
)
from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.policy import init_actor_critic, policy_observation_size
from bello_mujoco.tools.run_motion_hardware import HardwareMotionToolConfig
from bello_mujoco.tools.train_motion_ppo import (
    MotionTrainingRunConfig,
    motion_checkpoint_metadata,
    resolve_motion_selection,
)
from bello_mujoco.train.ppo import (
    critic_observation_size,
    init_observation_norm_state,
    init_value_norm_state,
)


def load_level4_recipe() -> MotionTrainingRunConfig:
    clear_hydra()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_ROOT.parent / "experiments" / "train_motion"),
        ):
            config = compose(
                config_name="asap_single_foot_balance_level4",
                overrides=[f"hydra.searchpath=[file://{CONFIG_ROOT / 'train_motion'}]"],
            )
        return resolve_motion_selection(
            structured_config(MotionTrainingRunConfig, config)
        )
    finally:
        clear_hydra()


def load_runner_config() -> HardwareMotionToolConfig:
    clear_hydra()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_ROOT / "run_motion_hardware"),
        ):
            config = compose(
                config_name="default",
                overrides=["checkpoint_path=checkpoints/motion"],
            )
        return structured_config(HardwareMotionToolConfig, config)
    finally:
        clear_hydra()


@pytest.fixture(scope="module")
def level4_adapter() -> HardwareMotionPolicyAdapter:
    run_config = load_level4_recipe()
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
    return HardwareMotionPolicyAdapter(env, policy, metadata, normalization)


def reference_frame(adapter: HardwareMotionPolicyAdapter) -> HardwareFrame:
    reference = adapter.env._aligned_reference(
        jax.numpy.asarray(adapter.start_time_seconds),
        jax.numpy.zeros(2),
        jax.numpy.zeros(()),
    )
    orientation_wxyz = np.asarray(reference.qpos[3:7])
    return HardwareFrame(
        positions=np.asarray(adapter.env._reference_ctrl(reference)),
        velocities=np.zeros(adapter.env.action_size),
        orientation_xyzw=orientation_wxyz[[1, 2, 3, 0]],
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        group_states=("IDLE", "IDLE"),
    )


def test_level4_adapter_uses_new_motion_schema_and_physical_targets(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    frame = reference_frame(adapter)
    adapter.initialize(frame)

    result = adapter.step(frame)

    assert adapter.metadata.task == "motion_tracking"
    assert adapter.metadata.motion is not None
    assert adapter.metadata.ppo.policy.action_distribution == "gaussian"
    assert adapter.env.config.formulation == "beyondmimic"
    assert adapter.env.model.mj_model.nu == adapter.env.action_size == 25
    assert len(adapter.physical_coordinate_names) == 25
    assert result.observation.shape == (adapter.env.observation_size,)
    assert result.action.shape == (25,)
    assert result.physical_target.shape == (25,)
    assert np.all(np.isfinite(result.physical_target))
    assert result.estimated_linear_velocity == pytest.approx(0.0)
    assert result.estimated_position_error == pytest.approx(0.0, abs=1.0e-6)


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
    frame = reference_frame(adapter)
    frame = replace(frame, positions=adapter.nominal_physical_target.copy())
    require_nominal_pose(frame, adapter, 0.1)

    index = adapter.physical_coordinate_names.index("left_shoulder_yaw_joint")
    positions = frame.positions.copy()
    positions[index] += 0.8
    with pytest.raises(RuntimeError, match="left_shoulder_yaw_joint"):
        require_nominal_pose(replace(frame, positions=positions), adapter, 0.75)


def test_motion_runner_defaults_to_read_only_and_requires_confirmation() -> None:
    config = load_runner_config()

    assert config.checkpoint_path == Path("checkpoints/motion")
    assert not config.hardware.arm
    assert config.hardware.motion_path is None
    validate_hardware_motion_config(config.hardware)
    with pytest.raises(ValueError, match="operator_confirmation"):
        validate_hardware_motion_config(
            HardwareMotionConfig(arm=True, operator_confirmation="yes")
        )


def test_motion_checkpoint_validation_rejects_task_and_hand_mismatch(
    level4_adapter: HardwareMotionPolicyAdapter,
) -> None:
    adapter = level4_adapter
    with pytest.raises(ValueError, match="motion-tracking checkpoint"):
        validate_hardware_motion_checkpoint(
            adapter.env, replace(adapter.metadata, task="locomotion")
        )
    with pytest.raises(ValueError, match="sphere hands"):
        validate_hardware_motion_checkpoint(
            adapter.env,
            replace(
                adapter.metadata,
                asset=replace(adapter.metadata.asset, end_effector_sphere_mode="none"),
            ),
        )


def test_motion_hardware_quaternion_and_inertial_helpers() -> None:
    half = np.sqrt(0.5)
    xyzw = np.asarray((0.0, 0.0, half, half))

    assert yaw_from_xyzw(xyzw) == pytest.approx(np.pi / 2.0)
    assert yaw_from_wxyz(xyzw[[3, 0, 1, 2]]) == pytest.approx(np.pi / 2.0)
    assert wrap_angle(3.0 * np.pi) == pytest.approx(np.pi)
    assert rotate_vector_xyzw(xyzw, np.asarray((1.0, 0.0, 0.0))) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1.0e-7
    )
