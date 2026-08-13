from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
import hashlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.contracts import ActionLayout
from bello_mujoco.envs.twist import (
    TwistSamplingConfig,
    TwistSimulationConfig,
)
from bello_mujoco.envs.twist2 import (
    Twist2EnvConfig,
    Twist2ObservationContract,
    Twist2ParityContract,
    default_twist2_asset_config,
)
from bello_mujoco.hardware.runtime import HardwareFrame
from bello_mujoco.hardware.twist2 import (
    HardwareTwist2Config,
    HardwareTwist2Environment,
    HardwareTwist2PolicyAdapter,
    ReferenceDiscontinuity,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
    bind_twist2_hardware_coordinate_limits,
    control_physical_twist2,
    _wait_for_live_twist2_reference,
    validate_hardware_twist2_config,
    validate_twist2_reference,
)
from bello_mujoco.hardware.smplx_retarget import (
    GmrBelloRetargeter,
    GmrLiveRetargetConfig,
    LiveGmrReferenceSource,
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
    import_pinned_gmr_retargeter,
    load_smplx_standing_sample,
    validate_smplx_target_frame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.motion.twist2_dataset import Twist2DatasetAudit
from bello_mujoco.policy import ActorCriticConfig, init_actor_critic
from bello_mujoco.sim import WarpConfig, load_bello_model
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    Twist2CheckpointMetadata,
    TwistDatasetContract,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    init_observation_norm_state,
    init_value_norm_state,
)


FINGERPRINT = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def twist2_hardware_adapter() -> HardwareTwist2PolicyAdapter:
    asset = default_twist2_asset_config()
    simulation = TwistSimulationConfig()
    model = load_bello_model(
        asset,
        terrain=simulation.terrain,
        load_mjx=False,
        simulation_timestep=1.0 / 500.0,
    )
    joint_names = tuple(
        model.actuator_joint_names[index] for index in model.policy_ctrl_indices
    )
    contract = Twist2ObservationContract(joint_names)
    policy_config = ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        action_distribution="gaussian",
        architecture="twist2",
        layer_norm_final_hidden=True,
    )
    ppo = PPOConfig(
        total_timesteps=4096 * 24,
        num_envs=4096,
        num_steps=24,
        normalize_observations=True,
        policy=policy_config,
    )
    metadata = Twist2CheckpointMetadata(
        ppo=ppo,
        asset=asset,
        warp=WarpConfig(),
        obs_size=contract.actor_observation_size,
        critic_obs_size=contract.critic_observation_size,
        action_size=25,
        actor_observation_layout=contract.actor_layout,
        critic_observation_layout=contract.critic_layout,
        action_layout=ActionLayout(joint_names),
        simulation=simulation,
        twist2_env=Twist2EnvConfig(),
        sampling=TwistSamplingConfig(
            base_distribution="manifest_weight",
            curriculum_max_difficulty=10.0,
        ),
        observation_contract=contract,
        parity_contract=Twist2ParityContract(),
        dataset=TwistDatasetContract(
            index_path=Path("dataset/index.json"),
            schema_version=3,
            dataset_fingerprint=FINGERPRINT,
            robot_fingerprint=FINGERPRINT,
            gmr_revision="b" * 40,
            joint_names=joint_names,
            body_names=contract.key_body_names,
        ),
        dataset_audit=Twist2DatasetAudit(
            clip_count=1,
            train_clip_count=1,
            evaluation_clip_count=0,
            source_counts=(("amass", 1),),
            train_source_counts=(("amass", 1),),
            unit_manifest_weights=True,
            train_unit_manifest_weights=True,
            matches_released_training_composition=False,
        ),
        robot_fingerprint=FINGERPRINT,
        source_revision="c" * 40,
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        contract.actor_observation_size,
        25,
        policy_config,
        critic_obs_size=contract.critic_observation_size,
    )
    normalization = CheckpointNormalizationState(
        value=init_value_norm_state(),
        observation=init_observation_norm_state(contract.actor_observation_size),
        critic_observation=init_observation_norm_state(
            contract.critic_observation_size
        ),
    )
    env = HardwareTwist2Environment(model, metadata.twist2_env, contract)
    return HardwareTwist2PolicyAdapter(
        env, policy, metadata, normalization, HardwareTwist2Config()
    )


def hardware_frame(adapter: HardwareTwist2PolicyAdapter) -> HardwareFrame:
    target = adapter.nominal_physical_target
    return HardwareFrame(
        positions=target.copy(),
        velocities=np.zeros_like(target),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )


def reference_frame(
    adapter: HardwareTwist2PolicyAdapter,
    *,
    sequence: int = 0,
    first_joint_offset: float = 0.0,
    source_age_ms: int | None = None,
) -> Twist2ReferenceFrame:
    qpos = np.asarray(adapter.env.model.default_qpos, dtype=np.float64).copy()
    qpos[adapter.policy_qpos_indices[0]] += first_joint_offset
    return Twist2ReferenceFrame(
        qpos=qpos,
        qvel=np.zeros(adapter.env.model.mj_model.nv),
        sequence=sequence,
        source_age_ms=source_age_ms,
        received_at=time.monotonic(),
    )


def standing_smplx_frame(*, sequence: int = 0) -> SmplxTargetFrame:
    positions = {
        "pelvis": (0.0, 0.0, 0.94),
        "spine3": (0.0, 0.0, 1.30),
        "left_hip": (0.0, 0.10, 0.90),
        "left_knee": (0.0, 0.10, 0.50),
        "left_foot": (0.08, 0.10, 0.06),
        "right_hip": (0.0, -0.10, 0.90),
        "right_knee": (0.0, -0.10, 0.50),
        "right_foot": (0.08, -0.10, 0.06),
        "left_shoulder": (0.0, 0.22, 1.42),
        "left_elbow": (0.0, 0.24, 1.17),
        "left_wrist": (0.05, 0.22, 0.93),
        "right_shoulder": (0.0, -0.22, 1.42),
        "right_elbow": (0.0, -0.24, 1.17),
        "right_wrist": (0.05, -0.22, 0.93),
    }
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    return SmplxTargetFrame(
        transforms={
            name: (np.asarray(positions[name]), identity.copy())
            for name in SMPLX_GMR_BODY_NAMES
        },
        sequence=sequence,
        source_timestamp_ns=sequence * 33_333_333,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def synthetic_smplx_model(tmp_path: Path) -> Path:
    template = np.asarray(
        (
            (0.00, 1.00, 0.00),
            (0.10, 0.92, 0.00),
            (-0.10, 0.92, 0.00),
            (0.00, 1.12, 0.00),
            (0.10, 0.52, 0.00),
            (-0.10, 0.52, 0.00),
            (0.00, 1.25, 0.00),
            (0.10, 0.12, 0.00),
            (-0.10, 0.12, 0.00),
            (0.00, 1.38, 0.00),
            (0.10, 0.04, 0.12),
            (-0.10, 0.04, 0.12),
            (0.00, 1.55, 0.00),
            (0.10, 1.46, 0.00),
            (-0.10, 1.46, 0.00),
            (0.00, 1.70, 0.00),
            (0.22, 1.48, 0.00),
            (-0.22, 1.48, 0.00),
            (0.48, 1.48, 0.00),
            (-0.48, 1.48, 0.00),
            (0.72, 1.48, 0.00),
            (-0.72, 1.48, 0.00),
        ),
        dtype=np.float64,
    )
    joint_count = len(SMPLX_BODY_JOINT_NAMES)
    parents = SMPLX_BODY_JOINT_PARENTS.astype(np.int64).copy()
    parents[0] = 0
    path = tmp_path / "SMPLX_NEUTRAL.npz"
    np.savez(
        path,
        v_template=template,
        J_regressor=np.eye(joint_count, dtype=np.float64),
        kintree_table=np.stack((parents, np.arange(joint_count))),
    )
    return path


def test_twist2_hardware_reconstructs_exact_actor_history_contract(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    adapter.reset()

    first = adapter.step(frame, reference_frame(adapter, sequence=1))
    frame_size = adapter.env.contract.actor_frame_size
    command_size = adapter.env.contract.command_frame_size
    np.testing.assert_array_equal(
        first.observation[frame_size : frame_size * 11],
        np.zeros(frame_size * 10),
    )
    np.testing.assert_array_equal(first.observation[-command_size:], first.command)
    first_current = first.observation[:frame_size]

    second = adapter.step(
        frame,
        reference_frame(adapter, sequence=2, first_joint_offset=0.1),
    )
    current = second.observation[:frame_size]
    reset_history = np.repeat(first_current[None, :], 10, axis=0).reshape(-1)
    np.testing.assert_array_equal(
        second.observation[frame_size : frame_size * 11], reset_history
    )

    third = adapter.step(
        frame,
        reference_frame(adapter, sequence=3, first_joint_offset=0.2),
    )
    expected_history = np.repeat(current[None, :], 10, axis=0).reshape(-1)
    np.testing.assert_array_equal(
        third.observation[frame_size : frame_size * 11], expected_history
    )
    assert third.observation.shape == (1252,)
    assert third.policy_observation.shape == (1252,)
    assert third.action.shape == third.raw_action.shape == (25,)
    assert third.physical_target.shape == (25,)


def test_twist2_hardware_preserves_body_local_gyro_at_nonzero_yaw(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    half_sqrt_two = np.sqrt(0.5)
    angular_velocity = np.asarray((0.4, -0.2, 0.1))
    frame = replace(
        frame,
        orientation_xyzw=np.asarray((0.0, 0.0, half_sqrt_two, half_sqrt_two)),
        angular_velocity=angular_velocity,
    )

    proprioception = adapter.proprioception(frame)

    np.testing.assert_allclose(
        proprioception[:3],
        0.25 * angular_velocity,
        atol=1.0e-7,
    )


def test_twist2_reference_accepts_training_pose_inside_hard_joint_limit(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    config = HardwareTwist2Config()
    action_index = adapter.env.action_layout.names.index("left_ankle_pitch_joint")
    control_index = int(adapter.policy_indices[action_index])
    joint_id = int(adapter.env.model.actuator_joint_ids[control_index])
    hard_upper = float(adapter.env.model.mj_model.jnt_range[joint_id, 1])
    soft_upper = float(adapter.env.model.joint_target_high[control_index])
    assert soft_upper < hard_upper
    reference = reference_frame(adapter)
    qpos = reference.qpos.copy()
    qpos[adapter.policy_qpos_indices[action_index]] = 0.5 * (soft_upper + hard_upper)

    validate_twist2_reference(replace(reference, qpos=qpos), adapter.env, config)

    qpos[adapter.policy_qpos_indices[action_index]] = (
        hard_upper + config.max_reference_joint_limit_violation_radians + 0.001
    )
    with pytest.raises(ReferenceUnavailable, match="left_ankle_pitch_joint"):
        validate_twist2_reference(replace(reference, qpos=qpos), adapter.env, config)


def test_twist2_hardware_only_projects_unsafe_coupled_ankle_targets(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    raw_action = np.zeros(25, dtype=np.float32)
    direct_action_index = next(
        index
        for index, name in enumerate(adapter.env.action_layout.names)
        if "ankle" not in name
    )
    left_pitch_action_index = adapter.env.action_layout.names.index(
        "left_ankle_pitch_joint"
    )
    raw_action[direct_action_index] = 11.0
    raw_action[left_pitch_action_index] = -10.0
    adapter._infer = lambda observation: (observation, jnp.asarray(raw_action))
    adapter.reset()

    step = adapter.step(frame, reference_frame(adapter))
    proprioception = adapter.proprioception(frame)

    assert step.raw_action[direct_action_index] == pytest.approx(11.0)
    assert step.action[direct_action_index] == pytest.approx(10.0)
    assert step.action_clipped
    np.testing.assert_array_equal(proprioception[-25:], raw_action)
    limits = adapter._ctrl_limits
    assert np.all(step.physical_target >= limits[:, 0])
    assert np.all(step.physical_target <= limits[:, 1])

    requested_logical_target = adapter.default_joint_target.copy()
    requested_logical_target[adapter.policy_indices] += 0.5 * step.action
    requested_physical_target = adapter.transform @ requested_logical_target
    old_runner_target = np.clip(
        requested_physical_target,
        limits[:, 0],
        limits[:, 1],
    )
    ankle_motor_indices = {
        index
        for envelope in adapter._ankle_target_envelopes
        for index in envelope.motor_indices
    }
    direct_motor_indices = np.asarray(
        [
            index
            for index in range(len(adapter.physical_coordinate_names))
            if index not in ankle_motor_indices
        ]
    )
    np.testing.assert_array_equal(
        step.physical_target[direct_motor_indices],
        old_runner_target[direct_motor_indices],
    )

    left = next(
        envelope
        for envelope in adapter._ankle_target_envelopes
        if envelope.side == "left"
    )
    motor_indices = np.asarray(left.motor_indices)
    old_resolved = left.motor_to_joint @ old_runner_target[motor_indices]
    resolved = left.motor_to_joint @ step.physical_target[motor_indices]
    assert old_resolved[0] < left.joint_low[0]
    assert resolved[0] == pytest.approx(left.joint_low[0], abs=2.0e-9)
    assert resolved[1] == pytest.approx(old_resolved[1], abs=2.0e-9)
    assert resolved[0] < adapter.env.model.joint_target_low[left.logical_indices[0]]

    pitch_row = left.motor_to_joint[0]
    correction = step.physical_target[motor_indices] - old_runner_target[motor_indices]
    expected_correction = (
        (resolved[0] - old_resolved[0]) * pitch_row / float(pitch_row @ pitch_row)
    )
    np.testing.assert_allclose(correction, expected_correction, atol=1.0e-12)


def test_twist2_hardware_ankle_projection_handles_all_action_corners(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    ankle_action_indices = np.asarray(
        [
            adapter.env.action_layout.names.index(f"{side}_ankle_{axis}_joint")
            for side in ("left", "right")
            for axis in ("pitch", "roll")
        ]
    )
    for signs in (
        (-1.0, -1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0, -1.0),
        (1.0, 1.0, 1.0, 1.0),
    ):
        raw_action = np.zeros(25, dtype=np.float32)
        raw_action[ankle_action_indices] = 10.0 * np.asarray(signs)
        adapter._infer = lambda observation, action=raw_action: (
            observation,
            jnp.asarray(action),
        )
        adapter.reset()
        step = adapter.step(frame, reference_frame(adapter))
        for envelope in adapter._ankle_target_envelopes:
            indices = np.asarray(envelope.motor_indices)
            resolved = envelope.motor_to_joint @ step.physical_target[indices]
            assert np.all(resolved >= envelope.joint_low - 1.0e-12)
            assert np.all(resolved <= envelope.joint_high + 1.0e-12)


def test_twist2_reference_validation_is_fail_closed(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    config = HardwareTwist2Config(max_reference_age_ms=100)
    reference = reference_frame(adapter, source_age_ms=0)
    validate_twist2_reference(reference, adapter.env, config)

    with pytest.raises(ReferenceUnavailable, match="stale"):
        validate_twist2_reference(
            replace(reference, source_age_ms=101), adapter.env, config
        )
    with pytest.raises(ReferenceUnavailable, match="qpos"):
        validate_twist2_reference(
            replace(reference, qpos=np.zeros(2)), adapter.env, config
        )
    invalid_height = reference.qpos.copy()
    invalid_height[adapter.env.model.root_qpos_adr + 2] = 2.0
    with pytest.raises(ReferenceUnavailable, match="root height"):
        validate_twist2_reference(
            replace(reference, qpos=invalid_height), adapter.env, config
        )


def test_twist2_hardware_config_rejects_unsafe_ranges() -> None:
    validate_hardware_twist2_config(HardwareTwist2Config())
    with pytest.raises(ValueError, match="max_raw_action_abs"):
        validate_hardware_twist2_config(HardwareTwist2Config(max_raw_action_abs=9.0))
    with pytest.raises(ValueError, match="root-height bounds"):
        validate_hardware_twist2_config(
            HardwareTwist2Config(
                min_reference_root_height_m=1.0,
                max_reference_root_height_m=1.0,
            )
        )
    with pytest.raises(ValueError, match="qualification_frames"):
        validate_hardware_twist2_config(
            HardwareTwist2Config(fastsam_reference_qualification_frames=0)
        )
    with pytest.raises(ValueError, match="latency budget"):
        validate_hardware_twist2_config(HardwareTwist2Config(max_reference_age_ms=130))


def test_twist2_hardware_ranges_must_cover_model_targets(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    test_adapter = SimpleNamespace(
        physical_coordinate_names=adapter.physical_coordinate_names,
        _model_ctrl_limits=adapter._model_ctrl_limits.copy(),
        _ctrl_limits=adapter._model_ctrl_limits.copy(),
        _ankle_target_envelopes=adapter._ankle_target_envelopes,
    )
    motors = tuple(
        SimpleNamespace(
            coordinate=name,
            limits=SimpleNamespace(
                position_min=float(test_adapter._model_ctrl_limits[index, 0]),
                position_max=float(test_adapter._model_ctrl_limits[index, 1]),
            ),
        )
        for index, name in enumerate(test_adapter.physical_coordinate_names)
    )
    bind_twist2_hardware_coordinate_limits(
        {"all": SimpleNamespace(motors=motors)}, test_adapter
    )

    elbow_index = test_adapter.physical_coordinate_names.index("right_elbow_yaw_joint")
    rounded_elbow = SimpleNamespace(
        coordinate="right_elbow_yaw_joint",
        limits=SimpleNamespace(
            position_min=-2.0943951,
            position_max=0.523598776,
        ),
    )
    rounded_motors = (*motors[:elbow_index], rounded_elbow, *motors[elbow_index + 1 :])
    bind_twist2_hardware_coordinate_limits(
        {"all": SimpleNamespace(motors=rounded_motors)}, test_adapter
    )
    np.testing.assert_array_equal(
        test_adapter._ctrl_limits[elbow_index],
        (rounded_elbow.limits.position_min, rounded_elbow.limits.position_max),
    )

    first = motors[0]
    incompatible = SimpleNamespace(
        coordinate=first.coordinate,
        limits=SimpleNamespace(
            position_min=first.limits.position_min + 0.01,
            position_max=first.limits.position_max,
        ),
    )
    with pytest.raises(ValueError, match="exceeds the hardware range"):
        bind_twist2_hardware_coordinate_limits(
            {"all": SimpleNamespace(motors=(incompatible, *motors[1:]))}, test_adapter
        )


def test_pinned_gmr_import_avoids_optional_ui_and_torch_dependencies() -> None:
    retargeter_type = import_pinned_gmr_retargeter()

    assert retargeter_type.__name__ == "GeneralMotionRetargeting"
    assert "torch" not in sys.modules
    assert "_bello_pinned_gmr.robot_motion_viewer" not in sys.modules


def test_smplx_target_contract_requires_complete_global_transforms() -> None:
    frame = standing_smplx_frame()
    validated = validate_smplx_target_frame(frame, live=False)
    assert tuple(validated) == SMPLX_GMR_BODY_NAMES

    missing = dict(frame.transforms)
    missing.pop("left_wrist")
    with pytest.raises(ReferenceUnavailable, match="left_wrist"):
        validate_smplx_target_frame(replace(frame, transforms=missing), live=False)
    invalid = dict(frame.transforms)
    invalid["pelvis"] = (np.zeros(3), np.zeros(4))
    with pytest.raises(ReferenceUnavailable, match="pelvis.*norm"):
        validate_smplx_target_frame(replace(frame, transforms=invalid), live=False)


def test_pinned_gmr_produces_finite_standing_reference(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
    tmp_path: Path,
) -> None:
    adapter = twist2_hardware_adapter
    root = repository_root()
    revision = subprocess.run(
        ("git", "-C", str(root / "deps/GMR"), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    robot_source = (
        root / "deps/bello-stack/descriptions/bello/full_body/urdf/bello_full_body.urdf"
    )
    fingerprint = "sha256:" + hashlib.sha256(robot_source.read_bytes()).hexdigest()
    metadata = replace(
        adapter.metadata,
        dataset=replace(
            adapter.metadata.dataset,
            gmr_revision=revision,
            robot_fingerprint=fingerprint,
        ),
        robot_fingerprint=fingerprint,
    )
    retargeter = GmrBelloRetargeter(
        adapter.env,
        metadata,
        GmrLiveRetargetConfig(),
    )

    standing = retargeter.standing(
        load_smplx_standing_sample(synthetic_smplx_model(tmp_path))
    )

    assert standing.qpos.shape == (adapter.env.model.mj_model.nq,)
    assert standing.qvel.shape == (adapter.env.model.mj_model.nv,)
    assert np.all(np.isfinite(standing.qpos))
    assert np.all(standing.qvel == 0.0)
    assert standing.source_age_ms is None
    assert not np.allclose(standing.qpos, adapter.env.model.default_qpos)
    root = adapter.env.model.root_qpos_adr
    assert standing.qpos[root + 4] == pytest.approx(0.0, abs=1.0e-10)
    assert standing.qpos[root + 5] == pytest.approx(0.0, abs=1.0e-10)
    sole_z = retargeter._foot_sole_z(standing.qpos)
    assert np.min(sole_z) == pytest.approx(0.0, abs=1.0e-9)
    assert np.ptp(sole_z) < 0.005
    for name in (
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
    ):
        address = adapter.env.model.mj_model.joint(name).qposadr[0]
        assert standing.qpos[address] == pytest.approx(
            adapter.env.model.default_qpos[address], abs=1.0e-10
        )
    validate_twist2_reference(standing, adapter.env, HardwareTwist2Config())

    solver_qpos = retargeter.retargeter.configuration.data.qpos.copy()
    retargeter.validate_reference = lambda _reference: (_ for _ in ()).throw(
        ReferenceUnavailable("unsafe live candidate")
    )
    with pytest.raises(ReferenceUnavailable, match="unsafe live candidate"):
        retargeter.retarget(standing_smplx_frame(sequence=1))
    np.testing.assert_array_equal(
        retargeter.retargeter.configuration.data.qpos, solver_qpos
    )
    assert retargeter._last_sequence is None


def test_live_gmr_source_runs_retargeting_outside_policy_thread() -> None:
    source_frame = standing_smplx_frame(sequence=1)
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    calls: list[str] = []

    class FakeStream:
        delivered = False

        def standing_sample(self):
            return standing_smplx_frame()

        def wait_next(self, *, after_sequence, timeout_seconds):
            del after_sequence
            if not self.delivered:
                self.delivered = True
                return source_frame
            time.sleep(min(timeout_seconds, 0.001))
            return None

    class FakeRetargeter:
        def standing(self, _frame):
            return standing

        def retarget(self, frame):
            import threading

            calls.append(threading.current_thread().name)
            return replace(
                standing,
                sequence=frame.sequence,
                source_age_ms=0,
                received_at=time.monotonic(),
            )

        def reset_live(self):
            return None

    with LiveGmrReferenceSource(
        FakeStream(), FakeRetargeter(), poll_timeout_seconds=0.01
    ) as source:
        result = source.wait_until_ready(1.0)

        assert source.standing().sequence == 0
        assert source.latest().sequence == result.sequence == 1

    assert calls == ["bello-live-gmr"]


def test_live_gmr_source_keeps_last_valid_frame_during_transient_rejection() -> None:
    source_frame = standing_smplx_frame(sequence=1)
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    rejected = threading.Event()
    reset_calls: list[None] = []

    class FakeStream:
        calls = 0

        def standing_sample(self):
            return standing_smplx_frame()

        def wait_next(self, *, after_sequence, timeout_seconds):
            del after_sequence
            self.calls += 1
            if self.calls == 1:
                return source_frame
            if self.calls == 2:
                rejected.set()
                raise ReferenceUnavailable("one low-confidence wrist frame")
            time.sleep(min(timeout_seconds, 0.001))
            return None

    class FakeRetargeter:
        def standing(self, _frame):
            return standing

        def retarget(self, frame):
            return replace(
                standing,
                sequence=frame.sequence,
                source_age_ms=0,
                received_at=time.monotonic(),
            )

        def reset_live(self):
            reset_calls.append(None)

    with LiveGmrReferenceSource(
        FakeStream(), FakeRetargeter(), poll_timeout_seconds=0.01
    ) as source:
        accepted = source.wait_until_ready(1.0)
        assert rejected.wait(1.0)
        assert source.latest() is accepted

    assert reset_calls == []


def test_live_gmr_source_clears_reference_on_tracking_discontinuity() -> None:
    source_frame = standing_smplx_frame(sequence=1)
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    release_discontinuity = threading.Event()
    reset_complete = threading.Event()

    class FakeStream:
        delivered = False
        discontinued = False

        def standing_sample(self):
            return standing_smplx_frame()

        def wait_next(self, *, after_sequence, timeout_seconds):
            del after_sequence
            if not self.delivered:
                self.delivered = True
                return source_frame
            if not self.discontinued and release_discontinuity.wait(timeout_seconds):
                self.discontinued = True
                raise ReferenceDiscontinuity("tracked identity changed")
            time.sleep(min(timeout_seconds, 0.001))
            return None

    class FakeRetargeter:
        def standing(self, _frame):
            return standing

        def retarget(self, frame):
            return replace(
                standing,
                sequence=frame.sequence,
                source_age_ms=0,
                received_at=time.monotonic(),
            )

        def reset_live(self):
            reset_complete.set()

    with LiveGmrReferenceSource(
        FakeStream(), FakeRetargeter(), poll_timeout_seconds=0.01
    ) as source:
        source.wait_until_ready(1.0)
        release_discontinuity.set()
        assert reset_complete.wait(1.0)
        deadline = time.monotonic() + 1.0
        while True:
            try:
                source.latest()
            except ReferenceDiscontinuity as error:
                assert "identity changed" in str(error)
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)


def test_live_reference_gate_requires_distinct_qualified_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bello_mujoco.hardware import twist2 as twist2_hardware

    references = iter((1, 1, 2, 2, 3))

    class FakeReferences:
        def latest(self):
            return Twist2ReferenceFrame(
                qpos=np.zeros(1),
                qvel=np.zeros(1),
                sequence=next(references),
                source_age_ms=0,
                received_at=time.monotonic(),
            )

    standing_steps: list[None] = []
    monkeypatch.setattr(
        twist2_hardware,
        "validate_twist2_reference",
        lambda *_args, **_kwargs: None,
    )

    result = _wait_for_live_twist2_reference(
        FakeReferences(),
        SimpleNamespace(env=SimpleNamespace()),
        HardwareTwist2Config(
            fastsam_reference_qualification_frames=3,
            fastsam_reference_ready_timeout_seconds=1.0,
        ),
        lambda: standing_steps.append(None) is None,
    )

    assert result is not None
    assert result.sequence == 3
    assert len(standing_steps) == 4


def test_twist2_control_uses_normal_standing_and_returns_on_source_loss(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bello_mujoco.hardware import twist2 as twist2_hardware

    coordinate_names = ("left", "right")
    frame = HardwareFrame(
        positions=np.zeros(2),
        velocities=np.zeros(2),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    live = replace(standing, sequence=1)
    seen_sequences: list[int] = []
    resets: list[None] = []
    motion_confirmed = False

    class FakeAdapter:
        env = SimpleNamespace()
        physical_coordinate_names = coordinate_names
        coordinate_index = {"left": 0, "right": 1}
        nominal_physical_target = np.zeros(2)
        kp = np.ones(2)
        kd = np.ones(2)
        control_timestep = 0.02

        def reset(self):
            resets.append(None)

        def reference_physical_target(self, _reference):
            return np.zeros(2)

        def step(self, _frame, reference):
            seen_sequences.append(reference.sequence)
            values = np.zeros(2)
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_action=values,
                action=values,
                command=values,
                action_clipped=False,
                reference_sequence=reference.sequence,
                reference_age_ms=None,
            )

    class FakeReferences:
        calls = 0

        def latest(self):
            assert motion_confirmed
            self.calls += 1
            if self.calls == 1:
                return live
            raise ReferenceUnavailable("tracking lost")

    class FakeControl:
        previous_target = np.zeros(2)

        def ramp_to_target(self, _positions, target):
            self.previous_target = np.asarray(target)
            return frame

        def wait_for_run(self, _prompt, *, operator_input, validate_hold):
            del operator_input
            validate_hold(frame)
            return True

        def read_frame(self, *, max_tilt_radians=None):
            del max_tilt_radians
            return frame

        def send_target(self, target):
            self.previous_target = np.asarray(target)

        def pace(self):
            return None

        def reset_deadline(self):
            return None

        def drain(self):
            return None

    control = FakeControl()
    monkeypatch.setattr(
        twist2_hardware,
        "armed_control_session",
        lambda *_args, **_kwargs: nullcontext(control),
    )

    def confirm_motion(expected, _prompt, hold_step, _operator_input):
        nonlocal motion_confirmed
        assert expected == "MOTION"
        assert hold_step()
        motion_confirmed = True
        return True

    monkeypatch.setattr(
        twist2_hardware,
        "wait_for_operator_confirmation_while_holding",
        confirm_motion,
    )
    monkeypatch.setattr(
        twist2_hardware,
        "validate_twist2_reference",
        lambda *_args, **_kwargs: None,
    )

    control_physical_twist2(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        FakeReferences(),
        standing,
        frame,
        FakeAdapter(),
        HardwareTwist2Config(
            max_duration_seconds=0.05,
            policy_transition_seconds=0.001,
            fastsam_reference_qualification_frames=1,
        ),
    )

    assert seen_sequences[:2] == [0, 0]
    assert 1 in seen_sequences
    assert seen_sequences[-1] == 0
    assert len(resets) >= 3
    output = capsys.readouterr().out
    assert "reference became unavailable" in output
    assert "normal standing resumed" in output


def test_twist2_reference_transition_blends_from_previous_target() -> None:
    from bello_mujoco.hardware.twist2 import _transition_twist2_reference

    frame = SimpleNamespace()
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )

    class FakeAdapter:
        control_timestep = 0.02

        def reset(self):
            return None

        def step(self, _frame, _reference):
            values = np.asarray((4.0, -4.0))
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_action=values,
                action=values,
                command=values,
            )

    class FakeControl:
        previous_target = np.zeros(2)

        def __init__(self):
            self.sent_targets: list[np.ndarray] = []
            self.max_tilts: list[float | None] = []

        def read_frame(self, *, max_tilt_radians=None):
            self.max_tilts.append(max_tilt_radians)
            return frame

        def send_target(self, target):
            self.previous_target = np.asarray(target)
            self.sent_targets.append(self.previous_target.copy())

        def pace(self):
            return None

        def reset_deadline(self):
            return None

        def drain(self):
            return None

    control = FakeControl()
    _transition_twist2_reference(
        control,
        FakeAdapter(),
        lambda: standing,
        HardwareTwist2Config(policy_transition_seconds=0.08),
        phase="transition_to_standing",
        policy_log=None,
        run_started=0.0,
        max_tilt_radians=0.7,
    )

    expected = np.asarray(((1.0, -1.0), (2.0, -2.0), (3.0, -3.0), (4.0, -4.0)))
    np.testing.assert_allclose(control.sent_targets, expected)
    assert control.max_tilts == [0.7] * 4


def test_twist2_motion_transition_refreshes_after_qualified_snapshot() -> None:
    from bello_mujoco.hardware.twist2 import _transition_twist2_reference

    frame = SimpleNamespace()
    first = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=7,
        source_age_ms=0,
        received_at=time.monotonic(),
    )
    live = replace(first, sequence=8)
    seen: list[int] = []

    class FakeAdapter:
        control_timestep = 0.02

        def reset(self):
            return None

        def step(self, _frame, reference):
            seen.append(reference.sequence)
            values = np.zeros(2)
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_action=values,
                action=values,
                command=values,
            )

    class FakeControl:
        previous_target = np.zeros(2)

        def read_frame(self, *, max_tilt_radians=None):
            del max_tilt_radians
            return frame

        def send_target(self, target):
            self.previous_target = np.asarray(target)

        def pace(self):
            return None

        def reset_deadline(self):
            return None

        def drain(self):
            return None

    _transition_twist2_reference(
        FakeControl(),
        FakeAdapter(),
        lambda: live,
        HardwareTwist2Config(policy_transition_seconds=0.08),
        phase="transition_to_motion",
        policy_log=None,
        run_started=0.0,
        first_reference=first,
    )

    assert seen == [7, 8, 8, 8]


def test_twist2_reference_maps_to_physical_hold_target(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    reference = reference_frame(adapter, first_joint_offset=0.1)
    logical = adapter.default_joint_target.copy()
    logical[adapter.policy_indices] = reference.qpos[adapter.policy_qpos_indices]
    limits = np.asarray(adapter.env.model.mj_model.actuator_ctrlrange)
    expected = np.clip(
        adapter.transform @ logical,
        limits[:, 0],
        limits[:, 1],
    )

    np.testing.assert_allclose(adapter.reference_physical_target(reference), expected)


def test_twist2_checkpoint_nominal_reference_maps_to_nominal_target(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    standing = Twist2ReferenceFrame(
        qpos=np.asarray(adapter.env.model.default_qpos, dtype=np.float64).copy(),
        qvel=np.zeros(adapter.env.model.mj_model.nv),
        sequence=0,
        source_age_ms=None,
        received_at=time.monotonic(),
    )

    np.testing.assert_allclose(
        adapter.reference_physical_target(standing),
        adapter.nominal_physical_target,
    )
