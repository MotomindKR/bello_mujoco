from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import jax

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.training.checkpoint import CheckpointNormalizationState
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware import runtime as hardware_runtime
from bello_mujoco.hardware.locomotion import (
    HardwareLocomotionConfig,
    HardwarePolicyAdapter,
    UpperBodyFollowConfig,
    command_minimum_moving_speeds,
    command_ranges,
    control_physical_robot,
    hardware_command,
    run_hardware_locomotion,
    validate_command,
    validate_hardware_checkpoint,
    validate_hardware_config,
)
from bello_mujoco.hardware.runtime import (
    HOLD_CONFIRMATION,
    RUN_CONFIRMATION,
    HardwareFrame,
    actuator_coordinate_names,
    actuator_impedance_gains,
    armed_control_session,
    coordinate_targets,
    pace_hardware_loop,
    projected_gravity_from_xyzw,
    require_async_hardware_binding,
    require_idle_confirmation,
    require_operator_confirmation,
    resolve_command_submissions,
    send_targets,
    specific_force_from_gravity_removed_acceleration,
    start_hardware_controls,
    stop_and_confirm_idle,
    stop_hardware_controls,
    tilt_from_xyzw,
    validate_hardware_frame,
    validate_hardware_inventory,
    wait_for_hardware_frame,
    wait_for_group_state,
    wait_for_operator_confirmation_while_holding,
)
from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.policy import init_actor_critic, policy_observation_size
from bello_mujoco.training.locomotion import TrainingRunConfig, checkpoint_metadata
from bello_mujoco.training.ppo import (
    init_observation_norm_state,
    init_value_norm_state,
)
from bello_mujoco.gamepad import axis_to_range
from bello_mujoco.hardware.remote import RemoteCommandConfig, RemoteCommandInput
from bello_mujoco.hardware.twist2 import ReferenceUnavailable, Twist2ReferenceFrame


_TEST_SMPLX_MODEL = Path("SMPLX_NEUTRAL.npz")


def pose_hardware_config(**overrides) -> HardwareLocomotionConfig:
    return HardwareLocomotionConfig(
        smplx_neutral_model_path=_TEST_SMPLX_MODEL,
        **overrides,
    )


class FakeReferences:
    def __init__(self) -> None:
        self.sequence = 0

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        return False

    def standing(self):
        return SimpleNamespace(sequence=0, age_ms=lambda: None)

    def latest(self):
        self.sequence += 1
        return SimpleNamespace(sequence=self.sequence, age_ms=lambda: 0.0)


def remote_input(
    command=(0.0, 0.0, 0.0),
    *,
    service_connected=True,
    controller_connected=True,
    intent_armed=True,
    abort_active=False,
) -> RemoteCommandInput:
    return RemoteCommandInput(
        normalized_command=tuple(command),
        service_connected=service_connected,
        controller_connected=controller_connected,
        intent_armed=intent_armed,
        abort_active=abort_active,
    )


def compose_hardware_training_config() -> TrainingRunConfig:
    return locomotion_training_profile("hardware_locomotion")


def compose_hardware_runner_config() -> SimpleNamespace:
    return SimpleNamespace(hardware=pose_hardware_config())


def test_hardware_state_frame_preserves_order_and_consumer_age(
    monkeypatch,
) -> None:
    def vector(x: float, y: float, z: float) -> SimpleNamespace:
        return SimpleNamespace(x=x, y=y, z=z)

    state_frame = SimpleNamespace(
        received_monotonic_ns=1_000_000_000,
        round_trip_ns=6_000_000,
        sample_span_ns=9_500_000,
        motors=(
            SimpleNamespace(
                coordinate="right",
                position=2.0,
                velocity=0.2,
                sample_age_ns=1_000_000,
            ),
            SimpleNamespace(
                coordinate="left",
                position=1.0,
                velocity=0.1,
                sample_age_ns=2_000_000,
            ),
        ),
        imus=(
            SimpleNamespace(
                imu="pelvis",
                sample_age_ns=4_000_000,
                orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                angular_velocity=vector(1.0, 2.0, 3.0),
                linear_acceleration=vector(4.0, 5.0, 6.0),
            ),
        ),
        groups=(
            SimpleNamespace(
                group="upper",
                sample_age_ns=3_000_000,
                state=SimpleNamespace(name="DAMPED"),
                cause=SimpleNamespace(name="STOP"),
                detail="held",
            ),
            SimpleNamespace(
                group="lower",
                sample_age_ns=2_500_000,
                state=SimpleNamespace(name="RUNNING"),
                cause=SimpleNamespace(name="START"),
                detail="",
            ),
        ),
    )
    state_reader = SimpleNamespace(read=lambda **_kwargs: state_frame)
    monkeypatch.setattr(hardware_runtime.time, "monotonic_ns", lambda: 1_003_500_000)

    frame = wait_for_hardware_frame(
        state_reader,
        ("left", "right"),
        ("lower", "upper"),
        "pelvis",
        0.5,
    )

    np.testing.assert_array_equal(frame.positions, np.asarray((1.0, 2.0)))
    np.testing.assert_array_equal(frame.velocities, np.asarray((0.1, 0.2)))
    assert frame.group_states == ("RUNNING", "DAMPED")
    assert frame.group_causes == ("START", "STOP")
    assert frame.max_motor_age_ms == 11
    assert frame.imu_age_ms == 13
    assert frame.max_group_age_ms == 12
    assert frame.sample_span_ms == pytest.approx(9.5)
    assert frame.state_frame_round_trip_ms == pytest.approx(6.0)
    with pytest.raises(RuntimeError, match="motor telemetry is stale"):
        validate_hardware_frame(
            frame,
            HardwareLocomotionConfig(max_motor_age_ms=10),
        )
    with pytest.raises(RuntimeError, match="group telemetry is stale"):
        validate_hardware_frame(
            frame,
            HardwareLocomotionConfig(max_group_age_ms=11),
        )
    with pytest.raises(RuntimeError, match="hardware state sample span"):
        validate_hardware_frame(
            frame,
            HardwareLocomotionConfig(max_state_sample_span_ms=9.0),
        )


def test_group_state_wait_uses_bounded_polling(monkeypatch) -> None:
    frame = HardwareFrame(
        positions=np.zeros(1),
        velocities=np.zeros(1),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("DAMPED",),
        group_causes=("ENABLE",),
        group_details=("",),
    )
    frames = iter((frame, replace(frame, group_states=("RUNNING",))))
    reads = []
    sleeps = []

    def read(*_args, **_kwargs):
        reads.append(None)
        return next(frames)

    monkeypatch.setattr(hardware_runtime, "wait_for_hardware_frame", read)
    monkeypatch.setattr(hardware_runtime.time, "sleep", sleeps.append)

    wait_for_group_state(
        SimpleNamespace(),
        ("hip",),
        HardwareLocomotionConfig(groups=("lower_body",)),
        "RUNNING",
    )

    assert len(reads) == 2
    assert len(sleeps) == 1
    assert 0.0 < sleeps[0] <= hardware_runtime.GROUP_STATE_POLL_INTERVAL_SECONDS


@pytest.fixture(scope="module")
def hardware_env() -> BelloLocomotionEnv:
    config = compose_hardware_training_config()
    prepared = prepare_asset(config.asset_config)
    return BelloLocomotionEnv.load_prepared(
        prepared,
        warp=config.warp,
        batch_size=1,
        env_config=config.env_config,
        actuator_config=config.asset_config.actuator_config,
        reward_config=config.reward_config,
        command_config=config.command_config,
        done_config=config.done_config,
        load_mjx=False,
    )


def test_hardware_training_recipe_matches_sphere_hand_inventory(
    hardware_env: BelloLocomotionEnv,
) -> None:
    config = compose_hardware_training_config()

    assert config.asset_config.target_total_mass_kg is None
    assert config.asset_config.joint_availability.enabled_joint_names == {
        "waist_yaw_joint"
    }
    assert hardware_env.action_size == 13
    assert hardware_env.model.mj_model.nu == 25
    assert np.sum(hardware_env.model.mj_model.body_mass) == pytest.approx(
        55.0683575411785
    )

    metadata = SimpleNamespace(
        task="locomotion",
        asset=config.asset_config,
        ppo=config.config,
    )
    validate_hardware_checkpoint(hardware_env, metadata)


def test_hardware_actuator_mapping_round_trips_differential_ankles(
    hardware_env: BelloLocomotionEnv,
) -> None:
    physical_names = actuator_coordinate_names(hardware_env)
    logical_names = hardware_env.model.actuator_joint_names
    transform = np.asarray(hardware_env.model.actuator_coordinate_transform)
    logical = np.zeros(25)
    pitch_column = logical_names.index("left_ankle_pitch_joint")
    roll_column = logical_names.index("left_ankle_roll_joint")
    motor_1_row = physical_names.index("left_ankle_motor_1")
    motor_2_row = physical_names.index("left_ankle_motor_2")
    logical[pitch_column] = 0.3
    logical[roll_column] = -0.1

    physical = transform @ logical
    recovered = np.linalg.solve(transform, physical)

    assert len(physical_names) == 25
    assert physical[motor_1_row] == pytest.approx(-0.4)
    assert physical[motor_2_row] == pytest.approx(0.2)
    assert recovered == pytest.approx(logical)


def test_hardware_uses_generated_motor_impedance_gains(
    hardware_env: BelloLocomotionEnv,
) -> None:
    names = actuator_coordinate_names(hardware_env)
    kp, kd = actuator_impedance_gains(hardware_env)

    for name in (
        "left_ankle_motor_1",
        "left_ankle_motor_2",
        "right_ankle_motor_1",
        "right_ankle_motor_2",
    ):
        index = names.index(name)
        assert kp[index] == pytest.approx(11.0)
        assert kd[index] == pytest.approx(0.5)


def make_hardware_adapter(hardware_env: BelloLocomotionEnv) -> HardwarePolicyAdapter:
    config = compose_hardware_training_config()
    actor_size = policy_observation_size(
        hardware_env.observation_size, config.config.policy
    )
    critic_size = policy_observation_size(
        hardware_env.critic_observation_size, config.config.policy
    )
    metadata = checkpoint_metadata(
        config,
        hardware_env,
        policy_obs_size=actor_size,
        critic_policy_obs_size=critic_size,
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        actor_size,
        hardware_env.action_size,
        config.config.policy,
        critic_obs_size=critic_size,
    )
    normalization = CheckpointNormalizationState(
        value=init_value_norm_state(),
        observation=init_observation_norm_state(actor_size),
        critic_observation=init_observation_norm_state(critic_size),
    )
    return HardwarePolicyAdapter(hardware_env, policy, metadata, normalization)


def hardware_reference(
    adapter: HardwarePolicyAdapter,
    upper_body_positions: np.ndarray,
    *,
    sequence: int = 1,
    source_age_ms: int | None = 0,
    received_at: float | None = None,
) -> Twist2ReferenceFrame:
    qpos = np.asarray(adapter.env.model.default_qpos, dtype=np.float64).copy()
    qpos[adapter.upper_body_qpos_indices] = upper_body_positions
    return Twist2ReferenceFrame(
        qpos=qpos,
        qvel=np.zeros(adapter.env.model.mj_model.nv, dtype=np.float64),
        sequence=sequence,
        source_age_ms=source_age_ms,
        received_at=time.monotonic() if received_at is None else received_at,
    )


def test_hardware_adapter_reconstructs_actor_frame_and_targets(
    hardware_env: BelloLocomotionEnv,
) -> None:
    adapter = make_hardware_adapter(hardware_env)
    nominal = adapter.nominal_physical_target
    frame = HardwareFrame(
        positions=nominal,
        velocities=np.zeros_like(nominal),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
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
    )
    command = np.asarray((0.2, 0.0, 0.0), dtype=np.float32)

    result = adapter.step(frame, command)

    assert result.observation.shape == (54,)
    assert result.observation[:3] == pytest.approx((0.0, 0.0, 0.980665))
    assert result.observation[3:6] == pytest.approx(np.zeros(3))
    assert result.observation[6:9] == pytest.approx((0.0, 0.0, -1.0))
    assert result.observation[9:12] == pytest.approx(command)
    assert result.observation[12:15] == pytest.approx((0.0, 1.0, 1.0))
    assert result.action.shape == (13,)
    assert result.action == pytest.approx(np.clip(result.raw_action, -1.0, 1.0))
    assert result.physical_target.shape == (25,)
    assert np.all(np.isfinite(result.physical_target))
    assert result.physical_target == pytest.approx(
        np.asarray(hardware_env.action_to_ctrl(result.action, np.zeros(25))),
        abs=1.0e-6,
    )


def test_locomotion_overlays_only_slew_limited_upper_body_reference(
    hardware_env: BelloLocomotionEnv,
) -> None:
    adapter = make_hardware_adapter(hardware_env)
    upper = adapter.upper_body_indices
    default_upper = adapter.default_joint_target[upper]
    standing = hardware_reference(
        adapter,
        default_upper,
        sequence=0,
        source_age_ms=None,
    )
    low = hardware_env.model.joint_target_low[upper]
    high = hardware_env.model.joint_target_high[upper]
    desired_upper = np.clip(default_upper + 0.5, low + 0.01, high - 0.01)
    live = hardware_reference(adapter, desired_upper)
    adapter.set_upper_body_standing_reference(standing)
    nominal = adapter.nominal_physical_target
    frame = HardwareFrame(
        positions=nominal,
        velocities=np.zeros_like(nominal),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("START", "START"),
        group_details=("", ""),
    )

    joystick_command = np.asarray((0.0, 0.3, -0.2), dtype=np.float32)
    step = adapter.step(frame, joystick_command, live)
    logical_target = adapter.inverse_transform @ step.physical_target
    policy_only = np.asarray(
        hardware_env.action_to_ctrl(step.action, np.zeros(25)), dtype=np.float64
    )
    policy_logical_target = adapter.inverse_transform @ policy_only
    maximum_step = (
        adapter.upper_body_follow.max_target_speed_rps * adapter.control_timestep
    )
    transition_step = (
        adapter.control_timestep
        / adapter.upper_body_follow.connection_transition_duration_seconds
    )
    blended_live = default_upper + transition_step * (desired_upper - default_upper)
    expected_upper = default_upper + np.clip(
        blended_live - default_upper,
        -maximum_step,
        maximum_step,
    )

    assert logical_target[adapter.policy_indices] == pytest.approx(
        policy_logical_target[adapter.policy_indices], abs=1.0e-6
    )
    assert logical_target[upper] == pytest.approx(expected_upper, abs=1.0e-6)
    assert step.observation[9:12] == pytest.approx(joystick_command)
    assert step.upper_body_following
    assert step.reference_sequence == 1

    stale = hardware_reference(
        adapter,
        desired_upper,
        sequence=2,
        received_at=time.monotonic() - 1.1,
    )
    fallback = adapter.step(frame, joystick_command, stale)
    fallback_logical = adapter.inverse_transform @ fallback.physical_target
    blended_fallback = expected_upper + transition_step * (
        default_upper - expected_upper
    )
    expected_fallback = expected_upper + np.clip(
        blended_fallback - expected_upper,
        -maximum_step,
        maximum_step,
    )
    assert fallback_logical[upper] == pytest.approx(expected_fallback, abs=1.0e-6)
    assert fallback.observation[9:12] == pytest.approx(joystick_command)
    assert not fallback.upper_body_following
    assert fallback.reference_sequence is None

    recovered = adapter.step(
        frame,
        joystick_command,
        hardware_reference(adapter, desired_upper, sequence=3),
    )
    recovered_logical = adapter.inverse_transform @ recovered.physical_target
    blended_recovered = expected_fallback + transition_step * (
        desired_upper - expected_fallback
    )
    expected_recovered = expected_fallback + np.clip(
        blended_recovered - expected_fallback,
        -maximum_step,
        maximum_step,
    )
    assert recovered_logical[upper] == pytest.approx(expected_recovered, abs=1.0e-6)
    assert recovered.observation[9:12] == pytest.approx(joystick_command)
    assert recovered.upper_body_following
    assert recovered.reference_sequence == 3


def test_locomotion_applies_configured_and_clipped_elbow_yaw_offset(
    hardware_env: BelloLocomotionEnv,
) -> None:
    adapter = make_hardware_adapter(hardware_env)
    adapter.upper_body_follow = replace(
        adapter.upper_body_follow,
        left_elbow_yaw_offset_radians=np.pi / 2.0,
        right_elbow_yaw_offset_radians=-np.pi,
    )
    upper = adapter.upper_body_indices
    desired = adapter.default_joint_target[upper]
    low = hardware_env.model.joint_target_low[upper]
    high = hardware_env.model.joint_target_high[upper]
    target = adapter._reference_upper_body_target(
        hardware_reference(adapter, desired),
        live=True,
    )
    elbow_yaw = adapter.upper_body_elbow_yaw_indices
    expected = desired.copy()
    expected[elbow_yaw] += np.asarray((np.pi / 2.0, -np.pi))
    expected = np.clip(expected, low, high)

    assert target == pytest.approx(expected)
    assert target[elbow_yaw] == pytest.approx((high[elbow_yaw[0]], low[elbow_yaw[1]]))


def test_locomotion_rejects_stale_and_unsafe_upper_body_references(
    hardware_env: BelloLocomotionEnv,
) -> None:
    adapter = make_hardware_adapter(hardware_env)
    assert adapter.upper_body_follow.max_reference_age_ms == 1_000
    default_upper = adapter.default_joint_target[adapter.upper_body_indices]
    adapter.validate_upper_body_reference(
        hardware_reference(adapter, default_upper, source_age_ms=470)
    )
    stale = hardware_reference(
        adapter,
        default_upper,
        received_at=time.monotonic() - 1.1,
    )
    with pytest.raises(ReferenceUnavailable, match="stale"):
        adapter.validate_upper_body_reference(stale)

    unsafe = hardware_reference(adapter, default_upper)
    unsafe.qvel[adapter.upper_body_qvel_indices[0]] = 26.0
    with pytest.raises(ReferenceUnavailable, match="joint speed"):
        adapter.validate_upper_body_reference(unsafe)


def test_hardware_imu_terms_match_deployment_conventions() -> None:
    projected_gravity = projected_gravity_from_xyzw(np.asarray((0.0, 0.0, 0.0, 1.0)))
    assert projected_gravity == pytest.approx((0.0, 0.0, -1.0))
    assert specific_force_from_gravity_removed_acceleration(
        np.zeros(3), projected_gravity
    ) == pytest.approx((0.0, 0.0, 9.80665))

    tilted_gravity = projected_gravity_from_xyzw(
        np.asarray((0.2, -0.3, 0.1, 0.92736185))
    )
    expected_specific_force = np.asarray((1.5, -2.0, 8.0))
    gravity_removed = expected_specific_force + 9.80665 * tilted_gravity
    assert specific_force_from_gravity_removed_acceleration(
        gravity_removed, tilted_gravity
    ) == pytest.approx(expected_specific_force, abs=1.0e-6)
    assert tilt_from_xyzw(np.asarray((0.0, 0.0, 0.0, 1.0))) == pytest.approx(0.0)


def test_hardware_runner_defaults_to_read_only() -> None:
    config = compose_hardware_runner_config()

    assert not config.hardware.arm
    assert config.hardware.command_validity_seconds == pytest.approx(0.08)
    assert config.hardware.acknowledgement_timeout_seconds == pytest.approx(0.5)
    assert config.hardware.remote.endpoint == "http://127.0.0.1:50054"
    assert config.hardware.remote.staleness_ms == 200
    assert config.hardware.upper_body_follow.left_elbow_yaw_offset_radians == 0.0
    assert config.hardware.upper_body_follow.right_elbow_yaw_offset_radians == 0.0
    validate_hardware_config(config.hardware)
    validate_hardware_config(pose_hardware_config(arm=True))
    with pytest.raises(ValueError, match="smplx_neutral_model_path"):
        validate_hardware_config(HardwareLocomotionConfig())
    with pytest.raises(ValueError, match="plaintext"):
        validate_hardware_config(
            pose_hardware_config(
                remote=replace(RemoteCommandConfig(), endpoint="https://remote:50054")
            )
        )
    with pytest.raises(ValueError, match="staleness_ms"):
        validate_hardware_config(
            pose_hardware_config(remote=replace(RemoteCommandConfig(), staleness_ms=0))
        )
    with pytest.raises(ValueError, match="gain_scale"):
        validate_hardware_config(pose_hardware_config(gain_scale=0.0))
    with pytest.raises(ValueError, match="gain_scale"):
        validate_hardware_config(pose_hardware_config(gain_scale=1.01))
    with pytest.raises(ValueError, match="acknowledgement_timeout_seconds"):
        validate_hardware_config(
            pose_hardware_config(acknowledgement_timeout_seconds=0.08)
        )
    with pytest.raises(ValueError, match="connection_transition_duration_seconds"):
        validate_hardware_config(
            pose_hardware_config(
                upper_body_follow=replace(
                    UpperBodyFollowConfig(),
                    connection_transition_duration_seconds=0.0,
                )
            )
        )
    with pytest.raises(ValueError, match="left_elbow_yaw_offset_radians"):
        validate_hardware_config(
            pose_hardware_config(
                upper_body_follow=replace(
                    UpperBodyFollowConfig(),
                    left_elbow_yaw_offset_radians=np.nan,
                )
            )
        )


def test_coordinate_targets_use_policy_gains() -> None:
    adapter = SimpleNamespace(
        coordinate_index={"left": 0, "right": 1},
        kp=np.asarray((12.0, 34.0)),
        kd=np.asarray((0.5, 0.75)),
    )
    binding = SimpleNamespace(
        CoordinateTarget=SimpleNamespace(position=lambda value, kp, kd: (value, kp, kd))
    )

    targets = coordinate_targets(
        binding,
        ("left", "right"),
        np.asarray((0.2, -0.4)),
        adapter,
    )

    assert targets == {
        "left": (0.2, 12.0, 0.5),
        "right": (-0.4, 34.0, 0.75),
    }

    scaled_targets = coordinate_targets(
        binding,
        ("left", "right"),
        np.asarray((0.2, -0.4)),
        adapter,
        gain_scale=5.0 / 6.0,
    )

    assert scaled_targets["left"] == pytest.approx((0.2, 10.0, 5.0 / 12.0))
    assert scaled_targets["right"] == pytest.approx((-0.4, 85.0 / 3.0, 0.625))


def test_hardware_requires_exact_hold_and_run_confirmations() -> None:
    prompts = []
    require_operator_confirmation(
        HOLD_CONFIRMATION,
        "hold prompt",
        lambda prompt: prompts.append(prompt) or HOLD_CONFIRMATION,
    )

    release_input = threading.Event()
    hold_steps = []

    def operator_input(prompt: str) -> str:
        prompts.append(prompt)
        assert release_input.wait(timeout=1.0)
        return RUN_CONFIRMATION

    def hold_step() -> bool:
        hold_steps.append(len(hold_steps))
        if len(hold_steps) == 2:
            release_input.set()
        threading.Event().wait(0.001)
        return True

    assert wait_for_operator_confirmation_while_holding(
        RUN_CONFIRMATION,
        "run prompt",
        hold_step,
        operator_input,
    )
    assert prompts == ["hold prompt", "run prompt"]
    assert len(hold_steps) >= 2

    idle_prompts = []
    idle_responses = iter(("idle", "IDLE"))
    require_idle_confirmation(
        "idle prompt",
        lambda prompt: idle_prompts.append(prompt) or next(idle_responses),
    )
    assert idle_prompts == ["idle prompt", "idle prompt"]

    with pytest.raises(RuntimeError, match="expected exact 'HOLD'"):
        require_operator_confirmation(HOLD_CONFIRMATION, "", lambda _prompt: "hold")
    with pytest.raises(RuntimeError, match="expected exact 'RUN'"):
        wait_for_operator_confirmation_while_holding(
            RUN_CONFIRMATION,
            "",
            lambda: True,
            lambda _prompt: "run",
        )


def test_terminal_confirmation_can_stop_without_leaving_an_input_reader(
    monkeypatch,
) -> None:
    terminal = SimpleNamespace(
        isatty=lambda: True,
        readline=lambda: pytest.fail("stdin must not be read before it is ready"),
    )
    monkeypatch.setattr(hardware_runtime.sys, "stdin", terminal)
    monkeypatch.setattr(
        hardware_runtime.select,
        "select",
        lambda *_args: ([], [], []),
    )

    assert not wait_for_operator_confirmation_while_holding(
        RUN_CONFIRMATION,
        "run prompt",
        lambda: False,
    )


def test_shared_runtime_preserves_failure_while_stopping_running_groups(
    monkeypatch,
) -> None:
    with pytest.raises(RuntimeError, match="send_async"):
        require_async_hardware_binding(SimpleNamespace())
    require_async_hardware_binding(
        SimpleNamespace(
            ControlSession=SimpleNamespace(send_async=lambda: None),
            ControlState=SimpleNamespace(RUNNING="running"),
        )
    )

    stopped = []
    disabled = []
    released = []
    running = object()
    damped = object()

    def stop_lower():
        stopped.append("lower")
        controls["lower"].state = damped

    def disable(name):
        disabled.append(name)
        controls[name].state = object()

    controls = {
        "lower": SimpleNamespace(
            state=running,
            stop=stop_lower,
            disable=lambda: disable("lower"),
            close=lambda *, preserve_damping: released.append(
                ("lower", preserve_damping)
            ),
        ),
        "upper": SimpleNamespace(
            state=damped,
            stop=lambda: pytest.fail("already-damped group must not be stopped"),
            disable=lambda: disable("upper"),
            close=lambda *, preserve_damping: released.append(
                ("upper", preserve_damping)
            ),
        ),
    }
    groups = {
        name: SimpleNamespace(
            control=lambda *_args, _name=name, **_kwargs: controls[_name]
        )
        for name in controls
    }
    config = HardwareLocomotionConfig(groups=tuple(groups))
    adapter = SimpleNamespace(
        physical_coordinate_names=("left", "right"),
        control_timestep=0.02,
    )
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
        group_states=("IDLE", "IDLE"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )
    monkeypatch.setattr(hardware_runtime, "start_hardware_controls", lambda *_: None)
    monkeypatch.setattr(hardware_runtime, "wait_for_group_state", lambda *_: None)
    confirmations = []

    def confirm_idle(prompt):
        confirmations.append(
            (prompt, list(stopped), [control.state for control in controls.values()])
        )
        return "IDLE"

    with pytest.raises(RuntimeError, match="control failed"):
        with armed_control_session(
            SimpleNamespace(
                ControlState=SimpleNamespace(RUNNING=running, DAMPED=damped),
            ),
            groups,
            SimpleNamespace(),
            frame,
            adapter,
            config,
            thread_name="test-control",
            operator_input=confirm_idle,
        ):
            raise RuntimeError("control failed")

    assert stopped == ["lower"]
    assert disabled == ["lower", "upper"]
    assert released == [("lower", True), ("upper", True)]
    assert confirmations == [
        (
            "Hardware controls are damped. Support the robot, then type IDLE to "
            "disable motors and release control: ",
            ["lower"],
            [damped, damped],
        )
    ]


def test_locomotion_validates_nominal_pose_through_run_hold(monkeypatch) -> None:
    nominal = np.asarray((0.1, -0.2))
    frame = HardwareFrame(
        positions=nominal.copy(),
        velocities=np.zeros(2),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING",),
        group_causes=("START",),
        group_details=("control started",),
    )
    checks = []

    def ramp_to_nominal(_positions, *, before_step):
        assert before_step()
        return frame

    def wait_for_run(_prompt, *, operator_input, before_step, validate_hold):
        del operator_input
        assert before_step()
        validate_hold(frame)
        return True

    control = SimpleNamespace(
        previous_target=nominal.copy(),
        ramp_to_nominal=ramp_to_nominal,
        wait_for_run=wait_for_run,
        read_frame=lambda: frame,
        reset_deadline=lambda: None,
        drain=lambda: None,
    )
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.armed_control_session",
        lambda *_args, **_kwargs: nullcontext(control),
    )
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.require_nominal_pose",
        lambda checked_frame, _adapter: checks.append(checked_frame),
    )
    remote_inputs = iter(
        (
            remote_input(),
            remote_input(),
            remote_input(abort_active=True),
        )
    )

    adapter = SimpleNamespace(
        physical_coordinate_names=("left", "right"),
        nominal_physical_target=nominal,
        control_timestep=0.02,
        reset=lambda: None,
    )
    control_physical_robot(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        frame,
        adapter,
        HardwareLocomotionConfig(max_duration_seconds=None),
        SimpleNamespace(poll=lambda: next(remote_inputs)),
        SimpleNamespace(latest=lambda: pytest.fail("pose read after remote abort")),
    )

    assert len(checks) == 3
    assert all(checked is frame for checked in checks)


def test_hardware_command_limits() -> None:
    metadata = SimpleNamespace(
        command=CommandConfig(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.2, 0.2),
            ang_vel_yaw=(-0.3, 0.3),
        )
    )

    validate_command(np.asarray((0.5, -0.2, 0.3)), metadata)
    with pytest.raises(ValueError, match="exceeds checkpoint bounds"):
        validate_command(np.asarray((0.6, 0.0, 0.0)), metadata)


def test_hardware_remote_commands_fail_closed_and_use_checkpoint_bounds() -> None:
    metadata = SimpleNamespace(
        command=CommandConfig(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.2, 0.3),
            ang_vel_yaw=(-0.4, 0.6),
        )
    )
    assert command_ranges(metadata) == ((-0.5, 1.0), (-0.2, 0.3), (-0.4, 0.6))
    assert command_minimum_moving_speeds(metadata) == (0.2, 0.2, 0.2)
    assert hardware_command(
        metadata,
        remote_input(service_connected=False, controller_connected=False),
    ) == pytest.approx(np.zeros(3))
    assert hardware_command(
        metadata,
        remote_input((0.8, -0.1, 0.4)),
    ) == pytest.approx((-0.23, 0.28, 0.36))


def test_normalized_axis_mapping_preserves_standing_and_conditioned_ranges() -> None:
    assert axis_to_range(0.0, (0.3, 1.0), 0.2) == 0.0
    assert axis_to_range(0.5, (0.3, 1.0), 0.2) == pytest.approx(0.65)
    assert axis_to_range(-0.5, (0.3, 1.0), 0.2) == 0.0
    assert axis_to_range(-0.5, (-1.0, -0.3), 0.2) == pytest.approx(-0.65)
    assert axis_to_range(0.5, (-1.0, -0.3), 0.2) == 0.0
    assert axis_to_range(0.25, (-1.0, 1.0), 0.2) == pytest.approx(0.4)


def test_hardware_honors_held_remote_abort_before_connect(monkeypatch) -> None:
    adapter = SimpleNamespace(precompile=lambda: None)
    metadata = SimpleNamespace(command=CommandConfig())
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.load_hardware_policy",
        lambda *_args, **_kwargs: (adapter, metadata),
    )
    connect_calls = []

    def connect(*_args, **_kwargs):
        connect_calls.append(True)
        raise AssertionError("hardware abort connected to the robot")

    binding = SimpleNamespace(
        ControlSession=SimpleNamespace(send_async=lambda: None),
        ControlState=SimpleNamespace(RUNNING="running"),
        Robot=SimpleNamespace(connect=connect),
    )
    remote = SimpleNamespace(
        wait_until_ready=lambda _timeout: remote_input(abort_active=True),
        poll=lambda: remote_input(abort_active=True),
        close=lambda: None,
    )

    run_hardware_locomotion(
        Path("unused"),
        pose_hardware_config(arm=True),
        hardware_binding=binding,
        remote_factory=lambda *_args: remote,
    )

    assert not connect_calls


def test_hardware_requires_remote_sample_before_connect(monkeypatch) -> None:
    adapter = SimpleNamespace(precompile=lambda: None)
    metadata = SimpleNamespace(command=CommandConfig())
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.load_hardware_policy",
        lambda *_args, **_kwargs: (adapter, metadata),
    )
    connect_calls = []

    def connect(*_args, **_kwargs):
        connect_calls.append(True)
        raise AssertionError("hardware connected without a remote sample")

    def wait_until_ready(_timeout):
        raise TimeoutError("remote unavailable")

    binding = SimpleNamespace(Robot=SimpleNamespace(connect=connect))
    remote = SimpleNamespace(
        wait_until_ready=wait_until_ready,
        close=lambda: None,
    )

    with pytest.raises(TimeoutError, match="remote unavailable"):
        run_hardware_locomotion(
            Path("unused"),
            pose_hardware_config(),
            hardware_binding=binding,
            remote_factory=lambda *_args: remote,
        )

    assert not connect_calls


@pytest.mark.parametrize("abort_during_prompt", (True, False))
def test_remote_abort_around_hold_never_acquires_control(
    monkeypatch,
    hardware_preflight_factory,
    abort_during_prompt,
) -> None:
    coordinates = ("left_hip_pitch_joint", "right_hip_pitch_joint")
    hardware = hardware_preflight_factory(coordinates, (0.0, 0.0))
    hardware.binding.ControlSession = SimpleNamespace(send_async=lambda: None)
    hardware.binding.ControlState = SimpleNamespace(RUNNING="running")
    adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        env=SimpleNamespace(action_size=13),
        control_timestep=0.02,
        precompile=lambda _reference: None,
        set_upper_body_standing_reference=lambda _reference: None,
        validate_upper_body_reference=lambda _reference: None,
        reset=lambda: None,
        step=lambda _frame, _command, reference: SimpleNamespace(
            action=np.zeros(13),
            tilt_radians=0.0,
            reference_sequence=reference.sequence,
            reference_age_ms=reference.age_ms(),
        ),
    )
    metadata = SimpleNamespace(command=CommandConfig())
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.load_hardware_policy",
        lambda *_args, **_kwargs: (adapter, metadata),
    )

    def wait_for_hold(_expected, _prompt, before_step, _operator_input):
        continued = before_step()
        assert continued is not abort_during_prompt
        return continued

    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.wait_for_operator_confirmation_while_holding",
        wait_for_hold,
    )
    inputs = iter(
        (
            remote_input(),
            remote_input(),
            remote_input(),
            remote_input(abort_active=abort_during_prompt),
            remote_input(abort_active=True),
        )
    )
    remote = SimpleNamespace(
        wait_until_ready=lambda _timeout: next(inputs),
        poll=lambda: next(inputs),
        close=lambda: None,
    )

    run_hardware_locomotion(
        Path("unused"),
        pose_hardware_config(
            arm=True,
            allow_git_commit_mismatch=True,
            fastsam_reference_qualification_frames=1,
        ),
        hardware_binding=hardware.binding,
        remote_factory=lambda _config: remote,
        reference_source_factory=lambda *_args: FakeReferences(),
    )

    assert not hardware.control_requests


def test_hardware_groups_start_concurrently() -> None:
    barrier = threading.Barrier(2, timeout=1.0)
    started = []
    sent = []

    class Control:
        def __init__(self, name: str, state: object) -> None:
            self.name = name
            self.state = state

        def start(self, targets) -> None:
            assert targets == {self.name: self.name}
            barrier.wait()
            started.append(self.name)

        def send_async(self, targets):
            assert targets == {self.name: self.name}
            barrier.wait()
            sent.append(self.name)
            return SimpleNamespace(done=lambda: True, result=lambda: None)

        def stop(self) -> None:
            stopped.append(self.name)

        def close(self, *, preserve_damping: bool = False) -> None:
            closed.append((self.name, preserve_damping))

    names = ("lower_body", "upper_body")
    running = object()
    uncertain = object()
    controls = {
        "lower_body": Control("lower_body", running),
        "upper_body": Control("upper_body", uncertain),
    }
    stopped = []
    closed = []
    groups = {name: SimpleNamespace(coordinates=(name,)) for name in names}
    adapter = SimpleNamespace(
        coordinate_index={name: index for index, name in enumerate(names)},
        kp=np.ones(2),
        kd=np.ones(2),
        control_timestep=0.02,
    )
    binding = SimpleNamespace(
        CoordinateTarget=SimpleNamespace(
            position=lambda value, kp, kd: names[int(value)]
        ),
        ControlState=SimpleNamespace(RUNNING=running),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        start_hardware_controls(
            binding,
            controls,
            groups,
            np.asarray((0.0, 1.0)),
            adapter,
            HardwareLocomotionConfig(groups=names),
            executor,
        )
        pending_submissions = {}
        pending_receipts = []
        send_targets(
            binding,
            controls,
            groups,
            np.asarray((0.0, 1.0)),
            adapter,
            HardwareLocomotionConfig(groups=names),
            executor,
            pending_submissions,
            pending_receipts,
        )
        resolve_command_submissions(pending_submissions, pending_receipts)
        assert len(pending_receipts) == 2
        stop_hardware_controls(
            binding,
            controls,
            HardwareLocomotionConfig(groups=names),
            executor,
        )

    assert set(started) == set(names)
    assert set(sent) == set(names)
    assert stopped == ["lower_body"]
    assert closed == []


def test_hardware_target_submission_overlaps_control_loop() -> None:
    submission_started = threading.Event()
    release_submission = threading.Event()
    receipt = SimpleNamespace(done=lambda: True, result=lambda: None)

    class Control:
        def send_async(self, targets):
            assert targets == {"lower_body": "lower_body"}
            submission_started.set()
            assert release_submission.wait(timeout=1.0)
            return receipt

    adapter = SimpleNamespace(
        coordinate_index={"lower_body": 0},
        kp=np.ones(1),
        kd=np.ones(1),
        control_timestep=0.02,
    )
    binding = SimpleNamespace(
        CoordinateTarget=SimpleNamespace(position=lambda value, kp, kd: "lower_body")
    )
    config = HardwareLocomotionConfig(groups=("lower_body",))
    pending_submissions = {}
    pending_receipts = []

    with ThreadPoolExecutor(max_workers=1) as executor:
        send_targets(
            binding,
            {"lower_body": Control()},
            {"lower_body": SimpleNamespace(coordinates=("lower_body",))},
            np.asarray((0.0,)),
            adapter,
            config,
            executor,
            pending_submissions,
            pending_receipts,
        )
        assert submission_started.wait(timeout=1.0)
        assert not pending_receipts
        assert tuple(pending_submissions) == ("lower_body",)

        release_submission.set()
        resolve_command_submissions(pending_submissions, pending_receipts)

    assert pending_submissions == {}
    assert pending_receipts == [receipt]


def test_hardware_loop_rebases_after_each_late_iteration(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(hardware_runtime.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(
        hardware_runtime.time,
        "sleep",
        lambda duration: setattr(clock, "now", clock.now + duration),
    )
    config = HardwareLocomotionConfig()
    deadline = clock.now

    for _ in range(600):
        clock.now += 0.022
        deadline = pace_hardware_loop(
            deadline,
            0.02,
            config,
        )

    assert deadline == pytest.approx(clock.now)


def test_hardware_loop_rejects_one_excessively_late_iteration(monkeypatch) -> None:
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(hardware_runtime.time, "monotonic", lambda: clock.now)
    config = HardwareLocomotionConfig()
    deadline = clock.now

    with pytest.raises(RuntimeError, match="missed its deadline"):
        clock.now += 0.121
        pace_hardware_loop(
            deadline,
            0.02,
            config,
        )


def test_hardware_cleanup_stops_only_running_groups() -> None:
    stopped = []
    closed = []
    running = object()
    damped = object()
    controls = {
        "lower_body": SimpleNamespace(
            state=running, stop=lambda: stopped.append("lower_body")
        ),
        "upper_body": SimpleNamespace(
            state=damped,
            stop=lambda: pytest.fail("already-damped group must not be stopped"),
            close=lambda: closed.append("upper_body"),
        ),
    }
    binding = SimpleNamespace(ControlState=SimpleNamespace(RUNNING=running))

    with ThreadPoolExecutor(max_workers=2) as executor:
        stop_hardware_controls(
            binding,
            controls,
            HardwareLocomotionConfig(groups=tuple(controls)),
            executor,
        )

    assert stopped == ["lower_body"]
    assert closed == []


def test_cleanup_keeps_damped_group_until_idle_confirmation_when_peer_faults() -> None:
    running = object()
    damped = object()
    fault = object()
    ready = object()
    events = []

    def stop_lower() -> None:
        controls["lower_body"].state = damped
        events.append("lower damped")

    def stop_upper() -> None:
        controls["upper_body"].state = fault
        events.append("upper faulted")
        raise RuntimeError("upper stop rejected in fault")

    def disable_lower() -> None:
        events.append("lower disabled")
        controls["lower_body"].state = ready

    controls = {
        "lower_body": SimpleNamespace(
            state=running,
            stop=stop_lower,
            disable=disable_lower,
        ),
        "upper_body": SimpleNamespace(
            state=running,
            stop=stop_upper,
            disable=lambda: pytest.fail("faulted group must not be disabled"),
        ),
    }
    binding = SimpleNamespace(
        ControlState=SimpleNamespace(RUNNING=running, DAMPED=damped)
    )

    def confirm_idle(_prompt: str) -> str:
        events.append("idle confirmed")
        assert controls["lower_body"].state is damped
        assert controls["upper_body"].state is fault
        return "IDLE"

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(ExceptionGroup, match="hardware control cleanup failed"),
    ):
        stop_and_confirm_idle(
            binding,
            controls,
            HardwareLocomotionConfig(groups=tuple(controls)),
            executor,
            confirm_idle,
        )

    assert events.index("idle confirmed") < events.index("lower disabled")
    assert controls["lower_body"].state is ready
    assert controls["upper_body"].state is fault


def test_damped_group_failure_reports_cause_and_detail() -> None:
    frame = HardwareFrame(
        positions=np.zeros(1),
        velocities=np.zeros(1),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("DAMPED",),
        group_causes=("COMMAND_TIMEOUT",),
        group_details=("command watchdog expired; damping applied",),
    )

    with pytest.raises(RuntimeError) as failure:
        validate_hardware_frame(
            frame,
            HardwareLocomotionConfig(groups=("lower_body",)),
            expected_group_state="RUNNING",
        )

    message = str(failure.value)
    assert "lower_body=DAMPED" in message
    assert "cause=COMMAND_TIMEOUT" in message
    assert "command watchdog expired; damping applied" in message


def test_read_only_preflight_never_acquires_control(
    monkeypatch,
    capsys,
    hardware_preflight_factory,
) -> None:
    coordinates = ("left_hip_pitch_joint", "right_hip_pitch_joint")
    hardware = hardware_preflight_factory(
        coordinates,
        (2.014, 2.014),
    )
    adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        env=SimpleNamespace(action_size=13),
        control_timestep=0.02,
        precompile=lambda _reference: None,
        set_upper_body_standing_reference=lambda _reference: None,
        validate_upper_body_reference=lambda _reference: None,
    )
    preview_calls = []

    def preview(frame, command, reference):
        preview_calls.append(("step", frame, command.copy()))
        return SimpleNamespace(
            action=np.asarray((0.25, -0.5)),
            tilt_radians=0.0,
            reference_sequence=reference.sequence,
            reference_age_ms=reference.age_ms(),
        )

    adapter.step = preview
    adapter.reset = lambda: preview_calls.append(("reset",))
    metadata = SimpleNamespace(command=CommandConfig())
    load_calls = []

    def load_policy(path, **kwargs):
        load_calls.append((path, kwargs))
        return adapter, metadata

    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.load_hardware_policy",
        load_policy,
    )

    run_hardware_locomotion(
        Path("unused"),
        pose_hardware_config(
            allow_git_commit_mismatch=True,
            fastsam_reference_qualification_frames=1,
        ),
        hardware_binding=hardware.binding,
        remote_factory=lambda _config: SimpleNamespace(
            wait_until_ready=lambda _timeout: remote_input(
                service_connected=True,
                controller_connected=False,
                intent_armed=False,
            ),
            poll=lambda: remote_input(
                service_connected=True,
                controller_connected=False,
                intent_armed=False,
            ),
            close=lambda: None,
        ),
        reference_source_factory=lambda *_args: FakeReferences(),
    )

    output = capsys.readouterr().out
    assert load_calls == [
        (
            Path("unused"),
            {
                "allow_git_commit_mismatch": True,
                "upper_body_follow": UpperBodyFollowConfig(),
            },
        ),
    ]
    assert not hardware.control_requests
    assert len(preview_calls) == 2
    assert preview_calls[0][0] == "step"
    assert preview_calls[0][1].positions == pytest.approx((2.014, 2.014))
    assert preview_calls[0][2] == pytest.approx(np.zeros(3))
    assert preview_calls[1] == ("reset",)
    assert "'initial_pose_error_max': 2.014" in output
    assert "'preview_action_abs_max': 0.5" in output
    assert "no lease was acquired" in output


def test_inventory_rejects_coordinate_mismatch(hardware_preflight_factory) -> None:
    binding = hardware_preflight_factory(("left", "right"), (0.0, 0.0)).binding
    group = SimpleNamespace(
        name="lower_body",
        coordinates=("wrong",),
        control_rate_hz=100.0,
        initial_state=binding.GroupState.IDLE,
        motors=(),
    )
    robot = SimpleNamespace(
        group_names=("lower_body",),
        group=lambda name: group,
        inventory=SimpleNamespace(
            imus=(
                SimpleNamespace(
                    name="imu",
                    status=SimpleNamespace(state=binding.DeviceState.ONLINE),
                ),
            )
        ),
    )
    adapter = SimpleNamespace(
        physical_coordinate_names=("expected",),
        control_timestep=0.02,
    )

    with pytest.raises(ValueError, match="coordinate mismatch"):
        validate_hardware_inventory(
            robot,
            adapter,
            HardwareLocomotionConfig(groups=("lower_body",)),
            binding,
        )
