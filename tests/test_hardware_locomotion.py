from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from enum import Enum
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import jax

from bello_mujoco.assets_pipeline.prepare import prepare_asset
from bello_mujoco.training.checkpoint import CheckpointNormalizationState
from bello_mujoco.envs.locomotion import BelloLocomotionEnv
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware.locomotion import (
    ARM_CONFIRMATION,
    HardwareFrame,
    HardwareLocomotionConfig,
    HardwarePolicyAdapter,
    actuator_coordinate_names,
    actuator_impedance_gains,
    command_ranges,
    hardware_command,
    limit_target_step,
    projected_gravity_from_xyzw,
    run_hardware_locomotion,
    send_targets,
    specific_force_from_gravity_removed_acceleration,
    start_hardware_controls,
    stop_hardware_controls,
    tilt_from_xyzw,
    validate_command,
    validate_hardware_checkpoint,
    validate_hardware_config,
    validate_hardware_inventory,
    wait_for_policy_start,
)
from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.policy import init_actor_critic, policy_observation_size
from bello_mujoco.training.locomotion import TrainingRunConfig, checkpoint_metadata
from bello_mujoco.training.ppo import (
    init_observation_norm_state,
    init_value_norm_state,
)
from bello_mujoco.evaluation.locomotion import GamepadConfig, GamepadInput


def compose_hardware_training_config() -> TrainingRunConfig:
    return locomotion_training_profile("hardware_locomotion")


def compose_hardware_runner_config() -> SimpleNamespace:
    return SimpleNamespace(hardware=HardwareLocomotionConfig())


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

    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.asset_config.target_total_mass_kg == pytest.approx(60.0)
    assert config.asset_config.joint_availability.enabled_joint_names == {
        "waist_yaw_joint"
    }
    assert config.command_config.standing_probability == pytest.approx(0.1)
    assert config.command_config.minimum_moving_linear_speed == pytest.approx(0.2)
    assert config.command_config.minimum_moving_yaw_speed == pytest.approx(0.2)
    assert config.config.gamma == pytest.approx(0.995)
    assert config.config.gae_lambda == pytest.approx(0.98)
    weights = config.reward_config.weights
    assert weights.track_ang_vel_z == pytest.approx(1.0)
    assert weights.joint_pose == pytest.approx(-0.1)
    assert weights.lin_vel_z == pytest.approx(-2.0)
    assert weights.flight == pytest.approx(-2.0)
    assert weights.touchdown_air_time == pytest.approx(0.5)
    assert weights.torque_l1 == pytest.approx(-1.2)
    assert weights.torque_peak == pytest.approx(-12.0)
    assert weights.stand_foot_motion == pytest.approx(-1.0)
    assert weights.base_height == pytest.approx(-10.0)
    assert weights.flat_orientation == pytest.approx(-2.0)
    assert weights.termination == pytest.approx(-100.0)
    assert config.video_config.command == (0.0, 0.0, 0.0)
    assert hardware_env.action_size == 13
    assert hardware_env.model.mj_model.nu == 25
    assert np.sum(hardware_env.model.mj_model.body_mass) == pytest.approx(60.0)

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


def test_hardware_adapter_reconstructs_actor_frame_and_targets(
    hardware_env: BelloLocomotionEnv,
) -> None:
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
    adapter = HardwarePolicyAdapter(hardware_env, policy, metadata, normalization)
    nominal = adapter.nominal_physical_target
    frame = HardwareFrame(
        positions=nominal,
        velocities=np.zeros_like(nominal),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        group_states=("IDLE", "IDLE"),
    )
    command = np.asarray((0.2, 0.0, 0.0), dtype=np.float32)

    result = adapter.step(frame, command)

    assert result.observation.shape == (51,)
    assert result.observation[:3] == pytest.approx((0.0, 0.0, 0.980665))
    assert result.observation[3:6] == pytest.approx(np.zeros(3))
    assert result.observation[6:9] == pytest.approx((0.0, 0.0, -1.0))
    assert result.observation[9:12] == pytest.approx(command)
    assert result.action.shape == (13,)
    assert result.physical_target.shape == (25,)
    assert np.all(np.isfinite(result.physical_target))
    assert result.physical_target == pytest.approx(
        np.asarray(hardware_env.action_to_ctrl(result.action, np.zeros(25))),
        abs=1.0e-6,
    )


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


def test_hardware_runner_defaults_to_read_only_and_requires_confirmation() -> None:
    config = compose_hardware_runner_config()

    assert not config.hardware.arm
    assert config.hardware.command == (0.0, 0.0, 0.0)
    assert config.hardware.gamepad.enabled
    assert config.hardware.wait_for_start_button
    validate_hardware_config(config.hardware)

    with pytest.raises(ValueError, match="operator_confirmation"):
        validate_hardware_config(
            HardwareLocomotionConfig(arm=True, operator_confirmation="yes")
        )
    validate_hardware_config(
        HardwareLocomotionConfig(
            arm=True,
            operator_confirmation=ARM_CONFIRMATION,
        )
    )
    with pytest.raises(ValueError, match="must not exceed"):
        validate_hardware_config(HardwareLocomotionConfig(kp_scale=1.01))
    with pytest.raises(ValueError, match="controller_index"):
        validate_hardware_config(
            HardwareLocomotionConfig(
                gamepad=replace(GamepadConfig(), controller_index=-1)
            )
        )
    with pytest.raises(ValueError, match="wait_for_start_button"):
        validate_hardware_config(
            HardwareLocomotionConfig(gamepad=replace(GamepadConfig(), enabled=False))
        )


def test_hardware_waits_for_fresh_start_button_after_ramp(monkeypatch) -> None:
    inputs = iter(
        (
            GamepadInput(np.zeros(3, dtype=np.float32), connected=False),
            GamepadInput(np.zeros(3, dtype=np.float32), connected=True),
            GamepadInput(
                np.zeros(3, dtype=np.float32),
                connected=True,
                reset_requested=True,
            ),
        )
    )
    sent = []
    validated = []
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.send_targets",
        lambda *args: sent.append(np.asarray(args[3]).copy()),
    )
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.pace_hardware_loop",
        lambda deadline, timestep, config: deadline,
    )
    frame = SimpleNamespace()
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.wait_for_hardware_frame",
        lambda *args: frame,
    )
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.validate_hardware_frame",
        lambda value, config, expected_group_state: validated.append(
            (value, expected_group_state)
        ),
    )
    adapter = SimpleNamespace(
        control_timestep=0.02,
        physical_coordinate_names=("motor",),
    )
    binding = SimpleNamespace(StateUnavailableError=RuntimeError)
    gamepad = SimpleNamespace(poll=lambda: next(inputs))
    target = np.asarray((0.25,), dtype=np.float64)

    assert wait_for_policy_start(
        binding,
        {},
        {},
        object(),
        target,
        adapter,
        HardwareLocomotionConfig(groups=("group",)),
        gamepad,
        SimpleNamespace(),
        [],
    )
    assert len(sent) == 2
    assert sent == pytest.approx((target, target))
    assert validated == [(frame, "RUNNING"), (frame, "RUNNING")]


def test_hardware_command_and_target_step_limits() -> None:
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

    limited = limit_target_step(np.asarray((0.0, 0.0)), np.asarray((0.2, -0.1)), 0.05)
    assert limited == pytest.approx((0.05, -0.05))


def test_hardware_gamepad_commands_fail_closed_and_use_checkpoint_bounds() -> None:
    metadata = SimpleNamespace(
        command=CommandConfig(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.2, 0.3),
            ang_vel_yaw=(-0.4, 0.6),
        )
    )
    config = HardwareLocomotionConfig(command=(0.1, 0.1, 0.1))

    assert command_ranges(metadata) == ((-0.5, 1.0), (-0.2, 0.3), (-0.4, 0.6))
    assert hardware_command(
        config,
        metadata,
        GamepadInput(np.zeros(3, dtype=np.float32), connected=False),
    ) == pytest.approx(np.zeros(3))
    assert hardware_command(
        config,
        metadata,
        GamepadInput(np.asarray((0.8, -0.1, 0.4)), connected=True),
    ) == pytest.approx((0.8, -0.1, 0.4))


def test_hardware_groups_start_concurrently() -> None:
    barrier = threading.Barrier(2, timeout=1.0)
    started = []
    sent = []

    class Control:
        def __init__(self, name: str) -> None:
            self.name = name

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
            barrier.wait()
            stopped.append(self.name)

    names = ("lower_body", "upper_body")
    controls = {name: Control(name) for name in names}
    stopped = []
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
        )
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
        pending_receipts = []
        send_targets(
            binding,
            controls,
            groups,
            np.asarray((0.0, 1.0)),
            adapter,
            HardwareLocomotionConfig(groups=names),
            executor,
            pending_receipts,
        )
        assert len(pending_receipts) == 2
        stop_hardware_controls(
            controls,
            HardwareLocomotionConfig(groups=names),
            executor,
        )

    assert set(started) == set(names)
    assert set(sent) == set(names)
    assert set(stopped) == set(names)


class FakeGroupState(Enum):
    IDLE = 1


class FakeDeviceState(Enum):
    ONLINE = 1


class FakeUnavailableError(Exception):
    pass


def test_read_only_preflight_never_acquires_control(monkeypatch, capsys) -> None:
    coordinates = ("left_hip_pitch_joint", "right_hip_pitch_joint")
    groups = {
        "lower_body": SimpleNamespace(
            name="lower_body",
            coordinates=(coordinates[0],),
            control_rate_hz=100.0,
            initial_state=FakeGroupState.IDLE,
            motors=(
                SimpleNamespace(
                    name="left",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            ),
        ),
        "upper_body": SimpleNamespace(
            name="upper_body",
            coordinates=(coordinates[1],),
            control_rate_hz=100.0,
            initial_state=FakeGroupState.IDLE,
            motors=(
                SimpleNamespace(
                    name="right",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            ),
        ),
    }

    def vector(*values):
        return SimpleNamespace(x=values[0], y=values[1], z=values[2])

    states = SimpleNamespace(
        coordinate=lambda name: SimpleNamespace(
            position=2.014, velocity=0.0, feedback_age_ms=0
        ),
        imu=lambda name: SimpleNamespace(
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            angular_velocity=vector(0.0, 0.0, 0.0),
            linear_acceleration=vector(0.0, 0.0, 0.0),
            sample_age_ms=0,
        ),
        group=lambda name: SimpleNamespace(state=SimpleNamespace(name="IDLE")),
    )
    robot = SimpleNamespace(
        group_names=tuple(groups),
        group=lambda name: groups[name],
        inventory=SimpleNamespace(
            imus=(
                SimpleNamespace(
                    name="imu",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                ),
            )
        ),
        states=lambda **kwargs: nullcontext(states),
    )
    binding = SimpleNamespace(
        Robot=SimpleNamespace(connect=lambda *args, **kwargs: nullcontext(robot)),
        GroupState=FakeGroupState,
        DeviceState=FakeDeviceState,
        StateUnavailableError=FakeUnavailableError,
    )
    adapter = SimpleNamespace(
        physical_coordinate_names=coordinates,
        nominal_physical_target=np.zeros(2),
        env=SimpleNamespace(action_size=13),
        control_timestep=0.02,
        precompile=lambda: None,
    )
    metadata = SimpleNamespace(command=CommandConfig())
    monkeypatch.setattr(
        "bello_mujoco.hardware.locomotion.load_hardware_policy",
        lambda path: (adapter, metadata),
    )

    run_hardware_locomotion(
        Path("unused"),
        HardwareLocomotionConfig(),
        hardware_binding=binding,
        gamepad_factory=lambda config, ranges: SimpleNamespace(
            poll=lambda: GamepadInput(np.zeros(3, dtype=np.float32), connected=False),
            close=lambda: None,
        ),
    )

    output = capsys.readouterr().out
    assert "'initial_pose_error_max': 2.014" in output
    assert "no lease was acquired" in output


def test_inventory_rejects_coordinate_mismatch() -> None:
    binding = SimpleNamespace(
        GroupState=FakeGroupState,
        DeviceState=FakeDeviceState,
    )
    group = SimpleNamespace(
        name="lower_body",
        coordinates=("wrong",),
        control_rate_hz=100.0,
        initial_state=FakeGroupState.IDLE,
        motors=(),
    )
    robot = SimpleNamespace(
        group_names=("lower_body",),
        group=lambda name: group,
        inventory=SimpleNamespace(
            imus=(
                SimpleNamespace(
                    name="imu",
                    status=SimpleNamespace(state=FakeDeviceState.ONLINE),
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
