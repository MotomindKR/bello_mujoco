import os
import struct

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from bello_mujoco.assets_pipeline import stack_description_path
from bello_mujoco.assets_pipeline.prepare import PreparedAsset, prepare_asset
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.robstride.can import (
    CAN_FRAME,
    BusSpec,
    CanFrame,
    CommType,
    MotorCommand,
    MotorSpec,
    MotorTargetSpec,
    PARAM_LIMIT_TORQUE,
    PARAM_MECHANICAL_POSITION,
    RobstrideBusEmulator,
    RobstrideMotor,
    build_can_id,
    comm_type,
    data_field,
    float_to_uint,
    load_motor_models,
    load_robstride_description,
    load_urdf_motor_targets,
    target_id,
    uint_to_float,
)
from bello_mujoco.robstride.sim import (
    BelloRobstrideTransmission,
    configure_robstride_actuators,
)
from bello_mujoco.sim import SimulationPhysicsConfig, load_prepared_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.robstride.emulator import (
    _CommandLatencyQueue,
    EmulatorConfig,
    TerminalEnterInput,
    add_torso_hold,
    cycle_torso_release_reset,
    default_hardware_description,
    default_motor_model_catalog,
    reset_data,
    validate_config,
    validate_motor_coverage,
)


def motor_spec(
    name: str = "waist_yaw",
    *,
    motor_id: int = 53,
    model: str = "ROBSTRIDE_04",
    direction: int = -1,
) -> MotorSpec:
    motor_model = next(
        candidate
        for candidate in load_motor_models(default_motor_model_catalog())
        if candidate.name == model
    )
    if "_ankle_" in name:
        side = name.split("_", 1)[0]
        motor_index = name.rsplit("_", 1)[1]
        target = MotorTargetSpec(
            type="tendon",
            name=f"{side}_ankle_motor_{motor_index}",
            owned_joints=(
                f"{side}_ankle_pitch_joint",
                f"{side}_ankle_roll_joint",
            ),
            coupling_group=f"{side}_ankle",
            coupling_size=2,
        )
    elif "_knee_mix_" in name:
        side = name.split("_", 1)[0]
        target = MotorTargetSpec(type="joint", name=f"{side}_knee_joint")
    else:
        target = MotorTargetSpec(type="joint", name="waist_yaw_joint")
    return MotorSpec(
        name=name,
        motor_id=motor_id,
        bus="can0",
        model=motor_model,
        target=target,
        direction=direction,
    )


def test_extended_can_frame_round_trip() -> None:
    frame = CanFrame(build_can_id(CommType.MOTOR_ENABLE, 0, 53), bytes(range(8)))

    packed = frame.pack()

    assert len(packed) == CAN_FRAME.size == 16
    assert CanFrame.unpack(packed) == frame


def test_standard_can_frame_is_rejected() -> None:
    standard = CAN_FRAME.pack(0x123, 1, b"x".ljust(8, b"\0"))

    with pytest.raises(ValueError, match="extended CAN"):
        CanFrame.unpack(standard)


def test_can_id_layout_matches_robstride_private_protocol() -> None:
    can_id = build_can_id(CommType.PARAM_READ, 0x12AB, 0x35)

    assert can_id == 0x1112AB35
    assert comm_type(can_id) == CommType.PARAM_READ
    assert data_field(can_id) == 0x12AB
    assert target_id(can_id) == 0x35


@pytest.mark.parametrize(
    ("model", "velocity", "torque", "kp", "kd"),
    [
        ("ROBSTRIDE_00", 33.0, 14.0, 500.0, 5.0),
        ("ROBSTRIDE_03", 20.0, 60.0, 5000.0, 100.0),
        ("ROBSTRIDE_04", 15.0, 120.0, 5000.0, 100.0),
        ("ROBSTRIDE_06", 50.0, 36.0, 5000.0, 100.0),
    ],
)
def test_model_protocol_ranges_match_platform_driver(
    model: str, velocity: float, torque: float, kp: float, kd: float
) -> None:
    limits = next(
        candidate.protocol
        for candidate in load_motor_models(default_motor_model_catalog())
        if candidate.name == model
    )

    assert limits.velocity == (-velocity, velocity)
    assert limits.torque == (-torque, torque)
    assert limits.kp == (0.0, kp)
    assert limits.kd == (0.0, kd)


def test_motor_control_decodes_direction_and_impedance_fields() -> None:
    spec = motor_spec()
    motor = RobstrideMotor(spec)
    limits = motor.limits
    position = 0.4
    velocity = -1.5
    torque = 3.0
    kp = 80.0
    kd = 4.0
    raw_direction = spec.direction
    payload = struct.pack(
        ">HHHH",
        float_to_uint(position * raw_direction, *limits.position),
        float_to_uint(velocity * raw_direction, *limits.velocity),
        float_to_uint(kp, *limits.kp),
        float_to_uint(kd, *limits.kd),
    )
    frame = CanFrame(
        build_can_id(
            CommType.MOTOR_CONTROL,
            float_to_uint(torque * raw_direction, *limits.torque),
            spec.motor_id,
        ),
        payload,
    )

    motor.decode_control(frame)

    def resolution(bounds: tuple[float, float]) -> float:
        return (bounds[1] - bounds[0]) / 65535.0

    assert motor.command.position == pytest.approx(
        position, abs=resolution(limits.position)
    )
    assert motor.command.velocity == pytest.approx(
        velocity, abs=resolution(limits.velocity)
    )
    assert motor.command.torque == pytest.approx(torque, abs=resolution(limits.torque))
    assert motor.command.kp == pytest.approx(kp, abs=resolution(limits.kp))
    assert motor.command.kd == pytest.approx(kd, abs=resolution(limits.kd))


def test_motor_control_returns_type_2_feedback() -> None:
    spec = motor_spec()
    emulator = RobstrideBusEmulator(BusSpec("can0", "vcan0", 0), (spec,))
    frame = CanFrame(
        build_can_id(CommType.MOTOR_CONTROL, 0, spec.motor_id),
        bytes(8),
    )

    responses = emulator.handle_frame(frame)

    assert len(responses) == 1
    assert comm_type(responses[0].can_id) == CommType.MOTOR_FEEDBACK
    assert target_id(responses[0].can_id) == emulator.spec.host_id


def test_enable_stop_and_active_reporting_match_gateway_contract() -> None:
    spec = motor_spec()
    emulator = RobstrideBusEmulator(BusSpec("can0", "vcan0", 0), (spec,))

    enable = CanFrame(build_can_id(CommType.MOTOR_ENABLE, 0, spec.motor_id), bytes(8))
    feedback = emulator.handle_frame(enable)

    assert len(feedback) == 1
    assert comm_type(feedback[0].can_id) == CommType.MOTOR_FEEDBACK
    assert target_id(feedback[0].can_id) == 0
    assert data_field(feedback[0].can_id) & 0xFF == spec.motor_id
    assert data_field(feedback[0].can_id) >> 14 == 2

    active = CanFrame(
        build_can_id(CommType.ACTIVE_REPORT, 0, spec.motor_id),
        bytes((1, 2, 3, 4, 5, 6, 1, 0)),
    )
    emulator.handle_frame(active)
    reports = emulator.active_feedback()
    assert len(reports) == 1
    assert comm_type(reports[0].can_id) == CommType.ACTIVE_REPORT

    stop = CanFrame(build_can_id(CommType.MOTOR_STOP, 0, spec.motor_id), bytes(8))
    feedback = emulator.handle_frame(stop)[0]
    assert data_field(feedback.can_id) >> 14 == 0


def test_command_latency_queue_delays_only_motor_control() -> None:
    queue = _CommandLatencyQueue(0.01)
    control = CanFrame(build_can_id(CommType.MOTOR_CONTROL, 0, 53), bytes(8))
    enable = CanFrame(build_can_id(CommType.MOTOR_ENABLE, 0, 53), bytes(8))

    assert queue.enqueue("can0", control, received_at=4.0)
    assert not queue.enqueue("can0", enable, received_at=4.0)
    assert queue.pop_ready(4.009) == []
    assert queue.pop_ready(4.01) == [("can0", control)]

    queue.enqueue("can0", control, received_at=5.0)
    queue.clear()
    assert queue.pop_ready(6.0) == []


@pytest.mark.parametrize("latency", (-0.001, float("inf"), float("nan")))
def test_emulator_rejects_invalid_command_latency(latency: float) -> None:
    with pytest.raises(
        ValueError,
        match="command_latency_seconds must be finite and nonnegative",
    ):
        validate_config(EmulatorConfig(command_latency_seconds=latency))


def test_emulator_defaults_to_current_firmware_feedback_rate() -> None:
    assert EmulatorConfig().feedback_rate_hz == 100.0


def test_emulator_reset_disables_motors_and_clears_stale_commands() -> None:
    spec = motor_spec()
    emulator = RobstrideBusEmulator(BusSpec("can0", "vcan0", 0), (spec,))
    motor = emulator.motors[spec.motor_id]
    motor.mode_status = 2
    motor.applied_torque = 4.0
    motor.command = MotorCommand(
        position=1.0,
        velocity=2.0,
        torque=3.0,
        kp=80.0,
        kd=4.0,
    )
    motor.active_reporting = True

    emulator.reset_control_state()

    assert not motor.enabled
    assert motor.applied_torque == 0.0
    assert motor.command == MotorCommand()
    assert motor.active_reporting


def test_mechanical_position_read_uses_wire_direction_and_returns_feedback() -> None:
    spec = motor_spec(direction=-1)
    emulator = RobstrideBusEmulator(BusSpec("can0", "vcan0", 0), (spec,))
    emulator.motors[spec.motor_id].mechanical_position = 0.25
    read = CanFrame(
        build_can_id(CommType.PARAM_READ, 0, spec.motor_id),
        struct.pack("<H6x", PARAM_MECHANICAL_POSITION),
    )

    responses = emulator.handle_frame(read)

    assert len(responses) == 2
    assert struct.unpack_from("<f", responses[0].data, 4)[0] == pytest.approx(-0.25)
    assert comm_type(responses[1].can_id) == CommType.MOTOR_FEEDBACK


def test_float_encoding_clamps_and_round_trips() -> None:
    assert float_to_uint(-20.0, -10.0, 10.0) == 0
    assert float_to_uint(20.0, -10.0, 10.0) == 65535
    midpoint = float_to_uint(0.0, -10.0, 10.0)
    assert uint_to_float(midpoint, -10.0, 10.0) == pytest.approx(0.0, abs=0.001)


def test_canonical_hardware_description_drives_emulator_inventory() -> None:
    targets = load_urdf_motor_targets(
        stack_description_path("bello/full_body/urdf/bello_full_body.urdf")
    )
    config = load_robstride_description(
        default_hardware_description(),
        default_motor_model_catalog(),
        {"can0": "vcan0", "can1": "vcan1"},
    )
    motors = {motor.name: motor for motor in config.motors}

    assert config.buses == (
        BusSpec("can0", "vcan0", 0),
        BusSpec("can1", "vcan1", 0),
    )
    assert motors["waist_yaw"].bus == "can1"
    assert motors["waist_yaw"].target == MotorTargetSpec(
        type="joint", name="waist_yaw_joint"
    )
    assert motors["left_ankle_1"].target == MotorTargetSpec(
        type="tendon",
        name="left_ankle_motor_1",
        owned_joints=("left_ankle_roll_joint", "left_ankle_pitch_joint"),
        coupling_group="left_ankle_differential",
        coupling_size=2,
    )
    assert "left_ankle_pitch_joint" not in targets
    assert "left_ankle_roll_joint" not in targets
    assert "left_arm_7" not in motors
    assert "right_arm_7" not in motors


def test_hardware_description_requires_existing_robot_description(tmp_path) -> None:
    description = OmegaConf.load(default_hardware_description())
    description.robot_description = "missing.urdf"
    hardware_path = tmp_path / "hardware.yaml"
    OmegaConf.save(description, hardware_path)

    with pytest.raises(ValueError, match="must reference an existing file"):
        load_robstride_description(
            hardware_path,
            default_motor_model_catalog(),
            {"can0": "vcan0", "can1": "vcan1"},
        )


def test_emulator_optional_joints_require_matching_motor_assignments() -> None:
    config = load_robstride_description(
        default_hardware_description(),
        default_motor_model_catalog(),
        {"can0": "vcan0", "can1": "vcan1"},
    )
    validate_motor_coverage(
        JointAvailabilityConfig(waist_yaw=True),
        config.motors,
    )

    with pytest.raises(ValueError, match="require RobStride motor mappings"):
        validate_motor_coverage(
            JointAvailabilityConfig(waist_yaw=True),
            tuple(
                motor
                for motor in config.motors
                if not (
                    motor.target.type == "joint"
                    and motor.target.name == "waist_yaw_joint"
                )
            ),
        )
    with pytest.raises(ValueError, match="fixed by joint availability"):
        validate_motor_coverage(
            JointAvailabilityConfig(waist_yaw=False),
            config.motors,
        )


def test_default_emulator_sphere_hands_preserve_all_motor_coordinates() -> None:
    config = EmulatorConfig()
    robstride = load_robstride_description(
        default_hardware_description(),
        default_motor_model_catalog(),
        {"can0": "vcan0", "can1": "vcan1"},
    )
    prepared = prepare_asset(config.asset_config)
    model = load_prepared_bello_model(
        prepared,
        terrain=FLAT_TERRAIN,
        simulation_physics=config.simulation_physics,
        load_mjx=False,
    )
    configured = configure_robstride_actuators(model, robstride.motors)
    transmission = BelloRobstrideTransmission(
        configured,
        tuple(RobstrideMotor(spec) for spec in robstride.motors),
    )

    assert len(transmission.actuators) == len(robstride.motors)
    assert configured.mj_model.opt.timestep == pytest.approx(0.0025)
    assert configured.mj_model.nu == model.mj_model.nu + 2 * len(robstride.motors)
    for side in ("left", "right"):
        for motor_index in (1, 2):
            assert (
                mujoco.mj_name2id(
                    configured.mj_model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{side}_ankle_motor_{motor_index}_pos",
                )
                == -1
            )
    assert "left_wrist_pitch_joint" in transmission.mapped_joint_names
    assert "right_wrist_pitch_joint" in transmission.mapped_joint_names
    assert "left_end_effector_sphere_joint" not in transmission.mapped_joint_names
    assert "right_end_effector_sphere_joint" not in transmission.mapped_joint_names
    for body_name in ("l_end_effector_sphere_link", "r_end_effector_sphere_link"):
        assert (
            mujoco.mj_name2id(
                configured.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            >= 0
        )


def test_torso_guide_allows_five_centimeter_drop_without_sideways_motion(
    full_body_asset: PreparedAsset,
) -> None:
    original = load_prepared_bello_model(
        full_body_asset,
        simulation_physics=SimulationPhysicsConfig(timestep=0.0025),
        load_mjx=False,
    )
    model, torso_hold = add_torso_hold(original)
    assert model.mj_model.opt.timestep == pytest.approx(0.0025)
    data = mujoco.MjData(model.mj_model)
    reset_data(model, data)
    torso_hold.initialize(model.mj_model, data)
    initial_position = np.asarray(data.xpos[torso_hold.body_id]).copy()
    slider_range = model.mj_model.joint("bello_emulator_torso_slider").range

    assert slider_range == pytest.approx((-0.05, 0.0))

    for _ in range(100):
        mujoco.mj_step(model.mj_model, data)

    assert data.eq_active[torso_hold.equality_id]
    displacement = data.xpos[torso_hold.body_id] - initial_position
    assert np.linalg.norm(displacement[:2]) < 0.001
    assert -0.052 < displacement[2] < -0.045
    released, reset_performed = cycle_torso_release_reset(
        model, data, torso_hold, torso_released=False
    )
    assert released
    assert not reset_performed
    assert not data.eq_active[torso_hold.equality_id]
    assert not torso_hold.release(data)

    data.qpos[:] += 0.1
    data.time = 12.0
    released, reset_performed = cycle_torso_release_reset(
        model, data, torso_hold, torso_released=released
    )

    assert not released
    assert reset_performed
    assert data.time == 0.0
    assert data.qpos == pytest.approx(model.default_qpos)
    assert data.eq_active[torso_hold.equality_id]


def test_terminal_enter_is_detected_without_blocking() -> None:
    read_fd, write_fd = os.pipe()
    with os.fdopen(read_fd) as reader, os.fdopen(write_fd, "w") as writer:
        control = TerminalEnterInput(reader, enabled=True)

        assert not control.poll()
        writer.write("\n")
        writer.flush()
        assert control.poll()
        assert not control.poll()


def test_bello_transmission_maps_knee_and_differential_ankle(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, load_mjx=False)
    specs = (
        motor_spec("left_knee_mix_1", motor_id=64, direction=1),
        motor_spec("left_ankle_1", motor_id=67, model="ROBSTRIDE_06"),
        motor_spec("left_ankle_2", motor_id=66, model="ROBSTRIDE_06"),
    )
    motors = tuple(RobstrideMotor(spec) for spec in specs)
    model = configure_robstride_actuators(model, specs)
    transmission = BelloRobstrideTransmission(model, motors)
    data = mujoco.MjData(model.mj_model)
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    data.qpos[transmission.bindings["left_knee_joint"].qpos_index] = 0.4
    data.qpos[transmission.bindings["left_ankle_pitch_joint"].qpos_index] = 0.2
    data.qpos[transmission.bindings["left_ankle_roll_joint"].qpos_index] = 0.1
    mujoco.mj_forward(model.mj_model, data)

    transmission.update_motor_states(data)

    by_name = {motor.spec.name: motor for motor in motors}
    assert by_name["left_knee_mix_1"].mechanical_position == pytest.approx(0.4)
    assert by_name["left_ankle_1"].mechanical_position == pytest.approx(-0.1)
    assert by_name["left_ankle_2"].mechanical_position == pytest.approx(0.3)

    ankle_1 = transmission.tendons["left_ankle_1"]
    ankle_2 = transmission.tendons["left_ankle_2"]
    pitch_index = transmission.bindings["left_ankle_pitch_joint"].qvel_index
    roll_index = transmission.bindings["left_ankle_roll_joint"].qvel_index

    def tendon_row(tendon_id: int) -> np.ndarray:
        row = np.zeros(model.mj_model.nv)
        row_start = int(model.mj_model.ten_J_rowadr[tendon_id])
        row_stop = row_start + int(model.mj_model.ten_J_rownnz[tendon_id])
        columns = model.mj_model.ten_J_colind[row_start:row_stop]
        row[columns] = data.ten_J[row_start:row_stop]
        return row

    ankle_1_jacobian = tendon_row(ankle_1.tendon_id)
    ankle_2_jacobian = tendon_row(ankle_2.tendon_id)
    assert ankle_1_jacobian[pitch_index] == pytest.approx(-1.0)
    assert ankle_1_jacobian[roll_index] == pytest.approx(1.0)
    assert ankle_2_jacobian[pitch_index] == pytest.approx(1.0)
    assert ankle_2_jacobian[roll_index] == pytest.approx(1.0)

    for motor in motors:
        motor.mode_status = 2
    by_name["left_knee_mix_1"].command = MotorCommand(torque=3.0)
    by_name["left_ankle_1"].command = MotorCommand(torque=1.0)
    by_name["left_ankle_2"].command = MotorCommand(torque=2.0)
    transmission.apply_motor_controls(data)
    mujoco.mj_forward(model.mj_model, data)
    transmission.update_motor_states(data)

    assert np.all(data.qfrc_applied == 0.0)
    assert data.qfrc_actuator[
        transmission.bindings["left_knee_joint"].qvel_index
    ] == pytest.approx(3.0)
    assert data.qfrc_actuator[
        transmission.bindings["left_ankle_pitch_joint"].qvel_index
    ] == pytest.approx(1.0)
    assert data.qfrc_actuator[
        transmission.bindings["left_ankle_roll_joint"].qvel_index
    ] == pytest.approx(3.0)


def test_ankle_motors_require_configured_native_actuator_banks(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, load_mjx=False)
    motors = tuple(
        RobstrideMotor(spec)
        for spec in (
            motor_spec("left_ankle_1", motor_id=67, model="ROBSTRIDE_06"),
            motor_spec("left_ankle_2", motor_id=66, model="ROBSTRIDE_06"),
        )
    )

    with pytest.raises(ValueError, match="missing its native position actuator"):
        BelloRobstrideTransmission(model, motors)


def test_native_actuator_triple_uses_one_common_dynamic_torque_limit(
    full_body_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(full_body_asset, load_mjx=False)
    motor = RobstrideMotor(motor_spec())
    original_nu = model.mj_model.nu
    other_actuator_name = "left_hip_pitch_joint_pos"
    other_id = mujoco.mj_name2id(
        model.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, other_actuator_name
    )
    other_gain = float(model.mj_model.actuator_gainprm[other_id, 0])
    model = configure_robstride_actuators(model, (motor.spec,))
    transmission = BelloRobstrideTransmission(model, (motor,))

    assert model.mj_model.nu == original_nu + 2
    assert (
        mujoco.mj_name2id(
            model.mj_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "waist_yaw_joint_pos",
        )
        < 0
    )
    other_id = mujoco.mj_name2id(
        model.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, other_actuator_name
    )
    assert model.mj_model.actuator_gainprm[other_id, 0] == pytest.approx(other_gain)

    actuator = transmission.actuators[motor.spec.name]
    assert not any(
        model.mj_model.actuator_forcelimited[actuator_id]
        for actuator_id in (
            actuator.position_id,
            actuator.velocity_id,
            actuator.feedforward_id,
        )
    )

    data = mujoco.MjData(model.mj_model)
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    mujoco.mj_forward(model.mj_model, data)
    motor.mode_status = 2
    motor.params[PARAM_LIMIT_TORQUE] = 7.0
    motor.command = MotorCommand(
        position=1.0, velocity=2.0, torque=3.0, kp=80.0, kd=2.0
    )

    transmission.apply_motor_controls(data)
    mujoco.mj_forward(model.mj_model, data)
    transmission.update_motor_states(data)

    assert data.ctrl[actuator.position_id] == pytest.approx(1.0)
    assert data.ctrl[actuator.velocity_id] == pytest.approx(2.0)
    assert data.ctrl[actuator.feedforward_id] == pytest.approx(3.0)
    assert model.mj_model.actuator_gainprm[actuator.position_id, 0] == 80.0
    assert model.mj_model.actuator_gainprm[actuator.velocity_id, 0] == 2.0
    joint_id = mujoco.mj_name2id(
        model.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "waist_yaw_joint"
    )
    assert model.mj_model.jnt_actfrcrange[joint_id] == pytest.approx((-7.0, 7.0))
    qvel_index = transmission.bindings["waist_yaw_joint"].qvel_index
    assert data.qfrc_actuator[qvel_index] == pytest.approx(7.0)
    assert motor.applied_torque == pytest.approx(7.0)

    motor.mode_status = 0
    transmission.apply_motor_controls(data)
    assert model.mj_model.actuator_gainprm[actuator.position_id, 0] == 0.0
    assert model.mj_model.actuator_gainprm[actuator.velocity_id, 0] == 0.0
    assert data.ctrl[actuator.feedforward_id] == 0.0
