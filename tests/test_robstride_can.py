from pathlib import Path
import os
import struct

from hydra import compose, initialize_config_dir
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset, prepare_asset
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.robstride_can import (
    CAN_FRAME,
    BusSpec,
    CanFrame,
    CommType,
    MotorCommand,
    MotorSpec,
    MotorTargetSpec,
    PARAM_LIMIT_TORQUE,
    PARAM_MECHANICAL_POSITION,
    PARAM_RUN_MODE,
    RobstrideBusEmulator,
    RobstrideMotor,
    build_can_id,
    comm_type,
    data_field,
    float_to_uint,
    load_motor_kinds,
    load_robstride_configs,
    target_id,
    uint_to_float,
)
from bello_mujoco.robstride_sim import (
    BelloRobstrideTransmission,
    configure_robstride_actuators,
)
from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.sim import SimulationPhysicsConfig, load_prepared_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.tools.robstride_runner import (
    RobstrideRunnerConfig,
    TerminalEnterInput,
    add_torso_hold,
    cycle_torso_release_reset,
    default_motor_kind_config,
    default_motor_configs,
    reset_data,
    resolve_motor_config_path,
    validate_runner_motor_coverage,
    validate_config,
)


def motor_spec(
    name: str = "waist_yaw",
    *,
    motor_id: int = 53,
    model: str = "ROBSTRIDE_04",
    direction: int = -1,
) -> MotorSpec:
    kind = next(
        kind
        for kind in load_motor_kinds(default_motor_kind_config())
        if kind.name == model
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
        kind=kind,
        target=target,
        direction=direction,
        run_mode=0,
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
        ("ROBSTRIDE_05", 50.0, 5.5, 500.0, 5.0),
        ("ROBSTRIDE_06", 50.0, 36.0, 5000.0, 100.0),
    ],
)
def test_model_protocol_ranges_match_platform_driver(
    model: str, velocity: float, torque: float, kp: float, kd: float
) -> None:
    limits = next(
        kind.protocol
        for kind in load_motor_kinds(default_motor_kind_config())
        if kind.name == model
    )

    assert limits.velocity == (-velocity, velocity)
    assert limits.torque == (-torque, torque)
    assert limits.kp == (0.0, kp)
    assert limits.kd == (0.0, kd)


@pytest.mark.parametrize(
    ("name", "rated_torque", "peak_torque", "no_load_speed_rpm"),
    [
        ("ROBSTRIDE_00", 5.0, 14.0, 315.0),
        ("ROBSTRIDE_03", 21.0, 60.0, 195.0),
        ("ROBSTRIDE_04", 40.0, 120.0, 200.0),
        ("ROBSTRIDE_05", 1.8, 5.5, 480.0),
        ("ROBSTRIDE_06", 11.0, 36.0, 480.0),
    ],
)
def test_motor_kind_catalog_contains_published_hardware_limits(
    name: str,
    rated_torque: float,
    peak_torque: float,
    no_load_speed_rpm: float,
) -> None:
    kind = next(
        kind
        for kind in load_motor_kinds(default_motor_kind_config())
        if kind.name == name
    )

    assert kind.hardware.rated_torque == rated_torque
    assert kind.hardware.peak_torque == peak_torque
    assert kind.hardware.no_load_speed_rpm == no_load_speed_rpm
    assert max(abs(value) for value in kind.protocol.torque) == peak_torque


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


def test_parameter_write_ack_and_readback_match_platform_driver() -> None:
    spec = motor_spec()
    emulator = RobstrideBusEmulator(BusSpec("can0", "vcan0", 0), (spec,))
    write_payload = struct.pack("<H2xB3x", PARAM_RUN_MODE, 5)
    write = CanFrame(
        build_can_id(CommType.PARAM_WRITE, 0, spec.motor_id), write_payload
    )

    ack = emulator.handle_frame(write)[0]

    assert comm_type(ack.can_id) == CommType.PARAM_WRITE
    assert data_field(ack.can_id) == spec.motor_id
    assert ack.data == write_payload

    read = CanFrame(
        build_can_id(CommType.PARAM_READ, 0, spec.motor_id),
        struct.pack("<H6x", PARAM_RUN_MODE),
    )
    response = emulator.handle_frame(read)[0]
    assert comm_type(response.can_id) == CommType.PARAM_READ
    assert response.data[4] == 5


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


def test_load_robstride_configs_validates_and_preserves_bus_motor_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "actuators.yaml"
    path.write_text(
        """
buses:
  can0:
    interface: can0
    host_id: 7
motors:
  - name: waist_yaw
    id: 53
    bus: can0
    kind: ROBSTRIDE_04
    target:
      type: joint
      name: waist_yaw_joint
    direction: -1
    run_mode: mit
"""
    )

    config = load_robstride_configs(default_motor_kind_config(), (path,))

    assert config.buses == (BusSpec("can0", "can0", 7),)
    assert config.motors[0].motor_id == 53
    assert config.motors[0].direction == -1
    assert config.motors[0].run_mode == 0
    assert config.motors[0].kind.name == "ROBSTRIDE_04"
    assert config.motors[0].target.name == "waist_yaw_joint"


def test_default_motor_configs_are_repository_local_and_complete() -> None:
    paths = default_motor_configs()

    assert [path.name for path in paths] == [
        "lower_body.yaml",
        "upper_body_sphere_hands.yaml",
    ]
    assert all(path.exists() for path in paths)
    assert all(
        path.parent.name == "robstride" and path.parent.parent.name == "configs"
        for path in paths
    )

    config = load_robstride_configs(default_motor_kind_config(), paths)
    assert {kind.name for kind in config.motor_kinds} == {
        "ROBSTRIDE_00",
        "ROBSTRIDE_03",
        "ROBSTRIDE_04",
        "ROBSTRIDE_05",
        "ROBSTRIDE_06",
    }
    assert len(config.motors) == 25
    assert {bus.name for bus in config.buses} == {"can0", "can1"}
    assert all(motor.target.type in {"joint", "tendon"} for motor in config.motors)
    by_name = {motor.name: motor for motor in config.motors}
    assert by_name["left_arm_1"].target.name == "left_shoulder_pitch_joint"
    assert by_name["right_knee_mix_1"].target.name == "right_knee_joint"
    assert by_name["left_ankle_1"].target.name == "left_ankle_motor_1"
    assert "left_arm_7" not in by_name
    assert "right_arm_7" not in by_name


def test_motor_assignment_rejects_unknown_kind(tmp_path: Path) -> None:
    path = tmp_path / "actuators.yaml"
    path.write_text(
        """
buses:
  can0: {interface: can0, host_id: 0}
motors:
  - name: waist_yaw
    id: 53
    bus: can0
    kind: NOT_A_MOTOR
    target: {type: joint, name: waist_yaw_joint}
"""
    )

    with pytest.raises(ValueError, match="unknown kind"):
        load_robstride_configs(default_motor_kind_config(), (path,))


def test_motor_assignment_rejects_incomplete_tendon_coupling(tmp_path: Path) -> None:
    path = tmp_path / "actuators.yaml"
    path.write_text(
        """
buses:
  can0: {interface: can0, host_id: 0}
motors:
  - name: left_ankle_1
    id: 67
    bus: can0
    kind: ROBSTRIDE_06
    target:
      type: tendon
      name: left_ankle_motor_1
      owned_joints: [left_ankle_pitch_joint, left_ankle_roll_joint]
      coupling_group: left_ankle
      coupling_size: 2
"""
    )

    with pytest.raises(ValueError, match="requires 2 motors; got 1"):
        load_robstride_configs(default_motor_kind_config(), (path,))


def compose_runner_config(overrides: list[str] | None = None) -> RobstrideRunnerConfig:
    clear_hydra()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_ROOT / "robstride_runner"),
        ):
            config = compose(config_name="default", overrides=overrides or [])
        return structured_config(RobstrideRunnerConfig, config)
    finally:
        clear_hydra()


def test_default_runner_hydra_config_selects_vcan_imu_and_torso_guide() -> None:
    config = compose_runner_config()

    assert config.asset_config.asset == "full_body"
    assert config.asset_config.collision_mode == "box"
    assert config.asset_config.visual_mesh_face_budget == 10_000
    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.asset_config.imu_config.enabled
    assert config.asset_config.imu_config.body_name == "pelvis_link"
    availability = config.asset_config.joint_availability
    assert availability.waist_yaw
    assert not availability.waist_roll
    assert not availability.waist_pitch
    assert not availability.neck_yaw
    assert not availability.head_pitch
    assert config.motor_kind_config.name == "motor_kinds.yaml"
    assert config.simulation_physics.timestep == pytest.approx(0.0025)
    assert config.feedback_rate_hz == pytest.approx(100.0)
    assert config.imu_serial_rate_hz == pytest.approx(250.0)
    assert config.imu_serial_enabled
    assert config.imu_serial_device == Path("/dev/ttyIMU-mujoco")
    assert config.bus_interfaces == ("can0=vcan0", "can1=vcan1")
    assert [path.name for path in config.motor_configs] == [
        "lower_body.yaml",
        "upper_body_sphere_hands.yaml",
    ]
    assert config.torso_hold
    assert config.torso_drop_limit_m == pytest.approx(0.05)
    assert not config.headless
    assert config.real_time
    assert config.pose_service.enabled
    assert config.pose_service.listen == "0.0.0.0:50052"
    assert config.pose_service.reference_frame == "odom"
    assert config.pose_service.tracked_frame == "pelvis_link"
    assert config.pose_service.publish_rate_hz == pytest.approx(200.0)


def test_direct_runner_default_matches_hil_morphology() -> None:
    config = RobstrideRunnerConfig()

    assert config.asset_config.collision_mode == "box"
    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.simulation_physics.timestep == pytest.approx(0.0025)
    assert config.feedback_rate_hz == pytest.approx(100.0)
    assert config.imu_serial_rate_hz == pytest.approx(250.0)
    assert config.pose_service.listen == "0.0.0.0:50052"


def test_runner_hydra_can_enable_head_joints() -> None:
    config = compose_runner_config(
        [
            "asset_config.joint_availability.neck_yaw=true",
            "asset_config.joint_availability.head_pitch=true",
        ]
    )

    assert config.asset_config.joint_availability.neck_yaw
    assert config.asset_config.joint_availability.head_pitch


def test_runner_optional_joints_require_matching_motor_assignments() -> None:
    config = load_robstride_configs(
        default_motor_kind_config(), default_motor_configs()
    )
    validate_runner_motor_coverage(
        JointAvailabilityConfig(
            waist_yaw=True,
            waist_roll=False,
            waist_pitch=False,
            neck_yaw=False,
            head_pitch=False,
        ),
        config.motors,
    )

    with pytest.raises(ValueError, match="require RobStride motor mappings"):
        validate_runner_motor_coverage(
            JointAvailabilityConfig(
                waist_yaw=True,
                waist_roll=False,
                waist_pitch=False,
                neck_yaw=True,
                head_pitch=True,
            ),
            config.motors,
        )
    with pytest.raises(ValueError, match="fixed by joint availability"):
        validate_runner_motor_coverage(
            JointAvailabilityConfig(
                waist_yaw=False,
                waist_roll=False,
                waist_pitch=False,
                neck_yaw=False,
                head_pitch=False,
            ),
            config.motors,
        )


def test_runner_hydra_rejects_unknown_config_keys() -> None:
    clear_hydra()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(CONFIG_ROOT / "robstride_runner"),
        ):
            raw = compose(config_name="default", overrides=["+backend=mjx"])
        with pytest.raises(ValueError, match="unknown config key.*backend"):
            structured_config(RobstrideRunnerConfig, raw)
    finally:
        clear_hydra()


def test_runner_hydra_can_disable_imu() -> None:
    config = compose_runner_config(
        [
            "asset_config.imu_config.enabled=false",
            "imu_serial_enabled=false",
        ]
    )

    assert not config.asset_config.imu_config.enabled
    validate_config(config)


def test_runner_rejects_serial_output_without_an_imu_sensor() -> None:
    config = compose_runner_config(["asset_config.imu_config.enabled=false"])

    with pytest.raises(ValueError, match="imu_serial_enabled requires"):
        validate_config(config)


def test_runner_hydra_can_disable_ebimu_serial_device() -> None:
    config = compose_runner_config(["imu_serial_enabled=false"])

    assert not config.imu_serial_enabled


def test_runner_resolves_relative_motor_configs_from_bundled_config_root() -> None:
    assert resolve_motor_config_path(Path("robstride/motor_kinds.yaml")) == (
        default_motor_kind_config().resolve()
    )


def test_default_runner_sphere_hands_preserve_all_motor_coordinates() -> None:
    config = compose_runner_config()
    robstride = load_robstride_configs(
        resolve_motor_config_path(config.motor_kind_config),
        tuple(resolve_motor_config_path(path) for path in config.motor_configs),
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
    assert "left_wrist_yaw_joint" not in transmission.mapped_joint_names
    assert "right_wrist_yaw_joint" not in transmission.mapped_joint_names
    for body_name in ("l_end_effector_sphere_link", "r_end_effector_sphere_link"):
        assert (
            mujoco.mj_name2id(
                configured.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            >= 0
        )


def test_runner_can_select_detailed_hands_with_wrist_yaw_motors() -> None:
    config = compose_runner_config(
        [
            "asset_config.end_effector_sphere_mode=none",
            "motor_configs=[robstride/lower_body.yaml,robstride/upper_body.yaml]",
        ]
    )
    robstride = load_robstride_configs(
        resolve_motor_config_path(config.motor_kind_config),
        tuple(resolve_motor_config_path(path) for path in config.motor_configs),
    )
    prepared = prepare_asset(config.asset_config)
    model = load_prepared_bello_model(
        prepared,
        terrain=FLAT_TERRAIN,
        load_mjx=False,
    )
    configured = configure_robstride_actuators(model, robstride.motors)
    transmission = BelloRobstrideTransmission(
        configured,
        tuple(RobstrideMotor(spec) for spec in robstride.motors),
    )

    assert "left_wrist_yaw_joint" in transmission.mapped_joint_names
    assert "right_wrist_yaw_joint" in transmission.mapped_joint_names
    for body_name in ("l_end_effector_sphere_link", "r_end_effector_sphere_link"):
        assert (
            mujoco.mj_name2id(
                configured.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            == -1
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
    slider_range = model.mj_model.jnt_range[torso_hold.support_joint_id]

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


def test_release_timer_requires_an_initial_torso_hold() -> None:
    with pytest.raises(ValueError, match="requires torso_hold"):
        validate_config(
            RobstrideRunnerConfig(torso_hold=False, release_after_seconds=1.0)
        )


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
