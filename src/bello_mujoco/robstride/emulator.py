"""Emulate Bello RobStride devices with real-time native MuJoCo."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import select
import sys
import time
from typing import TextIO

import mujoco
import numpy as np

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    canonical_physical_asset_config,
    prepare_asset,
)
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.hardware.ebimu import EbimuSerialEmulator
from bello_mujoco.imu import (
    ImuNoiseConfig,
    ImuNoiseInjector,
    MujocoImu,
    validate_imu_noise_config,
)
from bello_mujoco.hardware.pose import PoseServiceConfig, PoseServiceServer
from bello_mujoco.robstride.can import (
    CanFrame,
    CommType,
    MotorSpec,
    RobstrideBusEmulator,
    SocketCanEndpoint,
    comm_type,
    load_robstride_description,
)
from bello_mujoco.robstride.sim import (
    BelloRobstrideTransmission,
    configure_robstride_actuators,
)
from bello_mujoco.robot import AssetName, robot_definition
from bello_mujoco.sim import (
    BelloModel,
    SimulationPhysicsConfig,
    load_prepared_bello_model,
    preserve_runtime_physics,
)
from bello_mujoco.terrain import FLAT_TERRAIN


def default_hardware_description(asset: AssetName) -> Path:
    return robot_definition(asset).hardware_description_path(robot_descriptions_root())


def default_motor_model_catalog(asset: AssetName) -> Path:
    return robot_definition(asset).motor_model_catalog_path(robot_descriptions_root())


def default_emulator_asset_config(asset: AssetName = "bello") -> AssetPrepConfig:
    return canonical_physical_asset_config(asset=asset, imu_enabled=True)


@dataclass(frozen=True)
class EmulatorConfig:
    """Runtime settings for the RobStride/MuJoCo hardware emulator."""

    bus_interfaces: tuple[str, ...] = ("can0=vcan0", "can1=vcan1")
    asset_config: AssetPrepConfig = field(default_factory=default_emulator_asset_config)
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig(
        timestep=0.0025
    )
    command_latency_seconds: float = 0.01
    motor_kp_scale: float = 1.0
    motor_kd_scale: float = 1.0
    video_path: Path | None = None
    feedback_rate_hz: float = 100.0
    temperature_c: float = 25.0
    imu_serial_rate_hz: float = 200.0
    imu_serial_device: Path = Path("/dev/ttyIMU-mujoco")
    imu_noise: ImuNoiseConfig = ImuNoiseConfig(
        enabled=True,
        orientation_std_radians=0.005,
        linear_acceleration_std_mps2=0.1,
        seed=0,
    )
    torso_drop_limit_m: float = 0.05
    pose_service: PoseServiceConfig = PoseServiceConfig()


@dataclass(frozen=True)
class TorsoHold:
    """Runtime handle for the torso's world-vertical guide constraint."""

    equality_id: int
    body_id: int
    support_body_id: int
    drop_limit_m: float

    def initialize(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.eq_active[self.equality_id] = False
        mujoco.mj_forward(model, data)

        position = np.asarray(data.xpos[self.body_id]).copy()
        quaternion = np.asarray(data.xquat[self.body_id]).copy()
        rotation = np.asarray(data.xmat[self.body_id]).reshape(3, 3)
        support_position = np.asarray(data.xpos[self.support_body_id]).copy()
        support_quaternion = np.asarray(data.xquat[self.support_body_id]).copy()
        relative_position = rotation.T @ (support_position - position)
        inverse_quaternion = quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))
        relative_quaternion = np.empty(4)
        mujoco.mju_mulQuat(relative_quaternion, inverse_quaternion, support_quaternion)

        model.eq_data[self.equality_id] = np.asarray(
            (
                0.0,
                0.0,
                0.0,
                *relative_position,
                *relative_quaternion,
                1.0,
            )
        )
        data.eq_active[self.equality_id] = True
        mujoco.mj_forward(model, data)

    def release(self, data: mujoco.MjData) -> bool:
        if not data.eq_active[self.equality_id]:
            return False
        data.eq_active[self.equality_id] = False
        return True


@dataclass
class EmulatorVideoCapture:
    """Collect free-torso states at video rate without slowing the control loop."""

    fps: int = 30
    qpos_frames: list[np.ndarray] = field(default_factory=list)
    next_frame_time: float | None = None

    def sample(self, data: mujoco.MjData, *, active: bool) -> None:
        if not active:
            self.next_frame_time = None
            return
        period = 1.0 / self.fps
        if self.next_frame_time is None:
            self.next_frame_time = float(data.time)
        if float(data.time) + 1.0e-12 < self.next_frame_time:
            return
        self.qpos_frames.append(np.asarray(data.qpos).copy())
        self.next_frame_time += period


@dataclass
class _CommandLatencyQueue:
    delay_seconds: float
    pending: deque[tuple[float, str, CanFrame]] = field(default_factory=deque)

    def enqueue(
        self,
        bus_name: str,
        frame: CanFrame,
        *,
        received_at: float,
    ) -> bool:
        if (
            self.delay_seconds == 0.0
            or comm_type(frame.can_id) != CommType.MOTOR_CONTROL
        ):
            return False
        self.pending.append((received_at + self.delay_seconds, bus_name, frame))
        return True

    def pop_ready(self, now: float) -> list[tuple[str, CanFrame]]:
        ready = []
        while self.pending and self.pending[0][0] <= now:
            _, bus_name, frame = self.pending.popleft()
            ready.append((bus_name, frame))
        return ready

    def clear(self) -> None:
        self.pending.clear()


@dataclass(frozen=True)
class _EmulatorResources:
    endpoints: dict[str, SocketCanEndpoint]
    imu_serial: EbimuSerialEmulator | None
    pose_server: PoseServiceServer | None
    pose_body_id: int


class TerminalEnterInput:
    """Nonblocking Enter detection for an interactive terminal."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        enabled: bool | None = None,
    ) -> None:
        self._stream = sys.stdin if stream is None else stream
        self._enabled = self._stream.isatty() if enabled is None else enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def poll(self) -> bool:
        if not self._enabled:
            return False
        try:
            readable, _, _ = select.select((self._stream,), (), (), 0.0)
        except (OSError, ValueError):
            self._enabled = False
            return False
        if not readable:
            return False
        if self._stream.readline() == "":
            self._enabled = False
            return False
        return True


def run_emulator(config: EmulatorConfig = EmulatorConfig()) -> None:
    validate_config(config)
    overrides = parse_bus_interfaces(config.bus_interfaces)
    robstride = load_robstride_description(
        default_hardware_description(config.asset_config.asset),
        default_motor_model_catalog(config.asset_config.asset),
        overrides,
    )
    validate_motor_coverage(
        config.asset_config.asset,
        config.asset_config.joint_availability,
        robstride.motors,
    )

    prepared = prepare_asset(config.asset_config)
    model = load_prepared_bello_model(
        prepared,
        terrain=FLAT_TERRAIN,
        simulation_physics=config.simulation_physics,
        actuator_config=config.asset_config.actuator_config,
        use_viewer_model=False,
        load_mjx=False,
    )
    model = configure_robstride_actuators(model, robstride.motors)
    model, torso_hold = add_torso_hold(model, config.torso_drop_limit_m)
    emulators = {
        bus.name: RobstrideBusEmulator(
            bus,
            tuple(motor for motor in robstride.motors if motor.bus == bus.name),
        )
        for bus in robstride.buses
    }
    for emulator in emulators.values():
        for motor in emulator.motors.values():
            motor.temperature = config.temperature_c

    transmission = BelloRobstrideTransmission(
        model,
        tuple(
            motor
            for emulator in emulators.values()
            for motor in emulator.motors.values()
        ),
    )
    run_simulation(model, emulators, transmission, config, torso_hold)


@contextmanager
def _open_emulator_resources(
    model: BelloModel,
    data: mujoco.MjData,
    emulators: dict[str, RobstrideBusEmulator],
    config: EmulatorConfig,
    imu_sensor: MujocoImu | None,
) -> Iterator[_EmulatorResources]:
    with ExitStack() as stack:
        pose_server = None
        pose_body_id = -1
        if config.pose_service.enabled:
            pose_body_id = mujoco.mj_name2id(
                model.mj_model,
                mujoco.mjtObj.mjOBJ_BODY,
                config.pose_service.tracked_frame,
            )
            if pose_body_id < 0:
                raise ValueError(
                    "pose_service.tracked_frame must name a MuJoCo body; missing "
                    f"{config.pose_service.tracked_frame!r}"
                )
            pose_server = stack.enter_context(PoseServiceServer(config.pose_service))
            publish_body_pose(pose_server, model.mj_model, data, pose_body_id)

        endpoints = {}
        for bus_name, emulator in emulators.items():
            try:
                endpoints[bus_name] = stack.enter_context(
                    SocketCanEndpoint(emulator.spec.interface)
                )
            except OSError as error:
                raise RuntimeError(
                    f"cannot open {emulator.spec.interface!r} for logical bus "
                    f"{bus_name!r}; create and bring up the VCAN interface first"
                ) from error

        imu_serial = (
            stack.enter_context(
                EbimuSerialEmulator(
                    config.imu_serial_device,
                    output_rate_hz=config.imu_serial_rate_hz,
                    temperature_c=config.temperature_c,
                )
            )
            if imu_sensor is not None
            else None
        )
        yield _EmulatorResources(
            endpoints=endpoints,
            imu_serial=imu_serial,
            pose_server=pose_server,
            pose_body_id=pose_body_id,
        )


def run_simulation(
    model: BelloModel,
    emulators: dict[str, RobstrideBusEmulator],
    transmission: BelloRobstrideTransmission,
    config: EmulatorConfig,
    torso_hold: TorsoHold,
) -> None:
    data = mujoco.MjData(model.mj_model)
    reset_data(model, data)
    torso_hold.initialize(model.mj_model, data)
    transmission.update_motor_states(data)
    imu_sensor = (
        MujocoImu.bind(model.mj_model)
        if config.asset_config.imu_config.enabled
        else None
    )
    imu_noise = ImuNoiseInjector(config.imu_noise)
    video_capture = EmulatorVideoCapture() if config.video_path is not None else None

    with _open_emulator_resources(
        model,
        data,
        emulators,
        config,
        imu_sensor,
    ) as resources:
        print_emulator_summary(model, emulators)
        print(f"  command latency: {config.command_latency_seconds * 1000:g} ms")
        print(
            "  effective motor gains: "
            f"kp x{config.motor_kp_scale:g}, kd x{config.motor_kd_scale:g}"
        )
        if config.video_path is not None:
            print(f"  video: record free-torso motion to {config.video_path}")
        fixed_joints = sorted(
            config.asset_config.joint_availability.disabled_joint_names
        )
        if fixed_joints:
            print("  fixed in URDF: " + ", ".join(fixed_joints))
        print(f"  torso: GUIDED (downward travel {torso_hold.drop_limit_m * 100:g} cm)")
        if imu_sensor is not None:
            print(f"  imu: {config.asset_config.imu_config.body_name}")
            if resources.imu_serial is not None:
                print(
                    f"  ebimu: E2BOX EBIMU-9DOFV6 at "
                    f"{config.imu_serial_rate_hz:g} Hz -> "
                    f"{config.imu_serial_device} (host endpoint /dev/ttyIMU)"
                )
                if config.imu_noise.enabled:
                    print(
                        "  ebimu noise: orientation std "
                        f"{config.imu_noise.orientation_std_radians:g} rad/axis, "
                        "linear acceleration std "
                        f"{config.imu_noise.linear_acceleration_std_mps2:g} "
                        f"m/s^2/axis, seed {config.imu_noise.seed}"
                    )
                else:
                    print("  ebimu noise: disabled")
            else:
                print("  ebimu: disabled")
        if resources.pose_server is not None:
            print(
                f"  pose: {config.pose_service.reference_frame}_from_"
                f"{config.pose_service.tracked_frame} at "
                f"{config.pose_service.publish_rate_hz:g} Hz -> "
                f"{config.pose_service.listen}"
            )
        terminal_enter = TerminalEnterInput()
        if terminal_enter.enabled:
            print("  keyboard: press Enter to release; press again to reset/guide")
        else:
            print("  keyboard: unavailable; torso guide remains engaged")
        try:
            _simulation_loop(
                model,
                data,
                emulators,
                transmission,
                config,
                torso_hold,
                imu_sensor,
                imu_noise,
                terminal_enter,
                resources,
                video_capture,
            )
        finally:
            if config.video_path is not None and video_capture is not None:
                render_emulator_video(
                    model,
                    video_capture.qpos_frames,
                    config.video_path,
                    torso_hold,
                    fps=video_capture.fps,
                )


def _simulation_loop(
    model: BelloModel,
    data: mujoco.MjData,
    emulators: dict[str, RobstrideBusEmulator],
    transmission: BelloRobstrideTransmission,
    config: EmulatorConfig,
    torso_hold: TorsoHold,
    imu_sensor: MujocoImu | None,
    imu_noise: ImuNoiseInjector,
    terminal_enter: TerminalEnterInput,
    resources: _EmulatorResources,
    video_capture: EmulatorVideoCapture | None = None,
) -> None:
    endpoints = resources.endpoints
    imu_serial = resources.imu_serial
    pose_server = resources.pose_server
    pose_body_id = resources.pose_body_id
    start_wall = time.monotonic()
    start_sim = float(data.time)
    feedback_period = 1.0 / config.feedback_rate_hz
    next_feedback = start_wall
    pose_period = (
        1.0 / config.pose_service.publish_rate_hz
        if pose_server is not None
        else math.inf
    )
    next_pose_sim = start_sim + pose_period
    delayed_commands = _CommandLatencyQueue(config.command_latency_seconds)
    torso_released = False
    while True:
        if terminal_enter.poll():
            torso_released, reset_performed = cycle_torso_release_reset(
                model,
                data,
                torso_hold,
                torso_released=torso_released,
            )
            if not reset_performed:
                print("Torso released; robot is now free")
                continue
            for emulator in emulators.values():
                emulator.reset_control_state()
            delayed_commands.clear()
            transmission.update_motor_states(data)
            start_wall = time.monotonic()
            start_sim = float(data.time)
            next_feedback = start_wall
            next_pose_sim = start_sim + pose_period
            if pose_server is not None:
                publish_body_pose(
                    pose_server,
                    model.mj_model,
                    data,
                    pose_body_id,
                )
            print("Robot reset to home state; torso guide engaged")
            continue
        for bus_name, endpoint in endpoints.items():
            emulator = emulators[bus_name]
            while (frame := endpoint.receive()) is not None:
                if delayed_commands.enqueue(
                    bus_name,
                    frame,
                    received_at=time.monotonic(),
                ):
                    continue
                for response in emulator.handle_frame(frame):
                    endpoint.send(response)

        for bus_name, frame in delayed_commands.pop_ready(time.monotonic()):
            for response in emulators[bus_name].handle_frame(frame):
                endpoints[bus_name].send(response)

        transmission.apply_motor_controls(
            data,
            kp_scale=config.motor_kp_scale,
            kd_scale=config.motor_kd_scale,
        )
        mujoco.mj_step(model.mj_model, data)
        transmission.update_motor_states(data)
        if video_capture is not None:
            video_capture.sample(data, active=torso_released)

        if imu_sensor is not None and imu_serial is not None:
            imu_serial.update(imu_noise.apply(imu_sensor.sample(data)))

        now = time.monotonic()
        if pose_server is not None and float(data.time) >= next_pose_sim:
            publish_body_pose(pose_server, model.mj_model, data, pose_body_id)
            next_pose_sim += pose_period
            if next_pose_sim < float(data.time) - pose_period:
                next_pose_sim = float(data.time) + pose_period
        if now >= next_feedback:
            for bus_name, endpoint in endpoints.items():
                for frame in emulators[bus_name].active_feedback():
                    endpoint.send(frame)
            next_feedback += feedback_period
            if next_feedback < now - feedback_period:
                next_feedback = now + feedback_period

        deadline = start_wall + (float(data.time) - start_sim)
        delay = deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)


def render_emulator_video(
    model: BelloModel,
    qpos_frames: list[np.ndarray],
    output_path: Path,
    torso_hold: TorsoHold,
    *,
    fps: int,
    width: int = 640,
    height: int = 480,
) -> None:
    """Render captured emulator states after the real-time loop has stopped."""

    if not qpos_frames:
        print("  video: no free-torso frames captured; no video written")
        return
    from bello_mujoco.evaluation.render import (
        CameraConfig,
        VideoWriter,
        make_camera,
        update_camera,
    )

    data = mujoco.MjData(model.mj_model)
    data.eq_active[torso_hold.equality_id] = False
    camera_config = CameraConfig(
        mode="follow",
        distance=3.0,
        azimuth=135.0,
        elevation=-14.0,
        lookat_offset=(0.0, 0.0, 0.35),
    )
    camera = make_camera(model.mj_model, camera_config)
    renderer = mujoco.Renderer(model.mj_model, width=width, height=height)
    writer = VideoWriter(output_path, width=width, height=height, fps=fps)
    try:
        for qpos in qpos_frames:
            data.qpos[:] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model.mj_model, data)
            update_camera(camera, model.mj_model, data, camera_config)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
        renderer.close()
    print(f"  video: wrote {len(qpos_frames)} frames to {output_path}")


def publish_body_pose(
    server: PoseServiceServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> None:
    """Copy one MuJoCo world-frame body state into the pose mailbox."""

    position = np.asarray(data.xpos[body_id])
    quaternion = np.asarray(data.xquat[body_id])
    world_velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        world_velocity,
        0,
    )
    local_velocity = np.empty(6, dtype=np.float64)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        local_velocity,
        1,
    )
    server.publish(
        tuple(float(value) for value in position),
        tuple(float(value) for value in quaternion),
        tuple(float(value) for value in world_velocity[3:]),
        tuple(float(value) for value in local_velocity[:3]),
        sample_monotonic_ns=time.monotonic_ns(),
    )


def reset_data(model: BelloModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model.mj_model, data)
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    data.ctrl[:] = model.default_ctrl
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model.mj_model, data)


def reset_simulation(
    model: BelloModel,
    data: mujoco.MjData,
    torso_hold: TorsoHold,
) -> None:
    """Restore the home state and re-engage the torso guide."""

    reset_data(model, data)
    torso_hold.initialize(model.mj_model, data)


def cycle_torso_release_reset(
    model: BelloModel,
    data: mujoco.MjData,
    torso_hold: TorsoHold,
    *,
    torso_released: bool,
) -> tuple[bool, bool]:
    """Cycle guided -> released -> reset/guided.

    Returns ``(torso_released, reset_performed)``.
    """

    if not torso_released and torso_hold.release(data):
        return True, False
    reset_simulation(model, data, torso_hold)
    return False, True


def add_torso_hold(
    model: BelloModel,
    drop_limit_m: float = 0.05,
    body_name: str = "torso_link",
) -> tuple[BelloModel, TorsoHold]:
    """Guide the torso along world Z from its initial height down to a limit."""

    if model.mjx_model is not None:
        raise ValueError("the runtime torso hold is supported only by CPU MuJoCo")
    if not math.isfinite(drop_limit_m) or drop_limit_m <= 0.0:
        raise ValueError("drop_limit_m must be finite and positive")
    if mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
        raise ValueError(f"cannot hold missing MuJoCo body {body_name!r}")

    spec = mujoco.MjSpec.from_file(str(model.xml_path))
    support_body_name = "bello_emulator_torso_support"
    support_joint_name = "bello_emulator_torso_slider"
    equality_name = "bello_emulator_torso_hold"
    support = spec.worldbody.add_body(name=support_body_name)
    support.explicitinertial = True
    support.mass = 0.1
    support.inertia = (1e-4, 1e-4, 1e-4)
    support.gravcomp = 1.0
    support.add_joint(
        name=support_joint_name,
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        axis=(0.0, 0.0, 1.0),
        limited=True,
        range=(-drop_limit_m, 0.0),
        solref_limit=(0.01, 1.0),
        solimp_limit=(0.99, 0.9999, 0.001, 0.5, 2.0),
    )
    spec.add_equality(
        name=equality_name,
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=body_name,
        name2=support_body_name,
        active=True,
        data=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        solref=(0.01, 1.0),
        solimp=(0.99, 0.9999, 0.001, 0.5, 2.0),
    )
    mj_model = spec.compile()
    preserve_runtime_physics(model.mj_model, mj_model)
    if (mj_model.nq, mj_model.nv, mj_model.nu) != (
        model.mj_model.nq + 1,
        model.mj_model.nv + 1,
        model.mj_model.nu,
    ):
        raise RuntimeError("adding the torso guide changed unexpected state dimensions")

    equality_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_EQUALITY, equality_name
    )
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    support_body_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, support_body_name
    )
    support_joint_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_JOINT, support_joint_name
    )
    if min(equality_id, body_id, support_body_id, support_joint_id) < 0:
        raise RuntimeError("MuJoCo did not compile the torso guide constraint")

    guided_model = replace(
        model,
        mj_model=mj_model,
        default_qpos=np.concatenate((model.default_qpos, np.zeros(1))),
        default_qvel=np.concatenate((model.default_qvel, np.zeros(1))),
    )
    return guided_model, TorsoHold(
        equality_id,
        body_id,
        support_body_id,
        drop_limit_m,
    )


def parse_bus_interfaces(values: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, interface = value.partition("=")
        if not separator or not name or not interface:
            raise ValueError(
                f"invalid bus interface override {value!r}; expected BUS=INTERFACE"
            )
        if name in result:
            raise ValueError(f"duplicate bus interface override for {name!r}")
        result[name] = interface
    return result


def validate_config(config: EmulatorConfig) -> None:
    if not config.bus_interfaces:
        raise ValueError("bus_interfaces must not be empty")
    parse_bus_interfaces(config.bus_interfaces)
    if not math.isfinite(config.feedback_rate_hz) or config.feedback_rate_hz <= 0.0:
        raise ValueError("feedback_rate_hz must be positive")
    if (
        not math.isfinite(config.command_latency_seconds)
        or config.command_latency_seconds < 0.0
    ):
        raise ValueError("command_latency_seconds must be finite and nonnegative")
    if not math.isfinite(config.motor_kp_scale) or config.motor_kp_scale <= 0.0:
        raise ValueError("motor_kp_scale must be finite and positive")
    if not math.isfinite(config.motor_kd_scale) or config.motor_kd_scale < 0.0:
        raise ValueError("motor_kd_scale must be finite and nonnegative")
    if not math.isfinite(config.temperature_c) or config.temperature_c < 0.0:
        raise ValueError("temperature_c must be nonnegative")
    if not math.isfinite(config.imu_serial_rate_hz) or config.imu_serial_rate_hz <= 0.0:
        raise ValueError("imu_serial_rate_hz must be finite and positive")
    if not str(config.imu_serial_device):
        raise ValueError("imu_serial_device must not be empty")
    if not config.asset_config.imu_config.enabled:
        raise ValueError("the emulator asset must include the IMU sensor")
    validate_imu_noise_config(config.imu_noise)
    if not math.isfinite(config.torso_drop_limit_m) or config.torso_drop_limit_m <= 0.0:
        raise ValueError("torso_drop_limit_m must be finite and positive")
    if config.pose_service.enabled:
        for name, value in (
            ("pose_service.listen", config.pose_service.listen),
            ("pose_service.reference_frame", config.pose_service.reference_frame),
            ("pose_service.tracked_frame", config.pose_service.tracked_frame),
        ):
            if not value or value != value.strip():
                raise ValueError(
                    f"{name} must be nonempty without surrounding whitespace"
                )
        if (
            not math.isfinite(config.pose_service.publish_rate_hz)
            or config.pose_service.publish_rate_hz <= 0.0
        ):
            raise ValueError("pose_service.publish_rate_hz must be finite and positive")


def validate_motor_coverage(
    asset: AssetName,
    availability: JointAvailabilityConfig,
    motors: tuple[MotorSpec, ...],
) -> None:
    mapped: set[str] = set()
    for motor in motors:
        if motor.target.type == "joint":
            mapped.add(motor.target.name)
        else:
            mapped.update(motor.target.owned_joints)
    configured = {actuator.joint_name for actuator in robot_definition(asset).actuators}
    disabled = set(availability.disabled_joint_names)
    enabled = configured - disabled
    missing_motors = enabled - mapped
    disabled_motors = mapped & disabled
    unknown_motors = mapped - configured
    if missing_motors:
        raise ValueError(
            "enabled robot joints require RobStride motor mappings: "
            + ", ".join(sorted(missing_motors))
        )
    if disabled_motors:
        raise ValueError(
            "RobStride motors target joints fixed by joint availability: "
            + ", ".join(sorted(disabled_motors))
        )
    if unknown_motors:
        raise ValueError(
            "RobStride motors target joints outside the robot definition: "
            + ", ".join(sorted(unknown_motors))
        )


def print_emulator_summary(
    model: BelloModel, emulators: dict[str, RobstrideBusEmulator]
) -> None:
    print("Bello RobStride MuJoCo emulator ready")
    print("  backend: MuJoCo CPU (mj_step; MJX/Warp disabled)")
    print(f"  timestep: {model.mj_model.opt.timestep:.6f} s")
    for name, emulator in emulators.items():
        ids = ", ".join(str(motor_id) for motor_id in sorted(emulator.motors))
        print(
            f"  {name}: {emulator.spec.interface} host={emulator.spec.host_id} "
            f"motors=[{ids}]"
        )
