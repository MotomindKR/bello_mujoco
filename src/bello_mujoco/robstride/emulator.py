"""Emulate Bello RobStride devices with real-time native MuJoCo."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import select
import sys
import time
from typing import Protocol, TextIO

import mujoco
import numpy as np

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    canonical_physical_asset_config,
    prepare_asset,
)
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.assets_pipeline import stack_description_path
from bello_mujoco.hardware.ebimu import EbimuSerialEmulator
from bello_mujoco.imu import (
    ImuNoiseConfig,
    ImuNoiseInjector,
    MujocoImu,
    validate_imu_noise_config,
)
from bello_mujoco.hardware.pose import PoseServiceConfig, PoseServiceServer
from bello_mujoco.robstride.can import (
    MotorSpec,
    RobstrideBusEmulator,
    SocketCanEndpoint,
    load_robstride_description,
)
from bello_mujoco.robstride.sim import (
    BelloRobstrideTransmission,
    configure_robstride_actuators,
)
from bello_mujoco.sim import (
    BelloModel,
    SimulationPhysicsConfig,
    load_prepared_bello_model,
    preserve_runtime_physics,
)
from bello_mujoco.terrain import FLAT_TERRAIN


def default_hardware_description() -> Path:
    return stack_description_path("bello/hardware.yaml")


def default_motor_model_catalog() -> Path:
    return stack_description_path("robstride/models.yaml")


def default_robot_urdf() -> Path:
    return stack_description_path("bello/full_body/urdf/bello_full_body.urdf")


def default_emulator_asset_config() -> AssetPrepConfig:
    return canonical_physical_asset_config(imu_enabled=True)


@dataclass(frozen=True)
class EmulatorConfig:
    """Runtime settings for the RobStride/MuJoCo hardware emulator."""

    bus_interfaces: tuple[str, ...] = ("can0=vcan0", "can1=vcan1")
    asset_config: AssetPrepConfig = field(default_factory=default_emulator_asset_config)
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig(
        timestep=0.0025
    )
    feedback_rate_hz: float = 100.0
    temperature_c: float = 25.0
    imu_serial_rate_hz: float = 250.0
    imu_serial_device: Path = Path("/dev/ttyIMU-mujoco")
    imu_noise: ImuNoiseConfig = ImuNoiseConfig(
        enabled=True,
        orientation_std_radians=0.005,
        linear_acceleration_std_mps2=0.1,
        seed=0,
    )
    torso_hold: bool = True
    torso_drop_limit_m: float = 0.05
    release_after_seconds: float | None = None
    pose_service: PoseServiceConfig = PoseServiceConfig()


class _PassiveViewer(Protocol):
    def is_running(self) -> bool: ...

    def sync(self) -> None: ...


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


class EmulatorControlWindow:
    """Small Pygame command window for one-way simulation controls."""

    _SIZE = (440, 215)

    def __init__(self, drop_limit_m: float) -> None:
        import pygame

        self._pygame = pygame
        self._closed = False
        try:
            pygame.display.init()
            pygame.font.init()
            self._surface = pygame.display.set_mode(self._SIZE)
            pygame.display.set_caption("Bello controls")
            self._title_font = pygame.font.Font(None, 32)
            self._body_font = pygame.font.Font(None, 23)
            self._button_font = pygame.font.Font(None, 27)
            self._button = pygame.Rect(45, 130, 350, 55)
            self._drop_limit_cm = drop_limit_m * 100.0
            self._released = False
            self._draw()
        except pygame.error as error:
            pygame.quit()
            raise RuntimeError(
                "cannot open the Bello emulator controls; a graphical display "
                "is required"
            ) from error
        except BaseException:
            pygame.quit()
            raise

    def __enter__(self) -> EmulatorControlWindow:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pygame.quit()

    def poll(self) -> tuple[bool, bool, bool]:
        """Return (release_requested, enter_requested, close_requested)."""

        release_requested = False
        enter_requested = False
        close_requested = False
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                close_requested = True
            elif event.type == self._pygame.KEYUP and event.key in (
                self._pygame.K_RETURN,
                self._pygame.K_KP_ENTER,
            ):
                enter_requested = True
            elif (
                event.type == self._pygame.MOUSEBUTTONUP
                and event.button == 1
                and not self._released
                and self._button.collidepoint(event.pos)
            ):
                self._released = True
                release_requested = True
                self._draw()
        return release_requested, enter_requested, close_requested

    def mark_released(self) -> None:
        if not self._released:
            self._released = True
            self._draw()

    def mark_guided(self) -> None:
        if self._released:
            self._released = False
            self._draw()

    def _draw(self) -> None:
        surface = self._surface
        surface.fill((28, 31, 36))
        title = self._title_font.render("Torso constraint", True, (240, 240, 240))
        surface.blit(title, (25, 20))
        status_text = (
            "RELEASED - robot is free"
            if self._released
            else f"GUIDED - vertical drop limited to {self._drop_limit_cm:g} cm"
        )
        status_color = (110, 210, 140) if self._released else (255, 205, 90)
        status = self._body_font.render(status_text, True, status_color)
        surface.blit(status, (25, 61))
        instruction = self._body_font.render(
            "Enter: release / reset and re-engage", True, (190, 195, 205)
        )
        surface.blit(instruction, (25, 91))

        button_color = (73, 78, 88) if self._released else (185, 63, 63)
        self._pygame.draw.rect(surface, button_color, self._button, border_radius=7)
        button_text = "Torso released" if self._released else "Release torso"
        label = self._button_font.render(button_text, True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=self._button.center))
        self._pygame.display.flip()


@dataclass(frozen=True)
class _EmulatorResources:
    endpoints: dict[str, SocketCanEndpoint]
    imu_serial: EbimuSerialEmulator | None
    pose_server: PoseServiceServer | None
    pose_body_id: int
    viewer: _PassiveViewer
    control_window: EmulatorControlWindow | None


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
        default_hardware_description(),
        default_motor_model_catalog(),
        default_robot_urdf(),
        overrides,
    )
    validate_motor_coverage(config.asset_config.joint_availability, robstride.motors)

    prepared = prepare_asset(config.asset_config)
    model = load_prepared_bello_model(
        prepared,
        terrain=FLAT_TERRAIN,
        simulation_physics=config.simulation_physics,
        actuator_config=config.asset_config.actuator_config,
        use_viewer_model=True,
        load_mjx=False,
    )
    model = configure_robstride_actuators(model, robstride.motors)
    torso_hold = None
    if config.torso_hold:
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


def _launch_passive_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    from mujoco import viewer

    return viewer.launch_passive(model, data)


@contextmanager
def _open_emulator_resources(
    model: BelloModel,
    data: mujoco.MjData,
    emulators: dict[str, RobstrideBusEmulator],
    config: EmulatorConfig,
    torso_hold: TorsoHold | None,
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
            publish_body_pose(pose_server, data, pose_body_id)

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
        viewer = stack.enter_context(_launch_passive_viewer(model.mj_model, data))
        control_window = (
            stack.enter_context(EmulatorControlWindow(torso_hold.drop_limit_m))
            if torso_hold is not None
            else None
        )
        yield _EmulatorResources(
            endpoints=endpoints,
            imu_serial=imu_serial,
            pose_server=pose_server,
            pose_body_id=pose_body_id,
            viewer=viewer,
            control_window=control_window,
        )


def run_simulation(
    model: BelloModel,
    emulators: dict[str, RobstrideBusEmulator],
    transmission: BelloRobstrideTransmission,
    config: EmulatorConfig,
    torso_hold: TorsoHold | None = None,
) -> None:
    data = mujoco.MjData(model.mj_model)
    reset_data(model, data)
    if torso_hold is not None:
        torso_hold.initialize(model.mj_model, data)
    transmission.update_motor_states(data)
    imu_sensor = (
        MujocoImu.bind(model.mj_model)
        if config.asset_config.imu_config.enabled
        else None
    )
    imu_noise = ImuNoiseInjector(config.imu_noise)

    with _open_emulator_resources(
        model,
        data,
        emulators,
        config,
        torso_hold,
        imu_sensor,
    ) as resources:
        print_emulator_summary(model, emulators)
        fixed_joints = sorted(
            config.asset_config.joint_availability.disabled_joint_names
        )
        if fixed_joints:
            print("  fixed in URDF: " + ", ".join(fixed_joints))
        if torso_hold is not None:
            if config.release_after_seconds is not None:
                release_note = (
                    f"release after {config.release_after_seconds:g} simulated seconds"
                )
            else:
                release_note = "use the Bello controls window to release"
            print(
                f"  torso: GUIDED (downward travel "
                f"{torso_hold.drop_limit_m * 100:g} cm; {release_note})"
            )
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
            action = (
                "cycle torso release and reset/guide"
                if torso_hold is not None
                else "reset the robot"
            )
            print(f"  keyboard: press Enter to {action}")
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
        )


def _simulation_loop(
    model: BelloModel,
    data: mujoco.MjData,
    emulators: dict[str, RobstrideBusEmulator],
    transmission: BelloRobstrideTransmission,
    config: EmulatorConfig,
    torso_hold: TorsoHold | None,
    imu_sensor: MujocoImu | None,
    imu_noise: ImuNoiseInjector,
    terminal_enter: TerminalEnterInput,
    resources: _EmulatorResources,
) -> None:
    endpoints = resources.endpoints
    viewer = resources.viewer
    control_window = resources.control_window
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
    steps = 0
    torso_released = torso_hold is None
    while True:
        if not viewer.is_running():
            break
        enter_requested = terminal_enter.poll()
        if control_window is not None:
            release_requested, window_enter_requested, close_requested = (
                control_window.poll()
            )
            enter_requested = enter_requested or window_enter_requested
            if close_requested:
                break
            if release_requested and torso_hold is not None:
                torso_released = torso_hold.release(data)
                if torso_released:
                    print("Torso released; robot is now free")
        if enter_requested:
            torso_released, reset_performed = cycle_torso_release_reset(
                model,
                data,
                torso_hold,
                torso_released=torso_released,
            )
            if not reset_performed:
                if control_window is not None:
                    control_window.mark_released()
                print("Torso released; robot is now free")
                continue
            for emulator in emulators.values():
                emulator.reset_control_state()
            transmission.update_motor_states(data)
            start_wall = time.monotonic()
            start_sim = float(data.time)
            next_feedback = start_wall
            next_pose_sim = start_sim + pose_period
            if pose_server is not None:
                publish_body_pose(
                    pose_server,
                    data,
                    pose_body_id,
                    produced_at=start_wall,
                )
            if control_window is not None:
                control_window.mark_guided()
            state = "free" if torso_hold is None else "torso guide engaged"
            print(f"Robot reset to home state; {state}")
            continue
        if (
            not torso_released
            and torso_hold is not None
            and config.release_after_seconds is not None
            and float(data.time) - start_sim >= config.release_after_seconds
        ):
            torso_released = torso_hold.release(data)
            if torso_released:
                print("Torso released; robot is now free")
                if control_window is not None:
                    control_window.mark_released()
        for bus_name, endpoint in endpoints.items():
            emulator = emulators[bus_name]
            while (frame := endpoint.receive()) is not None:
                for response in emulator.handle_frame(frame):
                    endpoint.send(response)

        transmission.apply_motor_controls(data)
        mujoco.mj_step(model.mj_model, data)
        transmission.update_motor_states(data)
        steps += 1

        if imu_sensor is not None and imu_serial is not None:
            imu_serial.update(imu_noise.apply(imu_sensor.sample(data)))

        now = time.monotonic()
        if pose_server is not None and float(data.time) >= next_pose_sim:
            publish_body_pose(pose_server, data, pose_body_id, produced_at=now)
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

        viewer.sync()
        deadline = start_wall + (float(data.time) - start_sim)
        delay = deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)


def publish_body_pose(
    server: PoseServiceServer,
    data: mujoco.MjData,
    body_id: int,
    *,
    produced_at: float | None = None,
) -> None:
    """Copy one MuJoCo world-from-body transform into the pose mailbox."""

    position = np.asarray(data.xpos[body_id])
    quaternion = np.asarray(data.xquat[body_id])
    server.publish(
        tuple(float(value) for value in position),
        tuple(float(value) for value in quaternion),
        produced_at=produced_at,
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
    torso_hold: TorsoHold | None,
) -> None:
    """Restore the home state and re-engage the configured torso guide."""

    reset_data(model, data)
    if torso_hold is not None:
        torso_hold.initialize(model.mj_model, data)


def cycle_torso_release_reset(
    model: BelloModel,
    data: mujoco.MjData,
    torso_hold: TorsoHold | None,
    *,
    torso_released: bool,
) -> tuple[bool, bool]:
    """Cycle guided -> released -> reset/guided.

    Returns ``(torso_released, reset_performed)``.
    """

    if torso_hold is not None and not torso_released and torso_hold.release(data):
        return True, False
    reset_simulation(model, data, torso_hold)
    return torso_hold is None, True


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
    if config.release_after_seconds is not None:
        if (
            not math.isfinite(config.release_after_seconds)
            or config.release_after_seconds < 0.0
        ):
            raise ValueError("release_after_seconds must be finite and nonnegative")
        if not config.torso_hold:
            raise ValueError("release_after_seconds requires torso_hold")
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
    availability: JointAvailabilityConfig, motors: tuple[MotorSpec, ...]
) -> None:
    mapped: set[str] = set()
    for motor in motors:
        if motor.target.type == "joint":
            mapped.add(motor.target.name)
        else:
            mapped.update(motor.target.owned_joints)
    optional_names = set(availability.by_name)
    mapped_optional = mapped & optional_names
    enabled_optional = availability.enabled_joint_names
    missing_motors = enabled_optional - mapped_optional
    disabled_motors = mapped_optional - enabled_optional
    if missing_motors:
        raise ValueError(
            "enabled optional joints require RobStride motor mappings: "
            + ", ".join(sorted(missing_motors))
        )
    if disabled_motors:
        raise ValueError(
            "RobStride motors target joints fixed by joint availability: "
            + ", ".join(sorted(disabled_motors))
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
