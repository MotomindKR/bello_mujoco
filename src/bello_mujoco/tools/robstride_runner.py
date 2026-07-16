"""Run Bello in MuJoCo as a set of RobStride devices on SocketCAN."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
import math
import sys
import time
from typing import Any

import mujoco
import numpy as np
from omegaconf import OmegaConf
import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.assets_pipeline.mjcf import (
    AnkleDifferentialConfig,
    ImuSensorConfig,
)
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.assets import project_data_path
from bello_mujoco.ebimu_serial import EbimuSerialEmulator
from bello_mujoco.imu import MujocoImu
from bello_mujoco.robstride_can import (
    MotorSpec,
    RobstrideBusEmulator,
    SocketCanEndpoint,
    load_robstride_configs,
)
from bello_mujoco.robstride_sim import BelloRobstrideTransmission
from bello_mujoco.sim import BelloModel, load_prepared_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN, TerrainConfig

def default_motor_configs() -> tuple[Path, ...]:
    root = project_data_path("configs") / "robstride"
    return root / "lower_body.yaml", root / "upper_body.yaml"


def default_motor_kind_config() -> Path:
    return project_data_path("configs") / "robstride" / "motor_kinds.yaml"


def default_runner_config() -> Path:
    return project_data_path("configs") / "robstride" / "runner.yaml"


@dataclass(frozen=True)
class RobstrideRunnerConfig:
    """Runtime settings for the RobStride/MuJoCo hardware emulator."""

    config_file: Path = field(default_factory=default_runner_config)
    motor_kind_config: Path = field(default_factory=default_motor_kind_config)
    motor_configs: tuple[Path, ...] = field(default_factory=default_motor_configs)
    bus_interfaces: tuple[str, ...] = ("can0=vcan0", "can1=vcan1")
    asset_config: AssetPrepConfig = AssetPrepConfig(
        asset="full_body",
        visual_mesh_face_budget=20_000,
        ankle_differential_config=AnkleDifferentialConfig(enabled=True),
        joint_availability=JointAvailabilityConfig(
            waist_yaw=True,
            waist_roll=False,
            waist_pitch=False,
            neck_yaw=False,
            head_pitch=False,
        ),
    )
    terrain: TerrainConfig = FLAT_TERRAIN
    feedback_rate_hz: float = 100.0
    temperature_c: float = 25.0
    headless: bool = False
    real_time: bool = True
    max_steps: int | None = None
    imu_serial_rate_hz: float = 200.0
    imu_serial_enabled: bool = True
    imu_serial_device: Path = Path("/dev/ttyIMU-mujoco")
    torso_hold: bool = True
    torso_drop_limit_m: float = 0.05
    release_after_seconds: float | None = None


@dataclass(frozen=True)
class TorsoHold:
    """Runtime handle for the torso's world-vertical guide constraint."""

    equality_id: int
    body_id: int
    support_body_id: int
    support_joint_id: int
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


class RunnerControlWindow:
    """Small Pygame command window for one-way simulation controls."""

    _SIZE = (440, 190)

    def __init__(self, drop_limit_m: float) -> None:
        import pygame

        self._pygame = pygame
        try:
            pygame.display.init()
            pygame.font.init()
            self._surface = pygame.display.set_mode(self._SIZE)
        except pygame.error as error:
            pygame.quit()
            raise RuntimeError(
                "cannot open the Bello command window; use --headless on a "
                "machine without a display"
            ) from error
        pygame.display.set_caption("Bello controls")
        self._title_font = pygame.font.Font(None, 32)
        self._body_font = pygame.font.Font(None, 23)
        self._button_font = pygame.font.Font(None, 27)
        self._button = pygame.Rect(45, 105, 350, 55)
        self._drop_limit_cm = drop_limit_m * 100.0
        self._released = False
        self._draw()

    def __enter__(self) -> RunnerControlWindow:
        return self

    def __exit__(self, *_args) -> None:
        self._pygame.quit()

    def poll(self) -> tuple[bool, bool]:
        """Return (release_requested, close_requested)."""

        release_requested = False
        close_requested = False
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                close_requested = True
            elif (
                event.type == self._pygame.MOUSEBUTTONUP
                and event.button == 1
                and not self._released
                and self._button.collidepoint(event.pos)
            ):
                self._released = True
                release_requested = True
                self._draw()
        return release_requested, close_requested

    def mark_released(self) -> None:
        if not self._released:
            self._released = True
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

        button_color = (73, 78, 88) if self._released else (185, 63, 63)
        self._pygame.draw.rect(surface, button_color, self._button, border_radius=7)
        button_text = "Torso released" if self._released else "Release torso"
        label = self._button_font.render(button_text, True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=self._button.center))
        self._pygame.display.flip()


def main(config: RobstrideRunnerConfig = RobstrideRunnerConfig()) -> None:
    validate_config(config)
    robstride = load_robstride_configs(config.motor_kind_config, config.motor_configs)
    validate_runner_motor_coverage(
        config.asset_config.joint_availability, robstride.motors
    )
    overrides = parse_bus_interfaces(config.bus_interfaces)
    unknown_buses = set(overrides) - {bus.name for bus in robstride.buses}
    if unknown_buses:
        raise ValueError(
            "bus interface overrides reference undefined buses: "
            + ", ".join(sorted(unknown_buses))
        )
    buses = tuple(
        replace(bus, interface=overrides.get(bus.name, bus.interface))
        for bus in robstride.buses
    )

    prepared = prepare_asset(config.asset_config)
    model = load_prepared_bello_model(
        prepared,
        terrain=config.terrain,
        actuator_config=config.asset_config.actuator_config,
        use_viewer_model=not config.headless,
        load_mjx=False,
    )
    torso_hold = None
    if config.torso_hold:
        model, torso_hold = add_torso_hold(model, config.torso_drop_limit_m)
    emulators = {
        bus.name: RobstrideBusEmulator(
            bus,
            tuple(motor for motor in robstride.motors if motor.bus == bus.name),
        )
        for bus in buses
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


def run_simulation(
    model: BelloModel,
    emulators: dict[str, RobstrideBusEmulator],
    transmission: BelloRobstrideTransmission,
    config: RobstrideRunnerConfig,
    torso_hold: TorsoHold | None = None,
) -> None:
    transmission.disable_mapped_position_actuators()
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

    with ExitStack() as stack:
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

        imu_serial = None
        if imu_sensor is not None and config.imu_serial_enabled:
            imu_serial = stack.enter_context(
                EbimuSerialEmulator(
                    config.imu_serial_device,
                    output_rate_hz=config.imu_serial_rate_hz,
                    temperature_c=config.temperature_c,
                )
            )

        viewer_context = nullcontext(None)
        if not config.headless:
            from mujoco import viewer as mj_viewer

            viewer_context = mj_viewer.launch_passive(model.mj_model, data)

        print_runner_summary(model, emulators)
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
            elif config.headless:
                release_note = "no automatic release"
            else:
                release_note = "use the Bello controls window to release"
            print(
                f"  torso: GUIDED (downward travel "
                f"{torso_hold.drop_limit_m * 100:g} cm; {release_note})"
            )
        if imu_sensor is not None:
            print(f"  imu: {config.asset_config.imu_config.body_name}")
            if imu_serial is not None:
                print(
                    f"  ebimu: E2BOX EBIMU-9DOFV6 at "
                    f"{config.imu_serial_rate_hz:g} Hz -> "
                    f"{config.imu_serial_device} (host endpoint /dev/ttyIMU)"
                )
            else:
                print("  ebimu: disabled")
        with viewer_context as viewer:
            control_window = None
            if torso_hold is not None and not config.headless:
                control_window = stack.enter_context(
                    RunnerControlWindow(torso_hold.drop_limit_m)
                )
            _simulation_loop(
                model,
                data,
                emulators,
                endpoints,
                transmission,
                config,
                viewer,
                torso_hold,
                control_window,
                imu_sensor,
                imu_serial,
            )


def _simulation_loop(
    model: BelloModel,
    data: mujoco.MjData,
    emulators: dict[str, RobstrideBusEmulator],
    endpoints: dict[str, SocketCanEndpoint],
    transmission: BelloRobstrideTransmission,
    config: RobstrideRunnerConfig,
    viewer,
    torso_hold: TorsoHold | None,
    control_window: RunnerControlWindow | None,
    imu_sensor: MujocoImu | None,
    imu_serial: EbimuSerialEmulator | None,
) -> None:
    start_wall = time.monotonic()
    start_sim = float(data.time)
    feedback_period = 1.0 / config.feedback_rate_hz
    next_feedback = start_wall
    steps = 0
    torso_released = torso_hold is None
    while config.max_steps is None or steps < config.max_steps:
        if viewer is not None and not viewer.is_running():
            break
        if control_window is not None:
            release_requested, close_requested = control_window.poll()
            if close_requested:
                break
            if release_requested and torso_hold is not None:
                torso_released = torso_hold.release(data)
                if torso_released:
                    print("Torso released; robot is now free")
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

        transmission.apply_motor_torques(data)
        mujoco.mj_step(model.mj_model, data)
        transmission.update_motor_states(data)
        steps += 1

        if imu_sensor is not None and imu_serial is not None:
            imu_serial.update(imu_sensor.sample(data))

        now = time.monotonic()
        if now >= next_feedback:
            for bus_name, endpoint in endpoints.items():
                for frame in emulators[bus_name].active_feedback():
                    endpoint.send(frame)
            next_feedback += feedback_period
            if next_feedback < now - feedback_period:
                next_feedback = now + feedback_period

        if viewer is not None:
            viewer.sync()
        if config.real_time:
            deadline = start_wall + (float(data.time) - start_sim)
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)


def reset_data(model: BelloModel, data: mujoco.MjData) -> None:
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    data.ctrl[:] = model.default_ctrl
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model.mj_model, data)


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
    support_body_name = "robstride_runner_torso_support"
    support_joint_name = "robstride_runner_torso_slider"
    equality_name = "robstride_runner_torso_hold"
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
        support_joint_id,
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


def validate_config(config: RobstrideRunnerConfig) -> None:
    if not math.isfinite(config.feedback_rate_hz) or config.feedback_rate_hz <= 0.0:
        raise ValueError("feedback_rate_hz must be positive")
    if not math.isfinite(config.temperature_c) or config.temperature_c < 0.0:
        raise ValueError("temperature_c must be nonnegative")
    if config.max_steps is not None and config.max_steps <= 0:
        raise ValueError("max_steps must be positive when provided")
    if not math.isfinite(config.imu_serial_rate_hz) or config.imu_serial_rate_hz <= 0.0:
        raise ValueError("imu_serial_rate_hz must be finite and positive")
    if config.imu_serial_enabled and not str(config.imu_serial_device):
        raise ValueError("imu_serial_device must not be empty")
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


def load_runner_config(path: Path) -> RobstrideRunnerConfig:
    """Load the human-facing CPU runner YAML into its runtime dataclass."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    root = _config_mapping(raw, "runner config")
    _validate_config_keys(
        root, {"simulation", "can", "imu", "torso", "joints"}, "runner config"
    )
    simulation = _config_section(root, "simulation")
    can = _config_section(root, "can")
    imu = _config_section(root, "imu")
    torso = _config_section(root, "torso")
    joints = _config_section(root, "joints")
    _validate_config_keys(
        simulation,
        {
            "backend",
            "asset",
            "collision_mode",
            "visual_mesh_face_budget",
            "end_effector_sphere_mode",
            "ankle_differential_tendons",
            "terrain",
            "headless",
            "real_time",
            "max_steps",
        },
        "simulation",
    )
    _validate_config_keys(
        can,
        {
            "motor_kind_config",
            "motor_configs",
            "bus_interfaces",
            "feedback_rate_hz",
            "temperature_c",
        },
        "can",
    )
    _validate_config_keys(
        imu,
        {
            "enabled",
            "body_name",
            "serial_rate_hz",
            "serial_enabled",
            "serial_device",
        },
        "imu",
    )
    _validate_config_keys(
        torso,
        {"enabled", "drop_limit_m", "release_after_seconds"},
        "torso",
    )
    _validate_config_keys(
        joints,
        {"waist_yaw", "waist_roll", "waist_pitch", "neck_yaw", "head_pitch"},
        "joints",
    )

    backend = simulation.get("backend")
    if backend != "mujoco_cpu":
        raise ValueError(
            "simulation.backend must be 'mujoco_cpu'; MJX is not supported by "
            "the RobStride runner"
        )
    terrain_name = simulation.get("terrain")
    if terrain_name != "flat":
        raise ValueError("simulation.terrain must be 'flat'")
    asset = _config_string(simulation, "asset")
    collision_mode = _config_string(simulation, "collision_mode")
    if collision_mode not in {"mesh", "foot-mesh", "box"}:
        raise ValueError(
            "simulation.collision_mode must be one of: mesh, foot-mesh, box"
        )
    sphere_mode = _config_string(simulation, "end_effector_sphere_mode")
    if sphere_mode not in {"none", "both", "right", "left", "both_articulated"}:
        raise ValueError(
            "simulation.end_effector_sphere_mode must be one of: "
            "none, both, right, left, both_articulated"
        )
    motor_config_values = can.get("motor_configs")
    if not isinstance(motor_config_values, list) or not motor_config_values:
        raise ValueError("can.motor_configs must be a nonempty list of paths")
    motor_configs = tuple(
        _resolve_config_path(path.parent, value, "can.motor_configs")
        for value in motor_config_values
    )
    motor_kind_config = _resolve_config_path(
        path.parent,
        can.get("motor_kind_config"),
        "can.motor_kind_config",
    )
    interface_values = _config_mapping(can.get("bus_interfaces"), "bus_interfaces")
    if not interface_values:
        raise ValueError("can.bus_interfaces must not be empty")
    bus_interfaces = tuple(
        f"{name}={_plain_string(interface, f'bus interface {name!r}')}"
        for name, interface in interface_values.items()
    )
    imu_enabled = _config_bool(imu, "enabled")
    imu_body_name = _config_string(imu, "body_name")

    config = RobstrideRunnerConfig(
        config_file=path,
        motor_kind_config=motor_kind_config,
        motor_configs=motor_configs,
        bus_interfaces=bus_interfaces,
        asset_config=AssetPrepConfig(
            asset=asset,
            collision_mode=collision_mode,
            visual_mesh_face_budget=_config_positive_int(
                simulation, "visual_mesh_face_budget"
            ),
            end_effector_sphere_mode=sphere_mode,
            imu_config=ImuSensorConfig(
                enabled=imu_enabled,
                body_name=imu_body_name,
            ),
            ankle_differential_config=AnkleDifferentialConfig(
                enabled=_config_bool(simulation, "ankle_differential_tendons")
            ),
            joint_availability=JointAvailabilityConfig(
                waist_yaw=_config_bool(joints, "waist_yaw"),
                waist_roll=_config_bool(joints, "waist_roll"),
                waist_pitch=_config_bool(joints, "waist_pitch"),
                neck_yaw=_config_bool(joints, "neck_yaw"),
                head_pitch=_config_bool(joints, "head_pitch"),
            ),
        ),
        terrain=FLAT_TERRAIN,
        feedback_rate_hz=_config_number(can, "feedback_rate_hz"),
        temperature_c=_config_number(can, "temperature_c"),
        headless=_config_bool(simulation, "headless"),
        real_time=_config_bool(simulation, "real_time"),
        max_steps=_config_optional_int(simulation, "max_steps"),
        imu_serial_rate_hz=_config_number(imu, "serial_rate_hz"),
        imu_serial_enabled=_config_bool(imu, "serial_enabled"),
        imu_serial_device=Path(_config_string(imu, "serial_device")),
        torso_hold=_config_bool(torso, "enabled"),
        torso_drop_limit_m=_config_number(torso, "drop_limit_m"),
        release_after_seconds=_config_optional_number(torso, "release_after_seconds"),
    )
    validate_config(config)
    return config


def validate_runner_motor_coverage(
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


def _config_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping")
    return value


def _config_section(root: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in root:
        raise ValueError(f"runner config is missing section {name!r}")
    return _config_mapping(root[name], name)


def _validate_config_keys(values: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(unknown))}")


def _plain_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _config_string(values: dict[str, Any], name: str) -> str:
    if name not in values:
        raise ValueError(f"missing config value {name!r}")
    return _plain_string(values[name], name)


def _config_bool(values: dict[str, Any], name: str) -> bool:
    value = values.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _config_number(values: dict[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _config_optional_number(values: dict[str, Any], name: str) -> float | None:
    if values.get(name) is None:
        return None
    return _config_number(values, name)


def _config_optional_int(values: dict[str, Any], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer or null")
    return value


def _config_positive_int(values: dict[str, Any], name: str) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolve_config_path(root: Path, value: Any, name: str) -> Path:
    path = Path(_plain_string(value, name)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _config_file_from_args(args: list[str]) -> Path:
    config_path = default_runner_config()
    for index, argument in enumerate(args):
        if argument == "--config-file":
            if index + 1 >= len(args):
                raise ValueError("--config-file requires a path")
            config_path = Path(args[index + 1])
        elif argument.startswith("--config-file="):
            config_path = Path(argument.partition("=")[2])
    return config_path


def print_runner_summary(
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


def entrypoint() -> None:
    args = sys.argv[1:]
    base = load_runner_config(_config_file_from_args(args))
    main(tyro.cli(RobstrideRunnerConfig, default=base, args=args))


if __name__ == "__main__":
    entrypoint()
