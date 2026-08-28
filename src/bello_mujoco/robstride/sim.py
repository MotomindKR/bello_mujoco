"""Map RobStride physical actuators onto the Bello MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from bello_mujoco.robstride.can import MotorSpec, RobstrideMotor
from bello_mujoco.sim import BelloModel, preserve_runtime_physics


@dataclass(frozen=True)
class JointBinding:
    qpos_index: int
    qvel_index: int


@dataclass(frozen=True)
class TendonBinding:
    motor_name: str
    tendon_id: int
    owned_joints: tuple[str, ...]


@dataclass(frozen=True)
class NativeActuatorBinding:
    position_id: int
    velocity_id: int
    feedforward_id: int
    force_sensor_address: int
    target_type: str
    target_id: int


def configure_robstride_actuators(
    model: BelloModel,
    specs: tuple[MotorSpec, ...],
) -> BelloModel:
    """Replace mapped position actuators with native MIT actuator triples."""

    if model.mjx_model is not None:
        raise ValueError("RobStride native actuators require a CPU-only MuJoCo model")

    spec = mujoco.MjSpec.from_file(str(model.xml_path))
    mapped_joint_names = {
        joint_name
        for motor in specs
        for joint_name in (
            (motor.target.name,)
            if motor.target.type == "joint"
            else motor.target.owned_joints
        )
    }
    mapped_tendon_names = {
        motor.target.name for motor in specs if motor.target.type == "tendon"
    }
    for actuator in list(spec.actuators):
        maps_joint = (
            actuator.trntype == mujoco.mjtTrn.mjTRN_JOINT
            and actuator.target in mapped_joint_names
        )
        maps_tendon = (
            actuator.trntype == mujoco.mjtTrn.mjTRN_TENDON
            and actuator.target in mapped_tendon_names
        )
        if maps_joint or maps_tendon:
            spec.delete(actuator)

    for motor in specs:
        target = motor.target
        max_torque = max(abs(value) for value in motor.model.protocol.torque)
        if target.type == "joint":
            coordinate = spec.joint(target.name)
            transmission = mujoco.mjtTrn.mjTRN_JOINT
        else:
            coordinate = spec.tendon(target.name)
            transmission = mujoco.mjtTrn.mjTRN_TENDON
        if coordinate is None:
            raise ValueError(
                f"motor {motor.name!r} maps to missing MuJoCo {target.type} "
                f"{target.name!r}"
            )
        coordinate.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        coordinate.actfrcrange = (-max_torque, max_torque)

        position = spec.add_actuator(
            name=_actuator_name(motor.name, "position"),
            trntype=transmission,
            target=target.name,
        )
        position.set_to_position(kp=0.0)
        velocity = spec.add_actuator(
            name=_actuator_name(motor.name, "velocity"),
            trntype=transmission,
            target=target.name,
        )
        velocity.set_to_velocity(kv=0.0)
        feedforward = spec.add_actuator(
            name=_actuator_name(motor.name, "feedforward"),
            trntype=transmission,
            target=target.name,
        )
        feedforward.set_to_motor()
        spec.add_sensor(
            name=_force_sensor_name(motor.name),
            type=(
                mujoco.mjtSensor.mjSENS_JOINTACTFRC
                if target.type == "joint"
                else mujoco.mjtSensor.mjSENS_TENDONACTFRC
            ),
            objtype=(
                mujoco.mjtObj.mjOBJ_JOINT
                if target.type == "joint"
                else mujoco.mjtObj.mjOBJ_TENDON
            ),
            objname=target.name,
        )

    runner_path = model.xml_path.with_name(
        f"{model.xml_path.stem}.robstride{model.xml_path.suffix}"
    )
    runner_path.write_text(spec.to_xml(), encoding="utf-8")
    mj_model = mujoco.MjModel.from_xml_path(str(runner_path))
    preserve_runtime_physics(model.mj_model, mj_model)
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    default_ctrl = (
        np.asarray(mj_model.key_ctrl[key_id], dtype=np.float32).copy()
        if key_id >= 0
        else np.zeros(mj_model.nu, dtype=np.float32)
    )
    return replace(
        model,
        mj_model=mj_model,
        xml_path=runner_path,
        default_ctrl=default_ctrl,
    )


class BelloRobstrideTransmission:
    """Physical-motor to native MuJoCo actuator transmission for Bello."""

    def __init__(
        self,
        model: BelloModel,
        motors: tuple[RobstrideMotor, ...],
    ) -> None:
        self.model = model
        self.motors = {motor.spec.name: motor for motor in motors}
        self.bindings = self._joint_bindings()
        self.direct: dict[str, str] = {}
        self.tendons: dict[str, TendonBinding] = {}
        self.actuators: dict[str, NativeActuatorBinding] = {}
        self._bind_motors(tuple(motor.spec for motor in motors))

    @property
    def mapped_joint_names(self) -> set[str]:
        names = set(self.direct.values())
        for binding in self.tendons.values():
            names.update(binding.owned_joints)
        return names

    def update_motor_states(self, data: mujoco.MjData) -> None:
        for motor_name, joint_name in self.direct.items():
            position, velocity = self._joint_position_velocity(data, joint_name)
            self._set_motor_state(motor_name, position, velocity)

        for binding in self.tendons.values():
            self._set_motor_state(
                binding.motor_name,
                float(data.ten_length[binding.tendon_id]),
                float(data.ten_velocity[binding.tendon_id]),
            )
        for motor_name, binding in self.actuators.items():
            self.motors[motor_name].applied_torque = float(
                data.sensordata[binding.force_sensor_address]
            )

    def apply_motor_controls(
        self,
        data: mujoco.MjData,
        *,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ) -> None:
        """Write MIT targets and effective gains to the native actuators."""

        self.update_motor_states(data)
        data.qfrc_applied[:] = 0.0
        for motor_name, motor in self.motors.items():
            binding = self.actuators[motor_name]
            enabled = motor.enabled
            kp = motor.command.kp * kp_scale if enabled else 0.0
            kd = motor.command.kd * kd_scale if enabled else 0.0

            mj_model = self.model.mj_model
            mj_model.actuator_gainprm[binding.position_id, 0] = kp
            mj_model.actuator_biasprm[binding.position_id, 1] = -kp
            mj_model.actuator_gainprm[binding.velocity_id, 0] = kd
            mj_model.actuator_biasprm[binding.velocity_id, 2] = -kd
            data.ctrl[binding.position_id] = (
                motor.command.position + motor.zero_offset if enabled else 0.0
            )
            data.ctrl[binding.velocity_id] = motor.command.velocity if enabled else 0.0
            data.ctrl[binding.feedforward_id] = motor.command.torque if enabled else 0.0
            self._set_common_force_limit(
                binding,
                motor.torque_limit,
            )

    def _joint_bindings(self) -> dict[str, JointBinding]:
        mj_model = self.model.mj_model
        result = {}
        for joint_id in range(mj_model.njnt):
            if mj_model.jnt_type[joint_id] not in {
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            }:
                continue
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is not None:
                result[name] = JointBinding(
                    qpos_index=int(mj_model.jnt_qposadr[joint_id]),
                    qvel_index=int(mj_model.jnt_dofadr[joint_id]),
                )
        return result

    def _bind_motors(self, specs: tuple[MotorSpec, ...]) -> None:
        for spec in specs:
            target = spec.target
            if target.type == "joint":
                self._require_joint(target.name, spec.name)
                self.direct[spec.name] = target.name
                target_id = mujoco.mj_name2id(
                    self.model.mj_model, mujoco.mjtObj.mjOBJ_JOINT, target.name
                )
            else:
                for joint_name in target.owned_joints:
                    self._require_joint(joint_name, spec.name)
                target_id = self._require_tendon(target.name, spec.name)
                self.tendons[spec.name] = TendonBinding(
                    motor_name=spec.name,
                    tendon_id=target_id,
                    owned_joints=target.owned_joints,
                )
            self.actuators[spec.name] = NativeActuatorBinding(
                position_id=self._require_actuator(spec.name, "position"),
                velocity_id=self._require_actuator(spec.name, "velocity"),
                feedforward_id=self._require_actuator(spec.name, "feedforward"),
                force_sensor_address=self._require_force_sensor(spec.name),
                target_type=target.type,
                target_id=target_id,
            )

        groups: dict[str, list[MotorSpec]] = {}
        for spec in specs:
            if spec.target.coupling_group is not None:
                groups.setdefault(spec.target.coupling_group, []).append(spec)
        for group_name, group_specs in groups.items():
            sizes = {spec.target.coupling_size for spec in group_specs}
            if len(sizes) != 1:
                raise ValueError(
                    f"coupling group {group_name!r} has inconsistent sizes"
                )
            expected = sizes.pop()
            if expected is None or len(group_specs) != expected:
                raise ValueError(
                    f"coupling group {group_name!r} requires {expected} motors; "
                    f"got {len(group_specs)}"
                )

    def _require_joint(self, joint_name: str, motor_name: str) -> None:
        if joint_name not in self.bindings:
            raise ValueError(
                f"motor {motor_name!r} maps to missing MuJoCo joint {joint_name!r}"
            )

    def _require_tendon(self, tendon_name: str, motor_name: str) -> int:
        tendon_id = mujoco.mj_name2id(
            self.model.mj_model, mujoco.mjtObj.mjOBJ_TENDON, tendon_name
        )
        if tendon_id < 0:
            raise ValueError(
                f"motor {motor_name!r} maps to missing MuJoCo tendon "
                f"{tendon_name!r}; the prepared asset is missing the required "
                "ankle transmission"
            )
        return tendon_id

    def _require_actuator(self, motor_name: str, term: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model.mj_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            _actuator_name(motor_name, term),
        )
        if actuator_id < 0:
            raise ValueError(
                f"motor {motor_name!r} is missing its native {term} actuator; "
                "configure the RobStride actuator bank before binding motors"
            )
        return actuator_id

    def _require_force_sensor(self, motor_name: str) -> int:
        sensor_id = mujoco.mj_name2id(
            self.model.mj_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            _force_sensor_name(motor_name),
        )
        if sensor_id < 0:
            raise ValueError(
                f"motor {motor_name!r} is missing its aggregate force sensor"
            )
        return int(self.model.mj_model.sensor_adr[sensor_id])

    def _joint_position_velocity(
        self, data: mujoco.MjData, joint_name: str
    ) -> tuple[float, float]:
        binding = self.bindings[joint_name]
        return float(data.qpos[binding.qpos_index]), float(
            data.qvel[binding.qvel_index]
        )

    def _set_common_force_limit(
        self, binding: NativeActuatorBinding, limit: float | None
    ) -> None:
        if binding.target_type == "joint":
            self.model.mj_model.jnt_actfrclimited[binding.target_id] = limit is not None
            if limit is not None:
                self.model.mj_model.jnt_actfrcrange[binding.target_id] = (-limit, limit)
        else:
            self.model.mj_model.tendon_actfrclimited[binding.target_id] = (
                limit is not None
            )
            if limit is not None:
                self.model.mj_model.tendon_actfrcrange[binding.target_id] = (
                    -limit,
                    limit,
                )

    def _set_motor_state(self, name: str, position: float, velocity: float) -> None:
        motor = self.motors[name]
        motor.mechanical_position = position
        motor.mechanical_velocity = velocity


def _actuator_name(motor_name: str, term: str) -> str:
    return f"robstride_{motor_name}_{term}"


def _force_sensor_name(motor_name: str) -> str:
    return f"robstride_{motor_name}_force"
