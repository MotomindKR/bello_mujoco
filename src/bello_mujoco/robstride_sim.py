"""Map RobStride physical actuators onto the Bello MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bello_mujoco.robstride_can import MotorSpec, RobstrideMotor
from bello_mujoco.sim import BelloModel


@dataclass(frozen=True)
class JointBinding:
    actuator_id: int
    qpos_index: int
    qvel_index: int


@dataclass(frozen=True)
class TendonBinding:
    motor_name: str
    tendon_id: int
    owned_joints: tuple[str, ...]


class BelloRobstrideTransmission:
    """Physical-motor to simulated-joint transmission for Bello."""

    def __init__(self, model: BelloModel, motors: tuple[RobstrideMotor, ...]) -> None:
        self.model = model
        self.motors = {motor.spec.name: motor for motor in motors}
        self.bindings = {
            name: JointBinding(
                actuator_id=index,
                qpos_index=int(model.actuator_qpos_indices[index]),
                qvel_index=int(model.actuator_qvel_indices[index]),
            )
            for index, name in enumerate(model.actuator_joint_names)
        }
        self.direct: dict[str, str] = {}
        self.tendons: dict[str, TendonBinding] = {}
        self._bind_motors(tuple(motor.spec for motor in motors))

    @property
    def mapped_joint_names(self) -> set[str]:
        names = set(self.direct.values())
        for binding in self.tendons.values():
            names.update(binding.owned_joints)
        return names

    def disable_mapped_position_actuators(self) -> None:
        """Let RobStride impedance torques own mapped joints."""

        mj_model = self.model.mj_model
        for joint_name in self.mapped_joint_names:
            actuator_id = self.bindings[joint_name].actuator_id
            mj_model.actuator_gainprm[actuator_id, :] = 0.0
            mj_model.actuator_biasprm[actuator_id, :] = 0.0

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

    def apply_motor_torques(self, data: mujoco.MjData) -> None:
        self.update_motor_states(data)
        joint_torques: dict[str, float] = {}
        for motor_name, joint_name in self.direct.items():
            joint_torques[joint_name] = (
                joint_torques.get(joint_name, 0.0)
                + self.motors[motor_name].compute_torque()
            )
        data.qfrc_applied[:] = 0.0
        for joint_name, torque in joint_torques.items():
            data.qfrc_applied[self.bindings[joint_name].qvel_index] += torque
        for binding in self.tendons.values():
            motor_torque = self.motors[binding.motor_name].compute_torque()
            self._apply_tendon_force(data, binding.tendon_id, motor_torque)

    def _bind_motors(self, specs: tuple[MotorSpec, ...]) -> None:
        for spec in specs:
            target = spec.target
            if target.type == "joint":
                self._require_joint(target.name, spec.name)
                self.direct[spec.name] = target.name
                continue
            for joint_name in target.owned_joints:
                self._require_joint(joint_name, spec.name)
            self.tendons[spec.name] = TendonBinding(
                motor_name=spec.name,
                tendon_id=self._require_tendon(target.name, spec.name),
                owned_joints=target.owned_joints,
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
                f"{tendon_name!r}; "
                "enable ankle_differential_config when preparing the asset"
            )
        return tendon_id

    def _joint_position_velocity(
        self, data: mujoco.MjData, joint_name: str
    ) -> tuple[float, float]:
        binding = self.bindings[joint_name]
        return float(data.qpos[binding.qpos_index]), float(
            data.qvel[binding.qvel_index]
        )

    def _apply_tendon_force(
        self, data: mujoco.MjData, tendon_id: int, force: float
    ) -> None:
        mj_model = self.model.mj_model
        row_start = int(mj_model.ten_J_rowadr[tendon_id])
        row_stop = row_start + int(mj_model.ten_J_rownnz[tendon_id])
        dof_indices = np.asarray(mj_model.ten_J_colind[row_start:row_stop])
        jacobian_values = np.asarray(data.ten_J[row_start:row_stop])
        data.qfrc_applied[dof_indices] += jacobian_values * force

    def _set_motor_state(self, name: str, position: float, velocity: float) -> None:
        motor = self.motors[name]
        motor.mechanical_position = position
        motor.mechanical_velocity = velocity
