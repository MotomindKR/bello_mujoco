"""Offline loaded-standing experiment using the existing Rust policy consumer."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import onnx
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from bello_mujoco.configuration import structured_config
from bello_mujoco.envs.actions import limit_motor_step, project_motor_target
from bello_mujoco.envs.feedback import ObservationLowPassFilterConfig
from bello_mujoco.envs.observations import filter_feedback, unpack_feedback
from bello_mujoco.evaluation.barbell import (
    BarbellConfig,
    foot_support,
    load_sweep,
    solve_pose,
)
from bello_mujoco.evaluation.barbell_hold import HoldConfig, compose_scene, run_hold
from bello_mujoco.imu import MujocoImu
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.models.target_limits import TargetLimits
from bello_mujoco.models.urdf import ImuSensorConfig
from bello_mujoco.sim import load_model


@dataclass(frozen=True)
class PolicyHoldConfig:
    policy: Path
    source: Path
    binary: Path
    hold: HoldConfig
    observation_delay_steps: int = 0
    bar_robot_contact: bool = True


class NativeController:
    """Supply simulated filtered telemetry; Rust owns observations and inference."""

    def __init__(self, config, metadata, model, initial_target, log):
        self.metadata = metadata
        self.imu = MujocoImu.bind(model)
        self.previous_feedback = None
        self.history = deque(maxlen=config.observation_delay_steps + 1)
        self.held = np.setdiff1d(np.arange(model.nu), metadata["policy_indices"])
        self.reference = np.asarray(initial_target, dtype=np.float32)
        self.previous_target = self.reference.copy()
        self.filtered = structured_config(
            ObservationLowPassFilterConfig,
            OmegaConf.create(metadata["environment"]["observation_low_pass_filter"]),
        )
        self.filter = jax.jit(
            lambda old, new: filter_feedback(
                old, new, metadata["control_timestep"], self.filtered
            )
        )
        limits = TargetLimits(
            np.asarray(metadata["motor_limits"]),
            np.asarray(metadata["joint_limits"]),
            np.asarray(metadata["joint_from_motor"]),
            np.asarray(metadata["position_step_limits"]),
        )
        limits.validate_reference(initial_target)
        self.limit = jax.jit(
            lambda target, previous: limit_motor_step(
                project_motor_target(target, jnp.asarray(self.reference), limits),
                previous,
                limits,
            )
        )
        self.constrained_commands = 0
        self.calls = 0
        self.log = log
        self.process = subprocess.Popen(
            [
                str(config.binary),
                "--policy",
                str(config.policy),
                "--source",
                str(config.source),
                "--telemetry",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            bufsize=1,
        )

    def __call__(self, data):
        sample = self.imu.sample(data)
        rotation = Rotation.from_quat(sample.orientation_xyzw)
        gravity = rotation.inv().apply([0, 0, -1])
        measured = jnp.asarray(
            np.r_[
                sample.linear_acceleration_xyz,
                sample.angular_velocity_xyz,
                gravity,
                data.actuator_length,
                data.actuator_velocity,
            ],
            dtype=jnp.float32,
        )
        self.previous_feedback = (
            measured
            if self.previous_feedback is None
            else self.filter(self.previous_feedback, measured)
        )
        # Reconstruct an orientation with the filtered tilt. Heading is unused
        # by this policy; the minimal correction retains the measured yaw nearby.
        acc, gyro, filtered_gravity, pos, vel = map(
            np.asarray, unpack_feedback(self.previous_feedback)
        )
        correction = Rotation.align_vectors([gravity], [filtered_gravity])[0]
        orientation = (rotation * correction).as_quat()
        feedback = dict(
            motor_position=pos.tolist(),
            motor_velocity=vel.tolist(),
            orientation_xyzw=orientation.tolist(),
            angular_velocity=gyro.tolist(),
            # Rust restores specific force from hardwared's gravity-removed value.
            linear_acceleration=(acc + 9.80665 * filtered_gravity).tolist(),
            command=[0, 0, 0],
            torso_command=[0, 0, 0],
            arm_target=(np.asarray(self.metadata["joint_from_motor"]) @ self.reference)[
                self.held
            ].tolist(),
        )
        if not self.history:
            self.history.extend([feedback] * self.history.maxlen)
        else:
            self.history.append(feedback)
        self.process.stdin.write(json.dumps(self.history[0]) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"native policy stopped; inspect {self.log.name}")
        result = json.loads(line)
        target = np.asarray(result["motor_target"], dtype=np.float32)
        target[self.held] = self.reference[self.held]
        limited = np.asarray(
            self.limit(jnp.asarray(target), jnp.asarray(self.previous_target))
        )
        self.constrained_commands += int(np.max(np.abs(limited - target)) > 1e-5)
        self.previous_target = limited
        self.calls += 1
        return limited

    def close(self):
        self.process.stdin.close()
        try:
            code = self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError(
                "native diagnostic process did not exit after EOF"
            ) from None
        finally:
            self.process.stdout.close()
        if code:
            raise RuntimeError(
                f"native policy exited with {code}; inspect {self.log.name}"
            )


def run_suite(config, output):
    if config.hold.controllers != ("pd",):
        raise ValueError(
            "locomotion diagnostics require controllers: [pd] for the held arms"
        )
    with ZipFile(config.policy) as bundle:
        graph = onnx.load_model_from_string(bundle.read("policy.onnx"))
        metadata = json.loads(
            next(p.value for p in graph.metadata_props if p.key == "bello")
        )
        policy_physics = mujoco.MjModel.from_xml_string(
            bundle.read("physics.xml").decode()
        )
    if metadata["task"] != "locomotion":
        raise ValueError("loaded standing requires a locomotion bundle")
    if not np.allclose(metadata["policy_from_hardware"], np.eye(3)):
        raise ValueError(
            "this diagnostic currently requires aligned policy/hardware IMU frames"
        )
    if config.observation_delay_steps < 0:
        raise ValueError("observation delay must be nonnegative")
    geometry = structured_config(BarbellConfig, OmegaConf.load(config.hold.geometry))
    if metadata["robot"] != geometry.asset:
        raise ValueError("barbell geometry and policy robot differ")
    robot = load_model(
        ModelConfig(
            asset=geometry.asset,
            target_total_mass_kg=geometry.robot_mass_kg,
            actuator_mode="policy",
            imu_config=ImuSensorConfig(
                enabled=True,
                body_name=metadata["hardware_imu_frame"],
            ),
        ),
        load_mjx=False,
        use_viewer_model=True,
        spawn_clearance=0,
    )
    model = robot.mj_model
    if list(robot.actuator_coordinate_names) != metadata["coordinates"]:
        raise ValueError("policy motor order differs from the robot")
    for actual, expected in (
        (model.actuator_gainprm[:, 0], metadata["kp"]),
        (-model.actuator_biasprm[:, 2], metadata["kd"]),
        (robot.actuator_coordinate_transform, metadata["motor_from_joint"]),
        (model.actuator_forcerange, policy_physics.actuator_forcerange),
        (model.jnt_actfrcrange, policy_physics.jnt_actfrcrange),
    ):
        if not np.allclose(actual, expected):
            raise ValueError(
                "policy gains, transmission or effort limits differ from the model"
            )
    data = mujoco.MjData(model)
    data.qpos[:] = robot.default_qpos
    mujoco.mj_forward(model, data)
    center, hands = solve_pose(robot, data, geometry)
    points, support = foot_support(robot, data)
    _, motors, joints, _ = load_sweep(robot, data, geometry, center, points, support)
    ctrl = robot.actuator_coordinate_transform @ data.qpos[robot.actuator_qpos_indices]
    policy_indices = np.asarray(metadata["policy_indices"])
    if not np.allclose(
        ctrl[policy_indices], np.asarray(metadata["default_position"])[policy_indices]
    ):
        raise ValueError(
            "policy default lower-body pose differs from the starting stance"
        )
    output.mkdir(parents=True, exist_ok=True)
    source = ET.parse(robot.xml_path).getroot()
    cases = []
    # The zero-payload control has the same arm pose but no bar or attachments.
    for mass in (0.0, *config.hold.payloads_kg):
        name = f"{mass:g}kg-locomotion"
        directory = output / name
        directory.mkdir(exist_ok=True)
        root = compose_scene(
            source,
            geometry,
            center,
            mass,
            data.qpos,
            ctrl,
            np.zeros(model.nu),
            config.hold,
        )
        if mass and not config.bar_robot_contact:
            contact = root.find("contact")
            excluded = {
                frozenset((node.get("body1"), node.get("body2")))
                for node in contact.findall("exclude")
            }
            for body in root.findall(".//body"):
                body_name = body.get("name")
                if (
                    body_name != "barbell"
                    and frozenset((body_name, "barbell")) not in excluded
                ):
                    ET.SubElement(contact, "exclude", body1=body_name, body2="barbell")
        path = directory / "scene.xml"
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        loaded = mujoco.MjModel.from_xml_path(str(path))
        with (directory / "policy.log").open("w") as log:
            controller = NativeController(config, metadata, loaded, ctrl, log)
            try:
                result = run_hold(
                    robot,
                    geometry,
                    config.hold,
                    loaded,
                    directory,
                    controller=controller,
                    control_timestep=metadata["control_timestep"],
                    command_delay_seconds=metadata["environment"][
                        "command_delay_seconds"
                    ],
                )
            finally:
                controller.close()
        case = dict(
            name=name,
            payload_kg=mass,
            controller="locomotion",
            policy_calls=controller.calls,
            constrained_commands=controller.constrained_commands,
            **result,
        )
        cases.append(case)
        print(
            f"{name}: {result['elapsed_s']:.3f}s, {result['stop_reason'] or 'completed'}",
            flush=True,
        )
    report = dict(
        geometry=asdict(geometry),
        hands=hands,
        motors=motors,
        joint_effort_limits=joints,
        cases=cases,
        policy=dict(
            path=str(config.policy),
            source_commit=metadata["git_commit"],
            binary=str(config.binary),
            control_timestep=metadata["control_timestep"],
            bundled_robot_mass_kg=float(policy_physics.body_mass.sum()),
            simulated_robot_mass_kg=float(model.body_mass.sum()),
            feedback_filter=asdict(controller.filtered),
            observation_delay_steps=config.observation_delay_steps,
            command_delay_seconds=metadata["environment"]["command_delay_seconds"],
            bar_robot_contact=config.bar_robot_contact,
        ),
        hold_config={**asdict(config.hold), "geometry": str(config.hold.geometry)},
        limitations=[
            "Existing native locomotion policy, zero walking command; arms hold the reviewed grip targets with unchanged PD gains and no feedforward.",
            "Clean simulated IMU at the canonical site, checkpoint feedback filtering and command delay; no CAN/serial timing, estimator, noise or model randomization.",
            "Current canonical robot mass is used, not the bundle's historical robot mass; both are recorded.",
            "Ideal shaft-axis hinges have unlimited force and transverse torque capacity, with no finger/slip, thermal or power-board model.",
            "The baseline has no bar or grip constraints. Loaded cases allow balance motion and stop at explicit diagnostic thresholds.",
            "A completed diagnostic is not deployment approval or validation of a clean-and-jerk trajectory.",
        ],
    )
    if not config.bar_robot_contact:
        report["limitations"].insert(
            0,
            "COUNTERFACTUAL: bar/robot contacts are disabled, permitting unphysical overlap. This only isolates contact assistance; it is not a feasible grasp/pose validation.",
        )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    shutil.copyfile(
        Path(__file__).with_name("barbell_hold_viewer.html"), output / "index.html"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_suite(
        structured_config(PolicyHoldConfig, OmegaConf.load(args.config)), args.output
    )


if __name__ == "__main__":
    main()
