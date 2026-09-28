"""Free-base barbell holding diagnostics with ideal shaft-axis hinges."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Literal
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from omegaconf import OmegaConf

from bello_mujoco.configuration import structured_config
from bello_mujoco.evaluation.barbell import (
    BarbellConfig,
    foot_support,
    load_sweep,
    solve_pose,
    visual_scene,
)
from bello_mujoco.models.barbell import add_barbell, grip_axis_errors
from bello_mujoco.models import repo_root
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.models.ratings import motor_continuous_torque
from bello_mujoco.sim import load_model
from bello_mujoco.terrain import FLAT_TERRAIN, add_plane, format_vec


@dataclass(frozen=True)
class HoldConfig:
    geometry: Path
    payloads_kg: tuple[float, ...]
    controllers: tuple[Literal["pd", "pd_static_feedforward"], ...]
    duration_s: float
    timestep_s: float
    sample_period_s: float
    grip_solref: tuple[float, float]
    max_bar_displacement_m: float
    max_root_drop_m: float
    max_root_tilt_rad: float
    max_grip_separation_m: float
    max_joint_violation_rad: float
    max_unexpected_penetration_m: float
    max_grip_axis_error_rad: float = 0.1

    def __post_init__(self):
        positive = (
            *self.payloads_kg,
            self.duration_s,
            self.timestep_s,
            self.sample_period_s,
            *self.grip_solref,
            self.max_bar_displacement_m,
            self.max_root_drop_m,
            self.max_root_tilt_rad,
            self.max_grip_separation_m,
            self.max_grip_axis_error_rad,
            self.max_joint_violation_rad,
            self.max_unexpected_penetration_m,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("hold parameters must be finite and positive")
        if not self.payloads_kg or not self.controllers:
            raise ValueError("at least one payload and controller are required")
        if len(set(self.payloads_kg)) != len(self.payloads_kg) or len(
            set(self.controllers)
        ) != len(self.controllers):
            raise ValueError("hold cases must be unique")
        if any(c not in {"pd", "pd_static_feedforward"} for c in self.controllers):
            raise ValueError("unknown diagnostic controller")
        for duration in (self.sample_period_s, self.duration_s):
            if not np.isclose(
                duration / self.timestep_s, round(duration / self.timestep_s)
            ):
                raise ValueError(
                    "duration and sample period must be timestep multiples"
                )
        if (
            self.sample_period_s < self.timestep_s
            or self.grip_solref[0] < 2 * self.timestep_s
        ):
            raise ValueError(
                "invalid sampling interval or underresolved grip constraint"
            )


def add_feedforward(root, torques):
    """Native position PD + constant bias, inside the original torque clamps."""
    actuators = root.find("actuator")
    if (
        actuators is None
        or len(actuators) != len(torques)
        or not np.isfinite(torques).all()
    ):
        raise ValueError("one finite feedforward torque per actuator is required")
    for node, torque in zip(actuators, torques, strict=True):
        if node.tag != "position" or node.get("forcelimited") != "true":
            raise ValueError("hold screen requires torque-limited position actuators")
        kp, kd = float(node.attrib.pop("kp")), float(node.attrib.pop("kv"))
        node.tag = "general"
        node.set("gaintype", "fixed")
        node.set("gainprm", format_vec((kp,)))
        node.set("biastype", "affine")
        node.set("biasprm", format_vec((torque, -kp, -kd)))
        node.set("ctrllimited", "true")


def compose_scene(source, geometry, center, mass, qpos, ctrl, feedforward, hold):
    root = deepcopy(source)
    option = root.find("option")
    option.set("timestep", str(hold.timestep_s))
    option.set("integrator", "implicitfast")
    option.set("iterations", "100")
    add_plane(root, FLAT_TERRAIN)
    if not np.isfinite(mass) or mass < 0:
        raise ValueError("payload mass must be finite and nonnegative")
    if mass:
        add_barbell(root, geometry, center, mass, hold.grip_solref)
    add_feedforward(root, feedforward)
    for old in root.findall("keyframe"):
        root.remove(old)
    keyframe = ET.SubElement(root, "keyframe")
    ET.SubElement(
        keyframe,
        "key",
        name="loaded_start",
        qpos=format_vec(np.r_[qpos, center, 1, 0, 0, 0] if mass else qpos),
        ctrl=format_vec(ctrl),
    )
    ET.indent(root, space="  ")
    return root


def grip_measurements(model, data, geometry):
    gaps, forces = [], []
    for hand in geometry.hands:
        a = model.site(f"{hand.side}_hand_grip").id
        b = model.site(f"{hand.side}_bar_grip").id
        gaps.append(float(np.linalg.norm(data.site_xpos[a] - data.site_xpos[b])))
        resultant = np.zeros(3)
        for anchor in ("grip", "grip_axis"):
            equality = model.equality(f"{hand.side}_ideal_{anchor}").id
            rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (
                data.efc_id == equality
            )
            if rows.sum() != 3:
                raise RuntimeError("hinge anchor must have exactly three force rows")
            resultant += data.efc_force[rows]
        forces.append(float(np.linalg.norm(resultant)))
    return np.asarray(gaps), np.asarray(forces)


def unexpected_contacts(model, data, foot_bodies):
    ground = model.geom("ground").id
    collisions = []
    for index, contact in enumerate(data.contact):
        if (
            contact.geom1 == ground and model.geom_bodyid[contact.geom2] in foot_bodies
        ) or (
            contact.geom2 == ground and model.geom_bodyid[contact.geom1] in foot_bodies
        ):
            continue
        if contact.dist < 0:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, wrench)
            collisions.append(
                {
                    "geom1": model.geom(contact.geom1).name,
                    "geom2": model.geom(contact.geom2).name,
                    "penetration_m": float(-contact.dist),
                    "force_on_geom2_world_n": (
                        contact.frame.reshape(3, 3).T @ wrench[:3]
                    ).tolist(),
                }
            )
    return collisions


def run_hold(
    robot,
    geometry,
    hold,
    model,
    output,
    *,
    controller=None,
    control_timestep=None,
    command_delay_seconds=0.0,
):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    root_id = model.body(robot.robot.root_body_name).id
    bar_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "barbell")
    loaded = bar_id >= 0
    foot_bodies = {
        model.body(robot.mj_model.body(int(i)).name).id for i in robot.foot_body_ids
    }
    joint_ids = np.array([model.joint(name).id for name in robot.actuator_joint_names])
    limited = np.flatnonzero(model.jnt_actfrclimited[joint_ids])
    effort_dofs = model.jnt_dofadr[joint_ids[limited]]
    effort_map = robot.actuator_coordinate_transform.T[limited]
    effort_bounds = model.jnt_actfrcrange[joint_ids[limited]]
    initial_root_z = float(data.xpos[root_id, 2])
    initial_bar = data.xpos[bar_id].copy() if loaded else None
    initial_root = data.xpos[root_id].copy()
    initial_collisions = unexpected_contacts(model, data, foot_bodies)
    initial_gap = grip_measurements(model, data, geometry)[0] if loaded else np.zeros(2)
    if loaded and np.max(grip_axis_errors(model, data)) > 0.01:
        raise ValueError(
            "hand axes do not align with the shaft in the loaded start pose"
        )
    if max(initial_gap) > 0.002:
        raise ValueError("grip sites do not coincide in the loaded start pose")
    if any(
        c["penetration_m"] > hold.max_unexpected_penetration_m
        for c in initial_collisions
    ):
        raise ValueError(
            f"loaded start has an unexpected collision: {initial_collisions}"
        )
    visual_scene(model, data).export(output / "scene.glb")
    bodies = [model.body(i).name for i in range(model.nbody)]
    visuals = {
        model.geom(i).name: int(model.geom_bodyid[i])
        for i in np.flatnonzero(model.geom_group == 1)
    }
    initial_transforms = np.c_[data.xpos, data.xquat].copy()
    peak = robot.actuator_peak_force
    max_torque = np.zeros(model.nu)
    saturation_s = np.zeros(model.nu)
    above_rated_s = np.zeros(model.nu)
    max_joint_torque = np.zeros(len(limited))
    max_joint_clipping = np.zeros(len(limited))
    joint_clipping_s = np.zeros(len(limited))
    rated = motor_continuous_torque(robot)
    max_grip_force = np.zeros(2)
    maxima = {
        "bar_displacement_m": 0.0,
        "root_drop_m": 0.0,
        "root_tilt_rad": 0.0,
        "grip_separation_m": 0.0,
        "grip_axis_error_rad": 0.0,
        "joint_violation_rad": 0.0,
        "unexpected_penetration_m": 0.0,
    }
    frames = []
    control_steps, delay_steps = 1, 0
    requested = data.ctrl.copy()
    if controller is not None:
        from bello_mujoco.envs.feedback import validate_command_delay

        control_steps = round(control_timestep / hold.timestep_s)
        if control_steps < 1 or not np.isclose(
            control_steps * hold.timestep_s, control_timestep
        ):
            raise ValueError("control period must be a positive physics-step multiple")
        validate_command_delay(command_delay_seconds, hold.timestep_s, control_steps)
        delay_steps = round(command_delay_seconds / hold.timestep_s)
    stride = round(hold.sample_period_s / hold.timestep_s)
    stop = None
    steps = round(hold.duration_s / hold.timestep_s)
    for step in range(steps + 1):
        if step:
            if controller is not None:
                substep = (step - 1) % control_steps
                if substep == 0:
                    requested = np.asarray(controller(data))
                    if (
                        requested.shape != (model.nu,)
                        or not np.isfinite(requested).all()
                    ):
                        raise ValueError("controller returned invalid motor targets")
                if substep == delay_steps:
                    data.ctrl[:] = requested
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or np.any(data.warning.number)
        ):
            stop = "nonfinite state or MuJoCo solver warning"
            break
        gap, grip_force = (
            grip_measurements(model, data, geometry)
            if loaded
            else (np.zeros(2), np.zeros(2))
        )
        q = data.qpos[model.jnt_qposadr[joint_ids]]
        bounds = model.jnt_range[joint_ids]
        joint_violation = max(
            float(np.max(bounds[:, 0] - q)), float(np.max(q - bounds[:, 1])), 0.0
        )
        collisions = unexpected_contacts(model, data, foot_bodies)
        metrics = {
            "bar_displacement_m": float(np.linalg.norm(data.xpos[bar_id] - initial_bar))
            if loaded
            else 0.0,
            "root_drop_m": max(initial_root_z - float(data.xpos[root_id, 2]), 0.0),
            "root_tilt_rad": float(
                np.arccos(np.clip(data.xmat[root_id].reshape(3, 3)[2, 2], -1, 1))
            ),
            "grip_separation_m": float(max(gap)),
            "grip_axis_error_rad": float(np.max(grip_axis_errors(model, data)))
            if loaded
            else 0.0,
            "joint_violation_rad": joint_violation,
            "unexpected_penetration_m": max(
                (c["penetration_m"] for c in collisions), default=0.0
            ),
        }
        for name, value in metrics.items():
            maxima[name] = max(maxima[name], value)
            if value > getattr(hold, f"max_{name}") and stop is None:
                stop = name
        torque = np.abs(data.actuator_force)
        requested_effort = effort_map @ data.actuator_force
        actual_effort = data.qfrc_actuator[effort_dofs]
        clipping = np.abs(requested_effort - actual_effort)
        if not np.allclose(
            actual_effort,
            np.clip(requested_effort, effort_bounds[:, 0], effort_bounds[:, 1]),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "diagnostic transmission disagrees with MuJoCo joint effort"
            )
        max_joint_torque = np.maximum(max_joint_torque, np.abs(actual_effort))
        max_joint_clipping = np.maximum(max_joint_clipping, clipping)
        max_torque = np.maximum(max_torque, torque)
        max_grip_force = np.maximum(max_grip_force, grip_force)
        if step:
            saturation_s += (torque >= 0.999 * peak) * hold.timestep_s
            above_rated_s += (torque > rated) * hold.timestep_s
            joint_clipping_s += (clipping > 1e-6) * hold.timestep_s
        if step % stride == 0 or stop or step == steps:
            frames.append(
                {
                    "time_s": float(data.time),
                    "body_pose": np.c_[data.xpos, data.xquat].round(7).tolist(),
                    "torque_nm": data.actuator_force.round(6).tolist(),
                    "joint_torque_nm": actual_effort.round(6).tolist(),
                    "joint_requested_torque_nm": requested_effort.round(6).tolist(),
                    "motor_target": data.ctrl.round(6).tolist(),
                    "motor_position": data.actuator_length.round(6).tolist(),
                    "motor_velocity": data.actuator_velocity.round(6).tolist(),
                    "grip_force_n": grip_force.round(4).tolist(),
                    "metrics": metrics,
                    "unexpected_contacts": collisions,
                }
            )
        if stop:
            break
    report = {
        "completed_hold": stop is None,
        "stop_reason": stop,
        "elapsed_s": float(data.time),
        "root_displacement_m": (data.xpos[root_id] - initial_root).tolist(),
        "maxima": maxima,
        "max_abs_torque_nm": max_torque.tolist(),
        "saturation_s": saturation_s.tolist(),
        "above_rated_s": above_rated_s.tolist(),
        "max_abs_joint_torque_nm": max_joint_torque.tolist(),
        "max_joint_clipping_nm": max_joint_clipping.tolist(),
        "joint_clipping_s": joint_clipping_s.tolist(),
        "max_grip_force_n": max_grip_force.tolist(),
        "bodies": bodies,
        "visual_bodies": visuals,
        "initial_body_pose": initial_transforms.tolist(),
        "frames": frames,
    }
    (output / "rollout.json").write_text(
        json.dumps(report, separators=(",", ":"), allow_nan=False)
    )
    return {
        k: v
        for k, v in report.items()
        if k not in {"bodies", "visual_bodies", "initial_body_pose", "frames"}
    }


def run_suite(config, output):
    geometry = structured_config(
        BarbellConfig, OmegaConf.load(repo_root() / config.geometry)
    )
    if not set(config.payloads_kg) <= set(geometry.payload_samples_kg):
        raise ValueError(
            "hold payloads must be included in the geometry's static samples"
        )
    robot = load_model(
        ModelConfig(
            asset=geometry.asset,
            actuator_mode="policy",
            target_total_mass_kg=geometry.robot_mass_kg,
        ),
        load_mjx=False,
        use_viewer_model=True,
        spawn_clearance=0,
    )
    data = mujoco.MjData(robot.mj_model)
    data.qpos[:] = robot.default_qpos
    mujoco.mj_forward(robot.mj_model, data)
    center, hands = solve_pose(robot, data, geometry)
    support, support_map = foot_support(robot, data)
    samples, motors, joints, _ = load_sweep(
        robot, data, geometry, center, support, support_map
    )
    source = ET.parse(robot.xml_path).getroot()
    ctrl = robot.actuator_coordinate_transform @ data.qpos[robot.actuator_qpos_indices]
    cases = []
    output.mkdir(parents=True, exist_ok=True)
    for mass in config.payloads_kg:
        sample = next(s for s in samples if s["bar_mass_kg"] == mass)
        if sample["torque_nm"] is None:
            raise ValueError(f"no static equilibrium at {mass} kg; revise pose first")
        for controller in config.controllers:
            name = f"{mass:g}kg-{controller}"
            directory = output / name
            directory.mkdir(exist_ok=True)
            feedforward = (
                np.array(sample["torque_nm"])
                if controller == "pd_static_feedforward"
                else np.zeros(robot.mj_model.nu)
            )
            root = compose_scene(
                source, geometry, center, mass, data.qpos, ctrl, feedforward, config
            )
            path = directory / "scene.xml"
            ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
            model = mujoco.MjModel.from_xml_path(str(path))
            result = run_hold(robot, geometry, config, model, directory)
            cases.append(
                {
                    "name": name,
                    "payload_kg": mass,
                    "controller": controller,
                    "feedforward_nm": feedforward.tolist(),
                    "static_effort_fraction": sample["maximum_effort_fraction"],
                    **result,
                }
            )
            print(
                f"{name}: {result['elapsed_s']:.3f}s, {result['stop_reason'] or 'completed'}",
                flush=True,
            )
    document = {
        "config": {**asdict(config), "geometry": str(config.geometry)},
        "geometry": asdict(geometry),
        "hands": hands,
        "motors": motors,
        "joint_effort_limits": joints,
        "cases": cases,
        "limitations": [
            "Ideal shaft-axis hinges have unlimited force and transverse torque capacity and do not model fingers or slip.",
            "No world/root/foot supports; original motor and combined joint effort limits are retained.",
            "Diagnostic PD with optional static feedforward is not a learned policy.",
            "No motor thermal/current/power-board model, and hand frames remain provisional.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )
    shutil.copyfile(
        Path(__file__).with_name("barbell_hold_viewer.html"), output / "index.html"
    )
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_suite(structured_config(HoldConfig, OmegaConf.load(args.config)), args.output)


if __name__ == "__main__":
    main()
