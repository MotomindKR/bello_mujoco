"""Prepare smooth EEF-load arm targets and screen their inverse dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.evaluation.barbell import foot_support
from bello_mujoco.evaluation.barbell_reference import smoothstep
from bello_mujoco.evaluation.statics import static_equilibrium
from bello_mujoco.sim import load_model
from bello_mujoco.terrain import FLAT_TERRAIN, format_vec
from bello_mujoco.models.urdf import end_effector_payload_body_ids


def arm_pd_model(robot, pitch: float = 0.0):
    """Native arm rig with the root/legs fixed at home; never used for training."""
    if robot.bar_body_id >= 0 or not robot.payload_body_ids.size:
        raise ValueError("arm PD screening requires independent EEF payloads")
    model = robot.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.default_qpos
    mujoco.mj_forward(model, data)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "arms.xml"
        mujoco.mj_saveLastXML(str(path), model)
        root = ET.parse(path).getroot()
    held = {robot.actuator_joint_names[i] for i in robot.hold_ctrl_indices}
    for body in root.iter("body"):
        joints = list(body.findall("joint"))
        if not joints or all(joint.get("name") in held for joint in joints):
            continue
        body_id = model.body(body.get("name")).id
        parent = model.body_parentid[body_id]
        rotation = data.xmat[parent].reshape(3, 3).T
        body.set("pos", format_vec(rotation @ (data.xpos[body_id] - data.xpos[parent])))
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(
            quaternion, (rotation @ data.xmat[body_id].reshape(3, 3)).ravel()
        )
        body.set("quat", format_vec(quaternion))
        for joint in joints:
            body.remove(joint)
    for tag in ("keyframe", "tendon"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    actuators = root.find("actuator")
    for actuator in list(actuators):
        if actuator.get("joint") not in held:
            actuators.remove(actuator)
    probe = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # Rotated gravity gives the same arm loads as pitching the fixed rig.
    gravity = np.linalg.norm(probe.opt.gravity)
    probe.opt.gravity[:] = [gravity * np.sin(pitch), 0, -gravity * np.cos(pitch)]
    return probe


def screen_arm_pd(robot, commands, mass, pitch=0.0, strength=1.0, reset_positions=()):
    model = arm_pd_model(robot, pitch)
    payload = end_effector_payload_body_ids(model)
    ratio = mass / model.body_mass[payload].sum()
    model.body_mass[payload] *= ratio
    model.body_inertia[payload] *= ratio
    model.actuator_gainprm[:, 0] *= strength
    model.actuator_biasprm[:, 1:3] *= strength
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
    indices = np.array([robot.actuator_joint_names.index(name) for name in names])
    qpos_ids = model.jnt_qposadr[model.actuator_trnid[:, 0]]
    qvel_ids = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    data.qpos[qpos_ids] = robot.default_qpos[robot.actuator_qpos_indices[indices]]
    for name, value in reset_positions:
        data.qpos[int(model.joint(name).qposadr[0])] = value
    substeps = round(0.02 / model.opt.timestep)
    if not np.isclose(substeps * model.opt.timestep, 0.02):
        raise ValueError("arm PD screen requires an integral 50 Hz control period")
    warmup = 100
    targets = np.r_[np.repeat(commands[:1], warmup, axis=0), commands]
    positions, velocities, accelerations = [], [], []
    peak_contact = peak_startup_contact = peak_effort = 0.0
    contact_frames = startup_contact_frames = 0
    max_joint_violation = 0.0
    force_limits = np.max(np.abs(model.actuator_forcerange), axis=1)
    robot_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] != 0}
    for frame, target in enumerate(targets):
        data.ctrl[:] = target[indices]
        frame_contact = 0.0
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            for contact_id, contact in enumerate(data.contact):
                if (
                    int(contact.geom1) not in robot_geoms
                    or int(contact.geom2) not in robot_geoms
                ):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_id, wrench)
                frame_contact = max(frame_contact, abs(wrench[0]))
        if frame < warmup:
            peak_startup_contact = max(peak_startup_contact, frame_contact)
            startup_contact_frames += frame_contact > 1.0
            continue
        peak_contact = max(peak_contact, frame_contact)
        contact_frames += frame_contact > 1.0
        peak_effort = max(
            peak_effort, float(np.max(np.abs(data.actuator_force) / force_limits))
        )
        actual = data.qpos[qpos_ids]
        limits = model.jnt_range[model.actuator_trnid[:, 0]]
        max_joint_violation = max(
            max_joint_violation,
            float(np.max(np.maximum(limits[:, 0] - actual, actual - limits[:, 1]))),
        )
        positions.append(actual.copy())
        velocities.append(data.qvel[qvel_ids].copy())
        accelerations.append(data.qacc[qvel_ids].copy())
    report = dict(
        mass_kg=mass,
        pitch_radians=pitch,
        actuator_strength=strength,
        contact_frames=int(contact_frames),
        peak_contact_force_n=peak_contact,
        startup_contact_frames=int(startup_contact_frames),
        startup_peak_contact_force_n=peak_startup_contact,
        maximum_effort_fraction=peak_effort,
        max_joint_violation_radians=max_joint_violation,
    )
    report["passed"] = (
        contact_frames == 0
        and startup_contact_frames == 0
        and peak_effort < 0.99
        and max_joint_violation < 0.001
    )
    return (
        report,
        np.asarray(positions),
        np.asarray(velocities),
        np.asarray(accelerations),
        indices,
    )


def prepare(output: Path) -> None:
    profile = locomotion_training_profile("bello_mini_payload_standing")
    robot = load_model(
        profile.asset_config, terrain=FLAT_TERRAIN, spawn_clearance=0, load_mjx=False
    )
    model = robot.mj_model
    data = mujoco.MjData(model)
    # One-second endpoint holds and four-second quintic transitions, at 50 Hz.
    phase = np.r_[
        np.zeros(50),
        smoothstep(np.linspace(0, 1, 201)),
        np.ones(50),
        smoothstep(np.linspace(1, 0, 201)),
        np.zeros(50),
    ]
    home = robot.default_qpos[robot.actuator_qpos_indices]
    defaults = ActuatorConfig.for_asset("bello_mini")
    rest = np.array(
        [defaults.joints[name].default_pos for name in robot.actuator_joint_names]
    )
    variants = [("hold", home.copy(), True)]
    for joint, amount in [
        ("shoulder_pitch", 0.06),
        ("elbow_pitch", 0.10),
        ("wrist_pitch", 0.10),
    ]:
        for sign in [-1, 1]:
            endpoint = home.copy()
            for side in ["left", "right"]:
                endpoint[robot.actuator_joint_names.index(f"{side}_{joint}_joint")] += (
                    sign * amount
                )
            variants.append((f"{joint}_{sign}", endpoint, True))
    for fraction in [0.5, 1.0]:
        endpoint = home.copy()
        held = robot.hold_ctrl_indices
        endpoint[held] += fraction * (rest[held] - home[held])
        variants.append((f"rest_{fraction}", endpoint, False))

    motor_map = np.zeros((model.nv, model.nu))
    motor_map[robot.actuator_qvel_indices] = robot.actuator_coordinate_transform.T
    limited = model.jnt_actfrclimited[robot.actuator_joint_ids].astype(bool)
    effort_map = np.r_[np.eye(model.nu), robot.actuator_coordinate_transform.T[limited]]
    bounds = np.r_[
        model.actuator_forcerange,
        model.jnt_actfrcrange[robot.actuator_joint_ids[limited]],
    ]
    nominal_mass = model.body_mass.copy()
    nominal_inertia = model.body_inertia.copy()
    targets, loaded, reports = [], [], []
    output.mkdir(parents=True, exist_ok=True)
    for name, endpoint, is_loaded in variants:
        model.body_mass[:] = nominal_mass
        model.body_inertia[:] = nominal_inertia
        mujoco.mj_setConst(model, data)
        positions = np.repeat(robot.default_qpos[None], len(phase), axis=0)
        positions[:, robot.actuator_qpos_indices] += phase[:, None] * (endpoint - home)
        commands = (
            positions[:, robot.actuator_qpos_indices]
            @ robot.actuator_coordinate_transform.T
        )
        for command in commands:
            robot.target_limits.validate_reference(command)
        screens = []
        for mass in [0.0, 5.0, 10.0, 15.0] if is_loaded else [0.0]:
            for pitch, strength in [(0.0, 1.0), (-0.15, 0.95), (0.15, 1.05)]:
                screen, actual, velocity, acceleration, indices = screen_arm_pd(
                    robot,
                    commands,
                    mass,
                    pitch,
                    strength,
                    profile.env_config.arm_reset_joint_positions,
                )
                screens.append(screen)
                if mass == (15.0 if is_loaded else 0.0) and pitch == 0:
                    positions[:, robot.actuator_qpos_indices[indices]] = actual
                    velocities = np.zeros((len(phase), model.nv))
                    accelerations = np.zeros_like(velocities)
                    velocities[:, robot.actuator_qvel_indices[indices]] = velocity
                    accelerations[:, robot.actuator_qvel_indices[indices]] = (
                        acceleration
                    )
        if not is_loaded:
            model.body_mass[robot.payload_body_ids] = 0
            model.body_inertia[robot.payload_body_ids] = 0
            mujoco.mj_setConst(model, data)
        worst = penetration = 0.0
        failed = []
        peak = np.zeros(model.nu)
        mass_matrix = np.zeros((model.nv, model.nv))
        for i in range(len(phase)):
            data.qpos[:] = positions[i]
            data.qvel[:] = velocities[i]
            mujoco.mj_forward(model, data)
            for contact in data.contact:
                bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
                if 0 in bodies and any(body in robot.foot_body_ids for body in bodies):
                    continue
                penetration = max(penetration, -float(contact.dist))
            # Collision checks at 50 Hz, inverse dynamics at 5 Hz plus the endpoint.
            if i % 10 and i != len(phase) - 1:
                continue
            points, _ = foot_support(robot, data)
            columns = []
            friction = float(np.min(model.geom_friction[robot.foot_geom_ids, 0]))
            for point in points:
                body = int(
                    robot.foot_body_ids[
                        np.argmin(
                            np.linalg.norm(
                                data.xpos[robot.foot_body_ids, :2] - point[:2], axis=1
                            )
                        )
                    ]
                )
                jacobian = np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacobian, None, point, body)
                for x, y in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    columns.append(
                        jacobian.T
                        @ [x * friction / np.sqrt(2), y * friction / np.sqrt(2), 1]
                    )
            mujoco.mj_fullM(model, data, mass_matrix)
            required = (
                mass_matrix @ accelerations[i] + data.qfrc_bias - data.qfrc_passive
            )
            solution = static_equilibrium(
                motor_map, np.column_stack(columns), required, effort_map, bounds
            )
            if solution is None:
                failed.append(i)
            else:
                worst = max(worst, solution.maximum_limit_fraction)
                peak = np.maximum(peak, np.abs(solution.motor_torque))
        result = dict(
            name=name,
            loaded=is_loaded,
            passed=not failed
            and worst <= 0.98
            and penetration <= 0.001
            and all(item["passed"] for item in screens),
            pd_screens=screens,
            maximum_effort_fraction=worst,
            max_unintended_penetration_m=penetration,
            no_equilibrium_frames=failed,
            peak_motor_torque_nm=peak.tolist(),
        )
        reports.append(result)
        print(json.dumps(result), flush=True)
        if result["passed"]:
            targets.append(commands)
            loaded.append(is_loaded)
        (output / "proxy_pose_screens.json").write_text(json.dumps(reports, indent=2))
    if not any(loaded) or all(loaded):
        raise ValueError("screen must pass both loaded and unloaded trajectories")
    np.savez_compressed(
        output / "standing.npz",
        targets=np.asarray(targets, dtype=np.float32),
        loaded=loaded,
        fps=50.0,
        passed=True,
        joint_names=np.array(robot.actuator_joint_names),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/payload-standing-15kg-v2")
    )
    prepare(parser.parse_args().output)
