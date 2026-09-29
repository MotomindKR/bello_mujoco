"""Prepare smooth EEF-load arm targets and screen their inverse dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.evaluation.barbell import foot_support
from bello_mujoco.evaluation.barbell_reference import smoothstep
from bello_mujoco.evaluation.statics import static_equilibrium
from bello_mujoco.sim import load_model
from bello_mujoco.terrain import FLAT_TERRAIN


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
        if not is_loaded:
            model.body_mass[robot.payload_body_ids] = 0
            model.body_inertia[robot.payload_body_ids] = 0
        mujoco.mj_setConst(model, data)
        positions = np.repeat(robot.default_qpos[None], len(phase), axis=0)
        positions[:, robot.actuator_qpos_indices] += phase[:, None] * (endpoint - home)
        commands = (
            positions[:, robot.actuator_qpos_indices]
            @ robot.actuator_coordinate_transform.T
        )
        for command in commands:
            robot.target_limits.validate_reference(command)
        velocities = np.zeros((len(phase), model.nv))
        for i in range(1, len(phase) - 1):
            mujoco.mj_differentiatePos(
                model, velocities[i], 0.04, positions[i - 1], positions[i + 1]
            )
        accelerations = np.gradient(velocities, 0.02, axis=0)
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
            passed=not failed and worst <= 1 and penetration <= 0.001,
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
        "--output", type=Path, default=Path("outputs/payload-standing-15kg")
    )
    prepare(parser.parse_args().output)
