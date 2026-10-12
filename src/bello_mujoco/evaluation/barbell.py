"""Offline barbell pose/load preview. Never opens a hardware connection."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

from bello_mujoco.configuration import structured_config
from bello_mujoco.evaluation.statics import static_equilibrium, two_end_weight_inertia
from bello_mujoco.models import repo_root, robot_descriptions_root
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.models.ratings import motor_continuous_torque
from bello_mujoco.sim import load_model


@dataclass(frozen=True)
class Hand:
    body: str
    side: str
    grip_local_m: tuple[float, float, float]
    palm_normal_local: tuple[float, float, float]
    shaft_axis_local: tuple[float, float, float] = (0.0, 1.0, 0.0)


@dataclass(frozen=True)
class BarbellConfig:
    asset: str
    bar_mass_kg: float
    bar_length_m: float
    shaft_diameter_m: float
    end_weight_radius_m: float
    end_weight_width_m: float
    grip_spacing_m: float
    torso_body: str
    bar_center_in_torso_m: tuple[float, float, float]
    hands: tuple[Hand, Hand]
    hand_rpy_rad: tuple[float, float, float]
    payload_samples_kg: tuple[float, ...]
    robot_mass_kg: float = 49.5

    def __post_init__(self):
        dimensions = (
            self.bar_mass_kg,
            self.robot_mass_kg,
            self.bar_length_m,
            self.shaft_diameter_m,
            self.end_weight_radius_m,
            self.end_weight_width_m,
            self.grip_spacing_m,
        )
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("bar mass and dimensions must be positive and finite")
        if self.grip_spacing_m >= self.bar_length_m - 2 * self.end_weight_width_m:
            raise ValueError("grips must lie between the end weights")
        if {h.side for h in self.hands} != {"left", "right"}:
            raise ValueError("one left and one right hand are required")
        if len({h.body for h in self.hands}) != 2:
            raise ValueError("hand bodies must be distinct")
        if any(
            not np.isclose(np.linalg.norm(h.palm_normal_local), 1.0) for h in self.hands
        ):
            raise ValueError("each assumed palm normal must be a finite unit vector")
        poses = [self.bar_center_in_torso_m, self.hand_rpy_rad]
        poses.extend(h.grip_local_m for h in self.hands)
        if not np.isfinite(poses).all():
            raise ValueError("pose coordinates must be finite")
        if (
            not self.payload_samples_kg
            or not np.isfinite(self.payload_samples_kg).all()
            or min(self.payload_samples_kg) < 0
        ):
            raise ValueError("payload samples must be finite and nonnegative")


def _point(data, body_id, local):
    return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ local


def _jacobian(model, data, body_id, position):
    jac = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jac, None, position, body_id)
    return jac


def solve_pose(robot, data, config, *, report_visual=True, wrist_pitch_targets=None):
    model = robot.mj_model
    torso = model.body(config.torso_body).id
    torso_rotation = data.xmat[torso].reshape(3, 3).copy()
    center = _point(data, torso, config.bar_center_in_torso_m)
    desired_rotation = Rotation.from_matrix(torso_rotation) * Rotation.from_euler(
        "xyz", config.hand_rpy_rad
    )
    diagnostics = []
    for hand in config.hands:
        if not np.allclose(
            desired_rotation.apply(hand.shaft_axis_local),
            torso_rotation[:, 1],
            atol=1e-6,
        ):
            raise ValueError("configured hand shaft axis does not align with the bar")
        if (
            abs(desired_rotation.apply(hand.palm_normal_local) @ torso_rotation[:, 1])
            > 1e-6
        ):
            raise ValueError("bar shaft must lie in the assumed palm plane")
        sign = 1 if hand.side == "left" else -1
        target = center + torso_rotation @ [0, sign * config.grip_spacing_m / 2, 0]
        joint_names = [
            f"{hand.side}_{part}_joint"
            for part in (
                "shoulder_pitch",
                "shoulder_roll",
                "shoulder_yaw",
                "elbow_pitch",
                "wrist_yaw",
                "wrist_pitch",
                "wrist_roll",
            )
        ]
        ids = [model.joint(name).id for name in joint_names]
        addresses = model.jnt_qposadr[ids]
        body_id = model.body(hand.body).id
        seed = np.asarray([0, sign * 0.02, 0, -np.pi / 2, 0, 0, 0])

        def residual(
            angles,
            *,
            addresses=addresses,
            body_id=body_id,
            hand=hand,
            target=target,
            seed=seed,
        ):
            data.qpos[addresses] = angles
            mujoco.mj_kinematics(model, data)
            position = _point(data, body_id, hand.grip_local_m)
            orientation = Rotation.from_matrix(data.xmat[body_id].reshape(3, 3))
            error = (desired_rotation.inv() * orientation).as_rotvec()
            orientation_or_wrist = (
                error
                if wrist_pitch_targets is None
                else np.array([angles[5] - wrist_pitch_targets[hand.side]])
            )
            return np.r_[
                position - target, 0.15 * orientation_or_wrist, 1e-5 * (angles - seed)
            ]

        result = least_squares(
            residual,
            np.clip(data.qpos[addresses].copy(), *model.jnt_range[ids].T),
            bounds=model.jnt_range[ids].T,
            max_nfev=1000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        error = residual(result.x)
        position_error = float(np.linalg.norm(error[:3]))
        actual_orientation = Rotation.from_matrix(data.xmat[body_id].reshape(3, 3))
        angle_error = float(
            np.linalg.norm((desired_rotation.inv() * actual_orientation).as_rotvec())
        )
        target_error = (
            angle_error
            if wrist_pitch_targets is None
            else abs(result.x[5] - wrist_pitch_targets[hand.side])
        )
        if (
            not result.success
            or position_error > 0.002
            or target_error > (0.03 if wrist_pitch_targets is None else 1e-4)
        ):
            raise ValueError(f"{hand.side} hand target unreachable: {result.message}")
        if not report_visual:
            continue
        spheres = np.flatnonzero(
            (model.geom_bodyid == body_id)
            & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)
            & (model.geom_group == 1)
        )
        if len(spheres) != 1:
            raise ValueError("grip preview requires exactly one visual hand sphere")
        sphere = int(spheres[0])
        grip = _point(data, body_id, hand.grip_local_m)
        diagnostics.append(
            {
                "body": hand.body,
                "position_m": _point(data, body_id, hand.grip_local_m).tolist(),
                "sphere_center_m": data.geom_xpos[sphere].tolist(),
                "sphere_radius_m": float(model.geom_size[sphere, 0]),
                "sphere_surface_to_shaft_center_m": float(
                    np.linalg.norm(grip - data.geom_xpos[sphere])
                    - model.geom_size[sphere, 0]
                ),
                "position_error_m": position_error,
                "orientation_error_rad": angle_error,
                "palm_normal_world": (
                    data.xmat[body_id].reshape(3, 3) @ hand.palm_normal_local
                ).tolist(),
                "forearm_angle_from_downward_vertical_rad": float(
                    np.arccos(
                        np.clip(
                            data.xmat[model.jnt_bodyid[ids[4]]].reshape(3, 3)[2, 2],
                            -1,
                            1,
                        )
                    )
                ),
                "joint_angles_rad": dict(
                    zip(joint_names, result.x.tolist(), strict=True)
                ),
            }
        )
    mujoco.mj_forward(model, data)
    robot.target_limits.validate_reference(
        robot.actuator_coordinate_transform @ data.qpos[robot.actuator_qpos_indices]
    )
    return center, diagnostics


def _geom_mesh(model, data, geom_id):
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[geom_id]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        mesh = trimesh.Trimesh(
            vertices=model.mesh_vert[va : va + vn].copy(),
            faces=model.mesh_face[fa : fa + fn].copy(),
            process=False,
        )
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        mesh = trimesh.creation.box(extents=2 * size)
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        mesh = trimesh.creation.cylinder(
            radius=size[0], height=2 * size[1], sections=64
        )
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=size[0])
    else:
        raise ValueError(f"unsupported preview geometry: {model.geom(geom_id).name}")
    transform = np.eye(4)
    transform[:3, :3] = data.geom_xmat[geom_id].reshape(3, 3)
    transform[:3, 3] = data.geom_xpos[geom_id]
    return mesh.apply_transform(transform)


def foot_support(robot, data):
    points, bodies = [], []
    model = robot.mj_model
    for body in robot.foot_body_ids:
        vertices = np.concatenate(
            [
                _geom_mesh(model, data, gid).vertices
                for gid in robot.foot_geom_ids
                if model.geom_bodyid[gid] == body
            ]
        )
        bottom = vertices[vertices[:, 2] < vertices[:, 2].min() + 1e-5]
        if len(bottom) < 3 or abs(bottom[:, 2].mean()) > 1e-4:
            raise ValueError("static screen requires both soles flat on z=0")
        bottom = bottom[ConvexHull(bottom[:, :2]).vertices]
        points.extend(bottom)
        bodies.extend([int(body)] * len(bottom))
    points = np.asarray(points)
    support_map = np.column_stack(
        [
            _jacobian(model, data, body, p)[2]
            for body, p in zip(bodies, points, strict=True)
        ]
    )
    return points, support_map


def load_sweep(robot, data, config, center, support_points, support_map):
    model = robot.mj_model
    gravity = model.opt.gravity
    if not np.allclose(gravity[:2], 0) or gravity[2] >= 0:
        raise ValueError("static screen requires downward vertical gravity")
    per_kg = np.zeros(model.nv)
    for hand in config.hands:
        bid = model.body(hand.body).id
        jac = _jacobian(model, data, bid, _point(data, bid, hand.grip_local_m))
        per_kg -= jac.T @ (gravity / 2)
    motor_map = np.zeros((model.nv, model.nu))
    motor_map[robot.actuator_qvel_indices] = robot.actuator_coordinate_transform.T
    ratings = motor_continuous_torque(robot).astype(float)
    joint_indices = np.flatnonzero(model.jnt_actfrclimited[robot.actuator_joint_ids])
    joint_map = robot.actuator_coordinate_transform.T[joint_indices]
    joint_bounds = model.jnt_actfrcrange[robot.actuator_joint_ids[joint_indices]]
    effort_map = np.vstack((np.eye(model.nu), joint_map))
    effort_bounds = np.vstack((model.actuator_forcerange, joint_bounds))
    joint_rows = [
        {
            "joint": robot.actuator_joint_names[i],
            "lower_nm": float(bounds[0]),
            "upper_nm": float(bounds[1]),
            "coupled": bool(np.count_nonzero(row) > 1),
        }
        for i, row, bounds in zip(joint_indices, joint_map, joint_bounds, strict=True)
    ]
    robot_mass = float(model.body_mass.sum())
    robot_com = model.body_mass @ data.xipos / robot_mass
    hull = ConvexHull(support_points[:, :2])
    samples = []
    for mass in sorted(set((*config.payload_samples_kg, config.bar_mass_kg))):
        combined_com = (robot_mass * robot_com + mass * center) / (robot_mass + mass)
        margin = -np.max(
            hull.equations[:, :2] @ combined_com[:2] + hull.equations[:, 2]
        )
        solution = static_equilibrium(
            motor_map,
            support_map,
            data.qfrc_bias + mass * per_kg,
            effort_map,
            effort_bounds,
        )
        samples.append(
            {
                "bar_mass_kg": mass,
                "combined_com_m": combined_com.tolist(),
                "support_margin_m": float(margin),
                "ground_equilibrium": solution is not None,
                "torque_nm": None
                if solution is None
                else solution.motor_torque.tolist(),
                "normal_force_n": None
                if solution is None
                else solution.normal_force.tolist(),
                "joint_torque_nm": None
                if solution is None
                else (joint_map @ solution.motor_torque).tolist(),
                "maximum_effort_fraction": None
                if solution is None
                else solution.maximum_limit_fraction,
                "residual": None if solution is None else solution.residual,
            }
        )
    hardware = OmegaConf.to_container(
        OmegaConf.load(
            robot.robot.hardware_description_path(robot_descriptions_root())
        ),
        resolve=True,
    )
    motors = {m["coordinate"]: m for m in hardware["motors"]}
    motor_rows = [
        {
            "coordinate": name,
            "model": motors[name]["model"],
            "rated_nm": float(ratings[i]),
            "peak_nm": float(robot.actuator_peak_force[i]),
        }
        for i, name in enumerate(robot.actuator_coordinate_names)
    ]
    return samples, motor_rows, joint_rows, robot_com


def visual_scene(model, data):
    scene = trimesh.Scene()
    for gid in np.flatnonzero(model.geom_group == 1):
        mesh = _geom_mesh(model, data, gid)
        mesh.visual.face_colors = np.rint(model.geom_rgba[gid] * 255).astype(np.uint8)
        scene.add_geometry(mesh, node_name=model.geom(gid).name)
    return scene


def export_scene(robot, data, config, center, output):
    scene = visual_scene(robot.mj_model, data)
    # Bar axis is Y. Its shaft is visual-only with negligible mass.
    rotation = Rotation.from_euler("x", np.pi / 2).as_matrix()
    pieces = [
        (
            "shaft",
            config.shaft_diameter_m / 2,
            config.bar_length_m,
            0,
            [180, 193, 206, 255],
        )
    ]
    offset = (config.bar_length_m - config.end_weight_width_m) / 2
    pieces.extend(
        (
            f"end_weight_{sign}",
            config.end_weight_radius_m,
            config.end_weight_width_m,
            sign * offset,
            [236, 159, 56, 255],
        )
        for sign in (-1, 1)
    )
    for name, radius, length, y, color in pieces:
        mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=64)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center + [0, y, 0]
        mesh.apply_transform(transform)
        mesh.visual.face_colors = color
        scene.add_geometry(mesh, node_name=name)
    scene.export(output / "scene.glb")


def build_report(config: BarbellConfig, output: Path):
    robot = load_model(
        ModelConfig(
            asset=config.asset,
            actuator_mode="policy",
            target_total_mass_kg=config.robot_mass_kg,
        ),
        load_mjx=False,
        use_viewer_model=True,
        spawn_clearance=0,
    )
    model = robot.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = robot.default_qpos
    mujoco.mj_forward(model, data)
    center, hands = solve_pose(robot, data, config)
    # The current screen and bar visual assume an upright torso and transverse bar.
    if not np.allclose(
        data.xmat[model.body(config.torso_body).id].reshape(3, 3), np.eye(3), atol=1e-6
    ):
        raise ValueError("barbell screen currently requires an upright torso")
    points, support_map = foot_support(robot, data)
    samples, motors, joint_rows, robot_com = load_sweep(
        robot, data, config, center, points, support_map
    )
    selected = next(s for s in samples if s["bar_mass_kg"] == config.bar_mass_kg)
    collisions = [
        {
            "geom1": model.geom(c.geom1).name,
            "geom2": model.geom(c.geom2).name,
            "penetration_m": float(-c.dist),
        }
        for c in data.contact
        if c.dist < -1e-5
    ]
    overloads = (
        []
        if selected["torque_nm"] is None
        else [
            m["coordinate"]
            for m, t in zip(motors, selected["torque_nm"], strict=True)
            if abs(t) > m["peak_nm"]
        ]
    )
    offset = (config.bar_length_m - config.end_weight_width_m) / 2
    joint_overloads = (
        []
        if selected["joint_torque_nm"] is None
        else [
            row["joint"]
            for row, torque in zip(joint_rows, selected["joint_torque_nm"], strict=True)
            if not row["lower_nm"] - 1e-6 <= torque <= row["upper_nm"] + 1e-6
        ]
    )
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
        ).strip(),
        "stack_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=robot_descriptions_root(), text=True
        ).strip(),
        "config": asdict(config),
        "robot_mass_kg": float(model.body_mass.sum()),
        "robot_com_m": robot_com.tolist(),
        "bar_center_world_m": center.tolist(),
        "bar_inertia_kg_m2": two_end_weight_inertia(
            config.bar_mass_kg,
            config.end_weight_radius_m,
            config.end_weight_width_m,
            offset,
        ).tolist(),
        "hands": hands,
        "qpos": data.qpos.tolist(),
        "motors": motors,
        "joint_effort_limits": joint_rows,
        "samples": samples,
        "support_points_m": points.tolist(),
        "robot_self_collisions": collisions,
        "peak_overloads": overloads,
        "joint_effort_overloads": joint_overloads,
        "static_screen_passed": selected["ground_equilibrium"]
        and not overloads
        and not joint_overloads
        and not collisions,
        "limitations": [
            "Gravity-only screen, not a trained rollout or permission to lift.",
            "Equal vertical load sharing; ideal grip, no finger/slip/contact validation.",
            "Hand origins/orientations are provisional sphere-proxy assumptions, not measured palms.",
            "End-weight diameter/thickness are user supplied; equal cylinders and flush placement are assumptions; shaft mass is neglected.",
            "Ground forces minimize maximum motor/joint effort utilization; leg loads are one nonunique optimistic distribution.",
            "Rated torques are repository references, not indefinite-hold thermal guarantees.",
            "No acceleration, bar-robot collision, tendon compliance, thermal, current or power-budget validation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    export_scene(robot, data, config, center, output)
    shutil.copyfile(
        Path(__file__).with_name("barbell_viewer.html"), output / "index.html"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        structured_config(BarbellConfig, OmegaConf.load(args.config)), args.output
    )
    print(f"Review: {args.output / 'index.html'}")
    print(f"Static screen passed: {report['static_screen_passed']}")
    print(
        f"Above combined joint torque limits: {', '.join(report['joint_effort_overloads']) or 'none'}"
    )
    print(
        f"Above configured peak torque: {', '.join(report['peak_overloads']) or 'none'}"
    )


if __name__ == "__main__":
    main()
