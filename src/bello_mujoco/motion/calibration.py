"""Canonical Mini pose calibration and constrained offline contact refinement."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import SimpleNamespace
from typing import Literal

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.motion.retarget_limits import (
    ContactLimitConfig,
    PlantedContactTask,
    RetargetContactLimit,
    RetargetJointLimit,
    RetargetRootTranslationLimit,
    project_contacts,
    rigid_support_geoms,
)
from bello_mujoco.sim import geom_lower_z


@dataclass(frozen=True)
class SourceContactConfig:
    height_m: float = 0.025
    height_band_m: float | None = None
    horizontal_speed_m_s: float = 0.15
    vertical_speed_m_s: float = 0.10
    angular_speed_rad_s: float = 0.5
    min_seconds: float = 0.1

    def __post_init__(self):
        values = (v for name, v in vars(self).items() if name != "height_band_m")
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("source contact thresholds must be positive and finite")
        if self.height_band_m is not None and (
            not math.isfinite(self.height_band_m)
            or not 0 < self.height_band_m <= self.height_m
        ):
            raise ValueError(
                "source height band must be positive and no greater than height_m"
            )


@dataclass(frozen=True)
class KinematicRetargetConfig:
    support_mode: Literal["feet", "hands_and_feet"] = "feet"
    initialization_mode: Literal["fitted_projection", "constrained_neutral"] = (
        "fitted_projection"
    )
    initialization_iterations: int = 40
    calibration_iterations: int = 5
    refinement_iterations: int = 3
    projection_iterations: int = 20
    contact_blend_frames: int = 3
    plant_position_cost: float = 1000.0
    maximum_support_correction_speed_m_s: float = 1.0
    contacts: ContactLimitConfig = field(default_factory=ContactLimitConfig)
    source_contacts: SourceContactConfig = field(default_factory=SourceContactConfig)
    hand_contacts: SourceContactConfig = field(
        default_factory=lambda: SourceContactConfig(height_m=0.12, height_band_m=0.025)
    )

    def __post_init__(self):
        if self.support_mode not in ("feet", "hands_and_feet"):
            raise ValueError("unknown retarget support mode")
        if self.initialization_mode not in ("fitted_projection", "constrained_neutral"):
            raise ValueError("unknown retarget initialization mode")
        if not math.isfinite(self.plant_position_cost) or self.plant_position_cost <= 0:
            raise ValueError("plant position cost must be positive and finite")
        if (
            not math.isfinite(self.maximum_support_correction_speed_m_s)
            or self.maximum_support_correction_speed_m_s <= 0
        ):
            raise ValueError("support correction speed must be positive and finite")
        for name in (
            "initialization_iterations",
            "calibration_iterations",
            "refinement_iterations",
            "projection_iterations",
            "contact_blend_frames",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


def contact_intervals(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return tuple(
        zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    )


def stationary_contacts(positions, quaternions, fps, config=SourceContactConfig()):
    """Stationary source proxies close to the known Z=0 source floor."""
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or positions.shape[1] == 0
        or len(positions) < 3
        or quaternions.shape != positions.shape[:2] + (4,)
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-6)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("source contacts require finite clocked support transforms")
    # LAFAN's source plane is Z=0. Allow negative capture bias, but never infer
    # a raised support plane merely because both feet stay above the floor.
    floor = min(0.0, float(np.quantile(positions[:, :, 2], 0.01)))
    velocity = np.gradient(positions, 1 / fps, axis=0)
    angular = np.empty(positions.shape[:2])
    for side in range(positions.shape[1]):
        rotation = Rotation.from_quat(quaternions[:, side], scalar_first=True)
        angular[1:, side] = (rotation[:-1].inv() * rotation[1:]).magnitude() * fps
    angular[0] = angular[1]
    mask = (
        (positions[:, :, 2] < floor + config.height_m)
        & (np.linalg.norm(velocity[:, :, :2], axis=-1) < config.horizontal_speed_m_s)
        & (np.abs(velocity[:, :, 2]) < config.vertical_speed_m_s)
        & (angular < config.angular_speed_rad_s)
    )
    for side in range(positions.shape[1]):
        if config.height_band_m is not None and mask[:, side].any():
            lower = np.quantile(positions[mask[:, side], side, 2], 0.1)
            mask[:, side] &= positions[:, side, 2] < lower + config.height_band_m
        for start, end in contact_intervals(mask[:, side]):
            if end - start < math.ceil(config.min_seconds * fps):
                mask[start:end, side] = False
    return mask


def source_contact_labels(targets, fps, config):
    """Label feet and optional wrists before GMR scaling or pose offsets."""
    groups = [("left_foot", "right_foot", config.source_contacts)]
    if config.support_mode == "hands_and_feet":
        groups.append(("left_wrist", "right_wrist", config.hand_contacts))
    masks = []
    for left, right, thresholds in groups:
        if missing := {left, right} - set(targets):
            raise ValueError(f"source supports are missing joints: {sorted(missing)}")
        masks.append(
            stationary_contacts(
                np.stack([targets[n][0] for n in (left, right)], axis=1),
                np.stack([targets[n][1] for n in (left, right)], axis=1),
                fps,
                thresholds,
            )
        )
    if len({len(mask) for mask in masks}) != 1:
        raise ValueError("source support clocks differ")
    return np.concatenate(masks, axis=1)


def calibration_frames(table):
    """Human hip rotation describes the thigh after all three hip hinges."""
    return tuple(
        (
            frame.replace("_hip_roll_link", "_hip_yaw_link")
            if source in ("left_hip", "right_hip")
            else frame,
            source,
            position,
            rotation,
        )
        for frame, (source, position, rotation, *_) in table.items()
    )


def _set_target(task, position, quaternion):
    task.set_target(
        mink.SE3.from_rotation_and_translation(mink.SO3(quaternion), position)
    )


def _solve(configuration, tasks, backend, iterations):
    dt = backend.model.opt.timestep
    for _ in range(iterations):
        velocity = mink.solve_ik(
            configuration,
            tasks,
            dt,
            backend.solver,
            damping=backend.damping,
            limits=backend.ik_limits,
        )
        configuration.integrate_inplace(velocity, dt)


class CalibratedRetargeter:
    """Fit anatomy as orientations, obtain canonical FK positions, then refine."""

    def __init__(self, model, backend, fps, config=KinematicRetargetConfig()):
        if model.robot.asset != "bello_mini":
            raise ValueError("canonical pose calibration is defined for bello_mini")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("retarget fps must be positive and finite")
        self.model, self.backend = model, backend
        native = model.mj_model
        if (
            native.njnt == 0
            or native.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or native.jnt_qposadr[0] != 0
        ):
            raise ValueError(
                "calibration requires the first joint to be a floating root"
            )
        self.fps, self.config = fps, config
        self.frames1 = calibration_frames(backend.ik_match_table1)
        self.frames2 = calibration_frames(backend.ik_match_table2)
        if not self.frames1 or backend.human_root_name not in {
            row[1] for row in self.frames1
        }:
            raise ValueError("calibration task table must include its human root")
        if {r[1]: r[0] for r in self.frames1} != {r[1]: r[0] for r in self.frames2}:
            raise ValueError("calibration requires matching source/body task tables")
        self.body_ids = np.asarray(
            [model.mj_model.body(frame).id for frame, *_ in self.frames2]
        )
        self.support_names = ("left_foot", "right_foot")
        if config.support_mode == "hands_and_feet":
            self.support_names += ("left_wrist", "right_wrist")
        source_indices = {source: i for i, (_, source, *_) in enumerate(self.frames2)}
        if missing := set(self.support_names) - set(source_indices):
            raise ValueError(
                f"calibration task table lacks supports: {sorted(missing)}"
            )
        self.support_indices = tuple(source_indices[n] for n in self.support_names)
        self.support_bodies = tuple(int(self.body_ids[i]) for i in self.support_indices)
        if self.support_bodies[:2] != tuple(model.foot_body_ids):
            raise ValueError("source feet must map to the canonical foot bodies")
        self.support_geoms = tuple(
            rigid_support_geoms(
                native,
                body,
                model.foot_geom_ids if index < 2 else model.robot_geom_ids,
            )
            for index, body in enumerate(self.support_bodies)
        )
        if any(not geoms for geoms in self.support_geoms):
            raise ValueError("calibration requires collision geoms for every support")
        self.bounds = RetargetJointLimit(model, 1 / fps)

    def _aligned_source(self, targets):
        required = {row[1] for row in self.frames1}
        if missing := required - set(targets):
            raise ValueError(f"calibration source is missing joints: {sorted(missing)}")
        count = None
        validated = {}
        for name in required:
            p, q = map(np.asarray, targets[name])
            if (
                p.ndim != 2
                or p.shape[1] != 3
                or len(p) < 3
                or q.shape != (len(p), 4)
                or not np.isfinite(p).all()
                or not np.isfinite(q).all()
                or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-6)
                or (count is not None and len(p) != count)
            ):
                raise ValueError(f"invalid clocked calibration transforms for {name}")
            count = len(p)
            validated[name] = p, q
        desired = {name: [] for name in required}
        for frame in range(count):
            self.backend.update_targets(
                {
                    n: [p[frame].copy(), q[frame].copy()]
                    for n, (p, q) in validated.items()
                }
            )
            for name in required:
                p, q = self.backend.scaled_human_data[name]
                desired[name].append(np.r_[p, q])
        return {name: np.asarray(values) for name, values in desired.items()}

    def _fit(self, desired):
        native = self.model.mj_model
        tasks = [
            mink.FrameTask(
                frame,
                "body",
                position_cost=position if source == self.backend.human_root_name else 0,
                orientation_cost=rotation,
                lm_damping=1,
            )
            for frame, source, position, rotation in self.frames1
        ]
        configuration = mink.Configuration(native)
        initial = self.model.default_qpos.copy()
        root = desired[self.backend.human_root_name][0]
        initial[:3], initial[3:7] = root[:3], root[3:]
        configuration.update(initial)
        bounds = RetargetJointLimit(self.model, 1 / self.fps)
        solver = SimpleNamespace(
            model=native,
            solver=self.backend.solver,
            damping=self.backend.damping,
            ik_limits=[mink.ConfigurationLimit(native), bounds],
        )
        poses, positions, heights, com_z = [], [], [], []
        for frame in range(len(desired[self.backend.human_root_name])):
            for task, (_, name, *_) in zip(tasks, self.frames1, strict=True):
                t = desired[name][frame]
                _set_target(task, t[:3], t[3:])
            _solve(
                configuration,
                tasks,
                solver,
                self.config.initialization_iterations
                if frame == 0
                else self.config.calibration_iterations,
            )
            poses.append(configuration.q.copy())
            positions.append(configuration.data.xpos[self.body_ids].copy())
            mujoco.mj_comPos(native, configuration.data)
            com_z.append(float(configuration.data.subtree_com[0, 2]))
            heights.append(
                [
                    min(geom_lower_z(native, configuration.data, g) for g in geoms)
                    for geoms in self.support_geoms
                ]
            )
        return (
            np.asarray(poses),
            np.asarray(positions),
            np.asarray(heights),
            np.asarray(com_z),
        )

    def _ground_alignment(self, poses, heights, mask):
        if mask.any():
            method = (
                "source_planted_supports"
                if self.config.support_mode == "hands_and_feet"
                else "source_planted_soles"
            )
            return float(np.median(heights[mask])), method
        native = self.model.mj_model
        data = mujoco.MjData(native)
        minima = []
        for pose in poses:
            data.qpos[:] = pose
            mujoco.mj_kinematics(native, data)
            minima.append(
                min(
                    geom_lower_z(native, data, int(g))
                    for g in self.model.robot_geom_ids
                )
            )
        return float(np.quantile(minima, 0.01)), "body_lower_envelope_no_foot_labels"

    def retarget(self, targets: Mapping, support_contacts):
        desired = self._aligned_source(targets)
        count = len(desired[self.backend.human_root_name])
        mask = np.asarray(support_contacts)
        if mask.shape != (count, len(self.support_names)) or mask.dtype != np.bool_:
            raise ValueError(
                "source support labels must be a frames-by-supports bool array"
            )
        poses, positions, heights, com_z = self._fit(desired)
        source_indices = {
            source: index for index, (_, source, *_) in enumerate(self.frames2)
        }
        thigh_distances = {}
        for side in ("left", "right"):
            hip, knee = side + "_hip", side + "_knee"
            if hip in source_indices and knee in source_indices:
                distance = np.linalg.norm(
                    positions[:, source_indices[hip]]
                    - positions[:, source_indices[knee]],
                    axis=-1,
                )
                thigh_distances[side] = {
                    "min": float(distance.min()),
                    "max": float(distance.max()),
                }
        ground, method = self._ground_alignment(poses, heights, mask)
        ground -= self.config.contacts.floor_clearance_m
        flight = np.zeros(count, dtype=bool)
        if self.config.support_mode == "hands_and_feet":
            acceleration = np.gradient(np.gradient(com_z, 1 / self.fps), 1 / self.fps)
            unsupported = ~mask.any(axis=1) & (heights.min(axis=1) - ground > 0.02)
            for start, stop in contact_intervals(unsupported):
                if (
                    stop - start >= 3
                    and np.mean(acceleration[start:stop])
                    < self.model.mj_model.opt.gravity[2] / 2
                ):
                    flight[start:stop] = True
        poses[:, 2] -= ground
        positions[:, :, 2] -= ground
        for values in desired.values():
            values[:, 2] -= ground
        native = self.model.mj_model
        tasks = [
            mink.FrameTask(frame, "body", position, rotation, lm_damping=1)
            for frame, _, position, rotation in self.frames2
        ]
        configuration = mink.Configuration(native)
        initial = poses[0].copy()
        if self.config.initialization_mode == "constrained_neutral":
            initial[self.model.actuator_qpos_indices] = self.model.default_qpos[
                self.model.actuator_qpos_indices
            ]
            configuration.update(initial)
            lowest = min(
                geom_lower_z(native, configuration.data, int(g))
                for g in self.model.robot_geom_ids
            )
            initial[2] += max(0.0, self.config.contacts.floor_clearance_m - lowest)
        configuration.update(initial)
        contact_limit = RetargetContactLimit(native, self.config.contacts)
        plants = PlantedContactTask(
            self.model,
            self.support_bodies,
            self.support_geoms,
            self.config.plant_position_cost,
            self.config.contact_blend_frames,
            self.config.contacts.floor_clearance_m,
            maximum_lowering_step_m=(
                self.config.maximum_support_correction_speed_m_s / self.fps
                if self.config.support_mode == "hands_and_feet"
                else None
            ),
        )
        root_limit = RetargetRootTranslationLimit(
            native, self.config.maximum_support_correction_speed_m_s / self.fps
        )
        self.bounds.previous = None
        solver = SimpleNamespace(
            model=native,
            configuration=configuration,
            solver=self.backend.solver,
            damping=self.backend.damping,
            ik_limits=[
                mink.ConfigurationLimit(native),
                self.bounds,
                contact_limit,
                *([root_limit] if self.config.support_mode == "hands_and_feet" else []),
            ],
        )
        # Contact anchors must start on a physically feasible configuration.
        # A fitted pose can penetrate the floor before its first constrained solve.
        project_contacts(
            solver,
            contact_limit,
            iterations=self.config.projection_iterations,
            tolerance=1e-6,
        )
        if self.config.initialization_mode == "constrained_neutral":
            for index, (task, (_, source, *_)) in enumerate(
                zip(tasks, self.frames2, strict=True)
            ):
                _set_target(task, positions[0, index], desired[source][0, 3:])
            _solve(configuration, tasks, solver, self.config.initialization_iterations)
            project_contacts(
                solver,
                contact_limit,
                iterations=self.config.projection_iterations,
                tolerance=1e-6,
            )
        result, plant_xy_errors, plant_z_errors = [], [], []
        height_offset = 0.0
        for frame in range(count):
            if frame:
                root_limit.previous = result[-1][:3].copy()
                root_limit.source_displacement = poses[frame, :3] - poses[frame - 1, :3]
            contact_targets = positions[frame].copy()
            plants.update(configuration, mask[frame])
            if self.config.support_mode == "hands_and_feet" and (
                mask[frame].any() or flight[frame]
            ):
                # Carry the support-induced height translation into unlabelled
                # flight instead of pulling the root back to its floating fit.
                contact_targets[:, 2] += height_offset
            for side, support in enumerate(self.support_indices):
                body = self.support_bodies[side]
                if body in plants.points:
                    local, anchor, _, _ = plants.points[body]
                    source = self.frames2[support][1]
                    contact_targets[support] = anchor - Rotation.from_quat(
                        desired[source][frame, 3:], scalar_first=True
                    ).apply(local)
            for index, (task, (_, source, *_)) in enumerate(
                zip(tasks, self.frames2, strict=True)
            ):
                _set_target(task, contact_targets[index], desired[source][frame, 3:])
            _solve(
                configuration,
                [*tasks, plants],
                solver,
                self.config.initialization_iterations
                if frame == 0
                else self.config.refinement_iterations,
            )
            result.append(
                project_contacts(
                    solver,
                    contact_limit,
                    iterations=self.config.projection_iterations,
                    tolerance=1e-6,
                )
            )
            self.bounds.previous = result[-1][self.model.actuator_qpos_indices].copy()
            if self.config.support_mode == "hands_and_feet":
                height_offset = float(result[-1][2] - poses[frame, 2])
            error = plants.compute_error(configuration).reshape(-1, 3)
            for index, body in enumerate(plants.bodies):
                if body in plants.points:
                    plant_xy_errors.append(float(np.linalg.norm(error[index, :2])))
                    plant_z_errors.append(float(abs(error[index, 2])))
        return np.asarray(result), {
            "support_mode": self.config.support_mode,
            "source_support_names": list(self.support_names),
            "initialization_mode": self.config.initialization_mode,
            "ground_alignment": method,
            "ground_translation_m": ground,
            "source_planted_frames": mask.sum(axis=0).tolist(),
            "source_flight_frames": int(flight.sum()),
            "planted_point_z_error_m": {
                "p95": float(np.quantile(plant_z_errors, 0.95)),
                "max": float(np.max(plant_z_errors)),
            }
            if plant_z_errors
            else None,
            "calibrated_hip_yaw_to_knee_distance_m": thigh_distances,
            "planted_point_xy_error_m": {
                "p95": float(np.quantile(plant_xy_errors, 0.95)),
                "max": float(np.max(plant_xy_errors)),
            }
            if plant_xy_errors
            else None,
        }
