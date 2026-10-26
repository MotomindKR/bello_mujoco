"""Canonical Mini pose calibration and constrained offline contact refinement."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import mink
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bello_mujoco.motion.retarget_limits import (
    ContactLimitConfig,
    RetargetContactLimit,
    RetargetJointLimit,
    project_contacts,
)
from bello_mujoco.sim import geom_lower_z


@dataclass(frozen=True)
class SourceContactConfig:
    height_m: float = 0.025
    horizontal_speed_m_s: float = 0.15
    vertical_speed_m_s: float = 0.10
    angular_speed_rad_s: float = 0.5
    min_seconds: float = 0.1

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError("source contact thresholds must be positive and finite")


@dataclass(frozen=True)
class KinematicRetargetConfig:
    initialization_iterations: int = 40
    calibration_iterations: int = 5
    refinement_iterations: int = 3
    projection_iterations: int = 20
    contact_blend_frames: int = 3
    contacts: ContactLimitConfig = field(default_factory=ContactLimitConfig)
    source_contacts: SourceContactConfig = field(default_factory=SourceContactConfig)

    def __post_init__(self):
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


def planted_feet(positions, quaternions, fps, config=SourceContactConfig()):
    """Conservative stationary toe proxies; absence never rejects a sequence."""
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    if (
        positions.ndim != 3
        or positions.shape[1:] != (2, 3)
        or len(positions) < 3
        or quaternions.shape != positions.shape[:2] + (4,)
        or not np.isfinite(positions).all()
        or not np.isfinite(quaternions).all()
        or not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-6)
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("source contacts require finite clocked two-foot transforms")
    # LAFAN's source plane is Z=0. Allow negative capture bias, but never infer
    # a raised support plane merely because both feet stay above the floor.
    floor = min(0.0, float(np.quantile(positions[:, :, 2], 0.01)))
    velocity = np.gradient(positions, 1 / fps, axis=0)
    angular = np.empty(positions.shape[:2])
    for side in range(2):
        rotation = Rotation.from_quat(quaternions[:, side], scalar_first=True)
        angular[1:, side] = (rotation[:-1].inv() * rotation[1:]).magnitude() * fps
    angular[0] = angular[1]
    mask = (
        (positions[:, :, 2] < floor + config.height_m)
        & (np.linalg.norm(velocity[:, :, :2], axis=-1) < config.horizontal_speed_m_s)
        & (np.abs(velocity[:, :, 2]) < config.vertical_speed_m_s)
        & (angular < config.angular_speed_rad_s)
    )
    for side in range(2):
        for start, end in contact_intervals(mask[:, side]):
            if end - start < math.ceil(config.min_seconds * fps):
                mask[start:end, side] = False
    return mask


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
        frame_indices = {frame: i for i, (frame, *_) in enumerate(self.frames2)}
        if not set(model.robot.foot_body_names).issubset(frame_indices):
            raise ValueError("calibration task table must include both foot bodies")
        self.foot_indices = tuple(frame_indices[n] for n in model.robot.foot_body_names)
        self.foot_geoms = tuple(
            tuple(
                int(g)
                for g in model.foot_geom_ids
                if model.mj_model.geom_bodyid[g] == body
            )
            for body in model.foot_body_ids
        )
        if any(not geoms for geoms in self.foot_geoms):
            raise ValueError("calibration requires both canonical foot sole geometries")
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
        poses, positions, heights = [], [], []
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
            heights.append(
                [
                    min(geom_lower_z(native, configuration.data, g) for g in geoms)
                    for geoms in self.foot_geoms
                ]
            )
        return np.asarray(poses), np.asarray(positions), np.asarray(heights)

    def _ground_alignment(self, poses, heights, mask):
        if mask.any():
            return float(np.median(heights[mask])), "source_planted_soles"
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

    def _contact_targets(self, positions, poses, mask):
        targets = positions.copy()
        native, data = self.model.mj_model, mujoco.MjData(self.model.mj_model)
        for side, foot in enumerate(self.foot_indices):
            for start, end in contact_intervals(mask[:, side]):
                anchor = np.median(targets[start:end, foot], axis=0)
                data.qpos[:] = poses[(start + end) // 2]
                mujoco.mj_kinematics(native, data)
                anchor[2] -= min(
                    geom_lower_z(native, data, g) for g in self.foot_geoms[side]
                )
                anchor[2] += self.config.contacts.floor_clearance_m
                indices = np.arange(end - start)
                weight = np.minimum(
                    np.minimum(indices + 1, end - start - indices)
                    / self.config.contact_blend_frames,
                    1,
                )
                targets[start:end, foot] = (1 - weight[:, None]) * targets[
                    start:end, foot
                ] + weight[:, None] * anchor
        return targets

    def retarget(self, targets: Mapping, foot_contacts):
        desired = self._aligned_source(targets)
        count = len(desired[self.backend.human_root_name])
        mask = np.asarray(foot_contacts)
        if mask.shape != (count, 2) or mask.dtype != np.bool_:
            raise ValueError(
                "source planted-foot labels must be a frames-by-two bool array"
            )
        poses, positions, heights = self._fit(desired)
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
        poses[:, 2] -= ground
        positions[:, :, 2] -= ground
        for values in desired.values():
            values[:, 2] -= ground
        contact_targets = self._contact_targets(positions, poses, mask)
        native = self.model.mj_model
        tasks = [
            mink.FrameTask(frame, "body", position, rotation, lm_damping=1)
            for frame, _, position, rotation in self.frames2
        ]
        configuration = mink.Configuration(native)
        configuration.update(poses[0])
        contact_limit = RetargetContactLimit(native, self.config.contacts)
        self.bounds.previous = None
        solver = SimpleNamespace(
            model=native,
            configuration=configuration,
            solver=self.backend.solver,
            damping=self.backend.damping,
            ik_limits=[mink.ConfigurationLimit(native), self.bounds, contact_limit],
        )
        result = []
        for frame in range(count):
            for index, (task, (_, source, *_)) in enumerate(
                zip(tasks, self.frames2, strict=True)
            ):
                _set_target(
                    task, contact_targets[frame, index], desired[source][frame, 3:]
                )
            _solve(
                configuration,
                tasks,
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
        return np.asarray(result), {
            "ground_alignment": method,
            "ground_translation_m": ground,
            "source_planted_frames": mask.sum(axis=0).tolist(),
            "calibrated_hip_yaw_to_knee_distance_m": thigh_distances,
        }
