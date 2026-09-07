"""Replay a reference through the selected robot's PD actuators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import mujoco
import numpy as np

from bello_mujoco.contracts import CONTROL_DECIMATION
from bello_mujoco.motion.clip import (
    MotionClip,
    load_motion_clip,
    motion_tracking_asset_config,
)
from bello_mujoco.robot import AssetName
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import BelloModel, load_bello_model
from bello_mujoco.terrain import FLAT_TERRAIN


@dataclass(frozen=True)
class MotionFeasibilityReport:
    motion_path: str
    source_sha256: str
    duration_seconds: float
    simulated_seconds: float
    control_steps: int
    completed: bool
    first_failure_time: float | None
    first_failure_reason: str | None
    root_position_rms_m: float
    root_orientation_rms_deg: float
    joint_position_rms_rad: float
    max_joint_position_error_rad: float
    actuator_force_peak_ratio: float
    actuator_force_saturated_fraction: float
    contact_step_fraction: float
    stance_foot_slip_rms_mps: float
    nonfinite: bool


def load_feasibility_model(asset: AssetName) -> BelloModel:
    return load_bello_model(
        motion_tracking_asset_config(asset=asset),
        terrain=FLAT_TERRAIN,
        load_mjx=False,
    )


def replay_motion_reference(
    motion_path: Path,
    *,
    asset: AssetName,
    decimation: int = CONTROL_DECIMATION,
) -> MotionFeasibilityReport:
    if decimation <= 0:
        raise ValueError("decimation must be positive")
    model = load_feasibility_model(asset)
    motion = load_motion_clip(motion_path, model)
    mj_model = model.mj_model
    data = mujoco.MjData(mj_model)
    qpos = np.asarray(motion.qpos)
    qvel = np.asarray(motion.qvel)
    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    data.ctrl[:] = model.default_ctrl
    mujoco.mj_forward(mj_model, data)

    control_timestep = decimation * mj_model.opt.timestep
    control_steps = math.ceil(motion.duration / control_timestep)
    root_position_errors = []
    root_orientation_errors = []
    joint_position_errors = []
    force_ratios = []
    slip_speeds = []
    contact_steps = 0
    first_failure_time = None
    first_failure_reason = None
    nonfinite = False
    root = model.root_qpos_adr
    actuator_qpos = model.actuator_qpos_indices
    executed_steps = 0
    for step in range(control_steps):
        reference_time = min((step + 1) * control_timestep, motion.duration)
        reference = interpolate_qpos(qpos, motion.fps, reference_time)
        data.ctrl[:] = physical_ctrl_from_logical_target(
            model, reference[actuator_qpos]
        )
        for _ in range(decimation):
            mujoco.mj_step(mj_model, data)
        executed_steps += 1

        root_position_error = np.linalg.norm(
            data.qpos[root : root + 3] - reference[root : root + 3]
        )
        root_orientation_error = quaternion_distance(
            data.qpos[root + 3 : root + 7], reference[root + 3 : root + 7]
        )
        joint_error = data.qpos[actuator_qpos] - reference[actuator_qpos]
        root_position_errors.append(root_position_error)
        root_orientation_errors.append(root_orientation_error)
        joint_position_errors.append(joint_error)

        force_limit = np.max(np.abs(mj_model.actuator_forcerange), axis=1)
        force_ratios.append(
            np.abs(data.actuator_force) / np.maximum(force_limit, 1.0e-6)
        )
        foot_contacts = contact_foot_bodies(model, data)
        if foot_contacts:
            contact_steps += 1
            slip_speeds.extend(
                foot_planar_speeds(mj_model, data, sorted(foot_contacts))
            )

        values = np.concatenate((data.qpos, data.qvel, data.actuator_force))
        if not np.all(np.isfinite(values)):
            nonfinite = True
        end_effector_error = tracked_end_effector_error(
            model, data, motion, reference_time
        )
        root_height_error = abs(data.qpos[root + 2] - reference[root + 2])
        failure_reason = None
        if root_height_error > 0.35:
            failure_reason = "root_height"
        elif root_orientation_error > 1.2:
            failure_reason = "root_orientation"
        elif end_effector_error > 0.65:
            failure_reason = "end_effector"
        elif nonfinite:
            failure_reason = "nonfinite"
        if failure_reason is not None:
            first_failure_time = reference_time
            first_failure_reason = failure_reason
            break

    position_errors = np.asarray(root_position_errors)
    orientation_errors = np.asarray(root_orientation_errors)
    joint_errors = np.asarray(joint_position_errors)
    ratios = np.asarray(force_ratios)
    slips = np.asarray(slip_speeds)
    simulated_seconds = executed_steps * control_timestep
    return MotionFeasibilityReport(
        motion_path=str(Path(motion_path).resolve()),
        source_sha256=motion.source_sha256,
        duration_seconds=motion.duration,
        simulated_seconds=simulated_seconds,
        control_steps=executed_steps,
        completed=first_failure_time is None,
        first_failure_time=first_failure_time,
        first_failure_reason=first_failure_reason,
        root_position_rms_m=float(np.sqrt(np.mean(position_errors**2))),
        root_orientation_rms_deg=float(
            np.degrees(np.sqrt(np.mean(orientation_errors**2)))
        ),
        joint_position_rms_rad=float(np.sqrt(np.mean(joint_errors**2))),
        max_joint_position_error_rad=float(np.max(np.abs(joint_errors))),
        actuator_force_peak_ratio=float(np.max(ratios)),
        actuator_force_saturated_fraction=float(np.mean(ratios >= 0.999)),
        contact_step_fraction=contact_steps / executed_steps,
        stance_foot_slip_rms_mps=(
            float(np.sqrt(np.mean(slips**2))) if slips.size else 0.0
        ),
        nonfinite=nonfinite,
    )


def physical_ctrl_from_logical_target(
    model: BelloModel, logical_target: np.ndarray
) -> np.ndarray:
    """Map logical joint coordinates into clamped physical motor controls."""

    logical_target = np.asarray(logical_target)
    expected_shape = (model.mj_model.nu,)
    if logical_target.shape != expected_shape:
        raise ValueError(
            f"logical target has shape {logical_target.shape}; expected {expected_shape}"
        )
    requested_ctrl = model.actuator_coordinate_transform @ logical_target
    ctrlrange = np.asarray(model.mj_model.actuator_ctrlrange)
    return np.clip(requested_ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


def interpolate_qpos(qpos: np.ndarray, fps: float, time: float) -> np.ndarray:
    frame = np.clip(time * fps, 0.0, qpos.shape[0] - 1)
    lower = int(np.floor(frame))
    upper = min(lower + 1, qpos.shape[0] - 1)
    alpha = frame - lower
    result = qpos[lower] + alpha * (qpos[upper] - qpos[lower])
    result[3:7] = quaternion_slerp(qpos[lower, 3:7], qpos[upper, 3:7], alpha)
    return result


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, 0.0, 1.0))
    if dot > 0.9995:
        result = (1.0 - alpha) * q0 + alpha * q1
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    result = (
        math.sin((1.0 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1
    ) / math.sin(theta)
    return result / np.linalg.norm(result)


def quaternion_distance(q0: np.ndarray, q1: np.ndarray) -> float:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0)))


def contact_foot_bodies(model: BelloModel, data: mujoco.MjData) -> set[int]:
    foot_geoms = set(int(value) for value in model.foot_geom_ids)
    terrain_geoms = set(int(value) for value in model.terrain_geom_ids)
    bodies = set()
    for contact in data.contact[: data.ncon]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if geom1 in foot_geoms and geom2 in terrain_geoms:
            bodies.add(int(model.mj_model.geom_bodyid[geom1]))
        elif geom2 in foot_geoms and geom1 in terrain_geoms:
            bodies.add(int(model.mj_model.geom_bodyid[geom2]))
    return bodies


def foot_planar_speeds(
    model: mujoco.MjModel, data: mujoco.MjData, body_ids: list[int]
) -> list[float]:
    velocity = np.empty(6, dtype=np.float64)
    speeds = []
    for body_id in body_ids:
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, velocity, 0
        )
        speeds.append(float(np.linalg.norm(velocity[3:5])))
    return speeds


def tracked_end_effector_error(
    model: BelloModel,
    data: mujoco.MjData,
    motion: MotionClip,
    time: float,
) -> float:
    frame = min(int(round(time * motion.fps)), motion.num_frames - 1)
    names = motion.body_names
    end_effector_names = motion_task_definition(
        model.robot.asset
    ).default_end_effector_body_names
    body_indices = [names.index(name) for name in end_effector_names]
    body_ids = [
        mujoco.mj_name2id(model.mj_model, mujoco.mjtObj.mjOBJ_BODY, names[index])
        for index in body_indices
    ]
    reference = np.asarray(motion.body_pos)[frame, body_indices]
    current = data.xpos[body_ids]
    return float(np.max(np.linalg.norm(current - reference, axis=-1)))
