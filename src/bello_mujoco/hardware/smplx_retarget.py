"""Pinned GMR boundary for live SMPL-X reference retargeting."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
from typing import Any, Protocol

import mujoco
import numpy as np

from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.hardware.twist2 import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.robot import AssetName, robot_definition
from bello_mujoco.robot_tasks import GmrRetargetDefinition, motion_task_definition


SMPLX_BODY_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

SMPLX_BODY_JOINT_PARENTS = np.asarray(
    (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19),
    dtype=np.int32,
)

# SMPL-X's model frame has +y up. This proper rotation produces the same
# z-up, x-forward, y-left convention used by the offline AMASS pipeline.
_SMPLX_MODEL_TO_WORLD = np.asarray(
    ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)
SMPLX_MODEL_TO_WORLD_WXYZ = (0.5, 0.5, -0.5, -0.5)

# Repository-owned, hand-authored relaxed stand in SMPL-X body-pose order.
# This is deliberately independent of licensed motion datasets.
_SMPLX_RELAXED_STANDING_BODY_POSE = np.asarray(
    (
        (-0.255, 0.04, 0.00),  # left_hip
        (-0.255, -0.04, 0.00),  # right_hip
        (0.06, 0.00, 0.00),  # spine1
        (0.71, -0.04, 0.00),  # left_knee
        (0.71, 0.04, 0.00),  # right_knee
        (0.04, 0.00, 0.00),  # spine2
        (-0.435, 0.08, 0.00),  # left_ankle
        (-0.435, -0.08, 0.00),  # right_ankle
        (0.02, 0.00, 0.00),  # spine3
        (0.00, 0.00, 0.00),  # left_foot
        (0.00, 0.00, 0.00),  # right_foot
        (0.00, 0.00, 0.00),  # neck
        (0.02, 0.06, -0.30),  # left_collar
        (0.02, -0.06, 0.30),  # right_collar
        (0.00, 0.00, 0.00),  # head
        (0.10, -0.18, -0.95),  # left_shoulder
        (0.10, 0.18, 0.95),  # right_shoulder
        (0.12, -0.30, 0.08),  # left_elbow
        (0.12, 0.30, -0.08),  # right_elbow
        (-0.10, -0.06, -0.10),  # left_wrist
        (-0.10, 0.06, 0.10),  # right_wrist
    ),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SmplxTargetFrame:
    """Global SMPL-X joint transforms in world coordinates and wxyz order."""

    transforms: Mapping[str, tuple[np.ndarray, np.ndarray]]
    sequence: int
    source_timestamp_ns: int
    source_age_ms: int
    received_at: float
    foot_contacts: np.ndarray | None = None


@dataclass(frozen=True)
class GmrSourceIdentity:
    """Pinned solver and canonical robot identities used by live GMR."""

    revision: str
    robot_fingerprint: str


class SmplxTargetStream(Protocol):
    """Blocking source boundary implemented by the corrected gRPC adapter."""

    def standing_sample(self) -> SmplxTargetFrame: ...

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> SmplxTargetFrame | None: ...


def load_smplx_standing_sample(
    model_path: Path,
    source_body_names: tuple[str, ...],
) -> SmplxTargetFrame:
    """Pose the supplied neutral SMPL-X skeleton in a relaxed normal stand."""

    path = model_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"neutral SMPL-X model does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as body:
            template = np.asarray(body["v_template"], dtype=np.float64)
            regressor = np.asarray(body["J_regressor"], dtype=np.float64)
            parents = np.asarray(body["kintree_table"][0], dtype=np.int64)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"invalid neutral SMPL-X model {path}: {error}") from error
    if template.ndim != 2 or template.shape[1] != 3:
        raise ValueError("neutral SMPL-X v_template must have shape (vertices, 3)")
    joint_count = len(SMPLX_BODY_JOINT_NAMES)
    if regressor.ndim != 2 or regressor.shape[0] < joint_count:
        raise ValueError("neutral SMPL-X J_regressor has too few body joints")
    if regressor.shape[1] != template.shape[0] or parents.size < joint_count:
        raise ValueError("neutral SMPL-X model arrays have incompatible dimensions")
    body_parents = parents[:joint_count].astype(np.int32, copy=True)
    body_parents[0] = -1
    if not np.array_equal(body_parents, SMPLX_BODY_JOINT_PARENTS):
        raise ValueError("neutral SMPL-X model has an unexpected body hierarchy")
    rest_positions = regressor[:joint_count] @ template
    local_rotvecs = np.concatenate(
        (np.zeros((1, 3), dtype=np.float64), _SMPLX_RELAXED_STANDING_BODY_POSE),
        axis=0,
    )
    positions = np.empty((joint_count, 3), dtype=np.float64)
    rotations = np.empty((joint_count, 3, 3), dtype=np.float64)
    quaternions = np.empty((joint_count, 4), dtype=np.float64)
    for index, parent in enumerate(body_parents):
        local_rotation = _rotation_matrix_from_rotvec(local_rotvecs[index])
        local_quaternion = _quaternion_from_rotvec(local_rotvecs[index])
        if parent < 0:
            positions[index] = 0.0
            rotations[index] = local_rotation
            quaternions[index] = local_quaternion
            continue
        positions[index] = positions[parent] + rotations[parent] @ (
            rest_positions[index] - rest_positions[parent]
        )
        rotations[index] = rotations[parent] @ local_rotation
        quaternions[index] = _quaternion_multiply(quaternions[parent], local_quaternion)
    positions = positions @ _SMPLX_MODEL_TO_WORLD.T
    quaternions = np.stack(
        [
            _quaternion_multiply(
                np.asarray(SMPLX_MODEL_TO_WORLD_WXYZ, dtype=np.float64), quaternion
            )
            for quaternion in quaternions
        ]
    )
    foot_indices = tuple(
        SMPLX_BODY_JOINT_NAMES.index(name) for name in ("left_foot", "right_foot")
    )
    positions[:, 2] -= float(np.min(positions[foot_indices, 2]))
    if not np.all(np.isfinite(positions)):
        raise ValueError("neutral SMPL-X model produced non-finite body joints")
    position_lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, positions, strict=True))
    quaternion_lookup = dict(zip(SMPLX_BODY_JOINT_NAMES, quaternions, strict=True))
    return SmplxTargetFrame(
        transforms={
            name: (
                position_lookup[name].copy(),
                quaternion_lookup[name].copy(),
            )
            for name in source_body_names
        },
        sequence=0,
        source_timestamp_ns=0,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def _rotation_matrix_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _quaternion_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-12:
        return np.asarray((1.0, 0.0, 0.0, 0.0))
    scale = math.sin(0.5 * angle) / angle
    return np.concatenate((np.asarray((math.cos(0.5 * angle),)), scale * vector))


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    result = np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )
    return result / np.linalg.norm(result)


@dataclass(frozen=True)
class GmrLiveRetargetConfig:
    """Real-time settings layered on a robot's pinned GMR contract."""

    human_height_m: float = 1.66
    task_profile: str | None = None
    convergence_tolerance: float = 1.0e-4
    max_source_interval_seconds: float = 0.2
    contact_max_speed_mps: float = 0.2
    contact_height_margin_m: float = 0.03

    def __post_init__(self) -> None:
        if self.task_profile is not None and (
            not isinstance(self.task_profile, str)
            or not self.task_profile
            or self.task_profile.strip() != self.task_profile
        ):
            raise ValueError(
                "GMR task_profile must be null or a nonempty trimmed string"
            )
        positive = {
            "human_height_m": self.human_height_m,
            "convergence_tolerance": self.convergence_tolerance,
            "max_source_interval_seconds": self.max_source_interval_seconds,
            "contact_max_speed_mps": self.contact_max_speed_mps,
            "contact_height_margin_m": self.contact_height_margin_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


class GmrRetargeter:
    """Stateful source-rate retargeter using the selected robot's GMR profile."""

    def __init__(
        self,
        env: Any,
        source_identity: GmrSourceIdentity,
        config: GmrLiveRetargetConfig = GmrLiveRetargetConfig(),
        validate_reference: Callable[[Twist2ReferenceFrame], None] | None = None,
    ) -> None:
        asset = env.model.robot.asset
        profile = motion_task_definition(asset).gmr
        gmr_root = require_pinned_gmr_revision(source_identity.revision)
        require_robot_fingerprint(source_identity.robot_fingerprint, asset)
        _configure_pinned_gmr_profile(profile, gmr_root)
        retargeter_type = import_pinned_gmr_retargeter()
        with redirect_stdout(io.StringIO()):
            retargeter = retargeter_type(
                src_human=profile.source_human,
                tgt_robot=profile.target_robot,
                actual_human_height=config.human_height_m,
                solver="daqp",
                damping=5.0e-1,
                verbose=False,
                use_velocity_limit=None,
                robot_xml_path=env.model.xml_path,
                task_profile=config.task_profile or profile.task_profile,
            )
        retargeter.max_iter = 0
        self.passes_per_frame = _positive_solver_count(
            retargeter.offline_solver_config, "passes_per_frame"
        )
        self.settling_rounds = _positive_solver_count(
            retargeter.offline_solver_config, "initial_settle_passes"
        )
        self.env = env
        self.profile = profile
        self.source_identity = source_identity
        self.config = config
        self.validate_reference = validate_reference
        self.retargeter = retargeter
        self.model = retargeter.model
        _validate_model_identity(self.model, env.model.mj_model)
        self._foot_geom_ids = tuple(retargeter.ground_clearance_geom_ids)
        if len(self._foot_geom_ids) != len(profile.source_foot_body_names):
            raise ValueError(
                "GMR ground-clearance geoms must match the configured source feet"
            )
        self._ground_data = mujoco.MjData(self.model)
        self._previous_qpos: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._previous_source_feet: np.ndarray | None = None
        self._source_floor_height: float | None = None
        self._live_ground_offset: float | None = None
        self._last_sequence: int | None = None
        self._cached: Twist2ReferenceFrame | None = None
        self._standing_qpos: np.ndarray | None = None

    def standing(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Settle the canonical SMPL-X standing sample through pinned GMR."""

        targets = validate_smplx_target_frame(
            frame, self.profile.source_body_names, live=False
        )
        previous = self._solve(targets)
        for _ in range(self.settling_rounds - 1):
            current = self._solve(targets)
            if np.max(np.abs(current - previous)) < self.config.convergence_tolerance:
                previous = current
                break
            previous = current
        previous = self._normalize_standing(previous)
        previous = self._ground_standing(previous)
        self.retargeter.configuration.update(previous)
        self._standing_qpos = previous.copy()
        self._reset_live_state()
        return Twist2ReferenceFrame(
            qpos=previous,
            qvel=np.zeros(self.model.nv, dtype=np.float64),
            sequence=frame.sequence,
            source_age_ms=None,
            received_at=frame.received_at,
        )

    def retarget(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Retarget one new live source frame, caching repeated source frames."""

        source_targets = validate_smplx_target_frame(
            frame, self.profile.source_body_names, live=True
        )
        if self._last_sequence is not None:
            if frame.sequence == self._last_sequence:
                assert self._cached is not None
                return self._cached
            if frame.sequence < self._last_sequence:
                raise ReferenceDiscontinuity(
                    "SMPL-X source sequence moved backwards: "
                    f"{frame.sequence} < {self._last_sequence}"
                )
        if (
            self._previous_timestamp_ns is not None
            and frame.source_timestamp_ns <= self._previous_timestamp_ns
        ):
            raise ReferenceDiscontinuity("SMPL-X source timestamp did not advance")
        dt = None
        if self._previous_timestamp_ns is not None:
            dt = (frame.source_timestamp_ns - self._previous_timestamp_ns) / 1.0e9
            if not math.isfinite(dt) or not (
                0.0 < dt <= self.config.max_source_interval_seconds
            ):
                raise ReferenceDiscontinuity(
                    f"SMPL-X source interval is invalid: {dt:.6f} seconds"
                )
        solver_qpos = self.retargeter.configuration.data.qpos.copy()
        previous_source_feet = (
            None
            if self._previous_source_feet is None
            else self._previous_source_feet.copy()
        )
        source_floor_height = self._source_floor_height
        live_ground_offset = self._live_ground_offset
        try:
            qpos = self._ground_live(
                self._solve(source_targets),
                source_targets,
                dt=dt,
                source_contacts=frame.foot_contacts,
            )
            if self._previous_qpos is None:
                qvel = np.zeros(self.model.nv, dtype=np.float64)
            else:
                assert dt is not None
                qvel = np.empty(self.model.nv, dtype=np.float64)
                mujoco.mj_differentiatePos(
                    self.model, qvel, dt, self._previous_qpos, qpos
                )
            if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
                raise ReferenceDiscontinuity("GMR produced a non-finite reference")
            result = Twist2ReferenceFrame(
                qpos=qpos,
                qvel=qvel,
                sequence=frame.sequence,
                source_age_ms=frame.source_age_ms,
                received_at=frame.received_at,
            )
            if self.validate_reference is not None:
                self.validate_reference(result)
        except BaseException:
            self.retargeter.configuration.update(solver_qpos)
            self._previous_source_feet = previous_source_feet
            self._source_floor_height = source_floor_height
            self._live_ground_offset = live_ground_offset
            raise
        self.retargeter.configuration.update(qpos)
        self._previous_qpos = qpos
        self._previous_timestamp_ns = frame.source_timestamp_ns
        self._last_sequence = frame.sequence
        self._cached = result
        return result

    def reset_live(self) -> None:
        """Re-anchor GMR after a tracking discontinuity."""

        if self._standing_qpos is None:
            raise RuntimeError("GMR standing reference has not been initialized")
        self.retargeter.configuration.update(self._standing_qpos)
        self._reset_live_state()

    def _solve(
        self,
        targets: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        qpos = None
        for _ in range(self.passes_per_frame):
            qpos = self.retargeter.retarget(_copy_targets(targets))
        assert qpos is not None
        result = np.asarray(qpos, dtype=np.float64)
        if result.shape != (self.model.nq,):
            raise ReferenceDiscontinuity(
                f"GMR returned qpos shape {result.shape}; expected {(self.model.nq,)}"
            )
        if not np.all(np.isfinite(result)):
            raise ReferenceDiscontinuity("GMR returned non-finite qpos")
        return result

    def _ground_standing(self, qpos: np.ndarray) -> np.ndarray:
        sole_z = self._foot_sole_z(qpos)
        grounded = np.asarray(qpos, dtype=np.float64).copy()
        grounded[self.env.model.root_qpos_adr + 2] -= float(np.min(sole_z))
        return grounded

    def _normalize_standing(self, qpos: np.ndarray) -> np.ndarray:
        """Keep configured standing joints at the policy's canonical stance."""

        normalized = np.asarray(qpos, dtype=np.float64).copy()
        for joint_name in self.profile.standing_default_joint_names:
            address = int(self.model.jnt_qposadr[self.model.joint(joint_name).id])
            normalized[address] = self.env.model.default_qpos[address]
        root = self.env.model.root_qpos_adr
        w, x, y, z = normalized[root + 3 : root + 7]
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        normalized[root + 3 : root + 7] = (
            math.cos(0.5 * yaw),
            0.0,
            0.0,
            math.sin(0.5 * yaw),
        )
        return normalized

    def _ground_live(
        self,
        qpos: np.ndarray,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
        *,
        dt: float | None,
        source_contacts: np.ndarray | None,
    ) -> np.ndarray:
        source_feet = np.stack(
            [
                np.asarray(targets[body_name][0])
                for body_name in self.profile.source_foot_body_names
            ]
        )
        if self._source_floor_height is None:
            self._source_floor_height = float(np.min(source_feet[:, 2]))
        if self._previous_source_feet is None or dt is None:
            source_speed = np.zeros(2, dtype=np.float64)
        else:
            source_speed = (
                np.linalg.norm(source_feet - self._previous_source_feet, axis=1) / dt
            )
        if source_contacts is None:
            contacts = (source_speed < self.config.contact_max_speed_mps) & (
                source_feet[:, 2]
                <= self._source_floor_height + self.config.contact_height_margin_m
            )
        else:
            contacts = np.asarray(source_contacts, dtype=bool)
            if contacts.shape != (2,):
                raise ReferenceDiscontinuity(
                    "SMPL-X source foot contacts must have shape (2,)"
                )
        self._previous_source_feet = source_feet
        sole_z = self._foot_sole_z(qpos)
        if np.any(contacts):
            self._live_ground_offset = float(np.min(sole_z[contacts]))
        elif self._live_ground_offset is None:
            self._live_ground_offset = float(np.min(sole_z))
        ground_offset = min(self._live_ground_offset, float(np.min(sole_z)))
        grounded = np.asarray(qpos, dtype=np.float64).copy()
        grounded[self.env.model.root_qpos_adr + 2] -= ground_offset
        return grounded

    def _foot_sole_z(self, qpos: np.ndarray) -> np.ndarray:
        self._ground_data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self._ground_data)
        heights = np.empty(len(self._foot_geom_ids), dtype=np.float64)
        for output, geom_id in enumerate(self._foot_geom_ids):
            rotation = self._ground_data.geom_xmat[geom_id].reshape(3, 3)
            vertical_extent = np.abs(rotation[2]) @ self.model.geom_size[geom_id]
            heights[output] = self._ground_data.geom_xpos[geom_id, 2] - vertical_extent
        return heights

    def _reset_live_state(self) -> None:
        self._previous_qpos = None
        self._previous_timestamp_ns = None
        self._previous_source_feet = None
        self._source_floor_height = None
        self._live_ground_offset = None
        self._last_sequence = None
        self._cached = None


class LiveGmrReferenceSource:
    """Run source-rate GMR in a worker while the policy reads latest-only data."""

    def __init__(
        self,
        stream: SmplxTargetStream,
        retargeter: GmrRetargeter,
        *,
        poll_timeout_seconds: float = 0.1,
    ) -> None:
        if not math.isfinite(poll_timeout_seconds) or poll_timeout_seconds <= 0.0:
            raise ValueError("GMR worker poll timeout must be positive and finite")
        self.stream = stream
        self.retargeter = retargeter
        self.poll_timeout_seconds = poll_timeout_seconds
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._standing: Twist2ReferenceFrame | None = None
        self._latest: Twist2ReferenceFrame | None = None
        self._unavailable: ReferenceUnavailable | None = None
        self._terminal_error: BaseException | None = None

    def __enter__(self) -> LiveGmrReferenceSource:
        if self._thread is not None:
            raise RuntimeError("GMR reference source is already running")
        self._standing = self.retargeter.standing(self.stream.standing_sample())
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="live-gmr",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_exception: object) -> bool:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(2.0, 2.0 * self.poll_timeout_seconds))
            if thread.is_alive():
                raise RuntimeError("live GMR worker did not stop")
        self._thread = None
        return False

    def standing(self) -> Twist2ReferenceFrame:
        if self._standing is None:
            raise RuntimeError("live GMR source is not running")
        return self._standing

    def latest(self) -> Twist2ReferenceFrame:
        with self._condition:
            if self._terminal_error is not None:
                raise RuntimeError("live GMR worker failed") from self._terminal_error
            if self._latest is not None:
                return self._latest
            if self._unavailable is not None:
                raise self._unavailable
            raise ReferenceUnavailable("no live SMPL-X frame has been retargeted")

    def wait_until_ready(self, timeout_seconds: float) -> Twist2ReferenceFrame:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("GMR ready timeout must be positive and finite")
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._latest is None and self._terminal_error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a live GMR reference")
                self._condition.wait(remaining)
            if self._terminal_error is not None:
                raise RuntimeError("live GMR worker failed") from self._terminal_error
            assert self._latest is not None
            return self._latest

    def _run(self) -> None:
        after_sequence = None
        while not self._stop.is_set():
            try:
                frame = self.stream.wait_next(
                    after_sequence=after_sequence,
                    timeout_seconds=self.poll_timeout_seconds,
                )
                if frame is None:
                    continue
                reference = self.retargeter.retarget(frame)
            except ReferenceDiscontinuity as error:
                try:
                    self.retargeter.reset_live()
                except BaseException as reset_error:
                    self._set_terminal(reset_error)
                    return
                after_sequence = None
                with self._condition:
                    self._latest = None
                    self._unavailable = error
                    self._condition.notify_all()
                continue
            except ReferenceUnavailable as error:
                with self._condition:
                    self._unavailable = error
                    self._condition.notify_all()
                continue
            except BaseException as error:
                self._set_terminal(error)
                return
            after_sequence = frame.sequence
            with self._condition:
                self._latest = reference
                self._unavailable = None
                self._condition.notify_all()

    def _set_terminal(self, error: BaseException) -> None:
        with self._condition:
            self._terminal_error = error
            self._condition.notify_all()


def validate_smplx_target_frame(
    frame: SmplxTargetFrame,
    source_body_names: tuple[str, ...],
    *,
    live: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = sorted(set(source_body_names) - set(frame.transforms))
    if missing:
        raise ReferenceUnavailable(
            "SMPL-X frame is missing GMR target joints: " + ", ".join(missing)
        )
    if frame.sequence < 0 or frame.source_age_ms < 0:
        raise ReferenceUnavailable("SMPL-X sequence and source age must be nonnegative")
    if live and frame.source_timestamp_ns <= 0:
        raise ReferenceUnavailable("live SMPL-X source timestamp must be positive")
    result = {}
    for name in source_body_names:
        position, quaternion = frame.transforms[name]
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} position must be a finite 3-vector"
            )
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} orientation must be a finite wxyz quaternion"
            )
        norm = float(np.linalg.norm(quaternion))
        if not 0.999 <= norm <= 1.001:
            raise ReferenceUnavailable(
                f"SMPL-X joint {name!r} quaternion norm is invalid: {norm}"
            )
        result[name] = (position.copy(), (quaternion / norm).copy())
    return result


def require_pinned_gmr_revision(expected: str) -> Path:
    root = repository_root()
    gmr_root = root / "deps" / "GMR"
    if len(expected) != 40 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError("checkpoint GMR revision must be 40 lowercase hex digits")
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", "deps/GMR").split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError("deps/GMR is not a pinned Git submodule")
    pinned = gitlink[2]
    current = _git_output(gmr_root, "rev-parse", "HEAD")
    if pinned != expected or current != expected:
        raise ValueError(
            "live-pose source identity, repository gitlink, and checked-out GMR "
            f"revision differ: expected={expected}, gitlink={pinned}, "
            f"checkout={current}"
        )
    status = _git_output(
        gmr_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError("hardware GMR submodule must be clean")
    return gmr_root


def require_robot_fingerprint(expected: str, asset: AssetName) -> Path:
    robot = robot_definition(asset)
    path = robot.source_urdf_path(robot_descriptions_root())
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"live-pose robot fingerprint differs from asset {asset!r}: "
            f"description: expected={expected}, current={actual}"
        )
    return path


def repository_gmr_source_identity(asset: AssetName) -> GmrSourceIdentity:
    """Resolve and validate the GMR/robot identity pinned by this checkout."""

    root = repository_root()
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", "deps/GMR").split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError("deps/GMR is not a pinned Git submodule")
    robot_path = robot_definition(asset).source_urdf_path(robot_descriptions_root())
    identity = GmrSourceIdentity(
        revision=gitlink[2],
        robot_fingerprint="sha256:"
        + hashlib.sha256(robot_path.read_bytes()).hexdigest(),
    )
    require_pinned_gmr_revision(identity.revision)
    require_robot_fingerprint(identity.robot_fingerprint, asset)
    return identity


def _configure_pinned_gmr_profile(
    profile: GmrRetargetDefinition,
    gmr_root: Path,
) -> None:
    import_pinned_gmr_retargeter()
    config_path = profile.config_path(gmr_root).resolve()
    try:
        config_path.relative_to(gmr_root.resolve())
    except ValueError as error:
        raise ValueError("GMR profile config escapes the pinned submodule") from error
    if not config_path.is_file():
        raise FileNotFoundError(f"GMR profile config does not exist: {config_path}")
    params = sys.modules["_bello_pinned_gmr.params"]
    params.IK_CONFIG_DICT.setdefault(profile.source_human, {})[profile.target_robot] = (
        config_path
    )


def _positive_solver_count(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"GMR offline_solver.{name} must be a positive integer")
    return value


def import_pinned_gmr_retargeter() -> type[Any]:
    """Import only GMR's solver module, avoiding its optional UI/torch surface."""

    package_name = "_bello_pinned_gmr"
    module_name = f"{package_name}.motion_retarget"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.GeneralMotionRetargeting
    gmr_package = repository_root() / "deps" / "GMR" / "general_motion_retargeting"
    package = types.ModuleType(package_name)
    package.__path__ = [str(gmr_package)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    try:
        spec = importlib.util.spec_from_file_location(
            module_name, gmr_package / "motion_retarget.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("could not load pinned GMR motion_retarget module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
        raise
    module_path = Path(module.__file__).resolve()
    try:
        module_path.relative_to(gmr_package.resolve())
    except ValueError as error:
        raise RuntimeError("GMR retargeter was not imported from deps/GMR") from error
    return module.GeneralMotionRetargeting


def _validate_model_identity(
    gmr_model: mujoco.MjModel,
    model: mujoco.MjModel,
) -> None:
    fields = ("nq", "nv", "nu", "njnt", "nbody")
    mismatched = [
        name for name in fields if getattr(gmr_model, name) != getattr(model, name)
    ]
    if mismatched:
        raise ValueError(
            "GMR and runtime robot models have different dimensions: "
            + ", ".join(mismatched)
        )
    for object_type, count in (
        (mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        (mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
    ):
        left = tuple(
            mujoco.mj_id2name(gmr_model, object_type, index) for index in range(count)
        )
        right = tuple(
            mujoco.mj_id2name(model, object_type, index) for index in range(count)
        )
        if left != right:
            raise ValueError("GMR and runtime robot model ordering differs")

    if not np.array_equal(gmr_model.jnt_type, model.jnt_type):
        raise ValueError("GMR and runtime robot joint types differ")
    if not np.array_equal(gmr_model.jnt_limited, model.jnt_limited):
        raise ValueError("GMR and runtime robot joint limit declarations differ")
    if not np.allclose(gmr_model.jnt_axis, model.jnt_axis, atol=1.0e-9, rtol=0.0):
        raise ValueError("GMR and runtime robot joint axes differ")
    limited = np.asarray(model.jnt_limited, dtype=bool)
    if not np.allclose(
        gmr_model.jnt_range[limited],
        model.jnt_range[limited],
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("GMR and runtime robot joint ranges differ")

    if not np.array_equal(gmr_model.body_parentid, model.body_parentid):
        raise ValueError("GMR and runtime robot body parent frames differ")
    position_match = np.all(
        np.isclose(gmr_model.body_pos, model.body_pos, atol=1.0e-9, rtol=0.0),
        axis=1,
    )
    quaternion_dot = np.abs(np.sum(gmr_model.body_quat * model.body_quat, axis=1))
    orientation_match = np.isclose(quaternion_dot, 1.0, atol=1.0e-9, rtol=0.0)
    frame_match = position_match & orientation_match
    if not np.all(frame_match):
        body_id = int(np.flatnonzero(~frame_match)[0])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        raise ValueError(
            f"GMR and runtime robot body {body_name!r} local frames differ"
        )


def _copy_targets(
    targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: (position.copy(), quaternion.copy())
        for name, (position, quaternion) in targets.items()
    }


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"git {' '.join(arguments)} failed" + (f": {detail}" if detail else "")
        )
    return result.stdout.strip()
