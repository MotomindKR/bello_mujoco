"""Pinned GMR boundary for live SMPL-X-to-Bello reference retargeting."""

from __future__ import annotations

from collections.abc import Mapping
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

from bello_mujoco.hardware.twist2 import (
    HardwareTwist2Environment,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.training.checkpoint import Twist2CheckpointMetadata


SMPLX_GMR_BODY_NAMES = (
    "pelvis",
    "spine3",
    "left_hip",
    "left_knee",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
)

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
_SMPLX_MODEL_TO_WORLD_WXYZ = np.asarray((0.5, 0.5, -0.5, -0.5))

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


class SmplxTargetStream(Protocol):
    """Blocking source boundary implemented by the corrected gRPC adapter."""

    def standing_sample(self) -> SmplxTargetFrame: ...

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> SmplxTargetFrame | None: ...


def load_smplx_standing_sample(model_path: Path) -> SmplxTargetFrame:
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
            _quaternion_multiply(_SMPLX_MODEL_TO_WORLD_WXYZ, quaternion)
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
            for name in SMPLX_GMR_BODY_NAMES
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
    """Versioned real-time subset of the offline Bello GMR solver settings."""

    human_height_m: float = 1.66
    passes_per_frame: int = 5
    settling_rounds: int = 10
    convergence_tolerance: float = 1.0e-4
    max_source_interval_seconds: float = 0.2
    max_joint_velocity_rps: float = 3.0 * math.pi
    max_joint_acceleration_rps2: float = 200.0

    def __post_init__(self) -> None:
        positive = {
            "human_height_m": self.human_height_m,
            "convergence_tolerance": self.convergence_tolerance,
            "max_source_interval_seconds": self.max_source_interval_seconds,
            "max_joint_velocity_rps": self.max_joint_velocity_rps,
            "max_joint_acceleration_rps2": self.max_joint_acceleration_rps2,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.passes_per_frame < 1 or self.settling_rounds < 1:
            raise ValueError("GMR pass and settling counts must be positive")


class GmrBelloRetargeter:
    """Stateful, source-rate GMR retargeter producing TWIST2 command frames."""

    def __init__(
        self,
        env: HardwareTwist2Environment,
        metadata: Twist2CheckpointMetadata,
        config: GmrLiveRetargetConfig = GmrLiveRetargetConfig(),
    ) -> None:
        require_pinned_gmr_revision(metadata.dataset.gmr_revision)
        require_robot_fingerprint(metadata.robot_fingerprint)
        retargeter_type = import_pinned_gmr_retargeter()
        with redirect_stdout(io.StringIO()):
            retargeter = retargeter_type(
                src_human="smplx",
                tgt_robot="bello",
                actual_human_height=config.human_height_m,
                solver="daqp",
                damping=5.0e-1,
                verbose=False,
                use_velocity_limit=True,
                robot_xml_path=env.model.xml_path,
            )
        retargeter.max_iter = 0
        self.env = env
        self.metadata = metadata
        self.config = config
        self.retargeter = retargeter
        self.model = retargeter.model
        _validate_model_identity(self.model, env.model.mj_model)
        self._joint_ranges = _scalar_joint_ranges(self.model)
        self._previous_qpos: np.ndarray | None = None
        self._previous_joint_velocity: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._last_sequence: int | None = None
        self._cached: Twist2ReferenceFrame | None = None
        self._standing_qpos: np.ndarray | None = None

    def standing(self, frame: SmplxTargetFrame) -> Twist2ReferenceFrame:
        """Settle the canonical SMPL-X standing sample through pinned GMR."""

        targets = validate_smplx_target_frame(frame, live=False)
        previous = self._solve(targets)
        for _ in range(self.config.settling_rounds - 1):
            current = self._solve(targets)
            if np.max(np.abs(current - previous)) < self.config.convergence_tolerance:
                previous = current
                break
            previous = current
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

        targets = validate_smplx_target_frame(frame, live=True)
        if self._last_sequence is not None:
            if frame.sequence == self._last_sequence:
                assert self._cached is not None
                return self._cached
            if frame.sequence < self._last_sequence:
                raise ReferenceUnavailable(
                    "SMPL-X source sequence moved backwards: "
                    f"{frame.sequence} < {self._last_sequence}"
                )
        if (
            self._previous_timestamp_ns is not None
            and frame.source_timestamp_ns <= self._previous_timestamp_ns
        ):
            raise ReferenceUnavailable("SMPL-X source timestamp did not advance")
        dt = None
        if self._previous_timestamp_ns is not None:
            dt = (frame.source_timestamp_ns - self._previous_timestamp_ns) / 1.0e9
            if not math.isfinite(dt) or not (
                0.0 < dt <= self.config.max_source_interval_seconds
            ):
                raise ReferenceUnavailable(
                    f"SMPL-X source interval is invalid: {dt:.6f} seconds"
                )
        candidate = self._solve(targets)
        if self._previous_qpos is None:
            qpos = candidate
            joint_velocity = np.zeros(self.model.nq - 7, dtype=np.float64)
            qvel = np.zeros(self.model.nv, dtype=np.float64)
        else:
            assert self._previous_joint_velocity is not None
            assert dt is not None
            qpos, joint_velocity = _limit_scalar_joint_candidate(
                candidate,
                self._previous_qpos,
                self._previous_joint_velocity,
                dt=dt,
                joint_ranges=self._joint_ranges,
                maximum_velocity=self.config.max_joint_velocity_rps,
                maximum_acceleration=self.config.max_joint_acceleration_rps2,
            )
            qvel = np.empty(self.model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(self.model, qvel, dt, self._previous_qpos, qpos)
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise ReferenceUnavailable("GMR produced a non-finite Bello reference")
        self.retargeter.configuration.update(qpos)
        result = Twist2ReferenceFrame(
            qpos=qpos,
            qvel=qvel,
            sequence=frame.sequence,
            source_age_ms=frame.source_age_ms,
            received_at=frame.received_at,
        )
        self._previous_qpos = qpos
        self._previous_joint_velocity = joint_velocity
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

    def _solve(self, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        qpos = None
        for _ in range(self.config.passes_per_frame):
            qpos = self.retargeter.retarget(_copy_targets(targets))
        assert qpos is not None
        result = np.asarray(qpos, dtype=np.float64)
        if result.shape != (self.model.nq,):
            raise ReferenceUnavailable(
                f"GMR returned qpos shape {result.shape}; expected {(self.model.nq,)}"
            )
        return result

    def _reset_live_state(self) -> None:
        self._previous_qpos = None
        self._previous_joint_velocity = None
        self._previous_timestamp_ns = None
        self._last_sequence = None
        self._cached = None


class LiveGmrReferenceSource:
    """Run source-rate GMR in a worker while the policy reads latest-only data."""

    def __init__(
        self,
        stream: SmplxTargetStream,
        retargeter: GmrBelloRetargeter,
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
            name="bello-live-gmr",
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
            if self._unavailable is not None:
                raise self._unavailable
            if self._latest is None:
                raise ReferenceUnavailable("no live SMPL-X frame has been retargeted")
            return self._latest

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
            if self._unavailable is not None:
                raise self._unavailable
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
            except ReferenceUnavailable as error:
                try:
                    self.retargeter.reset_live()
                except BaseException as reset_error:
                    self._set_terminal(reset_error)
                    return
                after_sequence = None
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
    *,
    live: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = sorted(set(SMPLX_GMR_BODY_NAMES) - set(frame.transforms))
    if missing:
        raise ReferenceUnavailable(
            "SMPL-X frame is missing GMR target joints: " + ", ".join(missing)
        )
    if frame.sequence < 0 or frame.source_age_ms < 0:
        raise ReferenceUnavailable("SMPL-X sequence and source age must be nonnegative")
    if live and frame.source_timestamp_ns <= 0:
        raise ReferenceUnavailable("live SMPL-X source timestamp must be positive")
    result = {}
    for name in SMPLX_GMR_BODY_NAMES:
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
            "TWIST2 checkpoint, repository gitlink, and checked-out GMR revision "
            f"differ: checkpoint={expected}, gitlink={pinned}, checkout={current}"
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


def require_robot_fingerprint(expected: str) -> Path:
    path = (
        repository_root()
        / "deps"
        / "bello-stack"
        / "descriptions"
        / "bello"
        / "full_body"
        / "urdf"
        / "bello_full_body.urdf"
    )
    actual = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "TWIST2 checkpoint robot fingerprint differs from the canonical "
            f"Bello description: checkpoint={expected}, current={actual}"
        )
    return path


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


def _validate_model_identity(gmr_model: mujoco.MjModel, model: mujoco.MjModel) -> None:
    fields = ("nq", "nv", "nu", "njnt", "nbody")
    mismatched = [
        name for name in fields if getattr(gmr_model, name) != getattr(model, name)
    ]
    if mismatched:
        raise ValueError(
            "GMR and TWIST2 hardware models have different dimensions: "
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
            raise ValueError("GMR and TWIST2 hardware model ordering differs")


def _scalar_joint_ranges(model: mujoco.MjModel) -> np.ndarray:
    joint_ids = sorted(
        (
            joint_id
            for joint_id in range(model.njnt)
            if model.jnt_type[joint_id]
            in {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
        ),
        key=lambda joint_id: int(model.jnt_qposadr[joint_id]),
    )
    if len(joint_ids) != model.nq - 7:
        raise ValueError("Bello GMR model must have a free root and scalar joints")
    return np.asarray(model.jnt_range[joint_ids], dtype=np.float64)


def _limit_scalar_joint_candidate(
    candidate: np.ndarray,
    previous_qpos: np.ndarray,
    previous_joint_velocity: np.ndarray,
    *,
    dt: float,
    joint_ranges: np.ndarray,
    maximum_velocity: float,
    maximum_acceleration: float,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = np.asarray(candidate, dtype=np.float64)
    previous_qpos = np.asarray(previous_qpos, dtype=np.float64)
    joint_count = candidate.shape[0] - 7
    if candidate.shape != previous_qpos.shape or candidate.ndim != 1:
        raise ReferenceUnavailable("GMR qpos candidates have incompatible shapes")
    if previous_joint_velocity.shape != (joint_count,) or joint_ranges.shape != (
        joint_count,
        2,
    ):
        raise ReferenceUnavailable("GMR temporal-limiter layout is incompatible")
    velocity_step = maximum_acceleration * dt
    desired_velocity = (candidate[7:] - previous_qpos[7:]) / dt
    lower_distance = np.maximum(previous_qpos[7:] - joint_ranges[:, 0], 0.0)
    upper_distance = np.maximum(joint_ranges[:, 1] - previous_qpos[7:], 0.0)
    lower_braking_velocity = velocity_step - np.sqrt(
        velocity_step**2 + 2.0 * maximum_acceleration * lower_distance
    )
    upper_braking_velocity = -velocity_step + np.sqrt(
        velocity_step**2 + 2.0 * maximum_acceleration * upper_distance
    )
    lower_velocity = np.maximum.reduce(
        (
            np.full(joint_count, -maximum_velocity),
            previous_joint_velocity - velocity_step,
            lower_braking_velocity,
        )
    )
    upper_velocity = np.minimum.reduce(
        (
            np.full(joint_count, maximum_velocity),
            previous_joint_velocity + velocity_step,
            upper_braking_velocity,
        )
    )
    if np.any(lower_velocity > upper_velocity + 1.0e-10):
        raise ReferenceUnavailable("GMR joint limiter has no feasible solution")
    joint_velocity = np.clip(desired_velocity, lower_velocity, upper_velocity)
    limited = candidate.copy()
    limited[7:] = previous_qpos[7:] + joint_velocity * dt
    if np.any(limited[7:] < joint_ranges[:, 0] - 1.0e-10) or np.any(
        limited[7:] > joint_ranges[:, 1] + 1.0e-10
    ):
        raise ReferenceUnavailable("GMR output violates Bello joint limits")
    return limited, joint_velocity


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
