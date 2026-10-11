"""Pinned upstream GMR boundary for named human body poses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import math
from pathlib import Path
import sys
import threading
import time
import types
from typing import Any, Protocol

import mujoco
import numpy as np

from bello_mujoco.envs.torso import (
    neutral_torso_quaternion,
    torso_command_from_quaternions,
)
from bello_mujoco.hardware.smplx_contract import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
)
from bello_mujoco.hardware.reference import (
    ReferenceDiscontinuity,
    ReferenceUnavailable,
    RobotReferenceFrame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.robot_tasks import GmrRetargetDefinition, motion_task_definition

from bello_mujoco.hardware.smplx_kinematics import (
    MODEL_TO_WORLD as _SMPLX_MODEL_TO_WORLD,
    MODEL_TO_WORLD_WXYZ as SMPLX_MODEL_TO_WORLD_WXYZ,
    rotation_matrix_from_rotvec,
    quaternion_from_rotvec,
    quaternion_multiply,
)

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
class HumanTargetFrame:
    """Named global body transforms in the source profile's Z-up GMR frame."""

    transforms: Mapping[str, tuple[np.ndarray, np.ndarray]]
    sequence: int
    source_timestamp_ns: int
    source_age_ms: float
    received_at: float
    foot_contacts: np.ndarray | None = None
    track_id: str = ""
    smpl: tuple[np.ndarray, np.ndarray] | None = None

    def age_ms(self) -> float:
        return self.source_age_ms + (time.monotonic() - self.received_at) * 1000


class HumanTargetStream(Protocol):
    """Blocking source boundary implemented by the corrected gRPC adapter."""

    def wait_next(
        self,
        *,
        after_sequence: int | None,
        timeout_seconds: float,
    ) -> HumanTargetFrame | None: ...


def load_smplx_standing_sample(
    model_path: Path,
    source_body_names: tuple[str, ...],
) -> HumanTargetFrame:
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
        local_rotation = rotation_matrix_from_rotvec(local_rotvecs[index])
        local_quaternion = quaternion_from_rotvec(local_rotvecs[index])
        if parent < 0:
            positions[index] = 0.0
            rotations[index] = local_rotation
            quaternions[index] = local_quaternion
            continue
        positions[index] = positions[parent] + rotations[parent] @ (
            rest_positions[index] - rest_positions[parent]
        )
        rotations[index] = rotations[parent] @ local_rotation
        quaternions[index] = quaternion_multiply(quaternions[parent], local_quaternion)
    positions = positions @ _SMPLX_MODEL_TO_WORLD.T
    quaternions = np.stack(
        [
            quaternion_multiply(
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
    return HumanTargetFrame(
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


@dataclass(frozen=True)
class GmrLiveRetargetConfig:
    """Input timing and subject height for upstream GMR."""

    human_height_m: float = 1.66
    max_source_interval_seconds: float = 0.2

    def __post_init__(self) -> None:
        for name in ("human_height_m", "max_source_interval_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


class GmrRetargeter:
    """One upstream retarget call per source frame, without motion corrections."""

    def __init__(
        self,
        env: Any,
        config: GmrLiveRetargetConfig = GmrLiveRetargetConfig(),
        validate_reference: Callable[[RobotReferenceFrame], None] | None = None,
        *,
        profile: GmrRetargetDefinition | None = None,
    ) -> None:
        if profile is None:
            profile = motion_task_definition(env.model.robot.asset).gmr
        _configure_pinned_gmr_profile(
            profile, repository_root() / "deps" / "GMR", env.model.xml_path
        )
        with redirect_stdout(io.StringIO()):
            retargeter = import_pinned_gmr_retargeter()(
                src_human=profile.source_human,
                tgt_robot=profile.target_robot,
                actual_human_height=config.human_height_m,
                solver="daqp",
                damping=5.0e-1,
                verbose=False,
                use_velocity_limit=False,
            )
        self.profile = profile
        self.config = config
        self.validate_reference = validate_reference
        self.retargeter = retargeter
        self.model = retargeter.model
        _validate_model_identity(self.model, env.model.mj_model)
        self._torso_id = self.model.body(env.command_config.torso.body_name).id
        self._pelvis_id = self.model.body(
            env.model.robot.locomotion_command_frame_body_name
        ).id
        self._forward_axis = env.model.robot.locomotion_forward_axis
        self._neutral_torso = neutral_torso_quaternion(
            self.model, env.model.default_qpos, self._torso_id, self._pelvis_id
        )
        self._kinematics = mujoco.MjData(self.model)
        self._initialized = False
        self._previous_qpos: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._last_sequence: int | None = None
        self._cached: RobotReferenceFrame | None = None

    def _initialize(self, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        model = self.model
        root_body = model.body(self.retargeter.robot_root_name).id
        if (
            model.njnt == 0
            or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
            or model.jnt_qposadr[0] != 0
            or model.jnt_bodyid[0] != root_body
        ):
            raise ValueError("Configured GMR root must own the first free joint")
        self.retargeter.update_targets(_copy_targets(targets))
        position, quaternion = self.retargeter.scaled_human_data[
            self.retargeter.human_root_name
        ]
        initial = model.qpos0.copy()
        initial[:3], initial[3:7] = position, quaternion
        for name, angle in self.profile.initial_joint_positions:
            joint = model.joint(name)
            if (
                joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE
                or not joint.limited[0]
                or not joint.range[0] <= angle <= joint.range[1]
            ):
                raise ValueError(f"Invalid GMR initial joint position: {name}")
            initial[joint.qposadr[0]] = angle
        if not np.isfinite(initial).all():
            raise ReferenceDiscontinuity("Non-finite GMR initial pose")
        self.retargeter.configuration.update(initial)
        self._initialized = True

    def standing(self, frame: HumanTargetFrame) -> RobotReferenceFrame:
        targets = validate_human_target_frame(
            frame, self.profile.source_body_names, live=False
        )
        self._initialize(targets)
        qpos = self._solve(targets)
        result = RobotReferenceFrame(
            qpos=qpos,
            qvel=np.zeros(self.model.nv, dtype=np.float64),
            sequence=frame.sequence,
            source_age_ms=None,
            received_at=frame.received_at,
            torso_command=self._torso_command(qpos),
            smpl=frame.smpl,
        )
        if self.validate_reference is not None:
            self.validate_reference(result)
        self._reset_live_state()
        return result

    def retarget(self, frame: HumanTargetFrame) -> RobotReferenceFrame:
        targets = validate_human_target_frame(
            frame, self.profile.source_body_names, live=True
        )
        if self._last_sequence is not None:
            if frame.sequence == self._last_sequence:
                assert self._cached is not None
                return self._cached
            if frame.sequence < self._last_sequence:
                raise ReferenceDiscontinuity("Human source sequence moved backwards")
        dt = None
        if self._previous_timestamp_ns is not None:
            dt = (frame.source_timestamp_ns - self._previous_timestamp_ns) / 1.0e9
            if not math.isfinite(dt) or not (
                0.0 < dt <= self.config.max_source_interval_seconds
            ):
                raise ReferenceDiscontinuity(
                    f"Human source interval is invalid: {dt:.6f} seconds"
                )
        previous_solver_qpos = self.retargeter.configuration.data.qpos.copy()
        initialized = self._initialized
        try:
            qpos = self._solve(targets)
            qvel = np.zeros(self.model.nv, dtype=np.float64)
            if self._previous_qpos is not None:
                assert dt is not None
                mujoco.mj_differentiatePos(
                    self.model, qvel, dt, self._previous_qpos, qpos
                )
            if not np.isfinite(qvel).all():
                raise ReferenceDiscontinuity("GMR produced non-finite velocity")
            result = RobotReferenceFrame(
                qpos=qpos,
                qvel=qvel,
                sequence=frame.sequence,
                source_age_ms=frame.source_age_ms,
                received_at=frame.received_at,
                torso_command=self._torso_command(qpos),
                smpl=frame.smpl,
            )
            if self.validate_reference is not None:
                self.validate_reference(result)
        except BaseException:
            self.retargeter.configuration.update(previous_solver_qpos)
            self._initialized = initialized
            raise
        self._previous_qpos = qpos.copy()
        self._previous_timestamp_ns = frame.source_timestamp_ns
        self._last_sequence = frame.sequence
        self._cached = result
        return result

    def _torso_command(self, qpos: np.ndarray) -> np.ndarray:
        self._kinematics.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self._kinematics)
        return torso_command_from_quaternions(
            self._kinematics.xquat[self._torso_id],
            self._kinematics.xquat[self._pelvis_id],
            self._neutral_torso,
            self._forward_axis,
        )

    def _solve(self, targets: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        if not self._initialized:
            self._initialize(targets)
        qpos = np.asarray(
            self.retargeter.retarget(_copy_targets(targets)), dtype=np.float64
        ).copy()
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ReferenceDiscontinuity("GMR returned invalid qpos")
        return qpos

    def reset_live(self) -> None:
        self.retargeter.configuration.update(self.model.qpos0.copy())
        self._reset_live_state()

    def _reset_live_state(self) -> None:
        self._initialized = False
        self._previous_qpos = None
        self._previous_timestamp_ns = None
        self._last_sequence = None
        self._cached = None


class LiveGmrReferenceSource:
    """Run source-rate GMR in a worker while the policy reads latest-only data."""

    def __init__(
        self,
        stream: HumanTargetStream,
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
        self._latest: RobotReferenceFrame | None = None
        self._unavailable: ReferenceUnavailable | None = None
        self._terminal_error: BaseException | None = None

    def __enter__(self) -> LiveGmrReferenceSource:
        if self._thread is not None:
            raise RuntimeError("GMR reference source is already running")
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

    def latest(self) -> RobotReferenceFrame:
        with self._condition:
            if self._terminal_error is not None:
                raise RuntimeError("live GMR worker failed") from self._terminal_error
            if self._latest is not None:
                return self._latest
            if self._unavailable is not None:
                raise self._unavailable
            raise ReferenceUnavailable("no live human frame has been retargeted")

    def wait_until_ready(self, timeout_seconds: float) -> RobotReferenceFrame:
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
                    self._latest = None
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


def validate_human_target_frame(
    frame: HumanTargetFrame,
    source_body_names: tuple[str, ...],
    *,
    live: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    missing = sorted(set(source_body_names) - set(frame.transforms))
    if missing:
        raise ReferenceUnavailable(
            "Human frame is missing GMR target joints: " + ", ".join(missing)
        )
    if (
        frame.sequence < 0
        or not math.isfinite(frame.source_age_ms)
        or frame.source_age_ms < 0
        or not math.isfinite(frame.received_at)
    ):
        raise ReferenceUnavailable("Human sequence and source age must be nonnegative")
    if live and frame.source_timestamp_ns <= 0:
        raise ReferenceUnavailable("live human source timestamp must be positive")
    result = {}
    for name in source_body_names:
        position, quaternion = frame.transforms[name]
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ReferenceUnavailable(
                f"Human joint {name!r} position must be a finite 3-vector"
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


def _configure_pinned_gmr_profile(
    profile: GmrRetargetDefinition,
    gmr_root: Path,
    model_path: Path,
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
    params.ROBOT_XML_DICT[profile.target_robot] = Path(model_path)
    params.IK_CONFIG_DICT.setdefault(profile.source_human, {})[profile.target_robot] = (
        config_path
    )


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
