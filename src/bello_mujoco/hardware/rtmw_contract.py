"""Lightweight, exact-source RTMW gRPC contract used by hardware consumers."""

from __future__ import annotations

import importlib
import importlib.util
import math
from pathlib import Path
import string
import subprocess
import sys
from typing import Any, Protocol

from bello_mujoco.hardware.smplx_contract import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_FINGER_TIP_NAMES,
)
from bello_mujoco.repository import repository_root


RTMW_SCHEMA_VERSION = 1
RTMW_REVISION = "8bef0df43393ec9c755443e9a183451a06f2bf4f"
RTMW_REPRESENTATION = "smplx52_landmark_fit"
RTMW_SCHEMA_ID = "smplx_55_without_jaw_and_eyes_v1"
RTMW_ROTATION_SEMANTICS = (
    "derived_tree_consistent_landmark_ik;canonical_smplx_upper_body_projection_v1;"
    "anatomical_elbow_hinge_arm_chain_v2;"
    "palm_frame_wrist_orientation_v2;"
    "lower_body_hand_shape_and_unobserved_axial_twist_approximate"
)
RTMW_COORDINATE_SPACE = "camera_opencv_x_right_y_down_z_forward"
RTMW_SMPLX52_NAMES = SMPLX_BODY_JOINT_NAMES + tuple(
    f"{side}_{finger}{joint}"
    for side in ("left", "right")
    for finger in ("index", "middle", "pinky", "ring", "thumb")
    for joint in (1, 2, 3)
)
RTMW_SMPLX52_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
    23,
    20,
    25,
    26,
    20,
    28,
    29,
    20,
    31,
    32,
    20,
    34,
    35,
    21,
    37,
    38,
    21,
    40,
    41,
    21,
    43,
    44,
    21,
    46,
    47,
    21,
    49,
    50,
)
RTMW_FINGER_TIP_NAMES = SMPLX_FINGER_TIP_NAMES


class RtmwStreamConfig(Protocol):
    human_height_m: float


def require_pinned_rtmw_revision() -> Path:
    """Require the reviewed RTMW source checkout pinned by this repository."""

    root = repository_root()
    relative = "deps/RTMW-SMPLX-Teleop"
    submodule = root / relative
    gitlink = _git_output(root, "ls-tree", "HEAD", "--", relative).split()
    if len(gitlink) < 3 or gitlink[1] != "commit":
        raise RuntimeError(f"{relative} is not a pinned Git submodule")
    current = _git_output(submodule, "rev-parse", "HEAD")
    if gitlink[2] != RTMW_REVISION or current != RTMW_REVISION:
        raise ValueError(
            "RTMW gitlink and checkout must both equal the reviewed protocol "
            f"revision {RTMW_REVISION}"
        )
    if _git_output(submodule, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("hardware RTMW submodule must be clean")
    return submodule


def validate_pinned_rtmw_contract(submodule: Path) -> None:
    """Match the consumer's structural constants to the pinned producer source."""

    module_name = "_bello_pinned_rtmw_contracts"
    contract_path = submodule / "src" / "rtmw_teleop" / "contracts.py"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned RTMW contract module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    if tuple(module.SMPLX52_NAMES) != RTMW_SMPLX52_NAMES:
        raise ValueError("pinned RTMW SMPL-X joint order is incompatible")
    if tuple(int(value) for value in module.SMPLX52_PARENTS) != RTMW_SMPLX52_PARENTS:
        raise ValueError("pinned RTMW SMPL-X hierarchy is incompatible")
    if tuple(module.SMPLX_FINGER_TIP_NAMES) != RTMW_FINGER_TIP_NAMES:
        raise ValueError("pinned RTMW fingertip order is incompatible")
    if module.DERIVED_ROTATION_SEMANTICS != RTMW_ROTATION_SEMANTICS:
        raise ValueError("pinned RTMW rotation semantics are incompatible")


def import_pinned_rtmw_proto(submodule: Path | None = None) -> tuple[Any, Any]:
    """Import generated protobuf modules only from the reviewed submodule."""

    submodule = require_pinned_rtmw_revision() if submodule is None else submodule
    source_root = submodule / "src"
    source_string = str(source_root)
    if source_string not in sys.path:
        sys.path.insert(0, source_string)
    skeleton_pb2 = importlib.import_module("rtmw_teleop.proto.skeleton_pb2")
    skeleton_pb2_grpc = importlib.import_module("rtmw_teleop.proto.skeleton_pb2_grpc")
    for module in (skeleton_pb2, skeleton_pb2_grpc):
        path = Path(module.__file__).resolve()
        try:
            path.relative_to(submodule.resolve())
        except ValueError as error:
            raise RuntimeError(
                "RTMW protobuf module was not imported from the pinned submodule"
            ) from error
    return skeleton_pb2, skeleton_pb2_grpc


def validate_stream_info(message: Any, config: RtmwStreamConfig) -> None:
    if int(message.schema_version) != RTMW_SCHEMA_VERSION:
        raise ValueError(f"RTMW schema_version must be {RTMW_SCHEMA_VERSION}")
    if not message.stream_instance_id:
        raise ValueError("RTMW stream has no instance id")
    if len(message.config_sha256) != 64 or any(
        character not in string.hexdigits.lower() for character in message.config_sha256
    ):
        raise ValueError("RTMW stream has an invalid config SHA-256")
    if tuple(message.joint_names) != RTMW_SMPLX52_NAMES:
        raise ValueError("RTMW stream has an unexpected SMPL-X joint order")
    if tuple(message.joint_parents) != RTMW_SMPLX52_PARENTS:
        raise ValueError("RTMW stream has an unexpected SMPL-X hierarchy")
    exact = {
        "root_index": (int(message.root_index), 0),
        "coordinate_space": (message.coordinate_space, RTMW_COORDINATE_SPACE),
        "quaternion_order": (message.quaternion_order, "xyzw"),
        "rotation_semantics": (message.rotation_semantics, RTMW_ROTATION_SEMANTICS),
        "position_units": (message.position_units, "meters"),
        "representation": (message.representation, RTMW_REPRESENTATION),
        "schema_id": (message.schema_id, RTMW_SCHEMA_ID),
    }
    for name, (actual, expected) in exact.items():
        if actual != expected:
            raise ValueError(f"RTMW {name} is {actual!r}; expected {expected!r}")
    if not message.metric_scale:
        raise ValueError("RTMW hardware pose must have metric_scale=true")
    if not message.global_translation:
        raise ValueError("RTMW hardware pose must have global_translation=true")
    if message.global_depth_is_measured:
        raise ValueError(
            "RTMW monocular hardware pose must declare global_depth_is_measured=false"
        )
    if message.depth_provenance not in {
        "calibrated_camera+assumed_subject_height",
        "fixed_root_depth_assumption",
    }:
        raise ValueError(
            "RTMW hardware pose must use calibrated height-prior or fixed-depth "
            "reconstruction"
        )
    subject_height_m = float(message.subject_height_m)
    if not math.isfinite(subject_height_m) or not math.isclose(
        subject_height_m, config.human_height_m, abs_tol=1.0e-6
    ):
        raise ValueError(
            "RTMW producer subject height does not match rtmw_human_height_m: "
            f"{subject_height_m!r} != {config.human_height_m!r}"
        )


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


__all__ = (
    "RTMW_COORDINATE_SPACE",
    "RTMW_FINGER_TIP_NAMES",
    "RTMW_REPRESENTATION",
    "RTMW_REVISION",
    "RTMW_ROTATION_SEMANTICS",
    "RTMW_SCHEMA_ID",
    "RTMW_SCHEMA_VERSION",
    "RTMW_SMPLX52_NAMES",
    "RTMW_SMPLX52_PARENTS",
    "RtmwStreamConfig",
    "import_pinned_rtmw_proto",
    "require_pinned_rtmw_revision",
    "validate_pinned_rtmw_contract",
    "validate_stream_info",
)
