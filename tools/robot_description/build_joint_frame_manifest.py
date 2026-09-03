"""Extract source-bound Bello Mini joint-axis candidates from CAD circles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
REVIEW_CONTRACT = "bello_mini_joint_frames_v1"
MINIMUM_CIRCLE_RADIUS_MM = 4.0
AXIS_ALIGNMENT_TOLERANCE = 0.999
AXIS_CLUSTER_TOLERANCE_MM = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fcstd", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def column_major_matrix(values: list[float]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64).reshape(4, 4).T
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("invalid homogeneous occurrence transform")
    return matrix


def unit(vector: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(vector), dtype=np.float64)
    norm = np.linalg.norm(value)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / norm


def dominant_axis_line(
    candidates: Iterable[dict[str, Any]],
    *,
    expected_axis: Iterable[float],
    reference_point: Iterable[float],
    cluster_tolerance_mm: float = AXIS_CLUSTER_TOLERANCE_MM,
) -> dict[str, Any]:
    """Find the most strongly evidenced line parallel to ``expected_axis``."""

    axis = unit(expected_axis)
    aligned = []
    for candidate in candidates:
        candidate_axis = unit(candidate["axis"])
        alignment = abs(float(np.dot(candidate_axis, axis)))
        radius = float(candidate["radius_mm"])
        if alignment < AXIS_ALIGNMENT_TOLERANCE or radius < MINIMUM_CIRCLE_RADIUS_MM:
            continue
        center = np.asarray(candidate["center_mm"], dtype=np.float64)
        projected = center - np.dot(center, axis) * axis
        aligned.append((projected, radius))
    if not aligned:
        raise ValueError("no circular CAD edges align with the proposed joint axis")

    clusters: dict[tuple[int, int, int], list[tuple[np.ndarray, float]]] = defaultdict(
        list
    )
    for projected, radius in aligned:
        key = tuple(np.rint(projected / cluster_tolerance_mm).astype(int).tolist())
        clusters[key].append((projected, radius))

    def cluster_score(
        cluster: list[tuple[np.ndarray, float]],
    ) -> tuple[int, int, float, float]:
        radii = [item[1] for item in cluster]
        return (
            len(cluster),
            len({round(radius, 3) for radius in radii}),
            max(radii),
            sum(radii),
        )

    selected = max(clusters.values(), key=cluster_score)
    projected_centers = np.asarray([item[0] for item in selected])
    line_offset = projected_centers.mean(axis=0)
    dispersion = float(np.max(np.linalg.norm(projected_centers - line_offset, axis=1)))
    reference = np.asarray(tuple(reference_point), dtype=np.float64)
    origin = line_offset + np.dot(reference, axis) * axis
    perpendicular_reference_offset = float(np.linalg.norm(reference - origin))
    radii = [item[1] for item in selected]

    if len(selected) < 10 or len({round(radius, 3) for radius in radii}) < 3:
        raise ValueError("dominant shaft-axis cluster has insufficient CAD evidence")
    if dispersion > cluster_tolerance_mm:
        raise ValueError(f"shaft-axis cluster dispersion is too large: {dispersion}")
    if perpendicular_reference_offset > 1.0:
        raise ValueError(
            "dominant shaft axis is unexpectedly far from the motor center: "
            f"{perpendicular_reference_offset} mm"
        )

    return {
        "origin_mm": origin.tolist(),
        "axis_unsigned": axis.tolist(),
        "aligned_circle_count": len(aligned),
        "cluster_circle_count": len(selected),
        "unique_radius_count": len({round(radius, 3) for radius in radii}),
        "maximum_radius_mm": max(radii),
        "cluster_dispersion_mm": dispersion,
        "motor_center_perpendicular_offset_mm": perpendicular_reference_offset,
    }


def mirror_x(point: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(point), dtype=np.float64).copy()
    value[0] *= -1.0
    return value


def symmetrize_axis_origins(
    left_origin: Iterable[float],
    right_origin: Iterable[float],
    *,
    left_axis: Iterable[float],
    tolerance_mm: float = 0.1,
) -> tuple[list[float], list[float], float]:
    """Choose mirrored points on a bilaterally symmetric pair of axis lines."""

    left = np.asarray(tuple(left_origin), dtype=np.float64)
    mirrored_right = mirror_x(right_origin)
    axis = unit(left_axis)
    delta = mirrored_right - left
    perpendicular_residual = float(np.linalg.norm(delta - np.dot(delta, axis) * axis))
    if perpendicular_residual > tolerance_mm:
        raise ValueError(
            "bilateral joint axes are not symmetric: perpendicular residual "
            f"{perpendicular_residual} mm"
        )
    symmetric_left = (left + mirrored_right) / 2.0
    symmetric_right = mirror_x(symmetric_left)
    return symmetric_left.tolist(), symmetric_right.tolist(), perpendicular_residual


def circle_candidates(
    obj: Any, occurrence_transform: np.ndarray
) -> list[dict[str, Any]]:
    rotation = occurrence_transform[:3, :3]
    candidates = []
    for edge in obj.Shape.Edges:
        try:
            curve = edge.Curve
        except TypeError:
            continue
        if type(curve).__name__ != "Circle":
            continue
        center = occurrence_transform @ np.asarray(
            [curve.Center.x, curve.Center.y, curve.Center.z, 1.0]
        )
        axis = rotation @ np.asarray([curve.Axis.x, curve.Axis.y, curve.Axis.z])
        candidates.append(
            {
                "center_mm": center[:3].tolist(),
                "axis": unit(axis).tolist(),
                "radius_mm": float(curve.Radius),
            }
        )
    return candidates


def validate_approved_review(manifest: dict[str, Any], review: dict[str, Any]) -> None:
    if review.get("review_contract") != manifest["review_contract"]:
        raise ValueError("joint/link review contract does not match manifest")
    if review.get("source", {}).get("sha256") != manifest["source"]["sha256"]:
        raise ValueError("joint/link review STEP checksum does not match manifest")
    if not review.get("gate_decisions") or not all(review["gate_decisions"].values()):
        raise ValueError("joint/link topology gates are not all approved")
    expected_motors = {motor["id"] for motor in manifest["motors"]}
    approved_motors = {
        name
        for name, decision in review.get("motor_name_decisions", {}).items()
        if decision == "confirmed"
    }
    if approved_motors != expected_motors:
        raise ValueError("joint/link review does not approve every motor name")
    expected_joints = {joint["name"] for joint in manifest["joints"]}
    approved_joints = {
        name
        for name, decision in review.get("joint_link_decisions", {}).items()
        if decision == "confirmed"
    }
    if approved_joints != expected_joints:
        raise ValueError("joint/link review does not approve every joint/link row")


def main() -> None:
    try:
        import FreeCAD as App
    except ImportError as error:  # pragma: no cover - ordinary Python has no FreeCAD
        raise SystemExit(
            "run with FreeCAD's Python and its module directory on PYTHONPATH"
        ) from error

    args = parse_args()
    fcstd_path = args.fcstd.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    validate_approved_review(manifest, review)

    document = App.openDocument(str(fcstd_path))
    if document is None:
        raise RuntimeError(f"FreeCAD could not open {fcstd_path}")
    document.recompute()
    try:
        motors = {motor["id"]: motor for motor in manifest["motors"]}
        frame_joints = []
        for joint in manifest["joints"]:
            if joint["type"] != "revolute":
                continue
            if joint["coupling"] == "direct":
                motor = motors[joint["motor_ids"][0]]
                obj = document.getObject(motor["cad_object"])
                if obj is None:
                    raise KeyError(f"CAD motor object missing: {motor['cad_object']}")
                evidence = dominant_axis_line(
                    circle_candidates(
                        obj, column_major_matrix(motor["shape_transform"])
                    ),
                    expected_axis=joint["nominal_axis_cad"],
                    reference_point=motor["center_mm"],
                )
                origin = evidence.pop("origin_mm")
                origin_status = "cad_concentric_motor_circles"
            else:
                evidence = {
                    "bearing_evidence": joint["origin_evidence"],
                    "remote_motor_ids": joint["motor_ids"],
                }
                origin = joint["origin_cad_mm"]
                origin_status = joint["origin_status"]
            frame_joints.append(
                {
                    "name": joint["name"],
                    "parent_link": joint["parent_link"],
                    "child_link": joint["child_link"],
                    "side": joint["side"],
                    "station": joint["station"],
                    "coupling": joint["coupling"],
                    "motor_ids": joint["motor_ids"],
                    "origin_cad_mm": origin,
                    "axis_cad": joint["nominal_axis_cad"],
                    "axis_sign_status": "proposed_semantic_convention_pending_review",
                    "origin_status": origin_status,
                    "evidence": evidence,
                }
            )
    finally:
        App.closeDocument(document.Name)

    by_station_side = {
        (joint["station"], joint["side"]): joint for joint in frame_joints
    }
    bilateral_residuals = {}
    stations = sorted(
        joint["station"]
        for joint in frame_joints
        if joint["side"] == "left" and (joint["station"], "right") in by_station_side
    )
    for station in stations:
        left = by_station_side[(station, "left")]
        right = by_station_side[(station, "right")]
        left_origin, right_origin, residual = symmetrize_axis_origins(
            left["origin_cad_mm"],
            right["origin_cad_mm"],
            left_axis=left["axis_cad"],
        )
        left["origin_cad_mm"] = left_origin
        right["origin_cad_mm"] = right_origin
        left["bilateral_origin_policy"] = "mirrored_axis_line_gauge"
        right["bilateral_origin_policy"] = "mirrored_axis_line_gauge"
        bilateral_residuals[station] = residual

    waist_yaw = by_station_side[("waist_yaw", "center")]
    waist_roll = by_station_side[("waist_roll", "center")]
    yaw_origin = np.asarray(waist_yaw["origin_cad_mm"])
    roll_origin = np.asarray(waist_roll["origin_cad_mm"])
    waist_intersection = [roll_origin[0], roll_origin[1], yaw_origin[2]]
    waist_yaw["origin_cad_mm"] = waist_intersection
    waist_roll["origin_cad_mm"] = waist_intersection
    waist_yaw["origin_gauge_policy"] = "waist_axis_intersection"
    waist_roll["origin_gauge_policy"] = "waist_axis_intersection"

    direct_joint_count = sum(joint["coupling"] == "direct" for joint in frame_joints)
    linkage_joint_count = len(frame_joints) - direct_joint_count
    output = {
        "schema_version": SCHEMA_VERSION,
        "review_contract": REVIEW_CONTRACT,
        "review_stage": "joint_frames",
        "source": manifest["source"],
        "approved_joint_link_review": {
            "review_contract": review["review_contract"],
            "sha256": file_sha256(review_path),
        },
        "summary": {
            "movable_joint_count": len(frame_joints),
            "direct_cad_circle_axis_count": direct_joint_count,
            "linkage_bearing_axis_count": linkage_joint_count,
            "bilateral_pair_count": len(stations),
            "maximum_bilateral_axis_residual_mm": max(
                bilateral_residuals.values(), default=0.0
            ),
        },
        "coordinate_frame": manifest["coordinate_frame"],
        "origin_policy": (
            "origins are canonical points on physical axis lines; translation along "
            "a revolute axis is a frame gauge and is chosen to enforce bilateral "
            "symmetry, with both waist axes placed at their intersection"
        ),
        "axis_sign_policy": (
            "axis lines are CAD-derived; displayed positive directions are semantic "
            "proposals and require visual approval before URDF generation"
        ),
        "root": manifest["root"],
        "motors": manifest["motors"],
        "chains": manifest["chains"],
        "joints": frame_joints,
        "bilateral_axis_residuals_mm": bilateral_residuals,
    }
    atomic_write_json(args.output, output)
    print(f"wrote joint frame review manifest: {args.output}")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
