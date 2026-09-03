"""Audit Bello Mini sole primitives against bottom slices of the visual STLs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


SOLE_ROLE = "foot_contact_envelope"
RASTER_STEP_MM = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, required=True)
    parser.add_argument("--image-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ordered_section_polygon(mesh: trimesh.Trimesh, z_mm: float) -> np.ndarray:
    section = mesh.section(
        plane_origin=[0.0, 0.0, z_mm], plane_normal=[0.0, 0.0, 1.0]
    )
    if section is None or not section.entities:
        raise ValueError(f"foot mesh has no horizontal section at z={z_mm} mm")
    entity = max(section.entities, key=lambda value: len(value.points))
    polygon = np.asarray(section.vertices[entity.points], dtype=np.float64)[:, :2]
    if np.linalg.norm(polygon[0] - polygon[-1]) < 1e-6:
        polygon = polygon[:-1]
    if len(polygon) < 3:
        raise ValueError("foot section does not form a usable outline")
    return polygon


def points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape, dtype=bool)
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        crossing = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-300) + x1
        )
        inside ^= crossing
        x1, y1 = x2, y2
    return inside


def primitive_footprint_mask(
    x: np.ndarray, y: np.ndarray, primitives: list[dict[str, Any]]
) -> np.ndarray:
    included = np.zeros(x.shape, dtype=bool)
    for primitive in primitives:
        center_x, center_y, _ = (
            np.asarray(primitive["origin_xyz_m"], dtype=np.float64) * 1000.0
        )
        if primitive["type"] == "box":
            size = np.asarray(primitive["size_m"], dtype=np.float64) * 1000.0
            included |= (np.abs(x - center_x) <= size[0] * 0.5) & (
                np.abs(y - center_y) <= size[1] * 0.5
            )
        elif primitive["type"] == "cylinder":
            radius = float(primitive["radius_m"]) * 1000.0
            included |= (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
        else:
            raise ValueError(f"unsupported sole primitive: {primitive['type']}")
    return included


def primitive_bottom_mm(primitive: dict[str, Any]) -> float:
    center_z = float(primitive["origin_xyz_m"][2]) * 1000.0
    if primitive["type"] == "box":
        return center_z - float(primitive["size_m"][2]) * 500.0
    if primitive["type"] == "cylinder":
        return center_z - float(primitive["length_m"]) * 500.0
    raise ValueError(f"unsupported sole primitive: {primitive['type']}")


def audit_foot(
    mesh_path: Path, link_geometry: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    primitives = [
        value
        for value in link_geometry["collision_primitives"]
        if value["role"] == SOLE_ROLE
    ]
    if len(primitives) != 5:
        raise ValueError(
            f"{link_geometry['name']} expected five sole primitives, got {len(primitives)}"
        )
    bottom = min(primitive_bottom_mm(value) for value in primitives)
    top = max(
        float(value["origin_xyz_m"][2]) * 1000.0
        + (
            float(value["size_m"][2]) * 500.0
            if value["type"] == "box"
            else float(value["length_m"]) * 500.0
        )
        for value in primitives
    )
    slice_z = (bottom + top) * 0.5
    polygon = ordered_section_polygon(mesh, slice_z)
    lower = np.minimum(polygon.min(axis=0), np.asarray([-75.0, -50.0]))
    upper = np.maximum(polygon.max(axis=0), np.asarray([155.0, 50.0]))
    xs = np.arange(lower[0], upper[0], RASTER_STEP_MM)
    ys = np.arange(lower[1], upper[1], RASTER_STEP_MM)
    x_grid, y_grid = np.meshgrid(xs, ys)
    actual = points_in_polygon(x_grid.ravel(), y_grid.ravel(), polygon)
    fitted = primitive_footprint_mask(
        x_grid.ravel(), y_grid.ravel(), primitives
    )
    intersection = int(np.count_nonzero(actual & fitted))
    actual_count = int(np.count_nonzero(actual))
    fitted_count = int(np.count_nonzero(fitted))
    union = int(np.count_nonzero(actual | fitted))
    record = {
        "link": link_geometry["name"],
        "mesh": str(mesh_path),
        "mesh_bottom_z_mm": float(mesh.bounds[0, 2]),
        "collision_bottom_z_mm": bottom,
        "bottom_alignment_error_mm": abs(float(mesh.bounds[0, 2]) - bottom),
        "audit_slice_z_mm": slice_z,
        "outline_bounds_mm": {
            "min": polygon.min(axis=0).tolist(),
            "max": polygon.max(axis=0).tolist(),
        },
        "primitive_count": len(primitives),
        "intersection_over_union": intersection / union,
        "precision_within_stl_outline": intersection / fitted_count,
        "stl_outline_coverage": intersection / actual_count,
        "passed": (
            intersection / union >= 0.98
            and intersection / fitted_count >= 0.99
            and intersection / actual_count >= 0.98
            and abs(float(mesh.bounds[0, 2]) - bottom) <= 0.1
        ),
    }
    return record, polygon, primitives


def draw_audit(
    audits: list[tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]],
    output_path: Path,
) -> None:
    width, height = 1500, 720
    canvas = Image.new("RGB", (width, height), (9, 12, 17))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((28, 20), "Bello Mini foot soles - bottom view", fill=(240, 243, 247), font=font)
    draw.text(
        (28, 44),
        "white/gray = STL slice at z=-40 mm | green = fitted boxes | cyan = smooth heel/toe cylinders",
        fill=(174, 184, 198),
        font=font,
    )
    panel_width = 700
    scale = 2.55
    for index, (record, polygon, primitives) in enumerate(audits):
        panel_x = 32 + index * (panel_width + 32)
        panel_y = 94
        center_x = panel_x + panel_width * 0.5
        center_y = panel_y + 290

        def point(
            value: np.ndarray | tuple[float, float],
            origin_x: float = center_x,
            origin_y: float = center_y,
        ) -> tuple[float, float]:
            return (
                origin_x + float(value[0] - 40.0) * scale,
                origin_y - float(value[1]) * scale,
            )

        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + 500),
            radius=12,
            fill=(16, 21, 28),
            outline=(45, 54, 66),
            width=2,
        )
        stl_points = [point(value) for value in polygon]
        draw.polygon(stl_points, fill=(91, 100, 113), outline=(238, 242, 246))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for primitive in primitives:
            center = np.asarray(primitive["origin_xyz_m"][:2]) * 1000.0
            if primitive["type"] == "box":
                size = np.asarray(primitive["size_m"][:2]) * 1000.0
                low = point(center - size * 0.5)
                high = point(center + size * 0.5)
                box = (
                    min(low[0], high[0]),
                    min(low[1], high[1]),
                    max(low[0], high[0]),
                    max(low[1], high[1]),
                )
                overlay_draw.rectangle(
                    box, fill=(42, 232, 125, 100), outline=(71, 255, 154, 255), width=2
                )
            else:
                radius = float(primitive["radius_m"]) * 1000.0 * scale
                cx, cy = point(center)
                overlay_draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=(35, 205, 255, 100),
                    outline=(62, 220, 255, 255),
                    width=2,
                )
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        arrow_start = point(np.asarray([105.0, -53.0]))
        arrow_end = point(np.asarray([145.0, -53.0]))
        draw.line((*arrow_start, *arrow_end), fill=(255, 202, 84), width=3)
        draw.polygon(
            [
                arrow_end,
                (arrow_end[0] - 10, arrow_end[1] - 5),
                (arrow_end[0] - 10, arrow_end[1] + 5),
            ],
            fill=(255, 202, 84),
        )
        draw.text(
            (arrow_start[0], arrow_start[1] + 9),
            "+x forward",
            fill=(255, 202, 84),
            font=font,
        )
        metrics_y = panel_y + 520
        side = record["link"].split("_", 1)[0].title()
        draw.text((panel_x + 10, metrics_y), f"{side} sole", fill=(240, 243, 247), font=font)
        draw.text(
            (panel_x + 10, metrics_y + 24),
            (
                f"IoU {record['intersection_over_union'] * 100:.2f}% | "
                f"inside STL {record['precision_within_stl_outline'] * 100:.2f}% | "
                f"coverage {record['stl_outline_coverage'] * 100:.2f}%"
            ),
            fill=(113, 226, 167),
            font=font,
        )
        draw.text(
            (panel_x + 10, metrics_y + 47),
            (
                f"STL bottom {record['mesh_bottom_z_mm']:.2f} mm | "
                f"collision bottom {record['collision_bottom_z_mm']:.2f} mm"
            ),
            fill=(174, 184, 198),
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    geometry_path = args.geometry.expanduser().resolve()
    meshes_path = args.meshes.expanduser().resolve()
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    meshes = json.loads(meshes_path.read_text(encoding="utf-8"))
    geometry_by_link = {value["name"]: value for value in geometry["links"]}
    mesh_by_link = {value["link"]: value for value in meshes["meshes"]}
    audits = []
    for side in ("left", "right"):
        link = f"{side}_ankle_roll_link"
        audits.append(
            audit_foot(Path(mesh_by_link[link]["path"]), geometry_by_link[link])
        )
    records = [value[0] for value in audits]
    if not all(value["passed"] for value in records):
        raise ValueError("one or more foot sole collision fits failed")
    image_path = args.image_output.expanduser().resolve()
    draw_audit(audits, image_path)
    report = {
        "schema_version": 1,
        "review_contract": "bello_mini_foot_collision_audit_v1",
        "source_manifests": {
            "geometry_manifest_sha256": file_sha256(geometry_path),
            "visual_mesh_manifest_sha256": file_sha256(meshes_path),
        },
        "raster_step_mm": RASTER_STEP_MM,
        "feet": records,
        "passed": True,
    }
    report_path = args.report_output.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2))
    print(f"wrote foot collision review: {image_path}")


if __name__ == "__main__":
    main()
