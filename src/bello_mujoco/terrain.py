"""Runtime terrain composition for generated MuJoCo robot models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TerrainMode = Literal["none", "plane", "rough"]
SCENE_CACHE_VERSION = 2


@dataclass(frozen=True)
class TerrainConfig:
    """Runtime terrain settings applied after robot asset generation."""

    mode: TerrainMode = "none"
    name: str = "ground"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (20.0, 20.0, 0.05)
    rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)
    condim: int = 3
    seed: int = 0
    hfield_resolution: int = 65
    hfield_amplitude: float = 0.04
    hfield_base_thickness: float = 0.01
    hfield_frequency: float = 2.5
    slope_angle_range: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.slope_angle_range) != 2:
            raise ValueError(
                "slope_angle_range must contain two values, got "
                f"{self.slope_angle_range}"
            )
        low, high = self.slope_angle_range
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(
                "slope_angle_range must be finite and ordered, got "
                f"{self.slope_angle_range}"
            )
        if low <= -math.pi / 2.0 or high >= math.pi / 2.0:
            raise ValueError(
                "slope_angle_range must stay within (-pi/2, pi/2), got "
                f"{self.slope_angle_range}"
            )


ROBOT_ONLY_TERRAIN = TerrainConfig(mode="none")
FLAT_TERRAIN = TerrainConfig(mode="plane")


def runtime_mjcf_path(robot_mjcf_path: Path, config: TerrainConfig) -> Path:
    """Return a robot MJCF or a generated sibling scene MJCF for the terrain."""

    robot_mjcf_path = Path(robot_mjcf_path)
    if config.mode == "none":
        return robot_mjcf_path
    if config.mode not in {"plane", "rough"}:
        raise ValueError(f"unsupported terrain mode: {config.mode!r}")

    scene_path = robot_mjcf_path.with_name(
        f"{robot_mjcf_path.stem}.scene."
        f"{robot_mjcf_hash(robot_mjcf_path)}.{terrain_config_hash(config)}.xml"
    )
    if scene_path.exists():
        return scene_path

    root = ET.parse(robot_mjcf_path).getroot()
    if config.mode == "plane":
        add_plane(root, config)
    else:
        add_hfield(root, config)
    ET.indent(root, space="  ")
    tmp_path = scene_path.with_suffix(f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=True)
    tmp_path.replace(scene_path)
    return scene_path


def add_plane(root: ET.Element, config: TerrainConfig) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    existing = worldbody.find(f"./geom[@name='{config.name}']")
    if existing is not None:
        worldbody.remove(existing)
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": config.name,
                "type": "plane",
                "pos": format_vec(config.pos),
                "quat": format_vec(random_slope_quat(config)),
                "size": format_vec(config.size),
                "rgba": format_vec(config.rgba),
                "friction": format_vec(config.friction),
                "condim": str(config.condim),
            },
        ),
    )


def robot_mjcf_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(Path(path).read_bytes())
    return hasher.hexdigest()[:16]


def add_hfield(root: ET.Element, config: TerrainConfig) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF has no worldbody")
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    hfield_name = f"{config.name}_hfield"
    existing_hfield = asset.find(f"./hfield[@name='{hfield_name}']")
    if existing_hfield is not None:
        asset.remove(existing_hfield)
    existing_geom = worldbody.find(f"./geom[@name='{config.name}']")
    if existing_geom is not None:
        worldbody.remove(existing_geom)

    data = rough_height_data(config)
    ET.SubElement(
        asset,
        "hfield",
        {
            "name": hfield_name,
            "nrow": str(config.hfield_resolution),
            "ncol": str(config.hfield_resolution),
            "size": format_vec(
                (
                    config.size[0],
                    config.size[1],
                    config.hfield_amplitude,
                    config.hfield_base_thickness,
                )
            ),
            "elevation": data,
        },
    )
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": config.name,
                "type": "hfield",
                "hfield": hfield_name,
                "pos": format_vec(config.pos),
                "quat": format_vec(random_slope_quat(config)),
                "rgba": format_vec(config.rgba),
                "friction": format_vec(config.friction),
                "condim": str(config.condim),
            },
        ),
    )


def rough_height_data(config: TerrainConfig) -> str:
    if config.hfield_resolution < 2:
        raise ValueError("hfield_resolution must be at least 2")
    values: list[float] = []
    resolution = config.hfield_resolution
    rng = random.Random(config.seed)
    phase_x = rng.uniform(-math.pi, math.pi)
    phase_y = rng.uniform(-math.pi, math.pi)
    phase_diag = rng.uniform(-math.pi, math.pi)
    frequency_x = config.hfield_frequency * rng.uniform(0.8, 1.2)
    frequency_y = config.hfield_frequency * rng.uniform(0.8, 1.2)
    frequency_diag = config.hfield_frequency * rng.uniform(0.4, 0.8)
    for row in range(resolution):
        y = 2.0 * row / (resolution - 1) - 1.0
        for col in range(resolution):
            x = 2.0 * col / (resolution - 1) - 1.0
            height = (
                0.50
                + 0.25 * math.sin(frequency_x * math.pi * x + phase_x)
                + 0.20 * math.sin(frequency_y * math.pi * y + phase_y)
                + 0.05 * math.sin(frequency_diag * math.pi * (x + y) + phase_diag)
            )
            values.append(min(max(height, 0.0), 1.0))
    return " ".join(f"{value:.8g}" for value in values)


def terrain_surface_height(
    mj_model,
    terrain_geom_id: int,
    world_xy,
    *,
    geom_quat=None,
):
    """Return native-MuJoCo terrain elevation at world-frame XY positions."""

    import numpy as np

    return np.asarray(
        _terrain_surface_height(
            mj_model, terrain_geom_id, world_xy, np, geom_quat=geom_quat
        ),
        dtype=np.float32,
    )


def terrain_surface_height_jax(
    mj_model,
    terrain_geom_id: int,
    world_xy,
    *,
    geom_quat=None,
):
    """Return JAX-compatible terrain elevation at world-frame XY positions."""

    import jax.numpy as jnp

    return _terrain_surface_height(
        mj_model, terrain_geom_id, world_xy, jnp, geom_quat=geom_quat
    )


def _terrain_surface_height(
    mj_model, terrain_geom_id: int, world_xy, xp, *, geom_quat=None
):
    import mujoco
    import numpy as np

    geom_id = int(terrain_geom_id)
    geom_type = int(mj_model.geom_type[geom_id])
    if geom_type not in {
        int(mujoco.mjtGeom.mjGEOM_PLANE),
        int(mujoco.mjtGeom.mjGEOM_HFIELD),
    }:
        raise ValueError(
            f"terrain geom {geom_id} must be a plane or heightfield, got {geom_type}"
        )

    position = xp.asarray(np.asarray(mj_model.geom_pos[geom_id]), dtype=xp.float32)
    if geom_quat is None:
        geom_quat = np.asarray(mj_model.geom_quat[geom_id])
    rotation = _quat_rotation_matrix(xp.asarray(geom_quat, dtype=xp.float32), xp)
    world_xy = xp.asarray(world_xy, dtype=xp.float32)
    delta_xy = world_xy - position[:2]
    xy_inverse = xp.linalg.inv(rotation[:2, :2])
    local_xy = delta_xy @ xy_inverse.T

    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        local_height = xp.zeros(world_xy.shape[:-1], dtype=xp.float32)
    else:
        hfield_id = int(mj_model.geom_dataid[geom_id])
        row_count = int(mj_model.hfield_nrow[hfield_id])
        column_count = int(mj_model.hfield_ncol[hfield_id])
        data_address = int(mj_model.hfield_adr[hfield_id])
        sample_count = row_count * column_count
        height_data = xp.asarray(
            np.asarray(
                mj_model.hfield_data[data_address : data_address + sample_count]
            ).reshape(row_count, column_count),
            dtype=xp.float32,
        )
        hfield_size = xp.asarray(
            np.asarray(mj_model.hfield_size[hfield_id]), dtype=xp.float32
        )

        # A sloped heightfield couples local height back into local XY. Fixed
        # iterations converge rapidly for the intentionally slight slopes used here.
        for _ in range(4):
            local_height = _hfield_local_height(height_data, hfield_size, local_xy, xp)
            local_xy = (
                delta_xy - local_height[..., None] * rotation[:2, 2]
            ) @ xy_inverse.T
        local_height = _hfield_local_height(height_data, hfield_size, local_xy, xp)

    return position[2] + local_xy @ rotation[2, :2] + local_height * rotation[2, 2]


def _hfield_local_height(height_data, hfield_size, local_xy, xp):
    row_count, column_count = height_data.shape
    column = xp.clip(
        (local_xy[..., 0] / hfield_size[0] + 1.0) * 0.5 * (column_count - 1),
        0.0,
        float(column_count - 1),
    )
    row = xp.clip(
        (local_xy[..., 1] / hfield_size[1] + 1.0) * 0.5 * (row_count - 1),
        0.0,
        float(row_count - 1),
    )
    column0 = xp.clip(xp.floor(column).astype(int), 0, column_count - 2)
    row0 = xp.clip(xp.floor(row).astype(int), 0, row_count - 2)
    column_fraction = column - column0
    row_fraction = row - row0

    lower_left = height_data[row0, column0]
    lower_right = height_data[row0, column0 + 1]
    upper_left = height_data[row0 + 1, column0]
    upper_right = height_data[row0 + 1, column0 + 1]
    lower_triangle = (
        lower_left
        + (lower_right - lower_left) * column_fraction
        + (upper_right - lower_right) * row_fraction
    )
    upper_triangle = (
        lower_left
        + (upper_right - upper_left) * column_fraction
        + (upper_left - lower_left) * row_fraction
    )
    normalized_height = xp.where(
        column_fraction >= row_fraction, lower_triangle, upper_triangle
    )
    return normalized_height * hfield_size[2]


def _quat_rotation_matrix(quat_wxyz, xp):
    w, x, y, z = quat_wxyz
    return xp.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=xp.float32,
    )


def random_slope_angles(config: TerrainConfig) -> tuple[float, float]:
    """Return reproducible terrain roll and pitch angles in radians."""

    rng = random.Random(config.seed ^ 0x5F3759DF)
    low, high = config.slope_angle_range
    return rng.uniform(low, high), rng.uniform(low, high)


def slope_quat(roll: float, pitch: float) -> tuple[float, float, float, float]:
    """Return a MuJoCo wxyz quaternion for terrain roll and pitch."""

    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    return cr * cp, sr * cp, cr * sp, -sr * sp


def random_slope_quat(config: TerrainConfig) -> tuple[float, float, float, float]:
    """Return the MuJoCo wxyz quaternion for the seeded terrain incline."""

    return slope_quat(*random_slope_angles(config))


def terrain_config_hash(config: TerrainConfig) -> str:
    payload = {"cache_version": SCENE_CACHE_VERSION, **asdict(config)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
