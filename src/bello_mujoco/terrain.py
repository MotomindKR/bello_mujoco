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
SCENE_CACHE_VERSION = 1


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


ROBOT_ONLY_TERRAIN = TerrainConfig(mode="none")
FLAT_TERRAIN = TerrainConfig(mode="plane")
ROUGH_TERRAIN = TerrainConfig(mode="rough")


def runtime_mjcf_path(robot_mjcf_path: Path, config: TerrainConfig) -> Path:
    """Return a robot MJCF or a generated sibling scene MJCF for the terrain."""

    robot_mjcf_path = Path(robot_mjcf_path)
    if config.mode == "none":
        return robot_mjcf_path
    if config.mode not in {"plane", "rough"}:
        raise ValueError(f"unsupported terrain mode: {config.mode!r}")

    scene_path = robot_mjcf_path.with_name(
        f"{robot_mjcf_path.stem}.scene.{terrain_config_hash(config)}.xml"
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
                "size": format_vec(config.size),
                "rgba": format_vec(config.rgba),
                "friction": format_vec(config.friction),
                "condim": str(config.condim),
            },
        ),
    )


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
                    config.size[2],
                    config.hfield_base_thickness,
                )
            ),
        },
    ).text = data
    worldbody.insert(
        0,
        ET.Element(
            "geom",
            {
                "name": config.name,
                "type": "hfield",
                "hfield": hfield_name,
                "pos": format_vec(config.pos),
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
                + 0.05
                * math.sin(frequency_diag * math.pi * (x + y) + phase_diag)
            )
            values.append(
                config.hfield_amplitude
                * min(max(height, 0.0), 1.0)
                / max(config.size[2], 1e-6)
            )
    return " ".join(f"{value:.8g}" for value in values)


def terrain_config_hash(config: TerrainConfig) -> str:
    payload = {"cache_version": SCENE_CACHE_VERSION, **asdict(config)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
