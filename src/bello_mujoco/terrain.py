"""Runtime terrain composition for generated MuJoCo robot models."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

TerrainMode = Literal["none", "plane"]
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


ROBOT_ONLY_TERRAIN = TerrainConfig(mode="none")
FLAT_TERRAIN = TerrainConfig(mode="plane")


def runtime_mjcf_path(robot_mjcf_path: Path, config: TerrainConfig) -> Path:
    """Return a robot MJCF or a generated sibling scene MJCF for the terrain."""

    robot_mjcf_path = Path(robot_mjcf_path)
    if config.mode == "none":
        return robot_mjcf_path
    if config.mode != "plane":
        raise ValueError(f"unsupported terrain mode: {config.mode!r}")

    scene_path = robot_mjcf_path.with_name(
        f"{robot_mjcf_path.stem}.scene.{terrain_config_hash(config)}.xml"
    )
    if scene_path.exists():
        return scene_path

    root = ET.parse(robot_mjcf_path).getroot()
    add_plane(root, config)
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


def terrain_config_hash(config: TerrainConfig) -> str:
    payload = {"cache_version": SCENE_CACHE_VERSION, **asdict(config)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
