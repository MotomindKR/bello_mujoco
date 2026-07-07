from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bello_mujoco.assets import generated_assets_root, generated_model_path, generated_urdf_path
from bello_mujoco.assets_pipeline.mesh import copy_source_inputs, export_binary_simplified_mesh, rewrite_urdf_for_mujoco
from bello_mujoco.assets_pipeline.mjcf import CollisionMode, compile_urdf_to_mjcf

DEFAULT_SOURCE_ROOT = Path(
    "/home/gmmyung/motomind_platform/systems/motomind_isaaclab/"
    "motomind_isaaclab/assets/data/bello"
)


@dataclass(frozen=True)
class AssetPrepConfig:
    source_root: Path = DEFAULT_SOURCE_ROOT
    asset: Literal["full_body"] = "full_body"
    mesh_face_budget: int = 500
    collision_mode: CollisionMode = "foot-mesh"
    neighbor_exclusion_depth: int = 2
    copy_source: bool = True


@dataclass(frozen=True)
class PreparedAsset:
    name: str
    urdf_path: Path
    mjcf_path: Path
    mesh_count: int


def prepare_asset(config: AssetPrepConfig) -> PreparedAsset:
    source_root = config.source_root.expanduser().resolve()
    source_urdf = source_root / config.asset / "urdf" / f"bello_{config.asset}.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    if config.copy_source:
        copy_source_inputs(source_root)

    out_urdf = generated_urdf_path(config.asset)
    out_mesh_dir = generated_assets_root() / config.asset / "meshes"
    out_mjcf = generated_model_path(config.asset)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_mjcf.parent.mkdir(parents=True, exist_ok=True)

    mesh_sources = rewrite_urdf_for_mujoco(source_urdf, out_urdf)
    for src, dst_name in mesh_sources.items():
        export_binary_simplified_mesh(src, out_mesh_dir / dst_name, face_budget=config.mesh_face_budget)

    compile_urdf_to_mjcf(
        out_urdf,
        out_mjcf,
        collision_mode=config.collision_mode,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
    )
    return PreparedAsset(config.asset, out_urdf, out_mjcf, len(mesh_sources))
