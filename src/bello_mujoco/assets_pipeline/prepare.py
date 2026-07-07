from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bello_mujoco.assets import (
    generated_assets_root,
    generated_model_path,
    generated_urdf_path,
    generated_viewer_model_path,
    source_assets_root,
)
from bello_mujoco.assets_pipeline.mesh import (
    EndEffectorSphereMode,
    export_binary_simplified_mesh,
    rewrite_urdf_for_mujoco,
)
from bello_mujoco.assets_pipeline.mjcf import (
    FOOT_MESH_NAMES,
    CollisionMode,
    compile_urdf_to_mjcf,
)

AssetName = Literal["full_body", "lower_body", "upper_body", "down_body"]


@dataclass(frozen=True)
class AssetPrepConfig:
    asset: AssetName = "full_body"
    output_root: Path | None = None
    mesh_face_budget: int = 1_000
    foot_collision_mesh_face_budget: int = 1_000
    visual_mesh_face_budget: int = 200_000
    collision_mode: CollisionMode = "mesh"
    end_effector_sphere_mode: EndEffectorSphereMode = "none"
    neighbor_exclusion_depth: int = 2


@dataclass(frozen=True)
class PreparedAsset:
    name: str
    urdf_path: Path
    mjcf_path: Path
    viewer_mjcf_path: Path
    collision_mesh_count: int
    visual_mesh_count: int


def prepare_asset(config: AssetPrepConfig) -> PreparedAsset:
    asset = canonical_asset(config.asset)
    validate_asset_config(asset, config)
    source_root = source_assets_root()
    source_urdf = source_root / asset / "urdf" / f"bello_{asset}.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    generated_root = generated_assets_root(config.output_root)
    out_urdf = generated_urdf_path(asset, config.output_root)
    out_mesh_dir = generated_root / asset / "meshes"
    out_visual_mesh_dir = generated_root / asset / "visual_meshes"
    out_mjcf = generated_model_path(asset, config.output_root)
    out_viewer_mjcf = generated_viewer_model_path(asset, config.output_root)
    shutil.rmtree(out_mesh_dir, ignore_errors=True)
    shutil.rmtree(out_visual_mesh_dir, ignore_errors=True)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_visual_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_mjcf.parent.mkdir(parents=True, exist_ok=True)

    mesh_sources = rewrite_urdf_for_mujoco(
        source_urdf,
        out_urdf,
        end_effector_sphere_mode=config.end_effector_sphere_mode,
    )
    for src, dst_name in mesh_sources.collision.items():
        if (
            config.collision_mode in {"foot-mesh", "mesh"}
            and src.stem in FOOT_MESH_NAMES
        ):
            export_binary_simplified_mesh(
                src,
                out_mesh_dir / dst_name,
                face_budget=config.foot_collision_mesh_face_budget,
                aggression=5.0,
                weld_vertices=True,
            )
        else:
            export_binary_simplified_mesh(
                src,
                out_mesh_dir / dst_name,
                face_budget=config.mesh_face_budget,
                aggression=5.0 if config.collision_mode == "mesh" else 10.0,
                weld_vertices=config.collision_mode == "mesh",
            )
    for src, dst_name in mesh_sources.visual.items():
        export_binary_simplified_mesh(
            src,
            out_visual_mesh_dir / dst_name,
            face_budget=config.visual_mesh_face_budget,
            aggression=5.0,
            weld_vertices=True,
        )

    compile_urdf_to_mjcf(
        out_urdf,
        out_mjcf,
        collision_mode=config.collision_mode,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
        include_visuals=False,
    )
    compile_urdf_to_mjcf(
        out_urdf,
        out_viewer_mjcf,
        collision_mode=config.collision_mode,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
        include_visuals=True,
    )
    return PreparedAsset(
        asset,
        out_urdf,
        out_mjcf,
        out_viewer_mjcf,
        len(mesh_sources.collision),
        len(mesh_sources.visual),
    )


def canonical_asset(
    asset: AssetName,
) -> Literal["full_body", "lower_body", "upper_body"]:
    return "lower_body" if asset == "down_body" else asset


def validate_asset_config(
    asset: Literal["full_body", "lower_body", "upper_body"],
    config: AssetPrepConfig,
) -> None:
    if asset == "lower_body" and config.end_effector_sphere_mode != "none":
        raise ValueError(
            "end-effector sphere replacement requires arm joints and is only "
            "valid for full_body or upper_body assets"
        )
