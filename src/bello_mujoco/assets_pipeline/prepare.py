from __future__ import annotations

import fcntl
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from bello_mujoco.assets import (
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
from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.contracts import SIM_TIMESTEP

AssetName = Literal["full_body", "lower_body", "upper_body"]
ASSET_CACHE_VERSION = 4


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
    actuator_config: ActuatorConfig = field(default_factory=ActuatorConfig)


@dataclass(frozen=True)
class PreparedAsset:
    name: str
    urdf_path: Path
    mjcf_path: Path
    viewer_mjcf_path: Path
    collision_mesh_count: int
    visual_mesh_count: int


def prepare_asset(config: AssetPrepConfig) -> PreparedAsset:
    asset = config.asset
    validate_asset_config(asset, config)
    output_root = resolved_output_root(config, asset)
    lock_path = output_root.parent / f"{output_root.name}.lock"
    with asset_generation_lock(lock_path):
        return prepare_asset_unlocked(config, asset, output_root)


def prepare_asset_unlocked(
    config: AssetPrepConfig,
    asset: Literal["full_body", "lower_body", "upper_body"],
    output_root: Path,
) -> PreparedAsset:
    source_root = source_assets_root()
    source_urdf = source_root / asset / "urdf" / f"bello_{asset}.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    out_urdf = generated_urdf_path(asset, output_root)
    out_mesh_dir = output_root / asset / "meshes"
    out_visual_mesh_dir = output_root / asset / "visual_meshes"
    out_mjcf = generated_model_path(asset, output_root)
    out_viewer_mjcf = generated_viewer_model_path(asset, output_root)
    manifest_path = output_root / asset / "asset_manifest.json"
    cache_key = asset_config_hash(config, asset)
    cached = read_prepared_asset_manifest(
        asset, out_urdf, out_mjcf, out_viewer_mjcf, manifest_path, cache_key
    )
    if cached is not None:
        return cached

    shutil.rmtree(out_mesh_dir, ignore_errors=True)
    shutil.rmtree(out_visual_mesh_dir, ignore_errors=True)
    manifest_path.unlink(missing_ok=True)
    out_urdf.parent.mkdir(parents=True, exist_ok=True)
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_visual_mesh_dir.mkdir(parents=True, exist_ok=True)
    out_mjcf.parent.mkdir(parents=True, exist_ok=True)

    mesh_sources = rewrite_urdf_for_mujoco(
        source_urdf,
        out_urdf,
        end_effector_sphere_mode=config.end_effector_sphere_mode,
    )
    actuated_joint_order = actuated_urdf_joint_names(out_urdf)
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
        actuator_config=config.actuator_config,
        actuated_joint_order=actuated_joint_order,
    )
    compile_urdf_to_mjcf(
        out_urdf,
        out_viewer_mjcf,
        collision_mode=config.collision_mode,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
        include_visuals=True,
        actuator_config=config.actuator_config,
        actuated_joint_order=actuated_joint_order,
    )
    prepared = PreparedAsset(
        asset,
        out_urdf,
        out_mjcf,
        out_viewer_mjcf,
        len(mesh_sources.collision),
        len(mesh_sources.visual),
    )
    write_prepared_asset_manifest(prepared, manifest_path, cache_key)
    return prepared


def actuated_urdf_joint_names(urdf_path: Path) -> tuple[str, ...]:
    root = ET.parse(urdf_path).getroot()
    names: list[str] = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        joint_type = joint.get("type")
        if name is None or joint_type in {None, "fixed"}:
            continue
        names.append(name)
    return tuple(names)


def resolved_output_root(
    config: AssetPrepConfig,
    asset: Literal["full_body", "lower_body", "upper_body"],
) -> Path:
    if config.output_root is not None:
        return config.output_root
    return (
        Path(tempfile.gettempdir())
        / "bello-mujoco-assets"
        / asset_config_hash(config, asset)
    )


def asset_config_hash(
    config: AssetPrepConfig,
    asset: Literal["full_body", "lower_body", "upper_body"],
) -> str:
    payload = {
        "cache_version": ASSET_CACHE_VERSION,
        "asset": asset,
        "mesh_face_budget": config.mesh_face_budget,
        "foot_collision_mesh_face_budget": config.foot_collision_mesh_face_budget,
        "visual_mesh_face_budget": config.visual_mesh_face_budget,
        "collision_mode": config.collision_mode,
        "end_effector_sphere_mode": config.end_effector_sphere_mode,
        "neighbor_exclusion_depth": config.neighbor_exclusion_depth,
        "sim_timestep": SIM_TIMESTEP,
        "actuator_config": asdict(config.actuator_config),
        "source": source_asset_fingerprint(asset),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def source_asset_fingerprint(
    asset: Literal["full_body", "lower_body", "upper_body"],
) -> str:
    source_root = source_assets_root() / asset
    hasher = hashlib.sha256()
    for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        hasher.update(path.relative_to(source_root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


@contextmanager
def asset_generation_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def read_prepared_asset_manifest(
    asset: str,
    urdf_path: Path,
    mjcf_path: Path,
    viewer_mjcf_path: Path,
    manifest_path: Path,
    cache_key: str,
) -> PreparedAsset | None:
    if not (
        manifest_path.exists()
        and urdf_path.exists()
        and mjcf_path.exists()
        and viewer_mjcf_path.exists()
        and mjcf_mesh_files_exist(mjcf_path)
        and mjcf_mesh_files_exist(viewer_mjcf_path)
    ):
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    if data.get("name") != asset:
        return None
    if data.get("cache_key") != cache_key:
        return None
    try:
        return PreparedAsset(
            asset,
            urdf_path,
            mjcf_path,
            viewer_mjcf_path,
            int(data["collision_mesh_count"]),
            int(data["visual_mesh_count"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def write_prepared_asset_manifest(
    prepared: PreparedAsset, manifest_path: Path, cache_key: str
) -> None:
    data = {
        "cache_key": cache_key,
        "name": prepared.name,
        "collision_mesh_count": prepared.collision_mesh_count,
        "visual_mesh_count": prepared.visual_mesh_count,
    }
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, sort_keys=True))
    tmp_path.replace(manifest_path)


def mjcf_mesh_files_exist(mjcf_path: Path) -> bool:
    try:
        root = ET.parse(mjcf_path).getroot()
    except ET.ParseError:
        return False
    for mesh in root.findall("./asset/mesh"):
        file_attr = mesh.get("file")
        if file_attr is None:
            continue
        if not (mjcf_path.parent / file_attr).resolve().exists():
            return False
    return True


def validate_asset_config(
    asset: Literal["full_body", "lower_body", "upper_body"],
    config: AssetPrepConfig,
) -> None:
    if asset == "lower_body" and config.end_effector_sphere_mode != "none":
        raise ValueError(
            "end-effector sphere replacement requires arm joints and is only "
            "valid for full_body or upper_body assets"
        )
