from __future__ import annotations

import atexit
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shutil
import threading
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Literal, TextIO

from bello_mujoco.assets_pipeline import (
    generated_model_path,
    generated_urdf_path,
    generated_viewer_model_path,
    source_assets_root,
)
from bello_mujoco.assets_pipeline.mesh import (
    EndEffectorSphereMode,
    JointAvailabilityConfig,
    export_binary_simplified_mesh,
    rewrite_urdf_for_mujoco,
)
from bello_mujoco.assets_pipeline.mjcf import (
    AnkleDifferentialConfig,
    CollisionMode,
    ImuSensorConfig,
    compile_urdf_to_mjcf,
)
from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.contracts import SIM_TIMESTEP

AssetName = Literal["full_body"]
DEFAULT_ASSET_CACHE_LIMIT = 8
_CACHE_KEY_PATTERN = re.compile(r"[0-9a-f]{16}\Z")
_GENERATOR_SOURCE_PATHS = (
    "actuators.py",
    "assets_pipeline/__init__.py",
    "assets_pipeline/mesh.py",
    "assets_pipeline/mjcf.py",
    "assets_pipeline/prepare.py",
    "contracts.py",
    "spatial.py",
)
_GENERATOR_DISTRIBUTIONS = (
    "fast-simplification",
    "mujoco",
    "numpy",
    "trimesh",
)
_DEFAULT_CACHE_GUARD = threading.Lock()
_DEFAULT_CACHE_LEASES: dict[Path, TextIO] = {}
_DEFAULT_CACHE_ASSETS: dict[Path, PreparedAsset] = {}
SPHERE_REMOVED_ACTUATED_JOINTS = frozenset(
    {
        "right_wrist_yaw_joint",
        "left_wrist_yaw_joint",
        "right_finger_right_slide_joint",
        "right_finger_left_slide_joint",
        "left_finger_right_slide_joint",
        "left_finger_left_slide_joint",
    }
)


def _close_default_asset_cache_leases() -> None:
    with _DEFAULT_CACHE_GUARD:
        for lease in _DEFAULT_CACHE_LEASES.values():
            lease.close()
        _DEFAULT_CACHE_LEASES.clear()
        _DEFAULT_CACHE_ASSETS.clear()


atexit.register(_close_default_asset_cache_leases)


@dataclass(frozen=True)
class AssetPrepConfig:
    asset: AssetName = "full_body"
    output_root: Path | None = None
    mesh_face_budget: int = 1_000
    visual_mesh_face_budget: int = 200_000
    collision_mode: CollisionMode = "mesh"
    end_effector_sphere_mode: EndEffectorSphereMode = "none"
    target_total_mass_kg: float | None = None
    neighbor_exclusion_depth: int = 2
    enforce_policy_effort_limits: bool = True
    imu_config: ImuSensorConfig = ImuSensorConfig()
    ankle_differential_config: AnkleDifferentialConfig = AnkleDifferentialConfig()
    joint_availability: JointAvailabilityConfig = JointAvailabilityConfig()
    actuator_config: ActuatorConfig = field(default_factory=ActuatorConfig)

    def __post_init__(self) -> None:
        if self.asset != "full_body":
            raise ValueError(f"asset must be 'full_body', got {self.asset!r}")
        if self.collision_mode not in {"mesh", "box"}:
            raise ValueError(
                f"collision_mode must be 'mesh' or 'box', got {self.collision_mode!r}"
            )
        if self.end_effector_sphere_mode not in {"none", "both"}:
            raise ValueError(
                "end_effector_sphere_mode must be 'none' or 'both', got "
                f"{self.end_effector_sphere_mode!r}"
            )
        if self.target_total_mass_kg is not None and (
            not math.isfinite(self.target_total_mass_kg)
            or self.target_total_mass_kg <= 0.0
        ):
            raise ValueError(
                "target_total_mass_kg must be positive and finite when set, got "
                f"{self.target_total_mass_kg}"
            )
        removed_joints = set(self.joint_availability.disabled_joint_names)
        if self.end_effector_sphere_mode == "both":
            removed_joints.update(SPHERE_REMOVED_ACTUATED_JOINTS)
        active_joints = {
            name: config
            for name, config in self.actuator_config.joints.items()
            if name not in removed_joints
        }
        if len(active_joints) != len(self.actuator_config.joints):
            object.__setattr__(
                self,
                "actuator_config",
                ActuatorConfig(joints=active_joints),
            )


def canonical_physical_asset_config(
    *,
    imu_enabled: bool = False,
) -> AssetPrepConfig:
    return AssetPrepConfig(
        collision_mode="box",
        end_effector_sphere_mode="both",
        target_total_mass_kg=60.0,
        imu_config=ImuSensorConfig(
            enabled=imu_enabled,
            body_name="pelvis_link",
        ),
        joint_availability=JointAvailabilityConfig(
            waist_yaw=True,
            waist_roll=False,
            waist_pitch=False,
            neck_yaw=False,
            head_pitch=False,
        ),
    )


@dataclass(frozen=True)
class PreparedAsset:
    name: AssetName
    urdf_path: Path
    mjcf_path: Path
    viewer_mjcf_path: Path
    collision_mesh_count: int
    visual_mesh_count: int


def prepare_asset(config: AssetPrepConfig) -> PreparedAsset:
    output_root = resolved_output_root(config)
    lock_path = output_root.parent / f"{output_root.name}.lock"
    if config.output_root is None:
        return prepare_default_cached_asset(config, output_root, lock_path)
    with asset_generation_lock(lock_path):
        return prepare_asset_unlocked(config, output_root)


def prepare_default_cached_asset(
    config: AssetPrepConfig,
    output_root: Path,
    lock_path: Path,
) -> PreparedAsset:
    with _DEFAULT_CACHE_GUARD:
        cached = _DEFAULT_CACHE_ASSETS.get(output_root)
        if cached is not None and all(
            path.exists()
            for path in (
                cached.urdf_path,
                cached.mjcf_path,
                cached.viewer_mjcf_path,
            )
        ):
            os.utime(output_root)
            return cached
        if cached is not None:
            _DEFAULT_CACHE_ASSETS.pop(output_root)
            _DEFAULT_CACHE_LEASES.pop(output_root).close()

        with asset_generation_lock(lock_path):
            prepared = prepare_asset_unlocked(config, output_root)
            os.utime(output_root)
            lease_path = output_root.parent / f"{output_root.name}.lease"
            lease = lease_path.open("w")
            fcntl.flock(lease.fileno(), fcntl.LOCK_SH)
        _DEFAULT_CACHE_ASSETS[output_root] = prepared
        _DEFAULT_CACHE_LEASES[output_root] = lease

    prune_default_asset_cache(output_root.parent, keep=output_root)
    return prepared


def prepare_asset_unlocked(
    config: AssetPrepConfig,
    output_root: Path,
) -> PreparedAsset:
    asset = "full_body"
    source_root = source_assets_root()
    source_urdf = source_root / asset / "urdf" / f"bello_{asset}.urdf"
    if not source_urdf.exists():
        raise FileNotFoundError(source_urdf)

    out_urdf = generated_urdf_path(output_root)
    out_mesh_dir = output_root / asset / "meshes"
    out_visual_mesh_dir = output_root / asset / "visual_meshes"
    out_mjcf = generated_model_path(output_root)
    out_viewer_mjcf = generated_viewer_model_path(output_root)
    manifest_path = output_root / asset / "asset_manifest.json"
    cache_key = asset_config_hash(config)
    cached = read_prepared_asset_manifest(
        out_urdf, out_mjcf, out_viewer_mjcf, manifest_path, cache_key
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
        joint_availability=config.joint_availability,
    )
    if config.target_total_mass_kg is not None:
        scale_urdf_total_mass(out_urdf, config.target_total_mass_kg)
    actuated_joint_order = actuated_urdf_joint_names(out_urdf)
    validate_actuator_joint_contract(config.actuator_config, actuated_joint_order)
    for src, dst_name in mesh_sources.collision.items():
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
        enforce_policy_effort_limits=config.enforce_policy_effort_limits,
        imu_config=config.imu_config,
        ankle_differential_config=config.ankle_differential_config,
    )
    compile_urdf_to_mjcf(
        out_urdf,
        out_viewer_mjcf,
        collision_mode=config.collision_mode,
        neighbor_exclusion_depth=config.neighbor_exclusion_depth,
        include_visuals=True,
        actuator_config=config.actuator_config,
        actuated_joint_order=actuated_joint_order,
        enforce_policy_effort_limits=config.enforce_policy_effort_limits,
        imu_config=config.imu_config,
        ankle_differential_config=config.ankle_differential_config,
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


def scale_urdf_total_mass(urdf_path: Path, target_total_mass_kg: float) -> None:
    """Uniformly scale link masses and inertia tensors to a target robot mass."""

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    inertials: list[tuple[ET.Element, ET.Element]] = []
    total_mass = 0.0
    for link in root.findall("link"):
        mass = link.find("./inertial/mass")
        inertia = link.find("./inertial/inertia")
        if mass is None or inertia is None or "value" not in mass.attrib:
            raise ValueError(
                f"link {link.get('name', '<unnamed>')!r} is missing complete inertial data"
            )
        link_mass = float(mass.attrib["value"])
        if not math.isfinite(link_mass) or link_mass <= 0.0:
            raise ValueError(
                f"link {link.get('name', '<unnamed>')!r} has invalid mass {link_mass}"
            )
        total_mass += link_mass
        inertials.append((mass, inertia))

    if not inertials or not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ValueError(f"URDF has invalid total mass {total_mass}")

    scale = target_total_mass_kg / total_mass
    for mass, inertia in inertials:
        mass.set("value", f"{float(mass.attrib['value']) * scale:.17g}")
        for attribute in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"):
            if attribute not in inertia.attrib:
                raise ValueError(f"inertia tensor is missing {attribute!r}")
            value = float(inertia.attrib[attribute])
            if not math.isfinite(value):
                raise ValueError(f"inertia tensor has nonfinite {attribute}={value}")
            inertia.set(attribute, f"{value * scale:.17g}")

    ET.indent(tree, space="  ")
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)


def validate_actuator_joint_contract(
    config: ActuatorConfig,
    actuated_joint_order: tuple[str, ...],
) -> None:
    configured = set(config.joints)
    actuated = set(actuated_joint_order)
    missing = sorted(actuated - configured)
    unknown = sorted(configured - actuated)
    if missing:
        raise ValueError(f"missing actuator config for joints: {', '.join(missing)}")
    if unknown:
        raise ValueError(
            "actuator config contains joints absent from the prepared morphology: "
            + ", ".join(unknown)
        )


def resolved_output_root(config: AssetPrepConfig) -> Path:
    if config.output_root is not None:
        return config.output_root
    return default_asset_cache_root() / asset_config_hash(config)


def default_asset_cache_root() -> Path:
    configured_root = os.environ.get("XDG_CACHE_HOME")
    if configured_root:
        root = Path(configured_root).expanduser()
        if not root.is_absolute():
            raise ValueError("XDG_CACHE_HOME must be an absolute path")
    else:
        root = Path.home() / ".cache"
    return root / "bello-mujoco" / "assets"


def asset_config_hash(config: AssetPrepConfig) -> str:
    payload = {
        "asset": config.asset,
        "mesh_face_budget": config.mesh_face_budget,
        "visual_mesh_face_budget": config.visual_mesh_face_budget,
        "collision_mode": config.collision_mode,
        "end_effector_sphere_mode": config.end_effector_sphere_mode,
        "target_total_mass_kg": config.target_total_mass_kg,
        "neighbor_exclusion_depth": config.neighbor_exclusion_depth,
        "enforce_policy_effort_limits": config.enforce_policy_effort_limits,
        "imu_config": asdict(config.imu_config),
        "ankle_differential_config": asdict(config.ankle_differential_config),
        "joint_availability": asdict(config.joint_availability),
        "sim_timestep": SIM_TIMESTEP,
        "actuator_config": asdict(config.actuator_config),
        "generator": asset_generator_fingerprint(),
        "source": source_asset_fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


@lru_cache(maxsize=1)
def asset_generator_fingerprint() -> str:
    package_root = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for relative in _GENERATOR_SOURCE_PATHS:
        path = package_root / relative
        if not path.is_file():
            raise FileNotFoundError(
                f"asset generator source is missing from the installation: {path}"
            )
        hasher.update(relative.encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    hasher.update(f"python={platform.python_version()}".encode())
    hasher.update(b"\0")
    for distribution in _GENERATOR_DISTRIBUTIONS:
        hasher.update(f"{distribution}={distribution_version(distribution)}".encode())
        hasher.update(b"\0")
    return hasher.hexdigest()


def source_asset_fingerprint() -> str:
    asset = "full_body"
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


def prune_default_asset_cache(cache_root: Path, *, keep: Path) -> None:
    with asset_generation_lock(cache_root / ".prune.lock"):
        candidates = sorted(
            (
                path
                for path in cache_root.iterdir()
                if path.is_dir() and _CACHE_KEY_PATTERN.fullmatch(path.name)
            ),
            key=lambda path: path.stat().st_mtime,
        )
        remaining = len(candidates)
        for candidate in candidates:
            if remaining <= DEFAULT_ASSET_CACHE_LIMIT:
                break
            if candidate == keep:
                continue
            lock_path = cache_root / f"{candidate.name}.lock"
            with lock_path.open("w") as lock_file:
                try:
                    fcntl.flock(
                        lock_file.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB,
                    )
                except BlockingIOError:
                    continue
                lease_path = cache_root / f"{candidate.name}.lease"
                with lease_path.open("w") as lease_file:
                    try:
                        fcntl.flock(
                            lease_file.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB,
                        )
                    except BlockingIOError:
                        continue
                    try:
                        shutil.rmtree(candidate)
                    finally:
                        fcntl.flock(lease_file.fileno(), fcntl.LOCK_UN)
            remaining -= 1


def read_prepared_asset_manifest(
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
    if data.get("name") != "full_body":
        return None
    if data.get("cache_key") != cache_key:
        return None
    try:
        content_sha256 = prepared_asset_content_sha256(
            urdf_path,
            mjcf_path,
            viewer_mjcf_path,
            manifest_path,
        )
    except (ET.ParseError, OSError, ValueError):
        return None
    if data.get("content_sha256") != content_sha256:
        return None
    try:
        return PreparedAsset(
            "full_body",
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
        "content_sha256": prepared_asset_content_sha256(
            prepared.urdf_path,
            prepared.mjcf_path,
            prepared.viewer_mjcf_path,
            manifest_path,
        ),
    }
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, sort_keys=True))
    tmp_path.replace(manifest_path)


def prepared_asset_content_sha256(
    urdf_path: Path,
    mjcf_path: Path,
    viewer_mjcf_path: Path,
    manifest_path: Path,
) -> str:
    """Hash the canonical prepared XML and every runtime mesh it references."""

    asset_root = manifest_path.parent.resolve()
    documents = {
        urdf_path.resolve(),
        mjcf_path.resolve(),
        viewer_mjcf_path.resolve(),
    }
    files = set(documents)
    pending_xml = list(documents)
    parsed_xml: set[Path] = set()
    while pending_xml:
        document = pending_xml.pop()
        if document in parsed_xml:
            continue
        parsed_xml.add(document)
        root = ET.parse(document).getroot()
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1]
            reference = None
            if tag == "mesh":
                reference = element.get("filename") or element.get("file")
            elif tag in {"hfield", "include", "skin", "texture"}:
                reference = element.get("file")
            if not reference:
                continue
            if "://" in reference:
                raise ValueError(
                    f"prepared asset contains a nonlocal reference: {reference!r}"
                )
            referenced = (document.parent / reference).resolve()
            try:
                referenced.relative_to(asset_root)
            except ValueError as error:
                raise ValueError(
                    f"prepared asset references a file outside its cache: {referenced}"
                ) from error
            files.add(referenced)
            if tag == "include":
                pending_xml.append(referenced)

    hasher = hashlib.sha256()
    for path in sorted(files, key=lambda candidate: candidate.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(path)
        hasher.update(path.relative_to(asset_root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


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
