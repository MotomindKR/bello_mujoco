"""Project asset and bundled data paths."""

from __future__ import annotations

from contextlib import ExitStack
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path

_RESOURCE_STACK = ExitStack()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@cache
def package_data_path(name: str) -> Path:
    resource = files("bello_mujoco").joinpath(name)
    if not resource.is_dir():
        raise FileNotFoundError(
            f"packaged bello_mujoco data directory not found: {name}"
        )
    return _RESOURCE_STACK.enter_context(as_file(resource))


def project_data_path(name: str) -> Path:
    source_path = repo_root() / name
    if source_path.exists():
        return source_path
    return package_data_path(name)


def assets_root() -> Path:
    return project_data_path("assets")


def source_assets_root() -> Path:
    return assets_root() / "source" / "bello"


def generated_model_path(asset: str, output_root: Path) -> Path:
    return output_root / asset / "mjcf" / f"bello_{asset}.xml"


def generated_viewer_model_path(asset: str, output_root: Path) -> Path:
    return output_root / asset / "mjcf" / f"bello_{asset}_viewer.xml"


def generated_urdf_path(asset: str, output_root: Path) -> Path:
    return output_root / asset / "urdf" / f"bello_{asset}.urdf"
