"""Asset paths and preparation pipeline for Bello MuJoCo models."""

from __future__ import annotations

from contextlib import ExitStack
from functools import cache
from importlib.resources import as_file, files
from pathlib import Path

from bello_mujoco.robot import RobotDefinition

_RESOURCE_STACK = ExitStack()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def robot_descriptions_root() -> Path:
    checkout_path = repo_root() / "deps" / "bello-stack" / "descriptions"
    if checkout_path.is_dir():
        return checkout_path
    if (repo_root() / "pyproject.toml").is_file():
        raise RuntimeError(
            "the pinned bello-stack robot description is not initialized; "
            "run `git submodule update --init --recursive`"
        )
    return package_data_path("descriptions")


def stack_description_path(relative_path: str) -> Path:
    path = repo_root() / "deps" / "bello-stack" / "descriptions" / relative_path
    if not path.exists():
        raise RuntimeError(
            "this operation requires a source checkout with the pinned bello-stack "
            "submodule initialized; run `git submodule update --init --recursive`"
        )
    return path


def generated_model_path(output_root: Path, definition: RobotDefinition) -> Path:
    return definition.generated_mjcf_path(output_root)


def generated_viewer_model_path(output_root: Path, definition: RobotDefinition) -> Path:
    return definition.generated_mjcf_path(output_root, viewer=True)


def generated_urdf_path(output_root: Path, definition: RobotDefinition) -> Path:
    return definition.generated_urdf_path(output_root)
