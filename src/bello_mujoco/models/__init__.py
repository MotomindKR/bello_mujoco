"""Source-checkout paths; canonical descriptions are owned by bello-stack."""

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_data_path(name: str) -> Path:
    path = repo_root() / name
    if not path.is_dir():
        raise FileNotFoundError(f"source-checkout data directory is missing: {path}")
    return path


def robot_descriptions_root() -> Path:
    return project_data_path("deps/bello-stack/descriptions")


def stack_description_path(relative_path: str) -> Path:
    path = robot_descriptions_root() / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
