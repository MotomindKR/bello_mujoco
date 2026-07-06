"""Repository-local asset paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assets_root() -> Path:
    return repo_root() / "assets"


def source_assets_root() -> Path:
    return assets_root() / "source" / "bello"


def generated_assets_root() -> Path:
    return assets_root() / "generated" / "bello"


def generated_model_path(asset: str = "full_body") -> Path:
    return generated_assets_root() / asset / "mjcf" / f"bello_{asset}.xml"


def generated_urdf_path(asset: str = "full_body") -> Path:
    return generated_assets_root() / asset / "urdf" / f"bello_{asset}.urdf"
