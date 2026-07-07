"""Repository-local asset paths."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def assets_root() -> Path:
    return repo_root() / "assets"


def source_assets_root() -> Path:
    return assets_root() / "source" / "bello"


def generated_assets_root(output_root: Path | None = None) -> Path:
    return assets_root() / "generated" / "bello" if output_root is None else output_root


def generated_model_path(
    asset: str = "full_body", output_root: Path | None = None
) -> Path:
    return generated_assets_root(output_root) / asset / "mjcf" / f"bello_{asset}.xml"


def generated_viewer_model_path(
    asset: str = "full_body", output_root: Path | None = None
) -> Path:
    return (
        generated_assets_root(output_root)
        / asset
        / "mjcf"
        / f"bello_{asset}_viewer.xml"
    )


def generated_urdf_path(
    asset: str = "full_body", output_root: Path | None = None
) -> Path:
    return generated_assets_root(output_root) / asset / "urdf" / f"bello_{asset}.urdf"
