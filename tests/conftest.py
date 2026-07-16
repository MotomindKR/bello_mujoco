from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from bello_mujoco.assets_pipeline.mjcf import AnkleDifferentialConfig
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)


@pytest.fixture(scope="session")
def generated_test_assets() -> Iterator[dict[str, PreparedAsset]]:
    output_root = Path(tempfile.mkdtemp(prefix="bello-mujoco-test-assets-", dir="/tmp"))
    try:
        yield {
            asset: prepare_asset(AssetPrepConfig(asset=asset, output_root=output_root))
            for asset in ("full_body", "lower_body", "upper_body")
        }
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


@pytest.fixture(scope="session")
def full_body_asset(generated_test_assets: dict[str, PreparedAsset]) -> PreparedAsset:
    return generated_test_assets["full_body"]


@pytest.fixture(scope="session")
def full_body_ankle_tendon_asset() -> PreparedAsset:
    return prepare_asset(
        AssetPrepConfig(
            asset="full_body",
            ankle_differential_config=AnkleDifferentialConfig(enabled=True),
        )
    )
