from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)


@pytest.fixture(scope="session")
def full_body_asset() -> Iterator[PreparedAsset]:
    output_root = Path(tempfile.mkdtemp(prefix="bello-mujoco-test-assets-"))
    try:
        yield prepare_asset(AssetPrepConfig(output_root=output_root))
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
