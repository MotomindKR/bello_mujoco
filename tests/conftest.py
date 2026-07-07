from __future__ import annotations

import pytest

from bello_mujoco.assets import generated_model_path, generated_viewer_model_path
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset


@pytest.fixture(scope="session", autouse=True)
def ensure_generated_test_assets() -> None:
    for asset in ("full_body", "lower_body", "upper_body"):
        if (
            generated_model_path(asset).exists()
            and generated_viewer_model_path(asset).exists()
        ):
            continue
        prepare_asset(AssetPrepConfig(asset=asset))
