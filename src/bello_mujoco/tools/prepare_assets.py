from __future__ import annotations

import hydra
from omegaconf import DictConfig

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.hydra_config import config_path, structured_config


def run(config: AssetPrepConfig = AssetPrepConfig()) -> None:
    prepared = prepare_asset(config)
    print(
        f"{prepared.name}: {prepared.collision_mesh_count} collision meshes, "
        f"{prepared.visual_mesh_count} visual meshes -> {prepared.mjcf_path} "
        f"(viewer: {prepared.viewer_mjcf_path})"
    )


def entrypoint() -> None:
    hydra_entrypoint()


@hydra.main(
    version_base=None,
    config_path=config_path("prepare_assets"),
    config_name="default",
)
def hydra_entrypoint(config: DictConfig) -> None:
    run(structured_config(AssetPrepConfig, config))


if __name__ == "__main__":
    entrypoint()
