from __future__ import annotations

import tyro

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset


def run(config: AssetPrepConfig = AssetPrepConfig()) -> None:
    prepared = prepare_asset(config)
    print(
        f"{prepared.name}: {prepared.collision_mesh_count} collision meshes, "
        f"{prepared.visual_mesh_count} visual meshes -> {prepared.mjcf_path} "
        f"(viewer: {prepared.viewer_mjcf_path})"
    )


def entrypoint() -> None:
    run(tyro.cli(AssetPrepConfig))


if __name__ == "__main__":
    entrypoint()
