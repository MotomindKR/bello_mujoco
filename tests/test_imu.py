from pathlib import Path

import mujoco
import pytest

from bello_mujoco.assets_pipeline.mjcf import ImuSensorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.imu import MujocoImu
from bello_mujoco.sim import load_prepared_bello_model


def test_generated_mujoco_imu_is_bound_to_pelvis(tmp_path: Path) -> None:
    prepared = prepare_asset(
        AssetPrepConfig(
            asset="full_body",
            output_root=tmp_path,
            mesh_face_budget=100,
            visual_mesh_face_budget=100,
            imu_config=ImuSensorConfig(enabled=True, body_name="pelvis_link"),
        )
    )
    model = load_prepared_bello_model(prepared, load_mjx=False)
    imu = MujocoImu.bind(model.mj_model)
    data = mujoco.MjData(model.mj_model)
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    data.ctrl[:] = model.default_ctrl
    mujoco.mj_forward(model.mj_model, data)

    sample = imu.sample(data)

    assert sample.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert sample.angular_velocity_xyz == pytest.approx((0.0, 0.0, 0.0))
    assert sample.linear_acceleration_xyz == pytest.approx((0.0, 0.0, 0.0), abs=1.0e-8)
