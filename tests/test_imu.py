from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.mjcf import ImuSensorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, prepare_asset
from bello_mujoco.imu import ImuNoiseConfig, ImuNoiseInjector, ImuSample, MujocoImu
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv
from bello_mujoco.evaluation.render import mujoco_eval_state
from bello_mujoco.sim import load_prepared_bello_model
from bello_mujoco.terrain import ROBOT_ONLY_TERRAIN


def test_generated_mujoco_imu_is_bound_to_pelvis(tmp_path: Path) -> None:
    prepared = prepare_asset(
        AssetPrepConfig(
            asset="bello",
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

    root_qvel = model.root_qvel_adr
    data.qvel[root_qvel + 3 : root_qvel + 6] = (0.2, -0.3, 0.4)
    mujoco.mj_forward(model.mj_model, data)
    sample = imu.sample(data)

    env = BelloLocomotionEnv(model, config=BelloEnvConfig(terrain=ROBOT_ONLY_TERRAIN))
    zeros = jnp.zeros(2, dtype=jnp.float32)
    state = mujoco_eval_state(
        env,
        data,
        command=np.zeros(3, dtype=np.float32),
        last_action=np.zeros(env.action_size, dtype=np.float32),
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=0,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(0),
    )

    assert env._imu_angular_velocity(state) == pytest.approx(
        sample.angular_velocity_xyz, abs=1.0e-6
    )
    assert env._imu_linear_acceleration(state) == pytest.approx(
        sample.linear_acceleration_xyz, abs=1.0e-6
    )
    assert sample.angular_velocity_xyz != pytest.approx((0.0, 0.0, 0.0))


def test_imu_noise_is_reproducible_normalized_and_sensor_only() -> None:
    config = ImuNoiseConfig(
        enabled=True,
        orientation_std_radians=0.01,
        linear_acceleration_std_mps2=0.1,
        seed=7,
    )
    sample = ImuSample(
        simulation_time=1.25,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        angular_velocity_xyz=(1.0, 2.0, 3.0),
        linear_acceleration_xyz=(4.0, 5.0, 6.0),
        local_velocity_xyz=(7.0, 8.0, 9.0),
        world_position_xyz=(10.0, 11.0, 12.0),
    )

    first = ImuNoiseInjector(config).apply(sample)
    second = ImuNoiseInjector(config).apply(sample)

    assert first == second
    assert first.orientation_xyzw != sample.orientation_xyzw
    assert np.linalg.norm(first.orientation_xyzw) == pytest.approx(1.0)
    assert first.linear_acceleration_xyz != sample.linear_acceleration_xyz
    assert first.angular_velocity_xyz == sample.angular_velocity_xyz
    assert first.local_velocity_xyz == sample.local_velocity_xyz
    assert first.world_position_xyz == sample.world_position_xyz
    assert first.simulation_time == sample.simulation_time


def test_disabled_imu_noise_returns_original_sample() -> None:
    sample = ImuSample(
        simulation_time=0.0,
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        linear_acceleration_xyz=(0.0, 0.0, 0.0),
    )

    assert ImuNoiseInjector(ImuNoiseConfig()).apply(sample) is sample


@pytest.mark.parametrize(
    "config",
    [
        ImuNoiseConfig(enabled=True),
        ImuNoiseConfig(orientation_std_radians=-0.1),
        ImuNoiseConfig(linear_acceleration_std_mps2=float("inf")),
        ImuNoiseConfig(seed=-1),
    ],
)
def test_invalid_imu_noise_config_is_rejected(config: ImuNoiseConfig) -> None:
    with pytest.raises(ValueError, match="imu_noise"):
        ImuNoiseInjector(config)


def test_model_imu_frame_is_derived_from_generated_sensor_site(tmp_path: Path) -> None:
    prepared = prepare_asset(
        AssetPrepConfig(
            asset="bello",
            output_root=tmp_path / "generated-sensor",
            imu_config=ImuSensorConfig(enabled=True, body_name="torso_link"),
        )
    )
    model = load_prepared_bello_model(prepared, load_mjx=False)

    assert model.imu_body_name == "torso_link"
    with pytest.raises(ValueError, match="does not match the generated sensor site"):
        load_prepared_bello_model(
            prepared,
            imu_body_name="pelvis_link",
            load_mjx=False,
        )


def test_disabled_sensor_carries_asset_configured_imu_frame(tmp_path: Path) -> None:
    env = BelloLocomotionEnv.load(
        AssetPrepConfig(
            asset="bello",
            output_root=tmp_path / "disabled-sensor",
            imu_config=ImuSensorConfig(enabled=False, body_name="torso_link"),
        ),
        env_config=BelloEnvConfig(terrain=ROBOT_ONLY_TERRAIN),
        load_mjx=False,
    )

    assert env.model.imu_body_name == "torso_link"
