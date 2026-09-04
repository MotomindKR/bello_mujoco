from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import BelloEnvConfig, BelloLocomotionEnv
from bello_mujoco.sim import (
    BelloModel,
    SimulationPhysicsConfig,
    WarpConfig,
    load_prepared_bello_model,
)
from bello_mujoco.terrain import TerrainConfig, terrain_surface_height


MAX_SOLE_PENETRATION = 0.005
DROP_HEIGHT = 0.12
DROP_VELOCITY = -2.0
ROLLOUT_STEPS = 300


def contact_physics() -> SimulationPhysicsConfig:
    return SimulationPhysicsConfig(
        timestep=0.005,
        foot_terrain_margin=0.0125,
        foot_terrain_solref=(0.01, 1.0),
        foot_terrain_solimp=(0.98, 0.999, 0.001, 0.5, 2.0),
    )


def terrain_config(mode: str) -> TerrainConfig:
    return TerrainConfig(
        mode=mode,
        seed=4,
        hfield_resolution=17,
        hfield_amplitude=0.03,
        slope_angle_range=(-0.005, 0.005),
    )


def sole_clearance(model: BelloModel, qpos: np.ndarray) -> float:
    mj_model = model.mj_model
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)
    terrain_geom_id = int(model.terrain_geom_ids[0])
    clearances: list[float] = []
    for geom_id in model.foot_geom_ids:
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        if not name or not name.endswith("collision_box_1"):
            continue
        half_size = mj_model.geom_size[geom_id]
        sole_corners = np.asarray(
            [
                (x, y, -half_size[2])
                for x in (-half_size[0], half_size[0])
                for y in (-half_size[1], half_size[1])
            ]
        )
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        world_corners = data.geom_xpos[geom_id] + sole_corners @ rotation.T
        local_height = terrain_surface_height(
            mj_model, terrain_geom_id, world_corners[:, :2]
        )
        clearances.extend((world_corners[:, 2] - local_height).tolist())
    assert len(clearances) == 8
    return min(clearances)


def dropped_native_rollout(model: BelloModel) -> np.ndarray:
    mj_model = model.mj_model
    data = mujoco.MjData(mj_model)
    data.qpos[:] = model.default_qpos
    data.qvel[:] = model.default_qvel
    data.ctrl[:] = model.default_ctrl
    data.qpos[model.root_qpos_adr + 2] += DROP_HEIGHT
    data.qvel[model.root_qvel_adr + 2] = DROP_VELOCITY
    mujoco.mj_forward(mj_model, data)
    qpos = []
    for _ in range(ROLLOUT_STEPS):
        mujoco.mj_step(mj_model, data)
        qpos.append(data.qpos.copy())
    return np.asarray(qpos)


@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
def test_native_rollout_limits_geometric_sole_penetration(
    bello_asset: PreparedAsset, terrain_mode: str
) -> None:
    model = load_prepared_bello_model(
        bello_asset,
        terrain=terrain_config(terrain_mode),
        simulation_physics=contact_physics(),
        load_mjx=False,
    )

    minimum_clearance = min(
        sole_clearance(model, qpos) for qpos in dropped_native_rollout(model)
    )

    assert minimum_clearance >= -MAX_SOLE_PENETRATION


@pytest.mark.integration
@pytest.mark.parametrize("terrain_mode", ["plane", "rough"])
def test_mjx_warp_rollout_limits_geometric_sole_penetration(
    bello_asset: PreparedAsset, terrain_mode: str
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        bello_asset,
        warp=WarpConfig(contact_budget_per_env=64, njmax=2048),
        env_config=BelloEnvConfig(
            decimation=4,
            terrain=terrain_config(terrain_mode),
            simulation_physics=contact_physics(),
        ),
    )
    state, _ = env.reset()
    data = state.data.replace(
        qpos=state.data.qpos.at[env.model.root_qpos_adr + 2].add(DROP_HEIGHT),
        qvel=state.data.qvel.at[env.model.root_qvel_adr + 2].set(DROP_VELOCITY),
        ctrl=jnp.asarray(env.model.default_ctrl),
    )
    data = jax.jit(lambda value: mjx.forward(env.model.mjx_model, value))(data)

    @jax.jit
    def rollout(initial_data: mjx.Data) -> jax.Array:
        def step(data: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
            next_data = mjx.step(env.model.mjx_model, data)
            return next_data, next_data.qpos

        return jax.lax.scan(step, initial_data, None, length=ROLLOUT_STEPS)[1]

    qpos = np.asarray(rollout(data))
    minimum_clearance = min(
        sole_clearance(env.model, sample_qpos) for sample_qpos in qpos
    )

    assert env.control_timestep == pytest.approx(0.02)
    assert minimum_clearance >= -MAX_SOLE_PENETRATION
