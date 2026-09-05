from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.assets_pipeline import robot_descriptions_root
from bello_mujoco.assets_pipeline.prepare import PreparedAsset
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    foot_terrain_contact_membership,
)
from bello_mujoco.evaluation.render import (
    mujoco_foot_contact_and_normal_force,
    mujoco_undesired_contact,
)
from bello_mujoco.robot import (
    FootSoleGeometry,
    robot_definition,
    validate_robot_source_urdf,
)
from bello_mujoco.sim import load_prepared_bello_model
from bello_mujoco.terrain import TerrainConfig


SOLE_GEOM_NAMES = {
    "bello": (
        "left_ankle_roll_link_collision_box_1",
        "right_ankle_roll_link_collision_box_1",
    ),
    "bello_mini": (
        "left_ankle_roll_link_sole_heel_center",
        "left_ankle_roll_link_sole_main",
        "left_ankle_roll_link_sole_heel_negative_y",
        "left_ankle_roll_link_sole_heel_positive_y",
        "left_ankle_roll_link_sole_toe",
        "right_ankle_roll_link_sole_heel_center",
        "right_ankle_roll_link_sole_main",
        "right_ankle_roll_link_sole_heel_negative_y",
        "right_ankle_roll_link_sole_heel_positive_y",
        "right_ankle_roll_link_sole_toe",
    ),
}


@pytest.mark.parametrize(
    ("fixture_name", "asset", "housing_geom_names"),
    [
        (
            "bello_asset",
            "bello",
            (
                "left_ankle_roll_link_collision_box_2",
                "left_ankle_roll_link_collision_box_3",
                "left_ankle_roll_link_collision_cylinder_1",
                "left_ankle_roll_link_collision_cylinder_2",
                "right_ankle_roll_link_collision_box_2",
                "right_ankle_roll_link_collision_box_3",
                "right_ankle_roll_link_collision_cylinder_1",
                "right_ankle_roll_link_collision_cylinder_2",
            ),
        ),
        (
            "bello_mini_asset",
            "bello_mini",
            (
                "left_ankle_roll_link_ankle_housing",
                "right_ankle_roll_link_ankle_housing",
            ),
        ),
    ],
)
def test_model_resolves_only_configured_sole_geometries(
    request: pytest.FixtureRequest,
    fixture_name: str,
    asset: str,
    housing_geom_names: tuple[str, ...],
) -> None:
    prepared: PreparedAsset = request.getfixturevalue(fixture_name)
    scene = load_prepared_bello_model(prepared, load_mjx=False)
    model = scene.mj_model
    resolved_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id))
        for geom_id in scene.foot_geom_ids
    )
    nonfoot_geom_ids = set(int(geom_id) for geom_id in scene.nonfoot_robot_geom_ids)

    assert resolved_names == SOLE_GEOM_NAMES[asset]
    for name in housing_geom_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert geom_id not in scene.foot_geom_ids
        assert geom_id in nonfoot_geom_ids


def test_robot_source_validation_rejects_missing_sole_geometry() -> None:
    robot = robot_definition("bello_mini")
    invalid = replace(
        robot,
        foot_sole_geometries=(
            FootSoleGeometry(
                body_name=robot.foot_body_names[0],
                geom_names=("missing_left_sole",),
            ),
            robot.foot_sole_geometries[1],
        ),
    )

    with pytest.raises(ValueError, match="missing_left_sole"):
        validate_robot_source_urdf(
            invalid,
            robot.source_urdf_path(robot_descriptions_root()),
        )


def test_foot_terrain_membership_rejects_housing_and_nonterrain_contacts() -> None:
    # Geoms: terrain, left sole, left housing, right sole, another robot body.
    geom_body_ids = jnp.asarray([0, 10, 10, 20, 30], dtype=jnp.int32)
    contact_geom = jnp.asarray(
        [
            [0, 2],
            [0, 1],
            [3, 0],
            [1, 4],
        ],
        dtype=jnp.int32,
    )

    membership = foot_terrain_contact_membership(
        contact_geom,
        geom_body_ids,
        foot_body_ids=jnp.asarray([10, 20], dtype=jnp.int32),
        foot_geom_ids=jnp.asarray([1, 3], dtype=jnp.int32),
        terrain_geom_ids=jnp.asarray([0], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(
        np.asarray(membership),
        np.asarray(
            [
                [False, False],
                [True, False],
                [False, True],
                [False, False],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("fixture_name", "geom_name", "is_sole"),
    [
        ("bello_asset", "left_ankle_roll_link_collision_box_1", True),
        ("bello_asset", "left_ankle_roll_link_collision_box_2", False),
        ("bello_mini_asset", "left_ankle_roll_link_sole_main", True),
        ("bello_mini_asset", "left_ankle_roll_link_ankle_housing", False),
    ],
)
def test_native_contact_classification_uses_sole_geometry(
    request: pytest.FixtureRequest,
    fixture_name: str,
    geom_name: str,
    is_sole: bool,
) -> None:
    prepared: PreparedAsset = request.getfixturevalue(fixture_name)
    env = BelloLocomotionEnv.load_prepared(
        prepared,
        env_config=BelloEnvConfig(terrain=TerrainConfig(mode="plane")),
        load_mjx=False,
    )
    model = env.model.mj_model
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    terrain_geom_id = int(env.model.terrain_geom_ids[0])

    model.geom_contype[env.model.robot_geom_ids] = 0
    model.geom_conaffinity[env.model.robot_geom_ids] = 0
    model.geom_contype[geom_id] = 1
    model.geom_conaffinity[geom_id] = 1
    data = mujoco.MjData(model)
    data.qpos[:] = env.model.default_qpos
    data.qvel[:] = env.model.default_qvel
    mujoco.mj_forward(model, data)
    data.qpos[env.model.root_qpos_adr + 2] -= data.geom_xpos[geom_id, 2]
    mujoco.mj_forward(model, data)

    assert any(
        terrain_geom_id in (int(contact.geom1), int(contact.geom2))
        and geom_id in (int(contact.geom1), int(contact.geom2))
        for contact in data.contact[: data.ncon]
    )
    foot_contact, foot_force = mujoco_foot_contact_and_normal_force(env, data)
    undesired_contact = mujoco_undesired_contact(env, data)

    if is_sole:
        np.testing.assert_array_equal(foot_contact, [1.0, 0.0])
        assert foot_force[0] > env.config.contact.foot_contact_force_threshold
        assert undesired_contact == pytest.approx(0.0)
    else:
        np.testing.assert_array_equal(foot_contact, [0.0, 0.0])
        np.testing.assert_array_equal(foot_force, [0.0, 0.0])
        assert undesired_contact == pytest.approx(1.0)
