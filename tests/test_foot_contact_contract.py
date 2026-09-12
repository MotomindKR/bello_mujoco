from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.models import robot_descriptions_root
from bello_mujoco.envs.locomotion_config import BelloEnvConfig
from bello_mujoco.envs.geometry import foot_terrain_contact_membership
from bello_mujoco.evaluation.render import (
    mujoco_foot_contact_and_normal_force,
    mujoco_undesired_contact,
)
from bello_mujoco.robot import (
    FootSupportGeometry,
    robot_definition,
    validate_robot_source_urdf,
)


def test_robot_source_validation_rejects_missing_support_geometry() -> None:
    robot = robot_definition("bello_mini")
    invalid = replace(
        robot,
        foot_support_geometries=(
            FootSupportGeometry(
                body_name=robot.foot_body_names[0],
                geom_names=("missing_left_support",),
            ),
            robot.foot_support_geometries[1],
        ),
    )

    with pytest.raises(ValueError, match="missing_left_support"):
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


@pytest.mark.parametrize("is_support", [True, False])
def test_native_contact_classification_uses_robot_support_geometry(
    is_support: bool,
) -> None:
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="left" pos="0 0 .05"><freejoint/>
        <geom name="sole" type="sphere" size=".1"/>
        <geom name="housing" pos=".3 0 0" type="sphere" size=".1"/>
      </body>
      <body name="right" pos="1 0 1"><freejoint/>
        <geom name="right_sole" type="sphere" size=".1"/>
      </body>
    </worldbody></mujoco>""")
    sole, housing, right = (
        model.geom(name).id for name in ("sole", "housing", "right_sole")
    )
    left_body, right_body = model.body("left").id, model.body("right").id
    env = SimpleNamespace(
        config=BelloEnvConfig(),
        model=SimpleNamespace(
            mj_model=model,
            geom_body_ids=model.geom_bodyid,
            foot_body_ids=np.asarray([left_body, right_body]),
            foot_geom_ids=np.asarray([sole, right]),
            terrain_geom_ids=np.asarray([model.geom("floor").id]),
            nonfoot_robot_geom_ids=np.asarray([housing]),
            nonfoot_robot_body_ids=np.asarray([left_body]),
        ),
    )
    geom_id = sole if is_support else housing
    terrain_geom_id = int(env.model.terrain_geom_ids[0])
    model.geom_contype[[sole, housing, right]] = 0
    model.geom_conaffinity[[sole, housing, right]] = 0
    model.geom_contype[geom_id] = 1
    model.geom_conaffinity[geom_id] = 1
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert any(
        terrain_geom_id in (int(contact.geom1), int(contact.geom2))
        and geom_id in (int(contact.geom1), int(contact.geom2))
        for contact in data.contact[: data.ncon]
    )
    foot_contact, foot_force = mujoco_foot_contact_and_normal_force(env, data)
    undesired_contact = mujoco_undesired_contact(env, data)

    if is_support:
        np.testing.assert_array_equal(foot_contact, [1.0, 0.0])
        assert foot_force[0] > env.config.contact.foot_contact_force_threshold
        assert undesired_contact == pytest.approx(0.0)
    else:
        np.testing.assert_array_equal(foot_contact, [0.0, 0.0])
        np.testing.assert_array_equal(foot_force, [0.0, 0.0])
        assert undesired_contact == pytest.approx(1.0)
