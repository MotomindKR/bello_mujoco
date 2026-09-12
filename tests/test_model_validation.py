import mujoco
import pytest

from bello_mujoco.model_validation import (
    validate_home_pose,
    validate_robot_collision_geometry,
)


def model_at(position):
    return mujoco.MjModel.from_xml_string(f"""<mujoco>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="first" pos="0 0 .05"><freejoint/>
          <geom name="first_shape" type="sphere" size=".1"/>
        </body>
        <body name="second" pos="1 0 .05"><freejoint/>
          <geom name="second_shape" type="sphere" size=".1"/>
        </body>
      </worldbody>
      <keyframe><key name="home" qpos="0 0 .05 1 0 0 0 {position} 0 .05 1 0 0 0"/></keyframe>
    </mujoco>""")


def test_runtime_pose_validation_reports_colliding_geometry():
    with pytest.raises(ValueError, match="first_shape / second_shape"):
        validate_home_pose(model_at(0.15))


def test_runtime_pose_validation_ignores_floor_contact():
    validate_home_pose(model_at(0.3))


def test_runtime_pose_validation_requires_explicit_nominal_pose():
    with pytest.raises(ValueError, match="home pose"):
        validate_home_pose(mujoco.MjModel.from_xml_string("<mujoco/>"))


@pytest.mark.parametrize("collision", [False, True])
def test_runtime_rejects_robot_collision_meshes_but_allows_visuals(collision):
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <asset><mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
      <worldbody><body name="robot"><freejoint/>
        <geom name="shape" type="mesh" mesh="tetrahedron"
          contype="{int(collision)}" conaffinity="{int(collision)}"/>
      </body></worldbody>
    </mujoco>""")
    if collision:
        with pytest.raises(ValueError, match="canonical primitives.*shape"):
            validate_robot_collision_geometry(model)
    else:
        validate_robot_collision_geometry(model)
