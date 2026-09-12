from types import SimpleNamespace

import mujoco
import pytest

from bello_mujoco.model_validation import (
    validate_home_pose,
    validate_robot_collision_geometry,
)
from bello_mujoco.sim import geom_lower_z, logical_actuator_joint_names


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


@pytest.mark.parametrize("joint_type", ["hinge", "slide"])
def test_runtime_pose_validation_enforces_scalar_joint_limits(joint_type):
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <compiler angle="radian"/>
      <worldbody><body><joint name="limited" type="{joint_type}" range="-1 1"/>
        <geom size=".1"/>
      </body></worldbody>
      <keyframe><key name="home" qpos="1.5"/></keyframe>
    </mujoco>""")
    with pytest.raises(ValueError, match="limited.*outside"):
        validate_home_pose(model)


@pytest.mark.parametrize(
    "shape, size, bottom",
    [
        ("capsule", ".1 .2", 0.7),
        ("ellipsoid", ".1 .2 .3", 0.7),
        ("cylinder", ".1 .2", 0.8),
    ],
)
def test_curved_primitive_lower_bound_uses_its_shape(shape, size, bottom):
    model = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <worldbody><geom type="{shape}" size="{size}" pos="0 0 1"/></worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert geom_lower_z(model, data, 0) == pytest.approx(bottom)


def test_logical_mapping_recognizes_native_joint_and_tendon_transmissions():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <worldbody><body><joint name="direct"/><geom size=".1"/>
        <body pos="0 0 .3"><joint name="coupled"/><geom size=".1"/></body>
      </body></worldbody>
      <tendon><fixed name="differential"><joint joint="coupled" coef="1"/></fixed></tendon>
      <actuator><motor joint="direct"/><motor tendon="differential"/></actuator>
    </mujoco>""")
    robot = SimpleNamespace(
        coupled_actuators=(
            SimpleNamespace(tendon_name="differential", logical_joint_name="coupled"),
        )
    )
    assert logical_actuator_joint_names(model, robot) == ("direct", "coupled")


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
