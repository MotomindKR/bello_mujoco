"""Physical coordinate binding, independent of robot-specific YAML values."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bello_mujoco.models import ratings
from bello_mujoco.sim import BelloModel


@pytest.fixture
def binding(tmp_path, monkeypatch):
    model = mujoco.MjModel.from_xml_string("""
      <mujoco><worldbody><body>
        <joint name="pitch"/><geom size=".1"/>
        <body pos="0 0 .3"><joint name="roll" axis="1 0 0"/><geom size=".1"/></body>
      </body></worldbody>
      <tendon><fixed name="motor_sum"><joint joint="pitch" coef="1"/>
        <joint joint="roll" coef="1"/></fixed>
        <fixed name="motor_difference"><joint joint="pitch" coef="-1"/>
        <joint joint="roll" coef="1"/></fixed></tendon>
      <actuator><motor joint="pitch"/><motor tendon="motor_sum"/>
        <motor tendon="motor_difference"/></actuator></mujoco>
    """)
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text("""
motors:
  - {name: held_motor, coordinate: pitch, model: small}
  - {name: motor_a, coordinate: motor_sum, model: large}
  - {name: motor_b, coordinate: motor_difference, model: large}
""")
    resources = tmp_path / "robots"
    resources.mkdir()
    (resources / "motor_torque_ratings.yaml").write_text("small: 4.0\nlarge: 12.0\n")
    monkeypatch.setattr(ratings, "files", lambda _: tmp_path)
    wrapped = SimpleNamespace(
        mj_model=model,
        robot=SimpleNamespace(hardware_description_path=lambda _: hardware),
    )
    wrapped.actuator_coordinate_names = BelloModel.actuator_coordinate_names.fget(
        wrapped
    )
    return wrapped, hardware, resources


def test_joint_and_differential_motor_ratings_include_all_actuators(binding):
    model, _, _ = binding
    np.testing.assert_array_equal(ratings.motor_continuous_torque(model), [4, 12, 12])
    # Tendon forces are physical motor torques; do not rate logical pitch/roll twice.
    data = mujoco.MjData(model.mj_model)
    data.ctrl[:] = [0, 6, -6]
    mujoco.mj_forward(model.mj_model, data)
    np.testing.assert_allclose(data.actuator_force, [0, 6, -6])
    np.testing.assert_allclose(data.qfrc_actuator, [12, 0])


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "gear", "unknown", "nonfinite"]
)
def test_invalid_physical_motor_binding_fails_at_initialization(binding, problem):
    model, hardware, resources = binding
    if problem == "gear":
        model.mj_model.actuator_gear[0, 0] = 2
    elif problem == "missing":
        model.actuator_coordinate_names = model.actuator_coordinate_names[:-1]
    elif problem == "duplicate":
        hardware.write_text(
            hardware.read_text().replace("coordinate: motor_sum", "coordinate: pitch")
        )
    elif problem == "unknown":
        hardware.write_text(
            hardware.read_text().replace("model: small", "model: unknown")
        )
    else:
        (resources / "motor_torque_ratings.yaml").write_text(
            "small: .nan\nlarge: 12.0\n"
        )
    with pytest.raises(ValueError):
        ratings.motor_continuous_torque(model)
