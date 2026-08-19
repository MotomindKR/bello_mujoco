import pytest

from bello_mujoco.gamepad import gamepad_axes_to_command_axes


def test_gamepad_maps_forward_to_positive_y_and_lateral_to_positive_x() -> None:
    assert gamepad_axes_to_command_axes(0.5, -0.75, -0.25, 0.0) == pytest.approx(
        (0.5, 0.75, 0.25)
    )


def test_gamepad_command_axes_apply_the_configured_deadzone() -> None:
    assert gamepad_axes_to_command_axes(0.1, -0.1, 0.1, 0.1) == (0.0, 0.0, 0.0)
