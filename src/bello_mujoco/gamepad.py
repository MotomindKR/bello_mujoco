"""Optional gamepad input shared by evaluation and hardware deployment."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os

import numpy as np


@dataclass(frozen=True)
class GamepadConfig:
    enabled: bool = True
    controller_index: int = 0
    deadzone: float = 0.1


@dataclass(frozen=True)
class GamepadInput:
    command: np.ndarray
    connected: bool
    reset_requested: bool = False
    exit_requested: bool = False
    exit_held: bool = False
    connection_changed: bool = False
    controller_name: str | None = None

    @property
    def exit_active(self) -> bool:
        return self.exit_requested or self.exit_held


def apply_deadzone(value: float, deadzone: float) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    magnitude = abs(value)
    if magnitude <= deadzone:
        return 0.0
    return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def gamepad_axes_to_command_axes(
    left_x: float,
    left_y: float,
    right_x: float,
    deadzone: float,
    forward_axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map semantic SDL motion axes into the selected robot's command frame."""

    return forward_lateral_to_command_axes(
        apply_deadzone(-left_y, deadzone),
        apply_deadzone(left_x, deadzone),
        apply_deadzone(-right_x, deadzone),
        forward_axis,
    )


def forward_lateral_to_command_axes(
    forward: float,
    lateral: float,
    yaw: float,
    forward_axis: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Project semantic forward/right intent into a horizontal command frame."""

    axis = np.asarray(forward_axis, dtype=np.float64)
    if (
        axis.shape != (3,)
        or not np.all(np.isfinite(axis))
        or not np.isclose(axis[2], 0.0, rtol=0.0, atol=1.0e-6)
        or not np.isclose(np.linalg.norm(axis), 1.0, rtol=0.0, atol=1.0e-6)
    ):
        raise ValueError("forward_axis must be a normalized horizontal vector")
    right_axis = np.asarray((axis[1], -axis[0]), dtype=np.float64)
    planar = float(forward) * axis[:2] + float(lateral) * right_axis
    return (float(planar[0]), float(planar[1]), float(yaw))


def axis_to_range(
    value: float,
    bounds: tuple[float, float],
    minimum_moving_speed: float,
) -> float:
    value = float(np.clip(value, -1.0, 1.0))
    if value == 0.0:
        return 0.0

    low, high = bounds
    if value > 0.0:
        if high < minimum_moving_speed:
            return 0.0
        start = max(0.0, low, minimum_moving_speed)
        return start + value * (high - start)

    if low > -minimum_moving_speed:
        return 0.0
    start = min(0.0, high, -minimum_moving_speed)
    return start + -value * (low - start)


class GamepadCommandSource:
    def __init__(
        self,
        config: GamepadConfig,
        command_ranges: tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ],
        minimum_moving_speeds: tuple[float, float, float],
        forward_axis: tuple[float, float, float],
        *,
        controller_module=None,
        event_module=None,
        constants_module=None,
        pygame_module=None,
    ) -> None:
        display_initialized = False
        if controller_module is None or event_module is None:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            import pygame
            from pygame._sdl2 import controller

            if config.enabled:
                try:
                    pygame.display.init()
                    display_initialized = True
                except BaseException:
                    pygame.display.quit()
                    raise
            controller_module = controller
            event_module = pygame.event
            constants_module = pygame
            pygame_module = pygame
        elif constants_module is None:
            constants_module = controller_module
        self.config = config
        self.command_ranges = command_ranges
        self.minimum_moving_speeds = minimum_moving_speeds
        self.forward_axis = forward_axis
        self.controller_module = controller_module
        self.event_module = event_module
        self.constants = constants_module
        self.pygame = pygame_module
        self.controller = None
        self._was_connected = False
        self._reset_pressed = False
        self._exit_pressed = False
        self._closed = False
        if config.enabled:
            try:
                self.controller_module.init()
            except BaseException:
                try:
                    self.controller_module.quit()
                except BaseException:
                    pass
                if display_initialized:
                    try:
                        self.pygame.display.quit()
                    except BaseException:
                        pass
                raise

    def poll(self) -> GamepadInput:
        if not self.config.enabled:
            return GamepadInput(np.zeros(3, dtype=np.float32), connected=False)
        try:
            self.event_module.pump()
            self._refresh_controller()
            connected = self.controller is not None and self.controller.attached()
        except Exception:
            self._close_controller()
            connected = False
        connection_changed = connected != self._was_connected
        self._was_connected = connected
        if not connected:
            self._reset_pressed = False
            self._exit_pressed = False
            return GamepadInput(
                np.zeros(3, dtype=np.float32),
                connected=False,
                connection_changed=connection_changed,
            )

        try:
            module = self.constants
            left_x = self._axis(module.CONTROLLER_AXIS_LEFTX)
            left_y = self._axis(module.CONTROLLER_AXIS_LEFTY)
            right_x = self._axis(module.CONTROLLER_AXIS_RIGHTX)
            normalized = gamepad_axes_to_command_axes(
                left_x,
                left_y,
                right_x,
                self.config.deadzone,
                self.forward_axis,
            )
            command = np.asarray(
                [
                    axis_to_range(value, bounds, minimum_moving_speed)
                    for value, bounds, minimum_moving_speed in zip(
                        normalized,
                        self.command_ranges,
                        self.minimum_moving_speeds,
                        strict=True,
                    )
                ],
                dtype=np.float32,
            )
            reset_pressed = bool(self.controller.get_button(module.CONTROLLER_BUTTON_A))
            exit_pressed = bool(
                self.controller.get_button(module.CONTROLLER_BUTTON_START)
            )
            controller_name = self.controller.as_joystick().get_name()
        except Exception:
            self._close_controller()
            self._was_connected = False
            self._reset_pressed = False
            self._exit_pressed = False
            return GamepadInput(
                np.zeros(3, dtype=np.float32),
                connected=False,
                connection_changed=True,
            )
        result = GamepadInput(
            command,
            connected=True,
            reset_requested=reset_pressed and not self._reset_pressed,
            exit_requested=exit_pressed and not self._exit_pressed,
            exit_held=exit_pressed,
            connection_changed=connection_changed,
            controller_name=controller_name,
        )
        self._reset_pressed = reset_pressed
        self._exit_pressed = exit_pressed
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._close_controller()
            if self.config.enabled:
                self.controller_module.quit()
        finally:
            if self.pygame is not None and self.config.enabled:
                self.pygame.display.quit()

    def _axis(self, axis: int) -> float:
        raw = int(self.controller.get_axis(axis))
        divisor = 32767.0 if raw >= 0 else 32768.0
        return float(np.clip(raw / divisor, -1.0, 1.0))

    def _refresh_controller(self) -> None:
        if self.controller is not None:
            try:
                if self.controller.attached():
                    return
            except Exception:
                pass
        self._close_controller()
        index = self.config.controller_index
        if (
            index < self.controller_module.get_count()
            and self.controller_module.is_controller(index)
        ):
            try:
                self.controller = self.controller_module.Controller(index)
            except Exception:
                self.controller = None

    def _close_controller(self) -> None:
        if self.controller is None:
            return
        try:
            self.controller.quit()
        except Exception:
            pass
        self.controller = None
