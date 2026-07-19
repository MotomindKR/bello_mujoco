"""Shared timing, pose, observation, and action contracts."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ROOT_POS = (0.0, 0.0, 0.91)
DEFAULT_ROOT_QUAT = (1.0, 0.0, 0.0, 0.0)
DEFAULT_SPAWN_CLEARANCE = 0.01
BELLO_ACTION_CLIP = 1.0
SIM_TIMESTEP = 0.005
CONTROL_TIMESTEP = 0.02
CONTROL_DECIMATION = int(round(CONTROL_TIMESTEP / SIM_TIMESTEP))
EPISODE_SECONDS = 20.0
MAX_EPISODE_STEPS = int(round(EPISODE_SECONDS / CONTROL_TIMESTEP))


@dataclass(frozen=True)
class ObservationLayout:
    """Ordered, checkpointable observation component contract."""

    components: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.components)
        if not names or any(not name for name in names):
            raise ValueError("observation layout requires named components")
        if len(set(names)) != len(names):
            raise ValueError("observation layout component names must be unique")
        if any(type(size) is not int or size <= 0 for _, size in self.components):
            raise ValueError("observation layout component sizes must be positive")

    @property
    def size(self) -> int:
        return sum(size for _, size in self.components)


@dataclass(frozen=True)
class ActionLayout:
    """Ordered logical action contract."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names or any(not name for name in self.names):
            raise ValueError("action layout requires named actions")
        if len(set(self.names)) != len(self.names):
            raise ValueError("action layout names must be unique")

    @property
    def size(self) -> int:
        return len(self.names)
