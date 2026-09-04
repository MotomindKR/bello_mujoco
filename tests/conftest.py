from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
import shutil
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)


class FakeGroupState(Enum):
    IDLE = 1


class FakeDeviceState(Enum):
    ONLINE = 1


@dataclass
class HardwarePreflightHarness:
    binding: Any
    control_requests: list[str]


@pytest.fixture
def hardware_preflight_factory():
    def build(
        coordinates: tuple[str, str],
        positions: tuple[float, float],
    ) -> HardwarePreflightHarness:
        control_requests = []

        def group(name: str, coordinate: str):
            def control(*_args, **_kwargs):
                control_requests.append(name)
                raise AssertionError("read-only preflight acquired control")

            return SimpleNamespace(
                name=name,
                coordinates=(coordinate,),
                control_rate_hz=100.0,
                initial_state=FakeGroupState.IDLE,
                motors=(
                    SimpleNamespace(
                        name=coordinate,
                        status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                    ),
                ),
                control=control,
            )

        groups = {
            "lower_body": group("lower_body", coordinates[0]),
            "upper_body": group("upper_body", coordinates[1]),
        }
        position_by_name = dict(zip(coordinates, positions, strict=True))

        def vector(*values):
            return SimpleNamespace(x=values[0], y=values[1], z=values[2])

        state_frame = SimpleNamespace(
            received_monotonic_ns=time.monotonic_ns(),
            round_trip_ns=0,
            sample_span_ns=0,
            motors=tuple(
                SimpleNamespace(
                    coordinate=name,
                    position=position_by_name[name],
                    velocity=0.0,
                    sample_age_ns=0,
                )
                for name in coordinates
            ),
            imus=(
                SimpleNamespace(
                    imu="imu",
                    sample_age_ns=0,
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
                    angular_velocity=vector(0.0, 0.0, 0.0),
                    linear_acceleration=vector(0.0, 0.0, 0.0),
                ),
            ),
            groups=tuple(
                SimpleNamespace(
                    group=name,
                    sample_age_ns=0,
                    state=SimpleNamespace(name="IDLE"),
                    cause=SimpleNamespace(name="UNSPECIFIED"),
                    detail="",
                )
                for name in groups
            ),
        )

        def wait_for_frame(**_kwargs):
            state_frame.received_monotonic_ns = time.monotonic_ns()
            return state_frame

        state_reader = SimpleNamespace(
            read=wait_for_frame,
        )
        robot = SimpleNamespace(
            group_names=tuple(groups),
            group=lambda name: groups[name],
            inventory=SimpleNamespace(
                imus=(
                    SimpleNamespace(
                        name="imu",
                        status=SimpleNamespace(state=FakeDeviceState.ONLINE),
                    ),
                )
            ),
            states=lambda **_kwargs: state_reader,
        )
        binding = SimpleNamespace(
            Robot=SimpleNamespace(
                connect=lambda *_args, **_kwargs: nullcontext(robot),
                states=lambda **_kwargs: None,
            ),
            GroupState=FakeGroupState,
            DeviceState=FakeDeviceState,
        )
        return HardwarePreflightHarness(binding, control_requests)

    return build


@pytest.fixture(scope="session")
def bello_asset() -> Iterator[PreparedAsset]:
    output_root = Path(tempfile.mkdtemp(prefix="bello-mujoco-test-assets-"))
    try:
        yield prepare_asset(AssetPrepConfig(output_root=output_root))
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


@pytest.fixture(scope="session")
def bello_mini_asset() -> Iterator[PreparedAsset]:
    output_root = Path(tempfile.mkdtemp(prefix="bello-mini-test-assets-"))
    try:
        yield prepare_asset(
            AssetPrepConfig(asset="bello_mini", output_root=output_root)
        )
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
