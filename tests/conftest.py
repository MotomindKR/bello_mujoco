from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pytest

from bello_mujoco.models.compiler import (
    ModelConfig,
    CompiledRobot,
    compile_robot,
)

pytest_plugins = ["native_runtime"]


@pytest.fixture(scope="session")
def bello_asset() -> CompiledRobot:
    return compile_robot(ModelConfig())


@pytest.fixture(scope="session")
def bello_mini_asset() -> CompiledRobot:
    return compile_robot(ModelConfig(asset="bello_mini"))
