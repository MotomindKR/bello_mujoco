import os
import sys
import time

import pytest

from bello_mujoco.evaluation import software


def _environment():
    return os.getpid(), {
        key: os.environ.get(key) for key in software.SOFTWARE_ENVIRONMENT
    }


def _fail():
    raise ValueError("renderer failed")


def _crash():
    os._exit(7)


def test_software_rendering_rejects_non_linux(monkeypatch):
    monkeypatch.setattr(software.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="Linux/Lima"):
        software.run_software_evaluation(_environment)


def test_worker_uses_fresh_process_and_restores_parent_environment(monkeypatch):
    # Exercise process isolation without loading a graphics library on the host.
    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    before = {key: os.environ.get(key) for key in software.SOFTWARE_ENVIRONMENT}
    first_pid, first_env = software.run_software_evaluation(_environment)
    second_pid, second_env = software.run_software_evaluation(_environment)
    assert len({first_pid, second_pid, os.getpid()}) == 3
    assert first_env == second_env == software.SOFTWARE_ENVIRONMENT
    assert {key: os.environ.get(key) for key in before} == before


@pytest.mark.parametrize("function", [_fail, _crash])
def test_worker_errors_propagate(monkeypatch, function):
    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    with pytest.raises(RuntimeError, match="failed|exited"):
        software.run_software_evaluation(function)


def test_worker_timeout_cleans_up(monkeypatch):
    import multiprocessing as mp

    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    before = {child.pid for child in mp.active_children()}
    with pytest.raises(TimeoutError):
        software.run_software_evaluation(time.sleep, 30, timeout_seconds=0.1)
    assert {child.pid for child in mp.active_children()} == before


def test_configuration_rejects_late_backend_selection(monkeypatch):
    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    monkeypatch.setitem(sys.modules, "mujoco", object())
    with pytest.raises(RuntimeError, match="before importing"):
        software.configure_software_rendering()


def test_renderer_rejects_non_egl_context(monkeypatch):
    from types import SimpleNamespace
    import mujoco

    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    monkeypatch.setattr(mujoco, "GLContext", SimpleNamespace(__module__="mujoco.glfw"))
    with pytest.raises(RuntimeError, match="EGL backend"):
        software.validate_software_renderer()


@pytest.mark.parametrize("renderer", [b"llvmpipe (LLVM)", b"NVIDIA GPU", None])
def test_renderer_checks_actual_device_and_frees_context(monkeypatch, renderer):
    from types import SimpleNamespace

    calls = []

    class Context:
        __module__ = "mujoco.egl"

        def __init__(self, width, height):
            assert (width, height) == (1, 1)

        def make_current(self):
            calls.append("current")

        def free(self):
            calls.append("free")

    monkeypatch.setattr(software, "require_linux_software_rendering", lambda: None)
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace(GLContext=Context))
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(devices=lambda: [SimpleNamespace(platform="cpu")]),
    )
    monkeypatch.setitem(
        sys.modules,
        "OpenGL",
        SimpleNamespace(
            GL=SimpleNamespace(GL_RENDERER=1, glGetString=lambda _: renderer)
        ),
    )
    for key, value in software.SOFTWARE_ENVIRONMENT.items():
        monkeypatch.setenv(key, value)
    if renderer == b"llvmpipe (LLVM)":
        software.validate_software_renderer()
    else:
        with pytest.raises(RuntimeError, match="CPU llvmpipe"):
            software.validate_software_renderer()
    assert calls == ["current", "free"]
