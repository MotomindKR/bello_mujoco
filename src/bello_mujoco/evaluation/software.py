"""Isolated Linux software evaluation; safe to import before JAX or MuJoCo."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Any


SOFTWARE_ENVIRONMENT = {
    "JAX_PLATFORMS": "cpu",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
    "LIBGL_ALWAYS_SOFTWARE": "true",
    "GALLIUM_DRIVER": "llvmpipe",
    "LP_NUM_THREADS": "2",
    "__EGL_VENDOR_LIBRARY_FILENAMES": os.environ.get(
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    ),
}


def require_linux_software_rendering() -> None:
    if sys.platform != "linux":
        raise RuntimeError(
            "Evaluation videos require Linux/Lima with Mesa llvmpipe. "
            "Native macOS and GPU rasterization are not supported."
        )


def configure_software_rendering() -> None:
    require_linux_software_rendering()
    if "mujoco" in sys.modules or "OpenGL" in sys.modules or "jax" in sys.modules:
        raise RuntimeError("Configure software evaluation before importing JAX/MuJoCo")
    os.environ.update(SOFTWARE_ENVIRONMENT)


@contextmanager
def _child_environment() -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in SOFTWARE_ENVIRONMENT}
    try:
        os.environ.update(SOFTWARE_ENVIRONMENT)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_main(function: Callable, args: tuple, kwargs: dict, connection) -> None:
    try:
        connection.send((True, function(*args, **kwargs)))
    except BaseException:  # Worker failures must reach the training process.
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def run_software_evaluation(
    function: Callable,
    *args: Any,
    timeout_seconds: float = 300.0,
    **kwargs: Any,
) -> Any:
    """Run a picklable callable with host-only inputs in a fresh CPU process."""
    require_linux_software_rendering()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("evaluation timeout must be finite and positive")
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_main,
        args=(function, args, kwargs, sender),
        name="bello-software-evaluator",
    )
    started = False
    try:
        with _child_environment():
            process.start()
        started = True
        sender.close()
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"software evaluation exceeded {timeout_seconds:g}s")
            if receiver.poll(min(1.0, remaining)):
                try:
                    succeeded, result = receiver.recv()
                except EOFError as error:
                    raise RuntimeError(
                        "software evaluator exited without a result"
                    ) from error
                if not succeeded:
                    raise RuntimeError(f"software evaluation failed:\n{result}")
                return result
            if not process.is_alive():
                raise RuntimeError(
                    f"software evaluator exited with code {process.exitcode}"
                )
    finally:
        sender.close()
        receiver.close()
        if started:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join()
        process.close()


def validate_software_renderer() -> None:
    """Reject incorrectly initialized backends instead of silently using a GPU."""
    require_linux_software_rendering()
    import jax
    import mujoco

    if mujoco.GLContext.__module__ != "mujoco.egl":
        raise RuntimeError("Evaluation requires MuJoCo's EGL backend")
    if any(device.platform != "cpu" for device in jax.devices()):
        raise RuntimeError("Evaluation policy inference requires JAX CPU devices")
    if any(os.environ.get(key) != value for key, value in SOFTWARE_ENVIRONMENT.items()):
        raise RuntimeError(
            "Evaluation requires the software-only rendering environment"
        )
    from OpenGL import GL

    context = mujoco.GLContext(1, 1)
    try:
        context.make_current()
        renderer = GL.glGetString(GL.GL_RENDERER)
        if renderer is None or not renderer.lower().startswith(b"llvmpipe"):
            raise RuntimeError(f"Evaluation requires CPU llvmpipe, got {renderer!r}")
    finally:
        context.free()
