from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from bello_mujoco.cli import (
    EmulatorCommandConfig,
    EvalCommandConfig,
    HandsCommandConfig,
    HardwareCommandConfig,
    TrainCommandConfig,
    VastTrainCommandConfig,
    _hardware_policy_log_path,
    _reject_explicit_fields,
    _warp_overrides,
)
from bello_mujoco.configuration import (
    CONFIG_ROOT,
    command_config,
    coerce_value,
    locomotion_training_profile,
    motion_training_profile,
    structured_config,
)

COMMAND_SCHEMAS = {
    "train": TrainCommandConfig,
    "eval": EvalCommandConfig,
    "vast_train": VastTrainCommandConfig,
    "emulator": EmulatorCommandConfig,
    "hardware": HardwareCommandConfig,
    "hands": HandsCommandConfig,
}


def test_importing_command_surface_does_not_import_jax() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import bello_mujoco.cli; "
                "assert 'jax' not in sys.modules; "
                "assert 'bello_mujoco.evaluation.locomotion' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_training_entrypoint_uses_memory_sharing_jax_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    monkeypatch.delenv("TF_GPU_ALLOCATOR", raising=False)
    monkeypatch.setattr(cli, "_train_hydra", lambda: None)

    cli.train_entrypoint()

    assert cli.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert cli.os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"


def test_training_entrypoint_preserves_explicit_allocator_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    monkeypatch.setenv("TF_GPU_ALLOCATOR", "platform")
    monkeypatch.setattr(cli, "_train_hydra", lambda: None)

    cli.train_entrypoint()

    assert cli.os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    assert cli.os.environ["TF_GPU_ALLOCATOR"] == "platform"


def test_hardware_entrypoint_forces_cpu_backend() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys; "
                "os.environ['JAX_PLATFORMS'] = 'gpu'; "
                "os.environ['JAX_PLATFORM_NAME'] = 'gpu'; "
                "sys.argv = ['bello-run-hardware']; "
                "import bello_mujoco.cli as cli; "
                "cli._hardware_hydra = lambda: None; "
                "cli.hardware_entrypoint(); "
                "import jax; "
                "assert os.environ['JAX_PLATFORMS'] == 'cpu'; "
                "assert os.environ['JAX_PLATFORM_NAME'] == 'cpu'; "
                "assert jax.default_backend() == 'cpu'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_hands_entrypoint_forces_cpu_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli

    monkeypatch.setenv("JAX_PLATFORMS", "gpu")
    monkeypatch.setenv("JAX_PLATFORM_NAME", "gpu")
    monkeypatch.setattr(cli, "_hands_hydra", lambda: None)

    cli.hands_entrypoint()

    assert cli.os.environ["JAX_PLATFORMS"] == "cpu"
    assert cli.os.environ["JAX_PLATFORM_NAME"] == "cpu"


def test_hands_command_wires_rtmw_mapping_and_l6_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli
    from bello_mujoco.hardware import hand_control, rtmw_hands

    captured: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, config: object) -> None:
            captured["can"] = config

    def fake_run(config: object, transport_factory: object) -> None:
        captured["rtmw"] = config
        captured["transport"] = transport_factory(config)  # type: ignore[operator]

    monkeypatch.setattr(hand_control, "LinkerHandsCanTransport", FakeTransport)
    monkeypatch.setattr(rtmw_hands, "run_rtmw_hands", fake_run)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hands",
            overrides=(
                "protocol=linker_l6",
                "right_hand_enabled=false",
                "arm=true",
                "calibration_confirmed=true",
                "duration_seconds=2.5",
                "flexion_open_threshold=0.25",
                "flexion_close_threshold=0.75",
                "left_open_positions=[250,240,230,220,210,200]",
                "left_closed_positions=[10,20,30,40,50,60]",
            ),
        )

    cli._hands_hydra.__wrapped__(config)

    can = captured["can"]
    assert isinstance(can, hand_control.LinkerHandsCanConfig)
    assert can.model == "L6"
    assert can.interface == "can1"
    assert can.left_enabled and not can.right_enabled
    runtime = captured["rtmw"]
    assert isinstance(runtime, rtmw_hands.RtmwHandsConfig)
    assert runtime.max_duration_seconds == pytest.approx(2.5)
    assert runtime.left_enabled and not runtime.right_enabled
    assert runtime.arm and runtime.calibration_confirmed
    assert runtime.flexion_open_threshold == pytest.approx(0.25)
    assert runtime.flexion_close_threshold == pytest.approx(0.75)
    assert runtime.left_calibration.open_positions == (
        250,
        240,
        230,
        220,
        210,
        200,
    )
    assert runtime.left_calibration.closed_positions == (10, 20, 30, 40, 50, 60)
    assert runtime.geometry.thumb_adduction_range_degrees == pytest.approx(60.0)


def test_hardware_policy_log_is_disabled_by_null() -> None:
    assert _hardware_policy_log_path(None) is None


def test_hardware_policy_log_expands_explicit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert _hardware_policy_log_path(Path("~/policy.jsonl")) == (
        tmp_path / "policy.jsonl"
    )


@pytest.mark.parametrize(
    "value",
    (
        "can0=vcan0",
        {"can0": "vcan0"},
        1,
    ),
)
def test_tuple_config_rejects_non_sequences(value: object) -> None:
    with pytest.raises(TypeError, match="bus_interfaces must be a list or tuple"):
        coerce_value(tuple[str, ...], value, path="bus_interfaces")


def test_mapping_config_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="joints must be a mapping"):
        coerce_value(dict[str, int], [("left_hip", 1)], path="joints")


def test_irrelevant_explicit_override_is_rejected_even_at_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["bello-eval", "task=motion", "command=[0.0,0.0,0.0]"],
    )

    with pytest.raises(ValueError, match="task=motion does not accept: command"):
        _reject_explicit_fields(("command",), context="task=motion")


def test_contact_budget_command_override_is_explicit_and_positive() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        raw = compose(config_name="train", overrides=["contact_budget_per_env=64"])
    command = command_config(TrainCommandConfig, raw)
    warp = locomotion_training_profile("smoke").warp

    assert _warp_overrides(warp, command).contact_budget_per_env == 64
    with pytest.raises(ValueError, match="must be positive"):
        _warp_overrides(warp, replace(command, contact_budget_per_env=0))


def test_training_motion_selection_uses_requested_path(tmp_path: Path) -> None:
    path = tmp_path / "turn.npz"
    profile = motion_training_profile(path)

    assert profile.motion.directory == tmp_path
    assert profile.motion.sequence == "turn"


def test_public_command_configs_are_strict() -> None:
    for raw, rejected in (
        ({"experiment": "smoke", "typo": True}, "typo"),
        ({"experiment": "invalid"}, "experiment"),
        ({"experiment": None}, "experiment"),
    ):
        try:
            structured_config(TrainCommandConfig, OmegaConf.create(raw))
        except (TypeError, ValueError) as error:
            assert rejected in str(error)
        else:
            raise AssertionError(f"invalid public command config was accepted: {raw}")

    for config_type, raw in (
        (HardwareCommandConfig, {"arm": 1}),
        (HardwareCommandConfig, {"duration_seconds": True}),
        (HardwareCommandConfig, {"remote_staleness_ms": True}),
        (HardwareCommandConfig, {"joystick_command_scale": True}),
        (HardwareCommandConfig, {"joystick_linear_slew_rate_mps2": True}),
        (HardwareCommandConfig, {"joystick_yaw_slew_rate_rps2": True}),
        (HandsCommandConfig, {"arm": 1}),
        (HandsCommandConfig, {"calibration_confirmed": 1}),
        (HandsCommandConfig, {"control_rate_hz": True}),
        (HandsCommandConfig, {"max_reference_age_ms": True}),
    ):
        try:
            structured_config(config_type, OmegaConf.create(raw))
        except TypeError:
            pass
        else:
            raise AssertionError(f"invalid public command type was accepted: {raw}")
