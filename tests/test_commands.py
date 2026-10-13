from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from bello_mujoco.cli import (
    HandsCommandConfig,
    TrainCommandConfig,
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


def test_reference_command_wires_direct_body_transport_without_model(
    monkeypatch,
) -> None:
    from bello_mujoco import cli
    from bello_mujoco.hardware import reference_producer

    captured = []
    monkeypatch.setattr(reference_producer, "run", captured.append)
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_ROOT / "commands")
    ):
        config = compose(
            config_name="reference",
            overrides=(
                "xrobot.endpoint=synthetic:1234",
                "retarget.human_height_m=1.75",
            ),
        )
    cli._reference_hydra.__wrapped__(config)
    assert captured[0].xrobot.endpoint == "synthetic:1234"
    assert captured[0].retarget.human_height_m == 1.75
    legacy = OmegaConf.to_container(config, resolve=True)
    legacy["smplx"] = {"neutral_model_path": "/unused/model.npz"}
    with pytest.raises((TypeError, ValueError), match="smplx"):
        command_config(reference_producer.ProducerConfig, OmegaConf.create(legacy))


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


def test_export_cli_logging_preserves_clean_checkout(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("outputs/\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from bello_mujoco import cli, deployment; "
            "deployment.export_checkpoint = lambda config: 'export-dispatched'; "
            "cli.export_entrypoint()",
            "checkpoint_path=checkpoints/sample",
            "output_path=outputs/policy.zip",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "export-dispatched" in result.stdout
    assert (
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=tmp_path,
            text=True,
        )
        == ""
    )


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


def test_hands_command_wires_xrobot_mapping_and_l6_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bello_mujoco.cli as cli
    from bello_mujoco.hardware import hand_control, hand_server, xrobot_hands

    captured: dict[str, object] = {}

    class FakeTransport:
        def __init__(self, config: object, *, require_initial_feedback: bool) -> None:
            captured["can"] = config
            assert not require_initial_feedback

    def fake_run(
        config: object, server_config: object, transport_factory: object
    ) -> None:
        captured["xrobot"] = config
        captured["server"] = server_config
        captured["transport"] = transport_factory(config)  # type: ignore[operator]

    monkeypatch.setattr(hand_control, "LinkerHandsCanTransport", FakeTransport)
    monkeypatch.setattr(hand_server, "run_hand_server", fake_run)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / "commands"),
    ):
        config = compose(
            config_name="hands",
            overrides=(
                "protocol=linker_l6",
                "listen=127.0.0.1:50157",
                "lease_timeout_seconds=2.0",
                "xrobot.endpoint=127.0.0.1:50052",
                "right_hand_enabled=false",
                "arm=true",
                "calibration_confirmed=true",
                "duration_seconds=2.5",
                "smoothing_time_constant_ms=80.0",
                "max_raw_rate_per_second=300.0",
                "left_open_positions=[250,240,230,220,210,200]",
                "left_closed_positions=[10,20,30,40,50,60]",
            ),
        )

    cli._hands_hydra.__wrapped__(config)

    assert captured["server"] == hand_server.HandServerConfig(
        listen="127.0.0.1:50157", lease_timeout_seconds=2.0
    )

    can = captured["can"]
    assert isinstance(can, hand_control.LinkerHandsCanConfig)
    assert can.model == "L6"
    assert can.left_interface == "can2"
    assert can.right_interface == "can3"
    assert can.left_enabled and not can.right_enabled
    runtime = captured["xrobot"]
    assert isinstance(runtime, xrobot_hands.XRobotHandsConfig)
    assert runtime.max_duration_seconds == pytest.approx(2.5)
    assert runtime.left_enabled and not runtime.right_enabled
    assert runtime.arm and runtime.calibration_confirmed
    assert runtime.smoothing_time_constant_ms == pytest.approx(80.0)
    assert runtime.max_raw_rate_per_second == pytest.approx(300.0)
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


@pytest.mark.parametrize("name", ["motion", "bello_mini_motion"])
def test_training_motion_selection_uses_requested_path(tmp_path: Path, name) -> None:
    path = tmp_path / "turn.npz"
    profile = motion_training_profile(path, name=name)

    assert profile.motion.directory == tmp_path
    assert profile.motion.sequence == "turn"


def test_motion_overrides_are_applied_before_resolving_names(monkeypatch, tmp_path):
    from bello_mujoco import configuration

    monkeypatch.setattr(
        configuration,
        "_merged_profile",
        lambda _: OmegaConf.create(
            {
                "asset_config": {"asset": "bello"},
                "motion": {"directory": "unused", "sequence": "unused"},
                "checkpoint_config": {"name": "${motion.sequence}_checkpoint"},
            }
        ),
    )
    profile = motion_training_profile(tmp_path / "selected.npz")
    assert profile.checkpoint_config.name == "selected_checkpoint"


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
