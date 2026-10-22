from dataclasses import replace
from pathlib import Path
import ast
import subprocess

from hydra import compose, initialize_config_dir
import pytest

from bello_mujoco import cli
from bello_mujoco.configuration import CONFIG_ROOT
from bello_mujoco.training import vast_remote
from bello_mujoco.training.vast import VastTrainConfig, validate_config


def command(name, overrides):
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_ROOT / "commands")
    ):
        return compose(config_name=name, overrides=overrides)


@pytest.mark.parametrize(
    "profile",
    [
        "bello_mini_supermimic",
        "bello_mini_supermimic_smoke",
        "bello_mini_supermimic_pilot",
        "bello_mini_supermimic_amass",
    ],
)
def test_training_cli_routes_dataset_seed_and_resume_without_pretrained(
    profile, monkeypatch
):
    from bello_mujoco.training import supermimic

    captured = []
    monkeypatch.setattr(supermimic, "run_training", captured.append)
    overrides = [
        f"experiment={profile}",
        "motion_dataset=/tmp/corpus",
        "seed=19",
        "evaluation_clips=[held_out]",
        "resume_path=checkpoints/previous",
        "wandb_enabled=false",
        "checkpoint_directory=checkpoints/test",
    ]
    monkeypatch.setattr(cli.sys, "argv", ["bello-train", *overrides])
    cli._train_hydra.__wrapped__(command("train", overrides))
    (run,) = captured
    assert run.config.seed == run.environment.randomization.seed == 19
    assert run.dataset.directory == Path("/tmp/corpus").resolve()
    assert run.dataset.evaluation_clips == ("held_out",)
    assert run.checkpoint_config.resume_path == Path("checkpoints/previous")
    assert not run.wandb_config.enabled


@pytest.mark.parametrize(
    "invalid",
    ["pretrained_revision=abc", "motion_path=/tmp/clip.npz", "video_enabled=false"],
)
def test_training_cli_rejects_irrelevant_supermimic_flags(invalid, monkeypatch):
    overrides = [
        "experiment=bello_mini_supermimic",
        "motion_dataset=/tmp/corpus",
        invalid,
    ]
    monkeypatch.setattr(cli.sys, "argv", ["bello-train", *overrides])
    with pytest.raises(ValueError, match="does not accept"):
        cli._train_hydra.__wrapped__(command("train", overrides))


def test_evaluation_cli_routes_held_out_clips_and_first_frame_options(monkeypatch):
    from bello_mujoco.evaluation import supermimic

    captured = []
    monkeypatch.setattr(
        supermimic,
        "render_supermimic_checkpoint",
        lambda *a, **k: captured.append((a, k)),
    )
    overrides = [
        "task=supermimic",
        "checkpoint_path=checkpoints/test",
        "motion_dataset=/tmp/corpus",
        "evaluation_clips=[held_out]",
        "output_path=outputs/test",
        "stop_on_done=true",
    ]
    monkeypatch.setattr(cli.sys, "argv", ["bello-eval", *overrides])
    cli._eval_hydra.__wrapped__(command("eval", overrides))
    args, kwargs = captured[0]
    assert args[2] == ("held_out",)
    assert kwargs == {"steps": None, "stop_on_terminal": True}


@pytest.mark.parametrize(
    "profile",
    [
        "bello_mini_supermimic",
        "bello_mini_supermimic_smoke",
        "bello_mini_supermimic_pilot",
        "bello_mini_supermimic_amass",
    ],
)
def test_remote_config_accepts_pinned_dataset_and_rejects_wrong_training_inputs(
    profile,
):
    config = VastTrainConfig(
        experiment=profile,
        motion_dataset_artifact="entity/project/mini:v0",
        max_price=1.0,
        branch="feat/supermimic",
    )
    validate_config(config)
    for bad in (
        replace(config, pretrained_revision="a" * 40),
        replace(config, motion_path=Path("/tmp/clip.npz")),
        replace(config, motion_dataset_artifact="entity/project/mini:latest"),
        replace(config, end_effector_payload_mass_kg=1.0),
    ):
        with pytest.raises(ValueError):
            validate_config(bad)


def test_remote_script_parses_and_restores_improved_actor_before_full_training():
    script = vast_remote.REMOTE_TRAIN_SCRIPT
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    blocks = [part.split("\nPY", 1)[0] for part in script.split("<<'PY'\n")[1:]]
    for block in blocks:
        ast.parse(block)
    preflight = next(
        block for block in blocks if "supermimic_preflight_checkpoint_restored" in block
    )
    assert 'metadata["actor_updates"] < 1' in preflight
    assert "load_actor(" in preflight and "make_native_motion_step(env)" in preflight
    assert "export_checkpoint(ExportConfig(" in preflight
    assert script.index("supermimic_preflight_checkpoint_restored") < script.rindex(
        "uv run --no-sync bello-train"
    )
    evaluation = next(
        block for block in blocks if "supermimic_evaluation_artifact" in block
    )
    assert evaluation.index("configure_software_rendering()") < evaluation.index(
        "import wandb"
    )
    assert "evaluate_supermimic_corpus(" in evaluation
    assert "run.log_artifact(artifact).wait()" in evaluation
    assert script.index("supermimic_evaluation_artifact") > script.rindex(
        "uv run --no-sync bello-train"
    )
