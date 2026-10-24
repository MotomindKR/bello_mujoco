from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bello_mujoco.configuration import supermimic_training_profile
from bello_mujoco.evaluation import supermimic as evaluation
from bello_mujoco.supermimic.checkpoint import resume_contract
from bello_mujoco.training.vast import VastTrainConfig, validate_config


def test_parallel_evaluation_preserves_all_clips_and_equal_clip_weighting(
    tmp_path, monkeypatch
):
    launched = []

    def run(command, *, input, text, check):
        assert command[-1] == "bello_mujoco.evaluation.supermimic_worker"
        request = json.loads(input)
        launched.append(request)
        output = Path(request["output"])
        output.mkdir(parents=True)
        rows = {}
        for name in request["selected"]:
            value = int(name[1:]) / 10
            rows[name] = dict.fromkeys(
                (
                    "episode_return",
                    "normalized_return",
                    "completion_fraction",
                    "root_position_rms_m",
                    "root_orientation_rms_deg",
                    "joint_position_rms_rad",
                    "motor_peak_effort_ratio",
                    "limited_step_fraction",
                ),
                value,
            )
            rows[name]["terminal"] = False
        (output / "summary.json").write_text(json.dumps({"clips": rows}))
        for name in request["videos"]:
            (output / f"{name}.mp4").write_bytes(b"video")

    monkeypatch.setattr(evaluation.subprocess, "run", run)
    report = evaluation.evaluate_supermimic_corpus(
        "checkpoint",
        "corpus",
        tuple(f"c{i}" for i in range(5)),
        tmp_path / "results",
        video_clips=("c3",),
        workers=2,
    )
    assert sorted(len(request["selected"]) for request in launched) == [2, 3]
    assert report["summary"]["clip_count"] == 5
    assert report["summary"]["mean_normalized_return"] == pytest.approx(0.2)
    assert {p.name for p in (tmp_path / "results").iterdir()} == {
        "summary.json",
        "c3.mp4",
    }


def test_bounded_corpus_profile_and_operational_resume_budget():
    run = supermimic_training_profile(
        Path("corpus"), name="bello_mini_supermimic_amass"
    )
    assert run.max_training_seconds == 4500
    assert run.config.total_timesteps > run.config.warmup_steps * run.config.num_envs
    assert resume_contract(run) == resume_contract(replace(run, max_training_seconds=1))
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="max_training_seconds"):
            replace(run, max_training_seconds=invalid)


def test_remote_video_subset_and_worker_validation():
    config = VastTrainConfig(
        experiment="bello_mini_supermimic_amass",
        branch="feat/supermimic",
        motion_dataset_artifact="entity/project/corpus:v0",
        max_price=0.75,
        evaluation_clips=("held_out",),
        evaluation_video_clips=(),
        evaluation_workers=8,
    )
    validate_config(config)
    for bad in (
        replace(config, evaluation_workers=0),
        replace(config, evaluation_video_clips=("unknown",)),
        replace(config, evaluation_video_clips=("held_out", "held_out")),
    ):
        with pytest.raises(ValueError):
            validate_config(bad)
    with pytest.raises(ValueError, match="require SuperMimic"):
        validate_config(replace(config, experiment="bello_locomotion"))


def test_explicit_training_evaluation_does_not_relabel_training_as_held_out(tmp_path):
    (tmp_path / "sonic-corpus.json").write_text(
        json.dumps(
            {
                "clips": [
                    {"file": "train.npz", "split": "train"},
                    {"file": "test.npz", "split": "evaluation"},
                ]
            }
        )
    )
    config = SimpleNamespace(
        dataset=SimpleNamespace(directory=tmp_path, evaluation_clips=("test",))
    )
    evaluation.validate_evaluation_split(config, ("train",), "training")
    evaluation.validate_evaluation_split(config, ("test",), "held_out")
    for names, split in (
        (("train",), "held_out"),
        (("test",), "training"),
        (("unknown",), "training"),
    ):
        with pytest.raises(ValueError, match="recorded"):
            evaluation.validate_evaluation_split(config, names, split)
    report = evaluation._evaluation_report(
        {
            "train": {
                **dict.fromkeys(
                    (
                        "episode_return",
                        "normalized_return",
                        "completion_fraction",
                        "root_position_rms_m",
                        "root_orientation_rms_deg",
                        "joint_position_rms_rad",
                        "motor_peak_effort_ratio",
                        "limited_step_fraction",
                    ),
                    0.5,
                ),
                "terminal": False,
            }
        },
        split="training",
    )
    assert report["summary"]["split"] == "training"
    assert not report["summary"]["measures_held_out_generalization"]


def test_lafan_profile_uses_bounded_collection_without_shortening_references():
    run = supermimic_training_profile(
        Path("corpus"), name="bello_mini_supermimic_lafan"
    )
    assert run.config.max_episode_steps == 500
    assert run.networks.matmul_precision == "tensorfloat32"
    assert run.config.learning_rate_decay_updates == 250000
