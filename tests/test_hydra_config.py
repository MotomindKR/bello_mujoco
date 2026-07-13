from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.tools.benchmark_env import BenchmarkEnvConfig
from bello_mujoco.tools.eval_policy import EvalToolConfig
from bello_mujoco.tools.train_ppo import TrainingRunConfig
from bello_mujoco.train.ppo import validate_ppo_config


def compose_tool_config(group: str, config_name: str = "default", overrides=None):
    clear_hydra()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / group),
    ):
        return compose(config_name=config_name, overrides=overrides or [])


def compose_experiment_config(config_name: str):
    clear_hydra()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT.parent / "experiments" / "train"),
    ):
        return compose(
            config_name=config_name,
            overrides=[f"hydra.searchpath=[file://{CONFIG_ROOT / 'train'}]"],
        )


@pytest.mark.parametrize(
    "config_name",
    ["default", "rtx5090_4096x256"],
)
def test_train_hydra_configs_convert_to_valid_training_config(
    config_name: str,
) -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_tool_config("train", config_name),
    )

    validate_ppo_config(config.config)
    assert config.config.batch_size == config.config.num_envs * config.config.num_steps
    assert (
        config.config.minibatch_size * config.config.num_minibatches
        == config.config.batch_size
    )
    assert config.config.num_updates > 0


@pytest.mark.parametrize(
    "config_name",
    ["standing", "foot_distance", "locomotion", "locomotion_easy", "locomotion_hard"],
)
def test_experiment_train_configs_convert_to_valid_training_config(
    config_name: str,
) -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config(config_name),
    )

    validate_ppo_config(config.config)
    assert config.config.num_updates > 0


def test_hydra_overrides_apply_before_dataclass_conversion() -> None:
    config = structured_config(
        BenchmarkEnvConfig,
        compose_tool_config(
            "benchmark_env",
            overrides=["batch_size=1024", "env_config.terrain.mode=rough"],
        ),
    )

    assert config.batch_size == 1024
    assert config.env_config.terrain.mode == "rough"


def test_structured_config_rejects_unknown_keys() -> None:
    config = OmegaConf.create({"batch_size": 512, "step_count": 5, "unknown": True})

    with pytest.raises(ValueError, match="unknown config key.*unknown"):
        structured_config(BenchmarkEnvConfig, config)


def test_structured_config_rejects_nested_unknown_keys() -> None:
    config = compose_tool_config(
        "benchmark_env",
        overrides=["+env_config.observation.not_a_real_field=true"],
    )

    with pytest.raises(
        ValueError,
        match="unknown config key.*env_config\\.observation.*not_a_real_field",
    ):
        structured_config(BenchmarkEnvConfig, config)


def test_eval_config_accepts_checkpoint_override() -> None:
    config = structured_config(
        EvalToolConfig,
        compose_tool_config(
            "eval_policy",
            overrides=["checkpoint_path=checkpoints/ppo_seed_0"],
        ),
    )

    assert config.checkpoint_path is not None
    assert config.checkpoint_path.as_posix() == "checkpoints/ppo_seed_0"


def test_eval_config_accepts_gamepad_overrides() -> None:
    config = structured_config(
        EvalToolConfig,
        compose_tool_config(
            "eval_policy",
            overrides=[
                "checkpoint_path=checkpoints/ppo_seed_0",
                "eval_config.gamepad.enabled=false",
                "eval_config.gamepad.deadzone=0.2",
            ],
        ),
    )

    assert not config.eval_config.gamepad.enabled
    assert config.eval_config.gamepad.deadzone == pytest.approx(0.2)
