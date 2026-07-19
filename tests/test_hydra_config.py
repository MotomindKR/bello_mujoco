from __future__ import annotations

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
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


def test_default_train_hydra_config_is_valid() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_tool_config("train"),
    )

    validate_ppo_config(config.config)
    assert config.config.batch_size == config.config.num_envs * config.config.num_steps
    assert (
        config.config.minibatch_size * config.config.num_minibatches
        == config.config.batch_size
    )
    assert config.config.num_updates > 0


def test_default_training_config_uses_deployment_policy_contract() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_tool_config("train"),
    )

    assert len(config.asset_config.actuator_config.policy_joint_names) == 13
    assert len(config.asset_config.actuator_config.hold_joint_names) == 16
    assert config.env_config.root_link == "pelvis_link"
    assert config.env_config.action_clip == pytest.approx(1.0)
    assert not config.env_config.observation.privileged
    assert config.env_config.critic_observation.privileged
    assert config.config.policy.action_limit == pytest.approx(1.0)
    assert config.config.policy.normalize_observations


@pytest.mark.parametrize(
    "config_name",
    ["smoke", "locomotion", "locomotion_plane_slope"],
)
def test_maintained_experiment_configs_are_valid(config_name: str) -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config(config_name),
    )

    validate_ppo_config(config.config)
    assert config.config.num_updates > 0


def test_smoke_recipe_runs_one_real_update_without_outputs() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("smoke"),
    )

    assert not config.dry_run
    assert config.config.num_updates == 1
    assert config.config.batch_size == 4
    assert not config.checkpoint_config.enabled
    assert not config.wandb_config.enabled
    assert not config.video_config.enabled


def test_training_recipes_share_the_physical_ankle_transmission() -> None:
    baseline = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion"),
    )
    plane_slope = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_plane_slope"),
    )

    assert baseline.asset_config.ankle_differential_config == (
        plane_slope.asset_config.ankle_differential_config
    )
    assert baseline.env_config.terrain.mode == "plane"
    assert baseline.env_config.terrain.randomize_slope_per_world
    assert baseline.env_config.terrain.slope_angle_range == pytest.approx(
        (-0.034906585, 0.034906585)
    )
    assert not baseline.config.normalize_value


def test_hydra_overrides_apply_before_dataclass_conversion() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_tool_config(
            "train",
            overrides=["config.num_envs=128", "env_config.terrain.mode=rough"],
        ),
    )

    assert config.config.num_envs == 128
    assert config.env_config.terrain.mode == "rough"


def test_structured_config_rejects_unknown_keys() -> None:
    config = OmegaConf.create({"unknown": True})

    with pytest.raises(ValueError, match="unknown config key.*unknown"):
        structured_config(TrainingRunConfig, config)


def test_structured_config_rejects_nested_unknown_keys() -> None:
    config = compose_tool_config(
        "train",
        overrides=["+env_config.observation.not_a_real_field=true"],
    )

    with pytest.raises(
        ValueError,
        match="unknown config key.*env_config\\.observation.*not_a_real_field",
    ):
        structured_config(TrainingRunConfig, config)


@pytest.mark.parametrize("field", ["enabled", "actuate_tendons"])
def test_ankle_transmission_cannot_be_disabled(field: str) -> None:
    config = compose_tool_config(
        "train",
        overrides=[f"+asset_config.ankle_differential_config.{field}=false"],
    )

    with pytest.raises(
        ValueError,
        match=(
            f"unknown config key.*asset_config\\.ankle_differential_config.*{field}"
        ),
    ):
        structured_config(TrainingRunConfig, config)


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
