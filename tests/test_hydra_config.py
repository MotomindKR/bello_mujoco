from __future__ import annotations

from hydra import compose, initialize_config_dir

from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.tools.benchmark_env import BenchmarkEnvConfig
from bello_mujoco.tools.eval_policy import EvalToolConfig
from bello_mujoco.tools.train_ppo import TrainingRunConfig


def compose_tool_config(group: str, config_name: str = "default", overrides=None):
    clear_hydra()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_ROOT / group),
    ):
        return compose(config_name=config_name, overrides=overrides or [])


def test_train_hydra_default_converts_to_training_config() -> None:
    config = structured_config(TrainingRunConfig, compose_tool_config("train"))

    assert config.dry_run
    assert config.config.num_envs == 64
    assert config.env_config.terrain.mode == "plane"
    assert config.env_config.spawn.clearance == 0.01
    assert config.env_config.gait.cycle_time == 0.8
    assert config.env_config.gait.duty_factor == 0.55
    assert config.env_config.observation.gait_phase
    assert config.metric_config.stream_updates
    assert config.video_config.simulation_backend == "mujoco"


def test_rtx5090_train_preset_sets_measured_batch_shape() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_tool_config("train", "rtx5090_4096x256"),
    )

    assert not config.dry_run
    assert config.config.num_envs == 4096
    assert config.config.num_steps == 256
    assert config.config.batch_size == 1048576
    assert config.config.minibatch_size == 131072
    assert config.checkpoint_config.enabled


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
