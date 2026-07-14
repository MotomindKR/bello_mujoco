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
    [
        "standing",
        "foot_distance",
        "locomotion",
        "locomotion_easy",
        "locomotion_hard",
        "locomotion_clean",
        "locomotion_gait_base",
        "locomotion_gait",
    ],
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


def test_hard_locomotion_includes_standing_countermeasures() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_hard"),
    )

    assert config.env_config.observation.base_lin_vel
    assert config.command_config.standing_probability == pytest.approx(0.35)


def test_locomotion_tracks_pelvis_velocity() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_hard"),
    )

    assert config.reward_config.track_lin_vel_xy.link == "pelvis_link"
    assert config.reward_config.track_lin_vel_xy.frame_link == "pelvis_link"
    assert config.reward_config.track_ang_vel_z.link == "pelvis_link"
    assert config.reward_config.swing_foot_velocity.enabled
    assert config.reward_config.swing_foot_velocity.link == "pelvis_link"


def test_clean_locomotion_removes_gait_shaping_and_randomization() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_clean"),
    )

    assert config.env_config.root_link == "pelvis_link"
    assert config.env_config.observation.base_lin_vel
    assert not config.env_config.observation.gait_phase
    assert not config.env_config.observation.noise.enabled
    assert not config.env_config.random_push.enabled
    assert not config.command_config.curriculum.enabled
    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.config.learning_rate == pytest.approx(0.0001)
    assert config.config.schedule == "constant"
    assert config.config.policy.init_log_std == pytest.approx(-1.5)
    assert config.config.policy.max_log_std == pytest.approx(-0.5)
    assert config.warp.njmax == 2048
    assert config.env_config.action_clip == pytest.approx(1.0)
    assert config.asset_config.actuator_config.joints[
        "left_hip_pitch_joint"
    ].action_scale == pytest.approx(0.35)
    assert config.asset_config.actuator_config.joints[
        "left_ankle_roll_joint"
    ].action_scale == pytest.approx(0.06)
    assert config.command_config.lin_vel_x == (0.0, 0.0)
    assert config.command_config.lin_vel_y == pytest.approx((0.3, 0.3))
    assert config.command_config.ang_vel_yaw == (0.0, 0.0)
    assert config.command_config.standing_probability == pytest.approx(0.0)
    assert config.reward_config.track_lin_vel_xy.link == "pelvis_link"
    assert config.reward_config.track_lin_vel_xy.frame_link == "pelvis_link"
    assert config.reward_config.track_lin_vel_xy.weight == pytest.approx(2.0)
    assert config.reward_config.track_lin_vel_xy.sigma == pytest.approx(0.05)
    assert not config.reward_config.foot_slip.enabled
    assert not config.reward_config.foot_clearance.enabled
    assert not config.reward_config.swing_foot_velocity.enabled
    assert not config.reward_config.gait_contact.enabled
    assert not config.reward_config.feet_air_time.enabled
    assert config.reward_config.termination.weight == pytest.approx(-5.0)


def test_gait_base_retains_stable_stance_control() -> None:
    base = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_gait_base"),
    )

    assert base.config.learning_rate == pytest.approx(0.00005)
    assert base.config.schedule == "linear"
    assert base.config.target_kl == pytest.approx(0.015)
    assert base.config.max_grad_norm == pytest.approx(0.5)
    assert base.reward_config.action_rate.weight == pytest.approx(-0.05)
    assert base.reward_config.joint_acc.enabled
    assert base.reward_config.joint_acc.weight == pytest.approx(-0.000001)
    assert base.reward_config.feet_air_time.weight == pytest.approx(0.50)
    assert base.reward_config.stance_width.enabled
    assert base.reward_config.stance_width.link == "pelvis_link"
    assert base.reward_config.stance_width.min_width == pytest.approx(0.24)
    assert base.reward_config.stance_width.max_width == pytest.approx(0.34)
    assert not base.reward_config.gait_contact.enabled
    assert not base.reward_config.contact_timing.enabled
    assert not base.reward_config.swing_foot_velocity.enabled
    assert not base.reward_config.foot_slip.enabled
    assert base.env_config.contact.foot_contact_on_seconds == pytest.approx(0.0)


def test_gait_candidate_enables_explicit_anti_chatter_stack() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_gait"),
    )

    assert config.reward_config.stance_width.enabled
    assert config.env_config.contact.foot_contact_on_seconds == pytest.approx(0.04)
    assert config.env_config.contact.foot_contact_off_seconds == pytest.approx(0.04)
    assert config.env_config.gait.cycle_time == pytest.approx(1.0)
    assert config.env_config.observation.gait_phase
    assert config.env_config.critic_observation.gait_phase
    assert config.reward_config.gait_contact.enabled
    assert config.reward_config.gait_contact.weight == pytest.approx(0.35)
    assert config.reward_config.contact_timing.enabled
    assert config.reward_config.contact_timing.weight == pytest.approx(-0.10)
    assert config.reward_config.contact_timing.min_air_time == pytest.approx(0.20)
    assert config.reward_config.contact_timing.min_contact_time == pytest.approx(0.30)
    assert config.reward_config.swing_foot_velocity.enabled
    assert config.reward_config.swing_foot_velocity.weight == pytest.approx(0.25)
    assert config.reward_config.swing_foot_velocity.link == "pelvis_link"
    assert config.reward_config.foot_slip.enabled
    assert config.reward_config.foot_slip.weight == pytest.approx(-0.05)


def test_berkeley_port_matches_reference_recipe() -> None:
    config = structured_config(
        TrainingRunConfig,
        compose_experiment_config("locomotion_berkeley"),
    )

    assert config.config.total_timesteps == 150_000_000
    assert config.config.num_steps == 20
    assert config.config.num_minibatches == 32
    assert config.config.update_epochs == 4
    assert config.config.gamma == pytest.approx(0.97)
    assert config.config.learning_rate == pytest.approx(3e-4)
    assert config.config.target_kl == pytest.approx(0.03)
    assert config.config.policy.activation == "swish"
    assert config.config.policy.action_distribution == "tanh_gaussian"
    assert config.config.policy.normalize_observations
    assert config.env_config.gait.randomize_frequency
    assert config.env_config.gait.frequency_range == pytest.approx((1.25, 1.5))
    assert config.env_config.observation.foot_phase
    assert config.reward_config.touchdown_air_time.enabled
    assert config.reward_config.touchdown_air_time.weight == pytest.approx(2.0)
    assert config.reward_config.foot_phase.enabled
    assert config.reward_config.torque_l1.enabled
    assert config.reward_config.clip_negative_total
    assert config.reward_config.scale_termination_by_timestep


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
