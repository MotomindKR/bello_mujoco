from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from hydra import compose, initialize_config_dir

import bello_mujoco.tools.train_motion_ppo as train_motion
from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.mesh import JointAvailabilityConfig
from bello_mujoco.assets_pipeline.mjcf import AnkleDifferentialConfig
from bello_mujoco.checkpoint import CheckpointConfig
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.motion import motion_tracking_asset_config
from bello_mujoco.tools.train_motion_ppo import (
    MotionEvalVideoConfig,
    MotionSelectionConfig,
    MotionTrainingRunConfig,
    load_motion_resume,
    motion_checkpoint_metadata,
    motion_train_update_loop,
    render_motion_training_video,
    resolve_motion_selection,
    save_motion_checkpoint,
    should_save_motion_checkpoint,
    try_render_motion_checkpoint_video,
    upload_motion_checkpoint,
    validate_motion_checkpoint_config,
    validate_motion_resume,
    validate_motion_video_config,
)
from bello_mujoco.train.ppo import PPOConfig


ACTION_LAYOUT = ActionLayout(tuple(f"joint_{index}" for index in range(25)))
ACTOR_LAYOUT = ObservationLayout(
    (("reference_joint_position", 25), ("last_action", 25))
)
CRITIC_LAYOUT = ObservationLayout(
    ACTOR_LAYOUT.components + (("tracked_body_position_error", 6),)
)


def fake_motion_env(motion_path: Path):
    return SimpleNamespace(
        action_size=ACTION_LAYOUT.size,
        action_layout=ACTION_LAYOUT,
        observation_layout=ACTOR_LAYOUT,
        critic_observation_layout=CRITIC_LAYOUT,
        motion=SimpleNamespace(
            fps=50.0,
            num_frames=3,
            joint_names=ACTION_LAYOUT.names,
            body_names=("pelvis_link", "torso_link"),
            source_sha256="1" * 64,
            qpos=jnp.zeros((3, 32), dtype=jnp.float32),
            qvel=jnp.zeros((3, 31), dtype=jnp.float32),
        ),
    )


def motion_metadata_fixture(tmp_path: Path):
    motion_path = tmp_path / "motion.npz"
    motion_path.write_bytes(b"canonical motion bytes")
    env = fake_motion_env(motion_path)
    run_config = MotionTrainingRunConfig(
        config=PPOConfig(
            total_timesteps=16,
            num_envs=2,
            num_steps=2,
            num_minibatches=1,
        ),
        env_config=replace(
            MotionTrainingRunConfig().env_config,
            motion_path=motion_path,
        ),
    )
    policy_obs_size = ACTOR_LAYOUT.size * run_config.config.policy.obs_stack_size
    critic_policy_obs_size = (
        CRITIC_LAYOUT.size * run_config.config.policy.obs_stack_size
    )
    metadata = motion_checkpoint_metadata(
        run_config,
        env,
        policy_obs_size,
        critic_policy_obs_size,
    )
    return (
        motion_path,
        env,
        run_config,
        policy_obs_size,
        critic_policy_obs_size,
        metadata,
    )


def load_recipe(name: str) -> MotionTrainingRunConfig:
    clear_hydra()
    config_dir = CONFIG_ROOT.parent / "experiments" / "train_motion"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(
            config_name=name,
            overrides=[f"hydra.searchpath=[file://{CONFIG_ROOT / 'train_motion'}]"],
        )
    return structured_config(MotionTrainingRunConfig, config)


@pytest.mark.parametrize(
    (
        "recipe",
        "sequence",
        "motion_directory",
        "formulation",
        "num_envs",
        "distribution",
        "checkpoint_name",
    ),
    [
        (
            "walk_overfit",
            "walk1_subject1",
            Path("motions/local/lafan1_fixed_head"),
            "reference_residual",
            1024,
            "tanh_gaussian",
            "bello_walk_overfit_seed0",
        ),
        (
            "beyondmimic",
            "walk1_subject1",
            Path("motions/local/lafan1_fixed_head"),
            "beyondmimic",
            4096,
            "gaussian",
            "walk1_subject1_bello_beyondmimic_seed0",
        ),
        (
            "asap_single_foot_balance",
            "video_single_foot_balance_level1_filter_amass",
            Path("motions/asap_fixed_head"),
            "beyondmimic",
            4096,
            "gaussian",
            "asap_single_foot_balance_level1_bello_seed0",
        ),
        (
            "asap_single_foot_balance_level4",
            "video_single_foot_balance_level4_filter_amass",
            Path("motions/asap_fixed_head"),
            "beyondmimic",
            3584,
            "gaussian",
            "asap_single_foot_balance_level4_bello_seed0",
        ),
    ],
)
def test_all_motion_recipes_resolve(
    recipe: str,
    sequence: str,
    motion_directory: Path,
    formulation: str,
    num_envs: int,
    distribution: str,
    checkpoint_name: str,
    tmp_path: Path,
) -> None:
    recipe_config = load_recipe(recipe)
    (tmp_path / f"{sequence}.npz").touch()
    config = resolve_motion_selection(
        replace(
            recipe_config,
            motion=replace(recipe_config.motion, directory=tmp_path),
        )
    )

    assert config.motion.sequence == sequence
    assert recipe_config.motion.directory == motion_directory
    assert config.env_config.motion_path == config.motion.directory / f"{sequence}.npz"
    assert config.env_config.motion_path.is_file()
    assert config.env_config.formulation == formulation
    assert config.config.num_envs == num_envs
    assert config.config.policy.action_distribution == distribution
    assert config.checkpoint_config.name == checkpoint_name
    assert config.asset_config == motion_tracking_asset_config()


def test_hardware_motion_recipe_uses_bounded_60kg_contract() -> None:
    config = load_recipe("output_beyondmimic_hardware")

    assert config.motion.directory == Path("motions/custom")
    assert config.motion.sequence == "output_bello_beyondmimic"
    assert config.asset_config.target_total_mass_kg == pytest.approx(60.0)
    assert config.asset_config.enforce_policy_effort_limits
    assert (
        replace(config.asset_config, target_total_mass_kg=None)
        == motion_tracking_asset_config()
    )
    assert config.env_config.formulation == "beyondmimic_joint_limits"
    assert config.env_config.action_clip == pytest.approx(1.0)
    assert config.config.policy.action_distribution == "tanh_gaussian"
    assert config.config.policy.action_limit == pytest.approx(1.0)
    assert config.config.policy.init_log_std == pytest.approx(-1.5)
    assert config.config.policy.max_log_std == pytest.approx(-1.0)
    assert (
        config.checkpoint_config.name == "output_bello_beyondmimic_hardware_60kg_seed0"
    )


def test_motion_selection_resolves_sequence_name(tmp_path: Path) -> None:
    motion_path = tmp_path / "dance1_subject2.npz"
    motion_path.touch()
    run_config = MotionTrainingRunConfig(
        motion=MotionSelectionConfig(directory=tmp_path, sequence="dance1_subject2")
    )

    resolved = resolve_motion_selection(run_config)

    assert resolved.env_config.motion_path == motion_path


@pytest.mark.parametrize("sequence", ["", "dance.npz", "../dance", "a/b"])
def test_motion_selection_rejects_paths_and_suffixes(
    tmp_path: Path, sequence: str
) -> None:
    run_config = MotionTrainingRunConfig(
        motion=MotionSelectionConfig(directory=tmp_path, sequence=sequence)
    )
    with pytest.raises(ValueError, match="motion.sequence"):
        resolve_motion_selection(run_config)


def test_motion_selection_reports_available_sequences(tmp_path: Path) -> None:
    (tmp_path / "walk1_subject1.npz").touch()
    run_config = MotionTrainingRunConfig(
        motion=MotionSelectionConfig(directory=tmp_path, sequence="missing")
    )
    with pytest.raises(FileNotFoundError, match="walk1_subject1"):
        resolve_motion_selection(run_config)


def test_motion_checkpoint_metadata_names_task_layouts_and_clip_contract(
    tmp_path: Path,
) -> None:
    (
        motion_path,
        env,
        run_config,
        policy_obs_size,
        critic_policy_obs_size,
        metadata,
    ) = motion_metadata_fixture(tmp_path)

    assert metadata.task == "motion_tracking"
    assert metadata.reward is None
    assert metadata.command is None
    assert metadata.done is None
    assert metadata.env is None
    assert metadata.asset == motion_tracking_asset_config()
    assert metadata.asset.output_root is None
    assert metadata.obs_size == policy_obs_size
    assert metadata.critic_obs_size == critic_policy_obs_size
    assert metadata.action_size == env.action_size
    assert metadata.actor_observation_layout == ACTOR_LAYOUT
    assert metadata.critic_observation_layout == CRITIC_LAYOUT
    assert metadata.action_layout == ACTION_LAYOUT
    assert metadata.motion_env == run_config.env_config
    assert metadata.motion is not None
    assert metadata.motion.path == motion_path
    assert metadata.motion.content_sha256 == (
        "sha256:" + hashlib.sha256(motion_path.read_bytes()).hexdigest()
    )
    assert metadata.motion.source_sha256 == "1" * 64
    assert metadata.motion.joint_names == ACTION_LAYOUT.names
    assert metadata.motion.body_names == ("pelvis_link", "torso_link")
    assert (metadata.motion.qpos_size, metadata.motion.qvel_size) == (32, 31)


def test_motion_resume_validation_accepts_only_total_timestep_extension(
    tmp_path: Path,
) -> None:
    (
        _,
        env,
        run_config,
        policy_obs_size,
        critic_policy_obs_size,
        metadata,
    ) = motion_metadata_fixture(tmp_path)
    extended_config = replace(
        run_config,
        config=replace(
            run_config.config,
            total_timesteps=run_config.config.total_timesteps * 2,
        ),
    )

    validate_motion_resume(
        metadata,
        extended_config,
        env,
        metadata,
        policy_obs_size,
        critic_policy_obs_size,
    )

    with pytest.raises(ValueError, match="not a motion-tracking checkpoint"):
        validate_motion_resume(
            replace(metadata, task="locomotion", motion_env=None, motion=None),
            extended_config,
            env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )
    invalid_env = SimpleNamespace(**{**vars(env), "action_size": env.action_size + 1})
    with pytest.raises(ValueError, match="policy contract"):
        validate_motion_resume(
            metadata,
            extended_config,
            invalid_env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )


def test_motion_resume_validation_rejects_tampered_motion(tmp_path: Path) -> None:
    (
        motion_path,
        env,
        run_config,
        policy_obs_size,
        critic_policy_obs_size,
        metadata,
    ) = motion_metadata_fixture(tmp_path)
    motion_path.write_bytes(b"tampered motion bytes")

    with pytest.raises(ValueError, match="content_sha256"):
        validate_motion_resume(
            metadata,
            run_config,
            env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )


def test_load_motion_resume_uses_exact_orbax_runner_restore(
    monkeypatch, tmp_path: Path
) -> None:
    (
        _,
        env,
        run_config,
        policy_obs_size,
        critic_policy_obs_size,
        metadata,
    ) = motion_metadata_fixture(tmp_path)
    checkpoint_path = tmp_path / "checkpoint"
    restored = (object(), object(), object())
    validation_calls = []
    monkeypatch.setattr(train_motion, "load_checkpoint_metadata", lambda _: metadata)
    monkeypatch.setattr(
        train_motion,
        "validate_motion_resume",
        lambda *args: validation_calls.append(args),
    )
    monkeypatch.setattr(
        train_motion,
        "restore_runner_checkpoint",
        lambda path, restored_env: (*restored, metadata),
    )

    result = load_motion_resume(
        checkpoint_path,
        run_config,
        env,
        metadata,
        policy_obs_size,
        critic_policy_obs_size,
    )

    assert result == restored
    assert validation_calls == [
        (
            metadata,
            run_config,
            env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )
    ]
    assert (
        load_motion_resume(
            None,
            run_config,
            env,
            metadata,
            policy_obs_size,
            critic_policy_obs_size,
        )
        is None
    )


def test_save_motion_checkpoint_forwards_exact_runner_state(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    snapshot_calls = []
    monkeypatch.setattr(
        train_motion,
        "save_training_checkpoint",
        lambda *args: calls.append(args),
    )
    path = tmp_path / "checkpoint"
    runner_state = object()
    metadata = object()
    resolved_metadata = object()
    fingerprints = object()
    monkeypatch.setattr(
        train_motion,
        "motion_metadata_with_runner_state",
        lambda original, runner: (
            snapshot_calls.append((original, runner)) or resolved_metadata
        ),
    )

    save_motion_checkpoint(path, runner_state, metadata, fingerprints)

    assert snapshot_calls == [(metadata, runner_state)]
    assert calls == [(path, runner_state, resolved_metadata, fingerprints)]


def test_fixed_tracking_asset_validation_is_strict() -> None:
    canonical = motion_tracking_asset_config()
    train_motion._validate_fixed_tracking_asset(canonical)

    invalid = (
        (replace(canonical, collision_mode="mesh"), "collision_mode"),
        (
            replace(canonical, joint_availability=JointAvailabilityConfig()),
            "fixed waist",
        ),
        (replace(canonical, end_effector_sphere_mode="none"), "sphere"),
        (replace(canonical, actuator_config=ActuatorConfig()), "all-policy"),
        (
            replace(
                canonical,
                ankle_differential_config=AnkleDifferentialConfig(
                    motor_gain_scale=0.75
                ),
            ),
            "differential-ankle",
        ),
    )
    for config, message in invalid:
        with pytest.raises(ValueError, match=message):
            train_motion._validate_fixed_tracking_asset(config)


def test_motion_video_config_is_noop_when_disabled() -> None:
    validate_motion_video_config(
        MotionEvalVideoConfig(enabled=False, every_updates=0, steps=0)
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (MotionEvalVideoConfig(every_updates=0), "every_updates"),
        (MotionEvalVideoConfig(steps=0), "steps"),
        (MotionEvalVideoConfig(width=0), "width and height"),
        (MotionEvalVideoConfig(fps=0), "fps"),
        (MotionEvalVideoConfig(filename_prefix=""), "filename_prefix"),
        (MotionEvalVideoConfig(wandb_name=""), "wandb_name"),
    ],
)
def test_motion_video_config_rejects_invalid_values(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_motion_video_config(config)


def test_motion_training_video_passes_normalization_and_logs(
    monkeypatch, tmp_path: Path
) -> None:
    calls = {}
    runner_state = SimpleNamespace(
        update=jnp.asarray(7), observation_norm_state=object()
    )
    monkeypatch.setattr(
        train_motion,
        "runner_train_state",
        lambda state, static_policy: SimpleNamespace(policy="policy"),
    )

    def fake_render(env, policy, output_path, **kwargs):
        calls["render"] = (env, policy, output_path, kwargs)
        return {
            "rendered_steps": 17,
            "episode_return": 4.25,
            "terminal": False,
            "timeout": True,
            "completion_fraction": 1.0,
            "root_position_rms_m": 0.04,
            "root_orientation_rms_deg": 3.0,
            "joint_position_rms_rad": 0.12,
        }

    monkeypatch.setattr(train_motion, "render_motion_policy", fake_render)
    monkeypatch.setattr(
        train_motion,
        "track_video",
        lambda *args, **kwargs: calls.setdefault("video", (args, kwargs)),
    )
    monkeypatch.setattr(
        train_motion,
        "track_metric_snapshot",
        lambda *args, **kwargs: calls.setdefault("metrics", (args, kwargs)),
    )
    config = MotionEvalVideoConfig(
        output_dir=tmp_path,
        filename_prefix="wholebody",
        steps=20,
        width=320,
        height=240,
        fps=25,
        wandb_name="eval/bello",
    )
    ppo_config = PPOConfig()

    output_path = render_motion_training_video(
        "env",
        runner_state,
        "static",
        config,
        "run",
        env_steps=2048,
        obs_stack_size=4,
        ppo_config=ppo_config,
    )

    assert output_path == tmp_path / "wholebody_000007.mp4"
    assert calls["render"] == (
        "env",
        "policy",
        output_path,
        {
            "steps": 20,
            "width": 320,
            "height": 240,
            "fps": 25,
            "obs_stack_size": 4,
            "ppo_config": ppo_config,
            "observation_norm_state": runner_state.observation_norm_state,
        },
    )
    video_args, video_kwargs = calls["video"]
    assert video_args[:2] == ("run", output_path)
    assert video_kwargs["name"] == "eval/bello"
    assert video_kwargs["step"] == 2048
    metrics_args, _ = calls["metrics"]
    assert metrics_args[1]["completion_fraction"] == 1.0
    assert metrics_args[1]["joint_position_rms_rad"] == pytest.approx(0.12)
    assert "rendered_steps=17" in video_kwargs["caption"]
    assert "timeout=1" in video_kwargs["caption"]


def test_motion_update_loop_coordinates_checkpoint_and_video_schedules(
    monkeypatch, tmp_path: Path
) -> None:
    runner_state = SimpleNamespace(update=jnp.asarray(0, dtype=jnp.int32))
    saves = []
    uploads = []
    renders = []
    monkeypatch.setattr(
        train_motion,
        "update_runner_state",
        lambda state, *args: (
            SimpleNamespace(update=state.update + 1),
            {"loss": jnp.asarray(1.0)},
        ),
    )
    monkeypatch.setattr(
        train_motion, "training_update_range", lambda start, stop: range(start, stop)
    )
    monkeypatch.setattr(
        train_motion, "raise_on_invalid_training_metrics", lambda *a: None
    )
    monkeypatch.setattr(train_motion, "track_metric_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(
        train_motion,
        "save_motion_checkpoint",
        lambda path, state, *args: saves.append((path, int(state.update))),
    )
    monkeypatch.setattr(
        train_motion,
        "upload_motion_checkpoint",
        lambda run, path, root, config: uploads.append(path),
    )
    monkeypatch.setattr(
        train_motion,
        "try_render_motion_checkpoint_video",
        lambda worker, path, update, *args, **kwargs: renders.append(
            (worker, path, update, kwargs["env_steps"])
        ),
    )
    config = PPOConfig(
        total_timesteps=3,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
    )
    checkpoint_config = CheckpointConfig(enabled=True, every_updates=2)
    checkpoint_root = tmp_path / "checkpoints"
    video_config = MotionEvalVideoConfig(every_updates=2, render_final=True)

    output = motion_train_update_loop(
        "env",
        "worker",
        config,
        jnp.asarray([0, 1], dtype=jnp.uint32),
        video_config,
        "run",
        False,
        checkpoint_root=checkpoint_root,
        checkpoint_context=(object(), object(), checkpoint_config),
        initial_runner=(runner_state, "policy", "optimizer"),
    )

    assert int(output["runner_state"].update) == 3
    assert saves == [
        (checkpoint_root / "update_000002", 2),
        (checkpoint_root / "eval_000003", 3),
    ]
    assert uploads == [checkpoint_root / "update_000002"]
    assert renders == [
        ("worker", checkpoint_root / "update_000002", 2, 2),
        ("worker", checkpoint_root / "eval_000003", 3, 3),
    ]


def test_motion_checkpoint_schedule_is_periodic_or_disabled() -> None:
    enabled = CheckpointConfig(enabled=True, every_updates=100)
    assert not should_save_motion_checkpoint(0, enabled)
    assert not should_save_motion_checkpoint(99, enabled)
    assert should_save_motion_checkpoint(100, enabled)
    assert not should_save_motion_checkpoint(101, enabled)
    assert not should_save_motion_checkpoint(
        100, CheckpointConfig(enabled=False, every_updates=100)
    )
    assert not should_save_motion_checkpoint(
        100, CheckpointConfig(enabled=True, every_updates=None)
    )


def test_motion_checkpoint_upload_uses_versioned_wandb_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        train_motion,
        "track_checkpoint_artifact",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    checkpoint = tmp_path / "update_000500"
    config = CheckpointConfig(
        upload_wandb_artifact=True,
        wandb_artifact_name="balance-run",
        wandb_artifact_type="model",
    )

    upload_motion_checkpoint("run", checkpoint, tmp_path, config)

    assert calls == [
        (
            ("run", checkpoint),
            {"name": "balance-run", "artifact_type": "model"},
        )
    ]


def test_motion_checkpoint_config_rejects_invalid_interval_or_resume(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="every_updates"):
        validate_motion_checkpoint_config(
            CheckpointConfig(enabled=True, every_updates=0)
        )
    with pytest.raises(FileNotFoundError):
        validate_motion_checkpoint_config(
            CheckpointConfig(resume_path=tmp_path / "missing")
        )


def test_checkpoint_video_failures_are_isolated(monkeypatch, tmp_path: Path) -> None:
    config = MotionEvalVideoConfig()
    checkpoint = tmp_path / "checkpoint"

    with pytest.warns(RuntimeWarning, match="worker is unavailable at update 12"):
        assert (
            try_render_motion_checkpoint_video(
                None,
                checkpoint,
                12,
                config,
                "run",
                env_steps=100,
            )
            is None
        )

    monkeypatch.setattr(
        train_motion,
        "render_motion_checkpoint_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("worker crash")),
    )
    with pytest.warns(RuntimeWarning, match="update 13: worker crash"):
        assert (
            try_render_motion_checkpoint_video(
                "worker",
                checkpoint,
                13,
                config,
                "run",
                env_steps=100,
            )
            is None
        )
