from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.render import (
    RewardOverlayConfig,
    format_reward_overlay,
    sync_reward_overlay,
)
from bello_mujoco.envs.locomotion import BelloEnvConfig
from bello_mujoco.terrain import TerrainConfig
import bello_mujoco.tools.eval_policy as eval_policy
from bello_mujoco.tools.eval_policy import (
    EvalPolicyConfig,
    EvalRuntimeConfig,
    eval_runtime_configs,
    validate_eval_config,
    validate_checkpoint_shape,
)


def test_eval_runtime_configs_default_to_checkpoint_metadata() -> None:
    metadata = SimpleNamespace(
        asset=object(),
        env=object(),
        reward=object(),
        command=object(),
        done=object(),
        warp=object(),
    )

    assert eval_runtime_configs(metadata, EvalRuntimeConfig()) == (
        metadata.asset,
        metadata.env,
        metadata.reward,
        metadata.command,
        metadata.done,
        metadata.warp,
    )


def test_eval_runtime_configs_can_use_override_config() -> None:
    metadata = SimpleNamespace(
        asset=object(),
        env=BelloEnvConfig(decimation=5),
        reward=object(),
        command=object(),
        done=object(),
        warp=object(),
    )
    terrain = TerrainConfig(mode="rough", seed=9)
    config = EvalRuntimeConfig(
        source="override",
        env_config=BelloEnvConfig(terrain=terrain),
    )
    _, env_config, _, _, _, _ = eval_runtime_configs(metadata, config)

    assert env_config.decimation == 5
    assert env_config.terrain == terrain


def test_eval_runtime_override_merges_nested_actuator_dicts() -> None:
    base_joints = ActuatorConfig().by_name
    base_joints["waist_yaw_joint"] = JointActuatorConfig(default_pos=1.0)
    override_joints = ActuatorConfig().by_name
    override_joints["neck_yaw_joint"] = JointActuatorConfig(mode="hold")
    metadata = SimpleNamespace(
        asset=AssetPrepConfig(actuator_config=ActuatorConfig(joints=base_joints)),
        env=object(),
        reward=object(),
        command=object(),
        done=object(),
        warp=object(),
    )
    config = EvalRuntimeConfig(
        source="override",
        asset_config=AssetPrepConfig(
            actuator_config=ActuatorConfig(joints=override_joints)
        ),
    )

    asset_config, *_ = eval_runtime_configs(metadata, config)

    assert asset_config.actuator_config.joints["waist_yaw_joint"].default_pos == 1.0
    assert asset_config.actuator_config.joints["neck_yaw_joint"].mode == "hold"


def test_validate_checkpoint_shape_rejects_observation_mismatch() -> None:
    env = SimpleNamespace(observation_size=3, action_size=2)
    metadata = SimpleNamespace(obs_size=4, action_size=2)

    with pytest.raises(ValueError, match="observation_size=3"):
        validate_checkpoint_shape(env, metadata)


def test_validate_checkpoint_shape_rejects_action_mismatch() -> None:
    env = SimpleNamespace(observation_size=4, action_size=1)
    metadata = SimpleNamespace(obs_size=4, action_size=2)

    with pytest.raises(ValueError, match="action_size=1"):
        validate_checkpoint_shape(env, metadata)


def test_format_reward_overlay_includes_total_and_components() -> None:
    state = SimpleNamespace(
        episode_return=jnp.asarray(1.25),
        episode_length=jnp.asarray(7),
    )
    info = {
        "base_height": jnp.asarray(0.91),
        "reward/alive": jnp.asarray(0.15),
        "reward/energy": jnp.asarray(-0.02),
    }

    labels, values = format_reward_overlay(
        RewardOverlayConfig(precision=2),
        step_count=3,
        reward=jnp.asarray(0.5),
        done=jnp.asarray(False),
        state=state,
        info=info,
    )

    assert labels.splitlines() == [
        "step",
        "reward",
        "done",
        "episode_return",
        "episode_length",
        "base_height",
        "alive",
        "energy",
    ]
    assert values.splitlines() == [
        "3",
        "0.50",
        "False",
        "1.25",
        "7",
        "0.91",
        "0.15",
        "-0.02",
    ]


def test_format_reward_overlay_can_hide_all_components() -> None:
    state = SimpleNamespace(
        episode_return=jnp.asarray(1.25),
        episode_length=jnp.asarray(7),
    )
    labels, _ = format_reward_overlay(
        RewardOverlayConfig(max_reward_terms=0),
        step_count=3,
        reward=jnp.asarray(0.5),
        done=jnp.asarray(False),
        state=state,
        info={"reward/alive": jnp.asarray(0.15)},
    )

    assert "alive" not in labels.splitlines()


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EvalPolicyConfig(steps=0), "steps must be positive"),
        (EvalPolicyConfig(command=(0.0, 0.0)), "command must have three values"),
        (
            EvalPolicyConfig(
                reward_overlay=RewardOverlayConfig(max_reward_terms=-1),
            ),
            "max_reward_terms must be nonnegative",
        ),
    ],
)
def test_validate_eval_config_rejects_invalid_config(
    config: EvalPolicyConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_eval_config(config)


def test_eval_main_wires_checkpoint_env_and_viewer(monkeypatch) -> None:
    metadata = SimpleNamespace(
        asset="asset",
        env="env",
        reward="reward",
        command="command",
        done="done",
        warp="warp",
        obs_size=4,
        action_size=2,
    )
    train_state = SimpleNamespace(policy="policy")
    loaded = {}

    def fake_restore(path):
        loaded["checkpoint_path"] = path
        return train_state, metadata

    class FakeEnv:
        observation_size = 4
        action_size = 2

        @classmethod
        def load(cls, asset, **kwargs):
            loaded["load"] = (asset, kwargs)
            return cls()

    def fake_launch(env, policy, **kwargs):
        loaded["launch"] = (env, policy, kwargs)

    monkeypatch.setattr(eval_policy, "restore_training_checkpoint", fake_restore)
    monkeypatch.setattr(eval_policy, "BelloLocomotionEnv", FakeEnv)
    monkeypatch.setattr(eval_policy, "launch_policy_viewer", fake_launch)

    eval_policy.main("checkpoint")

    assert loaded["checkpoint_path"] == "checkpoint"
    assert loaded["load"] == (
        "asset",
        {
            "warp": "warp",
            "batch_size": 1,
            "env_config": "env",
            "reward_config": "reward",
            "command_config": "command",
            "done_config": "done",
        },
    )
    assert loaded["launch"][1] == "policy"


class FakeViewer:
    def __init__(self) -> None:
        self.texts = None
        self.cleared = False

    def set_texts(self, texts) -> None:
        self.texts = texts

    def clear_texts(self) -> None:
        self.cleared = True


def test_sync_reward_overlay_can_disable_texts() -> None:
    viewer = FakeViewer()

    sync_reward_overlay(
        viewer,
        RewardOverlayConfig(enabled=False),
        step_count=0,
        reward=jnp.asarray(0.0),
        done=jnp.asarray(False),
        state=SimpleNamespace(
            episode_return=jnp.asarray(0.0), episode_length=jnp.asarray(0)
        ),
        info={},
    )

    assert viewer.cleared
