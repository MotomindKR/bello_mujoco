from types import SimpleNamespace

import numpy as np
import pytest
from rerun.urdf import UrdfTree

from bello_mujoco.envs.locomotion import BelloEnvConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.rerun_eval import (
    GamepadCommandSource,
    GamepadConfig,
    apply_deadzone,
    axis_to_range,
    command_frame_linear_velocity,
    rough_terrain_mesh,
)
from bello_mujoco.terrain import TerrainConfig
from bello_mujoco.train.ppo import PPOConfig
import bello_mujoco.tools.eval_policy as eval_policy
from bello_mujoco.tools.eval_policy import (
    EvalPolicyConfig,
    EvalRuntimeConfig,
    expand_checkpoint_override,
    eval_runtime_configs,
    validate_checkpoint_shape,
    validate_eval_config,
)


def test_checkpoint_override_expands_home_before_hydra(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/Users/tester")

    assert expand_checkpoint_override("checkpoint_path=~/Downloads/run") == (
        "checkpoint_path=/Users/tester/Downloads/run"
    )
    assert expand_checkpoint_override("eval_config.steps=10") == (
        "eval_config.steps=10"
    )


def test_eval_runtime_configs_use_checkpoint_and_merge_overrides() -> None:
    metadata = SimpleNamespace(
        asset=AssetPrepConfig(),
        env=BelloEnvConfig(decimation=5),
        reward="reward",
        command="command",
        done="done",
        warp="warp",
    )
    assert eval_runtime_configs(metadata, EvalRuntimeConfig()) == (
        metadata.asset,
        metadata.env,
        "reward",
        "command",
        "done",
        "warp",
    )
    terrain = TerrainConfig(mode="rough", seed=9)
    config = EvalRuntimeConfig(
        source="override",
        env_config=BelloEnvConfig(terrain=terrain),
    )
    _, env_config, *_ = eval_runtime_configs(metadata, config)
    assert env_config.decimation == 5
    assert env_config.terrain == terrain


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EvalPolicyConfig(steps=0), "steps must be positive"),
        (EvalPolicyConfig(command=(0.0, 0.0)), "command must have three"),
        (EvalPolicyConfig(action_seed=-1), "action_seed must be nonnegative"),
        (
            EvalPolicyConfig(gamepad=GamepadConfig(controller_index=-1)),
            "controller_index must be nonnegative",
        ),
        (
            EvalPolicyConfig(gamepad=GamepadConfig(deadzone=1.0)),
            "deadzone must be in",
        ),
    ],
)
def test_validate_eval_config_rejects_invalid_values(config, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_eval_config(config)


def test_validate_checkpoint_shape_rejects_mismatches() -> None:
    metadata = SimpleNamespace(
        ppo=PPOConfig(policy=ActorCriticConfig(obs_stack_size=2)),
        obs_size=4,
        action_size=2,
    )
    with pytest.raises(ValueError, match="policy observation_size=6"):
        validate_checkpoint_shape(
            SimpleNamespace(observation_size=3, action_size=2), metadata
        )
    with pytest.raises(ValueError, match="action_size=1"):
        validate_checkpoint_shape(
            SimpleNamespace(observation_size=2, action_size=1), metadata
        )


def test_gamepad_axis_helpers_rescale_and_use_asymmetric_bounds() -> None:
    assert apply_deadzone(0.1, 0.1) == 0.0
    assert apply_deadzone(-0.55, 0.1) == pytest.approx(-0.5)
    assert axis_to_range(0.5, (-0.2, 0.8)) == pytest.approx(0.4)
    assert axis_to_range(-0.5, (-0.2, 0.8)) == pytest.approx(-0.1)


def test_command_velocity_uses_reward_yaw_frame() -> None:
    half_sqrt_two = np.sqrt(0.5)
    mj_data = SimpleNamespace(
        qpos=np.asarray([0.0, 0.0, 0.0, half_sqrt_two, 0.0, 0.0, half_sqrt_two]),
        qvel=np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )
    reward = object()
    env = SimpleNamespace(
        reward_config=SimpleNamespace(track_lin_vel_xy=reward),
        model=SimpleNamespace(root_qpos_adr=0, root_qvel_adr=0),
        _reward_link=lambda config: "bello_root",
        _reward_frame_link=lambda config: "bello_root",
    )

    velocity = command_frame_linear_velocity(env, mj_data)

    assert velocity == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


class FakeController:
    def __init__(self, module) -> None:
        self.module = module
        self.axes = {0: 0, 1: 0, 2: 0}
        self.buttons = {3: 0, 4: 0}
        self.is_attached = True

    def attached(self):
        return self.is_attached

    def get_axis(self, axis):
        return self.axes[axis]

    def get_button(self, button):
        return self.buttons[button]

    def as_joystick(self):
        return SimpleNamespace(get_name=lambda: "Fake Pad")

    def quit(self):
        self.is_attached = False


class FakeControllerModule:
    CONTROLLER_AXIS_LEFTX = 0
    CONTROLLER_AXIS_LEFTY = 1
    CONTROLLER_AXIS_RIGHTX = 2
    CONTROLLER_BUTTON_A = 3
    CONTROLLER_BUTTON_START = 4

    def __init__(self) -> None:
        self.available = True
        self.fail_open = False
        self.created = []

    def init(self):
        pass

    def quit(self):
        pass

    def get_count(self):
        return int(self.available)

    def is_controller(self, index):
        return self.available and index == 0

    def Controller(self, index):
        if self.fail_open:
            raise RuntimeError("HIDAPI device disconnected while opening")
        controller = FakeController(self)
        self.created.append(controller)
        return controller


def make_gamepad(module) -> GamepadCommandSource:
    return GamepadCommandSource(
        GamepadConfig(deadzone=0.1),
        ((-0.2, 0.8), (-0.4, 0.6), (-1.0, 0.5)),
        controller_module=module,
        event_module=SimpleNamespace(pump=lambda: None),
    )


def test_gamepad_maps_buttons_disconnects_and_reconnects() -> None:
    module = FakeControllerModule()
    gamepad = make_gamepad(module)
    assert gamepad.poll().connection_changed
    controller = module.created[0]
    controller.axes.update({0: 32767, 1: -32768, 2: 32767})
    controller.buttons[3] = 1
    state = gamepad.poll()
    assert np.allclose(state.command, [0.8, 0.6, -1.0])
    assert state.reset_requested
    assert not gamepad.poll().reset_requested

    module.available = False
    controller.is_attached = False
    disconnected = gamepad.poll()
    assert disconnected.connection_changed
    assert np.allclose(disconnected.command, 0.0)
    module.available = True
    assert gamepad.poll().connected
    assert len(module.created) == 2


def test_gamepad_treats_hid_open_race_as_disconnected() -> None:
    module = FakeControllerModule()
    module.fail_open = True
    gamepad = make_gamepad(module)

    disconnected = gamepad.poll()
    module.fail_open = False
    reconnected = gamepad.poll()

    assert not disconnected.connected
    assert np.allclose(disconnected.command, 0.0)
    assert reconnected.connected


def test_rough_terrain_mesh_matches_config_extent() -> None:
    config = TerrainConfig(
        mode="rough",
        hfield_resolution=3,
        size=(2.0, 3.0, 0.5),
        pos=(1.0, 2.0, 0.25),
    )
    vertices, triangles = rough_terrain_mesh(config)
    assert vertices.shape == (9, 3)
    assert triangles.shape == (8, 3)
    assert vertices[:, 0].min() == pytest.approx(-1.0)
    assert vertices[:, 0].max() == pytest.approx(3.0)


def test_rerun_urdf_prefix_is_a_visible_world_subtree(tmp_path) -> None:
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<robot name="test_robot">
  <link name="base">
    <visual name="base_visual">
      <geometry><box size="1 1 1"/></geometry>
    </visual>
  </link>
</robot>
"""
    )
    tree = UrdfTree.from_file_path(urdf_path, entity_path_prefix="world")

    assert tree.get_visual_geometry_paths("base") == [
        "/world/test_robot/visual_geometries/base/base_visual"
    ]


def test_eval_main_prepares_one_visual_env_and_starts_rerun(monkeypatch) -> None:
    metadata = SimpleNamespace(
        asset=AssetPrepConfig(),
        env="env",
        reward="reward",
        command="command",
        done="done",
        warp="warp",
        ppo=PPOConfig(policy=ActorCriticConfig(obs_stack_size=3)),
        obs_size=12,
        action_size=2,
    )
    prepared = SimpleNamespace(urdf_path="robot.urdf")
    calls = {}
    monkeypatch.setattr(
        eval_policy,
        "restore_training_checkpoint",
        lambda path: (SimpleNamespace(policy="policy"), metadata),
    )
    monkeypatch.setattr(eval_policy, "prepare_asset", lambda config: prepared)

    class FakeEnv:
        observation_size = 4
        action_size = 2

        @classmethod
        def load_prepared(cls, prepared_asset, **kwargs):
            calls["load"] = (prepared_asset, kwargs)
            return cls()

    monkeypatch.setattr(eval_policy, "BelloLocomotionEnv", FakeEnv)
    monkeypatch.setattr(
        eval_policy,
        "run_rerun_evaluation",
        lambda env, policy, **kwargs: calls.setdefault("run", (env, policy, kwargs)),
    )
    eval_policy.main("checkpoint")
    assert calls["load"][1]["use_viewer_model"]
    assert not calls["load"][1]["load_mjx"]
    assert calls["run"][2]["obs_stack_size"] == 3
    assert calls["run"][2]["gamepad_config"] == GamepadConfig()
