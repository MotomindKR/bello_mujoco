from dataclasses import dataclass, replace
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from rerun.urdf import UrdfTree

from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig, PreparedAsset
from bello_mujoco.checkpoint import CheckpointMetadata
from bello_mujoco.contracts import ActionLayout, ObservationLayout
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    DoneConfig,
)
from bello_mujoco.envs.rewards import CommandConfig, RewardConfig
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.render import (
    NativeDataImplementation,
    NativeDataView,
    mujoco_eval_state,
    mujoco_policy_observation,
)
from bello_mujoco.rerun_eval import (
    GamepadCommandSource,
    GamepadConfig,
    apply_deadzone,
    axis_to_range,
    rough_terrain_mesh,
)
from bello_mujoco.sim import WarpConfig
from bello_mujoco.terrain import TerrainConfig
from bello_mujoco.train.ppo import PPOConfig
import bello_mujoco.tools.eval_policy as eval_policy
from bello_mujoco.tools.eval_policy import (
    EvalPolicyConfig,
    expand_checkpoint_override,
    validate_checkpoint_shape,
    validate_eval_config,
)


@dataclass(frozen=True)
class EvalEnvContract:
    observation_size: int
    critic_observation_size: int
    action_size: int
    observation_layout: ObservationLayout
    critic_observation_layout: ObservationLayout
    action_layout: ActionLayout


def make_checkpoint_metadata(
    *,
    stack_size: int = 2,
    actor_frame_size: int = 3,
    critic_frame_size: int = 4,
    action_size: int = 2,
) -> CheckpointMetadata:
    actor_layout = ObservationLayout((("actor", actor_frame_size),))
    critic_layout = ObservationLayout((("critic", critic_frame_size),))
    action_layout = ActionLayout(
        tuple(f"action_{index}" for index in range(action_size))
    )
    return CheckpointMetadata(
        ppo=PPOConfig(
            policy=ActorCriticConfig(obs_stack_size=stack_size),
        ),
        asset=AssetPrepConfig(),
        reward=RewardConfig(),
        command=CommandConfig(),
        done=DoneConfig(),
        env=BelloEnvConfig(),
        warp=WarpConfig(),
        obs_size=actor_frame_size * stack_size,
        critic_obs_size=critic_frame_size * stack_size,
        action_size=action_size,
        actor_observation_layout=actor_layout,
        critic_observation_layout=critic_layout,
        action_layout=action_layout,
    )


def env_contract(metadata: CheckpointMetadata) -> EvalEnvContract:
    stack_size = metadata.ppo.policy.obs_stack_size
    return EvalEnvContract(
        observation_size=metadata.obs_size // stack_size,
        critic_observation_size=metadata.critic_obs_size // stack_size,
        action_size=metadata.action_size,
        observation_layout=metadata.actor_observation_layout,
        critic_observation_layout=metadata.critic_observation_layout,
        action_layout=metadata.action_layout,
    )


def test_checkpoint_override_expands_home_before_hydra(monkeypatch) -> None:
    monkeypatch.setenv("HOME", "/Users/tester")

    assert expand_checkpoint_override("checkpoint_path=~/Downloads/run") == (
        "checkpoint_path=/Users/tester/Downloads/run"
    )
    assert expand_checkpoint_override("eval_config.steps=10") == (
        "eval_config.steps=10"
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EvalPolicyConfig(steps=0), "steps must be positive"),
        (EvalPolicyConfig(command=(0.0, 0.0)), "command must have three"),
        (EvalPolicyConfig(action_seed=-1), "action_seed must be nonnegative"),
        (
            EvalPolicyConfig(simulation_backend="bad"),
            "simulation_backend must be",
        ),
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


def test_validate_checkpoint_shape_accepts_exact_typed_contract() -> None:
    metadata = make_checkpoint_metadata()

    validate_checkpoint_shape(env_contract(metadata), metadata)


def test_validate_checkpoint_shape_rejects_size_and_layout_mismatches() -> None:
    metadata = make_checkpoint_metadata()
    env = env_contract(metadata)

    with pytest.raises(ValueError, match="policy observation_size=6"):
        validate_checkpoint_shape(env, replace(metadata, obs_size=7))
    with pytest.raises(ValueError, match="critic observation_size=8"):
        validate_checkpoint_shape(env, replace(metadata, critic_obs_size=9))
    with pytest.raises(ValueError, match="action_size=1"):
        validate_checkpoint_shape(replace(env, action_size=1), metadata)
    with pytest.raises(ValueError, match="actor observation layout"):
        validate_checkpoint_shape(
            env,
            replace(
                metadata,
                actor_observation_layout=ObservationLayout((("other", 3),)),
            ),
        )
    with pytest.raises(ValueError, match="critic observation layout"):
        validate_checkpoint_shape(
            env,
            replace(
                metadata,
                critic_observation_layout=ObservationLayout((("other", 4),)),
            ),
        )
    with pytest.raises(ValueError, match="action layout"):
        validate_checkpoint_shape(
            env,
            replace(metadata, action_layout=ActionLayout(("other_0", "other_1"))),
        )


@pytest.mark.parametrize(
    ("backend", "load_mjx"),
    [("mujoco", False), ("mjx", True)],
)
def test_eval_main_uses_checkpoint_runtime_contract(
    monkeypatch, backend: str, load_mjx: bool
) -> None:
    metadata = make_checkpoint_metadata(stack_size=3)
    runtime_env = env_contract(metadata)
    prepared = PreparedAsset(
        name="full_body",
        urdf_path=Path("robot.urdf"),
        mjcf_path=Path("robot.xml"),
        viewer_mjcf_path=Path("robot_viewer.xml"),
        collision_mesh_count=0,
        visual_mesh_count=0,
    )
    policy = object()
    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class RestoredTrainState:
        policy: object

    def fake_prepare_asset(config: AssetPrepConfig) -> PreparedAsset:
        calls["asset"] = config
        return prepared

    class FakeLocomotionEnv:
        @classmethod
        def load_prepared(cls, prepared_asset, **kwargs):
            calls["load"] = (prepared_asset, kwargs)
            return runtime_env

    def fake_run(env, restored_policy, **kwargs) -> None:
        calls["run"] = (env, restored_policy, kwargs)

    monkeypatch.setattr(
        eval_policy,
        "restore_training_checkpoint",
        lambda path: (RestoredTrainState(policy), metadata),
    )
    monkeypatch.setattr(eval_policy, "prepare_asset", fake_prepare_asset)
    monkeypatch.setattr(eval_policy, "BelloLocomotionEnv", FakeLocomotionEnv)
    monkeypatch.setattr(eval_policy, "run_rerun_evaluation", fake_run)

    eval_policy.main(
        Path("checkpoint"),
        EvalPolicyConfig(simulation_backend=backend),
    )

    assert calls["asset"] == metadata.asset
    loaded_asset, load_kwargs = calls["load"]
    assert loaded_asset == prepared
    assert load_kwargs == {
        "warp": metadata.warp,
        "batch_size": 1,
        "env_config": metadata.env,
        "actuator_config": metadata.asset.actuator_config,
        "reward_config": metadata.reward,
        "command_config": metadata.command,
        "done_config": metadata.done,
        "use_viewer_model": True,
        "load_mjx": load_mjx,
    }
    run_env, run_policy, run_kwargs = calls["run"]
    assert run_env is runtime_env
    assert run_policy is policy
    assert run_kwargs["simulation_backend"] == backend
    assert run_kwargs["obs_stack_size"] == 3
    assert run_kwargs["urdf_path"] == prepared.urdf_path


def test_native_eval_adapter_is_typed_and_uses_shared_observation(
    full_body_asset: PreparedAsset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset, load_mjx=False)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = np.linspace(-0.1, 0.1, env.model.mj_model.nv)
    mujoco.mj_forward(env.model.mj_model, mj_data)
    command = np.asarray([0.3, -0.1, 0.2], dtype=np.float32)
    last_action = np.linspace(-0.2, 0.2, env.action_size, dtype=np.float32)
    zeros = jnp.zeros(env.model.foot_body_ids.size, dtype=jnp.float32)

    state = mujoco_eval_state(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        foot_air_time=zeros,
        foot_contact_time=zeros,
        foot_contact_debounce_time=zeros,
        last_foot_contact=zeros,
        step_count=7,
        episode_return=jnp.asarray(0.0),
        episode_track_lin_vel_xy=jnp.asarray(0.0),
        episode_length=jnp.asarray(7),
    )

    assert isinstance(state.data, NativeDataView)
    assert isinstance(state.data._impl, NativeDataImplementation)
    assert state.data.subtree_com.dtype == jnp.float32
    np.testing.assert_allclose(state.data.subtree_com, mj_data.subtree_com)
    np.testing.assert_allclose(
        state.data._impl.actuator_velocity,
        mj_data.actuator_velocity,
    )
    adapted = mujoco_policy_observation(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        step_count=7,
    )
    np.testing.assert_allclose(adapted, env.observation(state), atol=1.0e-6)


def test_gamepad_axis_helpers_rescale_and_use_asymmetric_bounds() -> None:
    assert apply_deadzone(0.1, 0.1) == 0.0
    assert apply_deadzone(-0.55, 0.1) == pytest.approx(-0.5)
    assert axis_to_range(0.5, (-0.2, 0.8)) == pytest.approx(0.4)
    assert axis_to_range(-0.5, (-0.2, 0.8)) == pytest.approx(-0.1)


class FakeJoystick:
    def get_name(self) -> str:
        return "Fake Pad"


class FakeEventModule:
    def pump(self) -> None:
        pass


class FakeController:
    def __init__(self) -> None:
        self.axes = {0: 0, 1: 0, 2: 0}
        self.buttons = {3: 0, 4: 0}
        self.is_attached = True

    def attached(self):
        return self.is_attached

    def get_axis(self, axis):
        return self.axes[axis]

    def get_button(self, button):
        return self.buttons[button]

    def as_joystick(self) -> FakeJoystick:
        return FakeJoystick()

    def quit(self) -> None:
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
        self.created: list[FakeController] = []

    def init(self) -> None:
        pass

    def quit(self) -> None:
        pass

    def get_count(self) -> int:
        return int(self.available)

    def is_controller(self, index: int) -> bool:
        return self.available and index == 0

    def Controller(self, index: int) -> FakeController:
        if self.fail_open:
            raise RuntimeError("HIDAPI device disconnected while opening")
        controller = FakeController()
        self.created.append(controller)
        return controller


def make_gamepad(module: FakeControllerModule) -> GamepadCommandSource:
    return GamepadCommandSource(
        GamepadConfig(deadzone=0.1),
        ((-0.2, 0.8), (-0.4, 0.6), (-1.0, 0.5)),
        controller_module=module,
        event_module=FakeEventModule(),
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
    assert np.ptp(vertices[:, 2]) == pytest.approx(config.hfield_amplitude)


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
