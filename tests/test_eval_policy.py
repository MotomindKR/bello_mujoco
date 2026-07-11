from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.render import (
    ActionOverlayConfig,
    CommandJoystick,
    JoystickConfig,
    RewardOverlayConfig,
    foot_geom_ids_for_bodies,
    format_action_overlay,
    format_reward_overlay,
    mujoco_action_to_ctrl,
    mujoco_foot_contact,
    mujoco_policy_observation,
    rollout_policy_action,
    sync_foot_contact_colors,
    sync_foot_contact_markers,
)
from bello_mujoco.envs.locomotion import (
    BelloEnvConfig,
    BelloLocomotionEnv,
    ObservationConfig,
    ObservationScaleConfig,
)
from bello_mujoco.terrain import TerrainConfig
from bello_mujoco.policy import ActorCriticConfig
from bello_mujoco.train.ppo import PPOConfig
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
    base_joints["waist_yaw_joint"] = replace(
        base_joints["waist_yaw_joint"], default_pos=1.0
    )
    override_joints = ActuatorConfig().by_name
    override_joints["neck_yaw_joint"] = replace(
        override_joints["neck_yaw_joint"], mode="hold"
    )
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
    metadata = SimpleNamespace(
        ppo=PPOConfig(policy=ActorCriticConfig(obs_stack_size=2)),
        obs_size=4,
        action_size=2,
    )

    with pytest.raises(ValueError, match="policy observation_size=6"):
        validate_checkpoint_shape(env, metadata)


def test_validate_checkpoint_shape_rejects_action_mismatch() -> None:
    env = SimpleNamespace(observation_size=4, action_size=1)
    metadata = SimpleNamespace(
        ppo=PPOConfig(policy=ActorCriticConfig(obs_stack_size=1)),
        obs_size=4,
        action_size=2,
    )

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


def test_format_reward_overlay_includes_command_values() -> None:
    state = SimpleNamespace(
        episode_return=jnp.asarray(0.0),
        episode_length=jnp.asarray(0),
    )

    labels, values = format_reward_overlay(
        RewardOverlayConfig(precision=2, show_components=False),
        step_count=0,
        reward=jnp.asarray(0.0),
        done=jnp.asarray(False),
        state=state,
        info={"command": jnp.asarray([0.1, 0.2, -0.3], dtype=jnp.float32)},
    )

    assert "cmd_x" in labels.splitlines()
    assert "cmd_y" in labels.splitlines()
    assert "cmd_yaw" in labels.splitlines()
    assert "0.20" in values.splitlines()


def test_format_action_overlay_includes_bar_graphs_and_saturation() -> None:
    env = SimpleNamespace(
        action_clip=10.0,
        model=SimpleNamespace(
            policy_ctrl_indices=np.asarray([0, 1], dtype=np.int32),
            actuator_joint_names=(
                "left_hip_pitch_joint",
                "right_ankle_roll_joint",
            ),
            mj_model=SimpleNamespace(
                actuator_ctrlrange=np.asarray(
                    [[-1.0, 1.0], [-0.5, 0.5]], dtype=np.float32
                )
            ),
        ),
    )

    labels, values = format_action_overlay(
        ActionOverlayConfig(max_actions=2, bar_width=7),
        env=env,
        action_mean=np.asarray([12.0, -5.0], dtype=np.float32),
        action=np.asarray([10.0, -5.0], dtype=np.float32),
        ctrl=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert labels.splitlines() == [
        "action_mean_abs_max",
        "action_abs_max",
        "ctrl_saturation",
        "h_pitch",
        "a_roll",
    ]
    value_lines = values.splitlines()
    assert value_lines[:3] == ["12.00", "10.00", "0.500"]
    assert "m +12.00" in value_lines[3]
    assert "a +10.00" in value_lines[3]
    assert "[---|###]" in value_lines[3]
    assert "m -5.00" in value_lines[4]


def test_command_joystick_updates_and_clips_command() -> None:
    joystick = CommandJoystick(
        np.zeros(3, dtype=np.float32),
        JoystickConfig(
            lin_vel_x_step=0.2,
            lin_vel_y_step=0.3,
            yaw_vel_step=0.4,
            lin_vel_x_limit=(-0.2, 0.2),
            lin_vel_y_limit=(-0.3, 0.3),
            yaw_vel_limit=(-0.4, 0.4),
        ),
        enabled=True,
    )

    joystick.key_callback(ord("W"))
    joystick.key_callback(ord("d"))
    joystick.key_callback(ord("Q"))
    joystick.key_callback(ord("W"))

    assert np.allclose(joystick.command, [0.2, 0.3, 0.4])

    joystick.key_callback(CommandJoystick.KEY_SPACE)

    assert np.allclose(joystick.command, [0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (EvalPolicyConfig(steps=0), "steps must be positive"),
        (EvalPolicyConfig(command=(0.0, 0.0)), "command must have three values"),
        (EvalPolicyConfig(action_seed=-1), "action_seed must be nonnegative"),
        (
            EvalPolicyConfig(
                reward_overlay=RewardOverlayConfig(max_reward_terms=-1),
            ),
            "max_reward_terms must be nonnegative",
        ),
        (
            EvalPolicyConfig(action_overlay=ActionOverlayConfig(max_actions=-1)),
            "action_overlay.max_actions must be nonnegative",
        ),
        (
            EvalPolicyConfig(action_overlay=ActionOverlayConfig(bar_width=0)),
            "action_overlay.bar_width must be positive",
        ),
        (
            EvalPolicyConfig(joystick=JoystickConfig(lin_vel_y_step=0.0)),
            "joystick.lin_vel_y_step must be positive",
        ),
        (
            EvalPolicyConfig(joystick=JoystickConfig(yaw_vel_limit=(1.0, -1.0))),
            "joystick.yaw_vel_limit must be finite and ordered",
        ),
    ],
)
def test_validate_eval_config_rejects_invalid_config(
    config: EvalPolicyConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_eval_config(config)


def test_mujoco_policy_observation_matches_env_shape(full_body_asset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(env.model.mj_model, mj_data)

    obs = mujoco_policy_observation(
        env,
        mj_data,
        command=jnp.zeros(3, dtype=jnp.float32),
        last_action=jnp.zeros(env.action_size, dtype=jnp.float32),
        step_count=0,
    )

    assert obs.shape == (env.observation_size,)
    assert jnp.all(jnp.isfinite(obs))


def test_mujoco_policy_observation_can_include_base_state(full_body_asset) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                base_lin_vel=True,
                base_ang_vel=False,
                projected_gravity=False,
                base_height=True,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(env.model.mj_model, mj_data)

    obs = mujoco_policy_observation(
        env,
        mj_data,
        command=jnp.zeros(3, dtype=jnp.float32),
        last_action=jnp.zeros(env.action_size, dtype=jnp.float32),
        step_count=0,
    )

    assert env.observation_size == 4
    assert obs.shape == (4,)
    assert jnp.allclose(obs[:3], jnp.zeros(3, dtype=jnp.float32))
    assert obs[3] == pytest.approx(mj_data.qpos[int(env.model.root_qpos_adr) + 2])


def test_mujoco_policy_observation_uses_configured_root_link(full_body_asset) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            root_link="torso_link",
            observation=ObservationConfig(
                base_lin_vel=True,
                base_ang_vel=False,
                projected_gravity=False,
                base_height=True,
                command=False,
                joint_pos=False,
                joint_vel=False,
                actuator_force=False,
                last_action=False,
            ),
        ),
    )
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = env.model.default_qpos
    mj_data.qvel[:] = env.model.default_qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(env.model.mj_model, mj_data)
    torso_id = mujoco.mj_name2id(
        env.model.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )

    obs = mujoco_policy_observation(
        env,
        mj_data,
        command=jnp.zeros(3, dtype=jnp.float32),
        last_action=jnp.zeros(env.action_size, dtype=jnp.float32),
        step_count=0,
    )

    assert torso_id >= 0
    assert env.observation_size == 4
    assert obs.shape == (4,)
    assert obs[3] == pytest.approx(mj_data.xpos[torso_id, 2])


def test_mujoco_policy_observation_matches_env_scaled_root_frame(
    full_body_asset,
) -> None:
    env = BelloLocomotionEnv.load_prepared(
        full_body_asset,
        env_config=BelloEnvConfig(
            observation=ObservationConfig(
                joint_mode="policy",
                base_lin_vel=True,
                base_ang_vel=True,
                projected_gravity=True,
                base_height=True,
                gait_phase=True,
                command=True,
                joint_pos=True,
                joint_vel=True,
                actuator_force=False,
                last_action=True,
                scale=ObservationScaleConfig(
                    base_lin_vel=2.0,
                    base_ang_vel=3.0,
                    projected_gravity=4.0,
                    base_height=5.0,
                    gait_phase=6.0,
                    command=7.0,
                    joint_pos=8.0,
                    joint_vel=9.0,
                    last_action=10.0,
                ),
            )
        ),
    )
    state, _ = env.reset()
    root_qpos = int(env.model.root_qpos_adr)
    root_qvel = int(env.model.root_qvel_adr)
    yaw = 0.5
    qpos = state.data.qpos.at[root_qpos + 3 : root_qpos + 7].set(
        jnp.asarray([jnp.cos(yaw / 2), 0.0, 0.0, jnp.sin(yaw / 2)], dtype=jnp.float32)
    )
    qvel = state.data.qvel.at[root_qvel : root_qvel + 6].set(
        jnp.asarray([0.2, -0.1, 0.0, 0.3, -0.4, 0.5], dtype=jnp.float32)
    )
    command = jnp.asarray([0.1, -0.2, 0.3], dtype=jnp.float32)
    last_action = jnp.linspace(-0.5, 0.5, env.action_size)
    state = state.replace(
        data=state.data.replace(qpos=qpos, qvel=qvel),
        command=command,
        last_action=last_action,
        step_count=jnp.asarray(3, dtype=jnp.int32),
    )
    mj_data = mujoco.MjData(env.model.mj_model)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = qvel
    mj_data.ctrl[:] = env.model.default_ctrl
    mujoco.mj_forward(env.model.mj_model, mj_data)

    expected = env.observation(state)
    actual = mujoco_policy_observation(
        env,
        mj_data,
        command=command,
        last_action=last_action,
        step_count=3,
    )

    assert jnp.allclose(actual, expected, atol=1.0e-5)


def test_mujoco_action_to_ctrl_matches_env_mapping(full_body_asset) -> None:
    env = BelloLocomotionEnv.load_prepared(full_body_asset)
    action = jnp.linspace(-0.5, 0.5, env.action_size)

    expected = env._action_to_ctrl(action)
    actual = mujoco_action_to_ctrl(env, jnp.asarray(action))

    assert jnp.allclose(actual, expected)


def test_rollout_policy_action_can_sample_stochastic_action() -> None:
    class FakePolicy:
        log_std = jnp.zeros(2, dtype=jnp.float32)
        min_log_std = -5.0
        max_log_std = 5.0

        def action_mean(self, obs):
            return jnp.asarray([0.25, -0.25], dtype=jnp.float32)

    obs_stack = jnp.zeros((2, 3), dtype=jnp.float32)
    deterministic = rollout_policy_action(FakePolicy(), obs_stack)
    sampled = rollout_policy_action(
        FakePolicy(),
        obs_stack,
        action_rng=jnp.asarray([0, 1], dtype=jnp.uint32),
        action_clip=0.5,
    )

    assert jnp.allclose(deterministic, jnp.asarray([0.25, -0.25]))
    assert not jnp.allclose(sampled, deterministic)
    assert jnp.max(jnp.abs(sampled)) <= 0.5


def test_sync_foot_contact_colors_marks_contacted_foot_red() -> None:
    model = SimpleNamespace(
        geom_bodyid=np.asarray([1, 1, 2, 3], dtype=np.int32),
        geom_rgba=np.asarray(
            [
                [0.1, 0.1, 0.1, 1.0],
                [0.2, 0.2, 0.2, 1.0],
                [0.3, 0.3, 0.3, 1.0],
                [0.4, 0.4, 0.4, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    base_rgba = model.geom_rgba.copy()
    foot_geoms = foot_geom_ids_for_bodies(model, np.asarray([1, 2], dtype=np.int32))

    sync_foot_contact_colors(
        model,
        base_rgba,
        foot_geoms,
        np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert np.allclose(model.geom_rgba[0], [1.0, 0.0, 0.0, 1.0])
    assert np.allclose(model.geom_rgba[1], [1.0, 0.0, 0.0, 1.0])
    assert np.allclose(model.geom_rgba[2], base_rgba[2])
    assert np.allclose(model.geom_rgba[3], base_rgba[3])

    sync_foot_contact_colors(
        model,
        base_rgba,
        foot_geoms,
        np.asarray([0.0, 1.0], dtype=np.float32),
    )

    assert np.allclose(model.geom_rgba[0], base_rgba[0])
    assert np.allclose(model.geom_rgba[1], base_rgba[1])
    assert np.allclose(model.geom_rgba[2], [1.0, 0.0, 0.0, 1.0])
    assert np.allclose(model.geom_rgba[3], base_rgba[3])


def test_sync_foot_contact_markers_draws_contacted_foot_overlay() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="left_foot" pos="0 0 0">
              <geom name="left_foot_geom" type="box" size="0.1 0.2 0.03"/>
            </body>
            <body name="right_foot" pos="1 0 0">
              <geom name="right_foot_geom" type="box" size="0.1 0.2 0.03"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=8)
    foot_geoms = (
        np.asarray([0], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
    )

    sync_foot_contact_markers(
        model,
        data,
        foot_geoms,
        np.asarray([0.0, 1.0], dtype=np.float32),
        scene,
    )

    assert scene.ngeom == 1
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_BOX
    assert np.allclose(scene.geoms[0].pos, data.geom_xpos[1])
    assert np.allclose(scene.geoms[0].rgba, [1.0, 0.0, 0.0, 1.0])

    sync_foot_contact_markers(
        model,
        data,
        foot_geoms,
        np.asarray([0.0, 0.0], dtype=np.float32),
        scene,
    )

    assert scene.ngeom == 0


def test_mujoco_foot_contact_reads_terrain_contacts() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="2 2 .1"/>
            <body name="left_foot" pos="-0.2 0 0.04">
              <freejoint/>
              <geom name="left_foot_geom" type="box" size=".1 .1 .04"/>
            </body>
            <body name="right_foot" pos="0.2 0 0.04">
              <freejoint/>
              <geom name="right_foot_geom" type="box" size=".1 .1 .04"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    foot_body_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot"),
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot"),
        ],
        dtype=np.int32,
    )
    env = SimpleNamespace(
        config=SimpleNamespace(
            contact=SimpleNamespace(foot_contact_force_threshold=1.0)
        ),
        model=SimpleNamespace(
            mj_model=model,
            foot_body_ids=foot_body_ids,
            terrain_geom_ids=np.asarray([0], dtype=np.int32),
            geom_body_ids=np.asarray(model.geom_bodyid, dtype=np.int32),
        )
    )

    assert np.allclose(mujoco_foot_contact(env, data), [1.0, 1.0])

    env.config.contact.foot_contact_force_threshold = 1.0e9

    assert np.allclose(mujoco_foot_contact(env, data), [0.0, 0.0])

    env.config.contact.foot_contact_force_threshold = 1.0
    env.model.terrain_geom_ids = np.asarray([], dtype=np.int32)

    assert np.allclose(mujoco_foot_contact(env, data), [0.0, 0.0])


def test_eval_main_wires_checkpoint_env_and_viewer(monkeypatch) -> None:
    metadata = SimpleNamespace(
        asset="asset",
        env="env",
        reward="reward",
        command="command",
        done="done",
        warp="warp",
        ppo=PPOConfig(policy=ActorCriticConfig(obs_stack_size=3)),
        obs_size=12,
        action_size=2,
    )
    train_state = SimpleNamespace(policy="policy")
    loaded = {"loads": []}

    def fake_restore(path):
        loaded["checkpoint_path"] = path
        return train_state, metadata

    class FakeEnv:
        observation_size = 4
        action_size = 2

        def __init__(self, *, use_viewer_model=False) -> None:
            self.use_viewer_model = use_viewer_model

        @classmethod
        def load(cls, asset, **kwargs):
            loaded["loads"].append((asset, kwargs))
            return cls(use_viewer_model=kwargs.get("use_viewer_model", False))

    def fake_launch(env, policy, **kwargs):
        loaded["launch"] = (env, policy, kwargs)

    monkeypatch.setattr(eval_policy, "restore_training_checkpoint", fake_restore)
    monkeypatch.setattr(eval_policy, "BelloLocomotionEnv", FakeEnv)
    monkeypatch.setattr(eval_policy, "launch_policy_viewer", fake_launch)

    eval_policy.main("checkpoint")

    assert loaded["checkpoint_path"] == "checkpoint"
    assert loaded["loads"] == [
        (
            "asset",
            {
                "warp": "warp",
                "batch_size": 1,
                "env_config": "env",
                "reward_config": "reward",
                "command_config": "command",
                "done_config": "done",
                "load_mjx": False,
            },
        ),
        (
            "asset",
            {
                "warp": "warp",
                "batch_size": 1,
                "env_config": "env",
                "reward_config": "reward",
                "command_config": "command",
                "done_config": "done",
                "use_viewer_model": True,
                "load_mjx": False,
            },
        ),
    ]
    assert loaded["launch"][0].use_viewer_model
    assert loaded["launch"][1] == "policy"
    assert loaded["launch"][2]["obs_stack_size"] == 3
    assert loaded["launch"][2]["joystick_config"] == JoystickConfig()
    assert loaded["launch"][2]["action_overlay"] == ActionOverlayConfig()
    assert not loaded["launch"][2]["stochastic_actions"]
    assert loaded["launch"][2]["action_seed"] == 0
