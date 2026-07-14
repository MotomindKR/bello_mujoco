from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionResetConfig,
    MotionTrackingEnvConfig,
)
from bello_mujoco.motion import (
    DEFAULT_TRACKED_BODY_NAMES,
    GMR_BELLO_JOINT_NAMES,
    MOTION_FORMAT_VERSION,
    load_motion_clip,
    sample_motion,
)
from bello_mujoco.sim import load_prepared_bello_model
from bello_mujoco.hydra_config import CONFIG_ROOT, clear_hydra, structured_config
from bello_mujoco.tools.import_gmr_motion import _build_qpos
from bello_mujoco.tools.train_motion_ppo import MotionTrainingRunConfig


@pytest.fixture(scope="module")
def fixed_waist_asset(tmp_path_factory) -> PreparedAsset:
    return prepare_asset(
        AssetPrepConfig(
            asset="full_body",
            output_root=tmp_path_factory.mktemp("fixed-waist-asset"),
            collision_mode="box",
            end_effector_sphere_mode="both",
            fixed_joint_names=("waist_roll_joint", "waist_pitch_joint"),
        )
    )


@pytest.fixture(scope="module")
def fixed_waist_motion(fixed_waist_asset: PreparedAsset, tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fixed-waist-motion") / "test_motion.npz"
    model = load_prepared_bello_model(fixed_waist_asset, load_mjx=False)
    mj_model = model.mj_model
    frame_count = 4
    qpos = np.repeat(model.default_qpos[None, :], frame_count, axis=0)
    qpos[:, 1] += np.linspace(0.0, 0.06, frame_count)
    qvel = np.zeros((frame_count, mj_model.nv), dtype=np.float64)
    qvel[:, 1] = 0.6
    body_ids = np.asarray(
        [mj_model.body(name).id for name in DEFAULT_TRACKED_BODY_NAMES]
    )
    body_pos = np.empty((frame_count, len(body_ids), 3))
    body_quat = np.empty((frame_count, len(body_ids), 4))
    body_lin_vel = np.empty((frame_count, len(body_ids), 3))
    body_ang_vel = np.empty((frame_count, len(body_ids), 3))
    data = mujoco.MjData(mj_model)
    velocity = np.empty(6)
    for frame in range(frame_count):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(mj_model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]
        for body_index, body_id in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                mj_model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
                velocity,
                0,
            )
            body_ang_vel[frame, body_index] = velocity[:3]
            body_lin_vel[frame, body_index] = velocity[3:]
    np.savez_compressed(
        path,
        format_version=np.asarray(MOTION_FORMAT_VERSION),
        fps=np.asarray(10.0),
        joint_names=np.asarray(model.actuator_joint_names),
        body_names=np.asarray(DEFAULT_TRACKED_BODY_NAMES),
        source_sha256=np.asarray("synthetic"),
        qpos=qpos,
        qvel=qvel,
        body_pos=body_pos,
        body_quat_wxyz=body_quat,
        body_lin_vel=body_lin_vel,
        body_ang_vel=body_ang_vel,
        foot_contacts=np.zeros((frame_count, 2)),
    )
    return path


def tracking_config(path: Path, *, random_start: bool = False):
    return MotionTrackingEnvConfig(
        motion_path=path,
        reset=MotionResetConfig(
            random_start=random_start,
            initial_xy_range=(0.0, 0.0),
            initial_yaw_range=(0.0, 0.0),
            joint_pos_noise=(0.0, 0.0),
            joint_vel_noise=(0.0, 0.0),
        ),
    )


def test_default_motion_training_config_has_fixed_waist_contract() -> None:
    clear_hydra()
    with initialize_config_dir(
        version_base=None, config_dir=str(CONFIG_ROOT / "train_motion")
    ):
        config = structured_config(
            MotionTrainingRunConfig, compose(config_name="default")
        )

    assert config.asset_config.fixed_joint_names == (
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    assert config.asset_config.end_effector_sphere_mode == "both"
    assert config.asset_config.collision_mode == "box"
    assert config.asset_config.actuator_config.default_action_scale_ratio == 0.5
    assert config.env_config.motion_path == Path(
        "motions/local/walk1_subject1_fixed_waist.npz"
    )
    assert config.video_config.enabled
    assert config.video_config.render_final
    assert config.video_config.log_wandb
    assert config.video_config.wandb_name == "eval/motion_policy"
    assert config.config.num_envs == 1024
    assert config.config.learning_rate == pytest.approx(3.0e-4)
    assert config.config.policy.init_log_std == pytest.approx(-1.0)
    assert config.config.policy.action_distribution == "tanh_gaussian"
    assert config.warp.njmax == 2048


def test_gmr_joint_columns_are_remapped_by_name(
    fixed_waist_asset: PreparedAsset,
) -> None:
    model = load_prepared_bello_model(fixed_waist_asset, load_mjx=False)
    source_joint_pos = np.arange(len(GMR_BELLO_JOINT_NAMES), dtype=np.float64)[None, :]
    qpos = _build_qpos(
        model,
        np.asarray([[0.0, 0.0, 1.0]]),
        np.asarray([[0.0, 0.0, 0.0, 1.0]]),
        source_joint_pos,
    )

    source_index = {name: index for index, name in enumerate(GMR_BELLO_JOINT_NAMES)}
    for name, address in zip(
        model.actuator_joint_names, model.actuator_qpos_indices, strict=True
    ):
        assert qpos[0, address] == source_index[name]


def test_motion_loader_and_shortest_arc_interpolation(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    model = load_prepared_bello_model(fixed_waist_asset, load_mjx=False)
    clip = load_motion_clip(fixed_waist_motion, model)
    clip = clip.__class__(
        **{**clip.__dict__, "qpos": clip.qpos.at[1, 3:7].multiply(-1)}
    )

    sample = jax.jit(lambda: sample_motion(clip, jnp.asarray(0.05)))()

    assert sample.qpos.shape == (model.mj_model.nq,)
    assert float(jnp.linalg.norm(sample.qpos[3:7])) == pytest.approx(1.0)
    assert np.allclose(sample.qpos[3:7], np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_tracking_reset_matches_reference_and_is_nonterminal(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, observation = env.reset(reference_time=jnp.asarray(0.1))
    reward, _ = env._reward(state, state)
    done = env._done_flags(state)

    assert observation.shape == (env.observation_size,)
    assert env.critic_observation(state).shape == (env.critic_observation_size,)
    assert float(reward) > 0.8
    assert not bool(done["done/any"])


def test_tracking_nonfinite_state_is_terminal_and_reward_is_sanitized(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    state = state.replace(
        data=state.data.replace(cvel=state.data.cvel.at[0, 0].set(jnp.nan))
    )
    done_flags = env._done_flags(state)
    reward, reward_info = env._reward(state, state)
    reward, reward_info = env._sanitize_nonfinite_reward(
        reward, reward_info, done_flags
    )

    assert bool(done_flags["done/nonfinite"])
    assert bool(done_flags["done/terminal"])
    assert float(reward) == 0.0
    assert all(float(value) == 0.0 for value in reward_info.values())


def test_tracking_joint_limit_cost_is_bounded(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion)
    )
    state, _ = env.reset(reference_time=jnp.asarray(0.1))
    indices = jnp.asarray(env.model.policy_qpos_indices)
    state = state.replace(
        data=state.data.replace(qpos=state.data.qpos.at[indices].set(1.0e6))
    )

    assert float(env._joint_limit_cost(state)) == 1.0


@pytest.mark.integration
def test_tracking_batch_reset_and_step_are_jittable(
    fixed_waist_asset: PreparedAsset, fixed_waist_motion: Path
) -> None:
    env = BelloMotionTrackingEnv.load_prepared(
        fixed_waist_asset, tracking_config(fixed_waist_motion), batch_size=2
    )
    state, observation = jax.jit(lambda key: env.reset_batch(2, key))(
        jax.random.PRNGKey(0)
    )
    next_state, next_observation, reward, done, info = jax.jit(env.step_batch)(
        state,
        jnp.zeros((2, env.action_size), dtype=jnp.float32),
        jax.random.PRNGKey(1),
    )

    assert observation.shape == (2, env.observation_size)
    assert next_observation.shape == (2, env.observation_size)
    assert next_state.last_action.shape == (2, env.action_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert info["done/terminal"].shape == (2,)
