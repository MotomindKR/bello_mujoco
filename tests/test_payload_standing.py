"""EEF-load proxy accounting and native standing reset contracts."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bello_mujoco.configuration import locomotion_training_profile
from bello_mujoco.models.barbell import sample_payloads
from bello_mujoco.models.urdf import end_effector_payload_body_ids
from bello_mujoco.sim import (
    ModelRandomizationConfig,
    _apply_native_model_sample,
    _randomized_model_fields,
    load_model,
)
from bello_mujoco.terrain import FLAT_TERRAIN


def test_proxy_worlds_split_total_load_and_refresh_native_constants():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root" pos="0 0 2"><freejoint/><geom size=".1" mass="40"/>
        <body name="torso"><joint name="hinge"/><geom size=".1" mass="9.5"/>
          <body name="left_payload" pos="0 .2 0"><inertial pos="0 0 0"
            mass="7.5" diaginertia=".01 .01 .01"/></body>
          <body name="right_payload" pos="0 -.2 0"><inertial pos="0 0 0"
            mass="7.5" diaginertia=".01 .01 .01"/></body>
        </body>
      </body></worldbody><actuator><position joint="hinge"/></actuator></mujoco>""")
    ids = end_effector_payload_body_ids(model)
    robot = SimpleNamespace(
        randomization_base_body_name="root",
        randomization_central_body_names=("root", "torso"),
        randomization_torso_body_name="torso",
    )
    config = ModelRandomizationConfig(
        enabled=True,
        payload_distribution="standing",
        payload_mass_range_kg=(5, 15),
        joint_inertia_scale_range=(0.8, 1.2),
    )
    args = dict(
        batch_size=64,
        actuator_joint_ids=np.array([model.joint("hinge").id]),
        policy_ctrl_indices=np.array([0]),
        robot=robot,
    )
    fields = _randomized_model_fields(model, config, **args)
    masses = sample_payloads(config.seed, 64, (5, 15))
    assert {0.0, 15.0}.issubset(set(masses))
    np.testing.assert_allclose(fields["payload_mass"], masses)
    for body in ids:
        np.testing.assert_allclose(fields["body_mass"][:, body], masses / 2)
        np.testing.assert_allclose(
            fields["body_inertia"][:, body],
            masses[:, None] / 15 * model.body_inertia[body],
            atol=1e-8,
        )
    np.testing.assert_allclose(fields["body_mass"].sum(axis=1), 49.5 + masses)
    for world in [int(np.flatnonzero(masses == mass)[0]) for mass in (0, 15)]:
        native = deepcopy(model)
        _apply_native_model_sample(native, fields, sample=world)
        for name in ("body_subtreemass", "body_invweight0", "dof_invweight0"):
            assert np.isfinite(fields[name]).all()
            np.testing.assert_allclose(
                fields[name][world], getattr(native, name), rtol=1e-6
            )
        data = mujoco.MjData(native)
        for _ in range(100):
            mujoco.mj_step(native, data)
        assert np.isfinite(data.qpos).all()
    fixed = _randomized_model_fields(
        model, replace(config, payload_distribution="fixed"), **args
    )
    np.testing.assert_allclose(fixed["payload_mass"], 15)
    np.testing.assert_allclose(fixed["body_mass"][:, ids], 7.5)


def test_payload_range_and_prefix():
    values = sample_payloads(1, 20000, (5, 15))
    np.testing.assert_array_equal(values[:100], sample_payloads(1, 100, (5, 15)))
    assert np.mean(values == 0) == pytest.approx(0.25, abs=0.015)
    assert np.mean(values == 15) == pytest.approx(0.25, abs=0.015)
    assert values[(values > 0) & (values < 15)].mean() == pytest.approx(10, abs=0.2)
    with pytest.raises(ValueError):
        ModelRandomizationConfig(payload_mass_range_kg=(15, 5))


def test_compiled_proxy_counts_robot_once_and_has_no_extra_free_body():
    profile = locomotion_training_profile("bello_mini_payload_standing")
    model = load_model(profile.asset_config, terrain=FLAT_TERRAIN, load_mjx=False)
    np.testing.assert_allclose(model.robot_mass, 49.5, atol=1e-5)
    np.testing.assert_allclose(model.payload_mass, 15)
    assert model.mj_model.body_mass.sum() == pytest.approx(64.5)
    assert model.bar_body_id == -1 and len(model.payload_body_ids) == 2
    assert model.mj_model.nq == 35 and model.mj_model.nv == 34
    assert len(model.policy_ctrl_indices) == 14
    walking = locomotion_training_profile("bello_mini_locomotion")
    assert walking.asset_config.end_effector_payload_mass_kg == 0
    assert walking.env_config.arm_pose_path is None


def test_proxy_arm_bank_native_reset_keeps_robot_root_intact(tmp_path):
    from bello_mujoco.envs.locomotion import BelloLocomotionEnv
    from bello_mujoco.evaluation.render import reset_mujoco_data

    profile = locomotion_training_profile("bello_mini_payload_standing")
    model = load_model(profile.asset_config, terrain=FLAT_TERRAIN, load_mjx=False)
    path = tmp_path / "standing.npz"
    np.savez_compressed(
        path,
        targets=np.tile(model.default_ctrl, (2, 2, 1)),
        loaded=[True, False],
        fps=50.0,
        passed=True,
        joint_names=model.actuator_joint_names,
    )
    config = replace(
        profile.env_config,
        arm_pose_path=path,
        model_randomization=ModelRandomizationConfig(),
    )
    env = BelloLocomotionEnv.load(
        profile.asset_config,
        env_config=config,
        reward_config=profile.reward_config,
        command_config=profile.command_config,
        done_config=profile.done_config,
        load_mjx=False,
    )
    data = mujoco.MjData(env.model.mj_model)
    reset_mujoco_data(env, data)
    root = env.model.root_qpos_adr
    np.testing.assert_allclose(
        data.qpos[root : root + 2], env.model.default_qpos[root : root + 2]
    )
    np.testing.assert_allclose(
        data.qpos[root + 3 : root + 7], env.model.default_qpos[root + 3 : root + 7]
    )
    assert np.isfinite(data.qpos).all()
    assert len(env.observed_indices) == 28
