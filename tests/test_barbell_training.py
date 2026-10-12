"""Synthetic contracts for a separately modeled payload and robot coordinates."""

from dataclasses import replace
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pytest

from bello_mujoco.envs.barbell import coherent_bar_reset, arm_target
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
    MotionResetConfig,
)
from bello_mujoco.models.barbell import (
    add_barbell,
    RigidBarConfig,
    Grip,
    sample_payloads,
    free_joint_coordinates,
)
from bello_mujoco.models.compiler import ModelConfig
from bello_mujoco.motion.clip import (
    MotionClip,
    sample_motion,
    save_motion_clip,
    load_motion_clip,
)
from bello_mujoco.sim import ModelRandomizationConfig, _randomized_model_fields


def mechanism(mass=30.0, *, rotated_hands=False):
    root = ET.fromstring("""<mujoco><worldbody>
    <body name="root" pos="0 0 1"><freejoint name="root_free"/><geom size=".1" mass="3"/>
      <body name="torso"><geom size=".05" mass="2"/>
        <body name="hinge"><joint name="hinge"/><geom size=".04" mass="1"/></body>
        <body name="left" pos="0 .2 0"><geom size=".01" mass=".1"/></body>
        <body name="right" pos="0 -.2 0"><geom size=".01" mass=".1"/></body>
      </body></body></worldbody><actuator><position joint="hinge" kp="1"/></actuator></mujoco>""")
    config = RigidBarConfig(
        hands=(Grip("left", "left", (0, 0, 0)), Grip("right", "right", (0, 0, 0)))
    )
    if rotated_hands:
        for side, sign in (("left", -1), ("right", 1)):
            root.find(f".//body[@name='{side}']").set(
                "quat", f"{2**-0.5} 0 0 {sign * 2**-0.5}"
            )
        config = replace(
            config,
            hands=(
                Grip("left", "left", (0, 0, 0), (-1, 0, 0), (0, -1, 0)),
                Grip("right", "right", (0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ),
        )
    add_barbell(root, config, (0, 0, 1), mass, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model


def test_mirrored_hand_axes_preserve_grip_hinge_and_detect_rotation():
    from bello_mujoco.models.barbell import grip_axis_errors

    model = mechanism(rotated_hands=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(grip_axis_errors(model, data), 0, atol=1e-12)
    for side in ("left", "right"):
        for anchor in ("grip", "grip_axis"):
            np.testing.assert_allclose(
                data.site_xpos[model.site(f"{side}_hand_{anchor}").id],
                data.site_xpos[model.site(f"{side}_bar_{anchor}").id],
                atol=1e-12,
            )
    model.body_quat[model.body("left").id] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(
        grip_axis_errors(model, data), [np.pi / 2, 0], atol=1e-12
    )


@pytest.mark.parametrize("axis", [(0, 0, 0), (2, 0, 0), (float("nan"), 0, 0)])
def test_invalid_grip_axis_rejected(axis):
    with pytest.raises(ValueError, match="unit vectors"):
        RigidBarConfig(
            hands=(Grip("left", "left", shaft_axis_local=axis), Grip("right", "right"))
        )


def test_shaft_normal_to_palm_is_rejected():
    with pytest.raises(ValueError, match="palm plane"):
        RigidBarConfig(
            hands=(
                Grip("left", "left", shaft_axis_local=(1, 0, 0)),
                Grip("right", "right"),
            )
        )


def test_payload_mixture_is_reproducible_and_world_prefix_stable():
    values = sample_payloads(4, 20000)
    np.testing.assert_array_equal(values[:100], sample_payloads(4, 100))
    assert np.mean(values == 0) == pytest.approx(0.25, abs=0.015)
    assert np.mean(values == 30) == pytest.approx(0.25, abs=0.015)
    uniform = values[(values != 0) & (values != 30)]
    assert uniform.min() >= 5 and uniform.max() <= 30
    assert uniform.mean() == pytest.approx(17.5, abs=0.3)


@pytest.mark.parametrize("mass", [15.0, 30.0])
def test_mass_accounted_once_and_generic_randomization_excludes_object(mass):
    model = mechanism(mass)
    robot = SimpleNamespace(
        randomization_base_body_name="root",
        randomization_central_body_names=("root", "torso"),
        randomization_torso_body_name="torso",
    )
    args = dict(
        batch_size=32,
        actuator_joint_ids=np.array([model.joint("hinge").id]),
        policy_ctrl_indices=np.array([0]),
        robot=robot,
    )
    fixed = _randomized_model_fields(
        model,
        ModelRandomizationConfig(
            enabled=True,
            joint_inertia_scale_range=(0.8, 1.2),
            torso_mass_scale_range=(0.8, 1.2),
        ),
        **args,
    )
    bar = model.body("barbell").id
    assert model.body_mass.sum() == pytest.approx(6.2 + mass)
    np.testing.assert_allclose(fixed["body_mass"][:, bar], mass)
    np.testing.assert_allclose(
        fixed["body_inertia"][:, bar], np.tile(model.body_inertia[bar], (32, 1))
    )
    varied = _randomized_model_fields(
        model,
        ModelRandomizationConfig(enabled=True, payload_distribution="standing"),
        **args,
    )
    positive = varied["payload_mass"] > 0
    np.testing.assert_allclose(
        varied["body_mass"][positive, bar], varied["payload_mass"][positive]
    )
    np.testing.assert_allclose(
        varied["body_inertia"][positive, bar],
        varied["payload_mass"][positive, None] / mass * model.body_inertia[bar],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        varied["body_mass"][:, :bar], np.tile(model.body_mass[:bar], (32, 1)), rtol=1e-6
    )
    np.testing.assert_allclose(
        varied["body_invweight0"][positive, bar, 0],
        1 / varied["payload_mass"][positive],
        rtol=1e-6,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        ModelConfig(barbell=RigidBarConfig(), end_effector_payload_mass_kg=1.0)


def test_loaded_reset_moves_both_grips_and_unloaded_parks_and_disconnects():
    model = mechanism()
    native = mujoco.MjData(model)
    data = mjx.put_data(model, native, impl="jax")
    meta = SimpleNamespace(
        root_qpos_adr=0,
        bar_qpos_adr=int(model.joint("barbell_free").qposadr[0]),
        default_qpos=native.qpos.copy(),
        payload_mass=np.array([30.0, 0.0]),
        grip_equality_ids=np.arange(model.neq),
    )
    data = data.replace(
        qpos=data.qpos.at[:3]
        .add(jnp.array([0.3, -0.2, 0.1]))
        .at[3:7]
        .set(jnp.array([2**-0.5, 0, 0, 2**-0.5]))
    )
    loaded = jax.jit(lambda d: coherent_bar_reset(meta, d, 0))(data)
    native.qpos[:] = loaded.qpos
    mujoco.mj_forward(model, native)
    for side in ("left", "right"):
        np.testing.assert_allclose(
            native.site_xpos[model.site(f"{side}_hand_grip").id],
            native.site_xpos[model.site(f"{side}_bar_grip").id],
            atol=1e-6,
        )
    assert np.asarray(loaded.eq_active).all()
    unloaded = jax.jit(lambda d: coherent_bar_reset(meta, d, 1))(data)
    assert not np.asarray(unloaded.eq_active).any()
    np.testing.assert_allclose(
        unloaded.qpos[meta.bar_qpos_adr : meta.bar_qpos_adr + 3], [10, 0, 0.15]
    )


@pytest.mark.parametrize("mass", [15.0, 30.0])
def test_extra_free_body_before_robot_roundtrip_and_rigid_alignment(tmp_path, mass):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="barbell"><freejoint name="barbell_free"/><geom size=".1" mass="30"/></body>
    <body name="robot" pos="1 0 0"><freejoint name="root"/><geom size=".1" mass="2"/>
    <body name="arm"><joint name="hinge"/><geom size=".1" mass="1"/></body></body>
    </worldbody></mujoco>""")
    model.body_mass[1] = mass
    data = mujoco.MjData(model)
    meta = SimpleNamespace(
        mj_model=model,
        bar_body_id=1,
        root_qpos_adr=7,
        root_qvel_adr=6,
        free_joint_coordinates=free_joint_coordinates(model),
        actuator_joint_names=("hinge",),
        actuator_qpos_indices=np.array([14]),
        actuator_qvel_indices=np.array([12]),
    )
    qpos = np.tile(data.qpos, (2, 1))
    qvel = np.zeros((2, 13), dtype=np.float32)
    clip = MotionClip(
        fps=50.0,
        joint_names=("hinge",),
        body_names=("barbell", "robot"),
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        body_pos=jnp.zeros((2, 2, 3)),
        body_quat=jnp.tile(jnp.array([1.0, 0, 0, 0]), (2, 2, 1)),
        body_lin_vel=jnp.zeros((2, 2, 3)),
        body_ang_vel=jnp.zeros((2, 2, 3)),
        foot_contacts=jnp.ones((2, 2)),
        object_mass_kg=mass,
        screen_passed=True,
        free_joint_qpos=(0, 7),
        joint_qpos_indices=(14,),
        joint_qvel_indices=(12,),
    )
    path = tmp_path / "clip.npz"
    save_motion_clip(path, clip)
    loaded = load_motion_clip(path, meta)
    np.testing.assert_allclose(sample_motion(loaded, jnp.array(0.005)).qpos, qpos[0])
    env = object.__new__(BelloMotionTrackingEnv)
    object.__setattr__(env, "model", meta)
    object.__setattr__(env, "motion", loaded)
    object.__setattr__(
        env,
        "config",
        MotionTrackingEnvConfig(
            reset=MotionResetConfig(
                root_position_noise=(0.1, 0.1, 0.1), root_rotation_noise=(0.1, 0.1, 0.1)
            )
        ),
    )
    aligned = jax.jit(env.aligned_reference)(
        jnp.array(0.0), jnp.array([2.0, 3.0]), jnp.array(np.pi / 2)
    )
    np.testing.assert_allclose(aligned.qpos[:3], [2, 3, 0], atol=1e-6)
    np.testing.assert_allclose(aligned.qpos[7:10], [2, 4, 0], atol=1e-6)
    q, v = jax.jit(env._perturb_root)(aligned.qpos, aligned.qvel, jax.random.PRNGKey(0))
    assert np.linalg.norm(q[:3] - q[7:10]) == pytest.approx(1.0, abs=1e-6)
    save_motion_clip(path, replace(clip, object_mass_kg=mass + 1))
    with pytest.raises(ValueError, match="configured bar mass"):
        load_motion_clip(path, meta)
    with pytest.raises(ValueError, match="configured bar mass"):
        load_motion_clip(path, meta, allow_unscreened_reference=True)
    save_motion_clip(path, replace(clip, screen_passed=False))
    with pytest.raises(ValueError, match="passed reference screen"):
        load_motion_clip(path, meta)
    exploratory = load_motion_clip(path, meta, allow_unscreened_reference=True)
    assert not exploratory.screen_passed
    assert exploratory.object_mass_kg == mass
    save_motion_clip(path, exploratory)
    with pytest.raises(ValueError, match="passed reference screen"):
        load_motion_clip(path, meta)


def test_loaded_arm_sampler_never_selects_rest_and_jit_vmap_agree():
    targets = np.array([[[1.0], [1.0]], [[2.0], [2.0]]], dtype=np.float32)
    bank = (targets, np.array([True, False]), 1.0)

    def sample(world, mass):
        return arm_target(bank, jnp.array(0), 0.02, world, mass, 42)

    loaded = jax.jit(jax.vmap(sample))(jnp.arange(32), jnp.full(32, 30.0))
    np.testing.assert_allclose(loaded, 1.0)
    unloaded = jax.jit(jax.vmap(sample))(jnp.arange(32), jnp.zeros(32))
    assert set(np.asarray(unloaded).ravel()) == {1.0, 2.0}


@pytest.mark.integration
@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="MJX-Warp throughput requires CUDA"
)
@pytest.mark.parametrize("rotated_hands", [False, True])
def test_warp_hinge_grips_world_loads_and_throughput(rotated_hands):
    import time
    import json

    root = ET.fromstring("""<mujoco><option timestep=".0025"/><worldbody>
    <body name="left" pos="0 .2 1"><geom size=".01"/></body>
    <body name="right" pos="0 -.2 1"><geom size=".01"/></body>
    </worldbody></mujoco>""")
    geometry = RigidBarConfig(
        hands=(Grip("left", "left", (0, 0, 0)), Grip("right", "right", (0, 0, 0)))
    )
    if rotated_hands:
        for side, sign in (("left", -1), ("right", 1)):
            root.find(f".//body[@name='{side}']").set(
                "quat", f"{2**-0.5} 0 0 {sign * 2**-0.5}"
            )
        geometry = replace(
            geometry,
            hands=(
                Grip("left", "left", (0, 0, 0), (-1, 0, 0), (0, -1, 0)),
                Grip("right", "right", (0, 0, 0), (1, 0, 0), (0, 1, 0)),
            ),
        )
    add_barbell(root, geometry, (0, 0, 1), 30.0, (0.02, 1.0))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    count = 256
    masses = sample_payloads(7, count)
    physical = np.where(masses > 0, masses, 30.0)
    body = model.body("barbell").id
    mass = np.tile(model.body_mass, (count, 1))
    mass[:, body] = physical
    inertia = np.tile(model.body_inertia, (count, 1, 1))
    inertia[:, body] *= (physical / 30)[:, None]
    inverse_weight = np.tile(model.body_invweight0, (count, 1, 1))
    inverse_weight[:, body] *= (30 / physical)[:, None]
    dof_weight = np.tile(model.dof_invweight0, (count, 1)) * (30 / physical)[:, None]
    batch_model = mjx.put_model(model, impl="warp").replace(
        body_mass=jnp.asarray(mass),
        body_inertia=jnp.asarray(inertia),
        body_invweight0=jnp.asarray(inverse_weight),
        dof_invweight0=jnp.asarray(dof_weight),
    )
    data = mjx.make_data(model, impl="warp", naconmax=count * 16, njmax=128)
    batch = jax.vmap(lambda _: data)(jnp.arange(count))
    batch = batch.replace(
        _impl=batch._impl.replace(nworld=count),
        eq_active=jnp.repeat(jnp.asarray(masses > 0)[:, None], model.neq, axis=1),
        qpos=batch.qpos.at[:, 0].set(jnp.where(masses > 0, 0.0, 10.0)),
    )

    def rollout(data):
        return jax.lax.fori_loop(0, 200, lambda _, d: mjx.step(batch_model, d), data)

    run = jax.jit(rollout)
    first = run(batch)
    jax.block_until_ready(first)
    start = time.perf_counter()
    result = run(batch)
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    assert np.isfinite(result.qpos).all()
    gaps = np.linalg.norm(
        np.asarray(result.site_xpos)[:, model.site("left_hand_grip").id]
        - np.asarray(result.site_xpos)[:, model.site("left_bar_grip").id],
        axis=1,
    )
    assert np.max(gaps[masses > 0]) < 0.015
    print(
        json.dumps(
            dict(
                worlds=count,
                physics_steps_per_second=count * 200 / elapsed,
                max_loaded_grip_error_m=float(np.max(gaps[masses > 0])),
            )
        )
    )


def test_motion_profile_keeps_fixed_object_and_all_policy_actions():
    from bello_mujoco.configuration import (
        locomotion_training_profile,
        motion_training_profile,
    )
    from bello_mujoco.training.motion import _validate_motion_asset

    standing = locomotion_training_profile("bello_mini_loaded_standing")
    walking = locomotion_training_profile("bello_mini_locomotion")
    motion = motion_training_profile(name="bello_mini_barbell_motion")
    _validate_motion_asset(motion.asset_config)
    assert (
        standing.asset_config.actuator_config.policy_joint_names
        == walking.asset_config.actuator_config.policy_joint_names
    )
    assert (
        walking.asset_config.barbell is None
        and walking.env_config.arm_pose_path is None
    )
    assert standing.env_config.model_randomization.payload_distribution == "standing"
    assert motion.env_config.model_randomization.payload_distribution == "fixed"
    assert motion.asset_config.barbell.mass_kg == 15.0
    assert motion.asset_config.target_total_mass_kg == 49.5
    with pytest.raises(ValueError, match="49.5 kg"):
        _validate_motion_asset(
            replace(
                motion.asset_config,
                target_total_mass_kg=48.0,
            )
        )


@pytest.mark.parametrize("rotated_hands", [False, True])
def test_contact_metrics_select_world_and_ignore_parked_prop(rotated_hands):
    from bello_mujoco.envs.barbell import bar_metrics

    model = mechanism(rotated_hands=rotated_hands)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_geom = int(model.body_geomadr[model.body("root").id])
    bar_geom = model.geom("barbell_shaft").id
    impl = SimpleNamespace(
        contact__geom=jnp.array([[root_geom, bar_geom]] * 2),
        contact__worldid=jnp.array([0, 1]),
        nacon=jnp.array([2]),
        contact__efc_address=jnp.array([[0], [1]]),
        efc__force=jnp.array([5.0, 7.0]),
        efc__type=jnp.array([6, 6]),
        efc__id=jnp.array([0, 1]),
        nefc=jnp.array([2]),
    )
    view = SimpleNamespace(
        xpos=jnp.asarray(data.xpos), xquat=jnp.asarray(data.xquat), _impl=impl
    )
    meta = SimpleNamespace(
        mj_model=model,
        bar_body_id=model.body("barbell").id,
        geom_body_ids=model.geom_bodyid,
        payload_mass=np.array([30.0, 0.0]),
    )
    result = jax.jit(lambda world: bar_metrics(meta, view, world))(0)
    assert result["barbell/unintended_contact_force_n"] == 5.0
    assert result["barbell/grip_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert result["barbell/grip_force_n"] == 0.0
    assert result["barbell/grip_axis_error_rad"] == pytest.approx(0.0, abs=1e-6)
    assert all(float(value) == 0.0 for value in bar_metrics(meta, view, 1).values())
