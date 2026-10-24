from dataclasses import replace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.state import PhysicalState


@pytest.fixture
def chain():
    model = mujoco.MjModel.from_xml_string("""
    <mujoco><compiler angle="radian"/><worldbody>
      <body name="object" pos="3 0 1"><freejoint/><geom size=".1"/></body>
      <body name="root" pos="0 0 1" quat=".8 .2 .3 .4">
        <freejoint/><geom pos=".12 .03 .04" size=".1"/>
        <body name="arm" pos=".2 -.1 .15" quat=".9 .1 .3 .2">
          <joint name="hinge" axis="1 2 3" pos=".02 -.03 .04" ref=".2"/>
          <geom pos=".03 .04 .2" size=".08"/>
          <body name="tip" pos=".04 .1 .3" quat=".7 -.1 .2 .3">
            <joint name="slide" type="slide" axis="1 0 1" ref=".03"/>
            <geom pos="-.02 .1 .05" size=".06"/>
            <body name="fixed" pos=".1 .2 -.1"><geom size=".02"/></body>
          </body>
        </body>
      </body>
    </worldbody></mujoco>""")
    names = ("fixed", "arm", "root", "tip")
    kin = Kinematics.from_model(model, ("slide", "hinge"), names, root_body_name="root")
    return model, kin, names


def check_native(model, kin, names, qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    state = kin.from_qpos(
        jnp.asarray(data.qpos, dtype=jnp.float32),
        jnp.asarray(data.qvel, dtype=jnp.float32),
    )
    result = jax.jit(lambda k, s: k.forward(s))(kin, state)
    ids = [model.body(name).id for name in names]
    np.testing.assert_allclose(result.position, data.xpos[ids], atol=2e-6)
    dots = np.sum(np.asarray(result.quaternion) * data.xquat[ids], axis=-1)
    np.testing.assert_allclose(np.abs(dots), 1, atol=2e-6)
    for i, body in enumerate(ids):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0
        )
        np.testing.assert_allclose(result.angular_velocity[i], velocity[:3], atol=2e-6)
        np.testing.assert_allclose(result.linear_velocity[i], velocity[3:], atol=2e-6)
    restored_q, restored_v = kin.to_qpos(state, jnp.asarray(qpos), jnp.asarray(qvel))
    np.testing.assert_allclose(restored_q, qpos, atol=2e-6)
    np.testing.assert_allclose(restored_v, qvel, atol=2e-6)


def test_poses_and_link_twists_match_native_with_offset_joints_and_free_addresses(
    chain,
):
    model, kin, names = chain
    rng = np.random.default_rng(1)
    for _ in range(5):
        qpos = model.qpos0.copy()
        qpos[kin.root_qpos_address : kin.root_qpos_address + 3] += rng.normal(size=3)
        qpos[np.asarray(kin.joint_qpos_address)] += rng.normal(size=2) * 0.3
        qvel = rng.normal(size=model.nv)
        check_native(model, kin, names, qpos, qvel)


def test_fk_is_equivariant_and_has_correct_velocity_derivatives(chain):
    model, kin, _ = chain
    state = kin.from_qpos(
        jnp.asarray(model.qpos0, dtype=jnp.float32),
        jnp.asarray(np.random.default_rng(3).normal(size=model.nv), dtype=jnp.float32),
    )
    yaw = rot.exp(jnp.array([0.0, 0.0, 0.7]))
    shift = jnp.array([0.3, -0.4, 0.0])
    moved = replace(
        state,
        position=rot.rotate(yaw, state.position) + shift,
        quaternion=rot.multiply(yaw, state.quaternion),
        linear_velocity=rot.rotate(yaw, state.linear_velocity),
        angular_velocity=rot.rotate(yaw, state.angular_velocity),
    )
    expected, actual = kin.forward(state), kin.forward(moved)
    np.testing.assert_allclose(
        actual.position, rot.rotate(yaw, expected.position) + shift, atol=1e-6
    )
    np.testing.assert_allclose(
        actual.linear_velocity, rot.rotate(yaw, expected.linear_velocity), atol=1e-6
    )

    def advance(t):
        return kin.forward(
            replace(
                state,
                position=state.position + t * state.linear_velocity,
                quaternion=rot.multiply(
                    rot.exp(t * state.angular_velocity), state.quaternion
                ),
                joint_position=state.joint_position + t * state.joint_velocity,
            )
        ).position

    _, derivative = jax.jvp(advance, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    np.testing.assert_allclose(derivative, expected.linear_velocity, atol=2e-6)
    difference = (advance(1e-3) - advance(-1e-3)) / 2e-3
    np.testing.assert_allclose(difference, derivative, atol=1e-4, rtol=1e-3)
    batched = jax.jit(jax.vmap(kin.forward))(
        jax.tree.map(lambda x: jnp.stack((x, x)), state)
    )
    np.testing.assert_allclose(batched.position[1], expected.position, atol=1e-6)


def test_rotations_and_state_roundtrip_have_finite_zero_angle_gradients():
    for vector in (jnp.zeros(3), jnp.array([1e-7, -1e-6, 0.2])):
        np.testing.assert_allclose(rot.log(rot.exp(vector)), vector, atol=1e-6)
        np.testing.assert_allclose(
            jax.jacfwd(lambda v: rot.log(rot.exp(v)))(vector), np.eye(3), atol=1e-5
        )
    identity = jnp.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(rot.log(identity), rot.log(-identity))
    state = PhysicalState(
        jnp.zeros(3), identity, jnp.ones(3), jnp.ones(3), jnp.zeros(2), jnp.ones(2)
    )
    restored = PhysicalState.unflatten(state.flatten(), 2)
    for a, b in zip(jax.tree.leaves(state), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fps,frame", [(50.0, 3.0), (100.0, 4.0), (25.0, 3.5)])
def test_cached_reference_targets_match_fk_after_alignment(chain, fps, frame):
    from types import SimpleNamespace

    from bello_mujoco.envs.supermimic import BelloSuperMimicEnv, align_reference
    from bello_mujoco.motion.clip import MotionClip
    from bello_mujoco.sonic.reference import ReferenceBank

    model, kin, _ = chain
    qpos = jnp.tile(jnp.asarray(model.qpos0), (10, 1))
    qpos = qpos.at[:, kin.joint_qpos_address].add(jnp.arange(10)[:, None] * 0.04)
    qvel = jnp.asarray(np.random.default_rng(9).normal(size=(10, model.nv)))
    physical = jax.vmap(kin.from_qpos)(qpos, qvel)
    bodies = jax.vmap(kin.forward)(physical)
    clip = MotionClip(
        fps,
        kin.joint_names,
        kin.body_names,
        qpos,
        qvel,
        *jax.tree.leaves(bodies),
        jnp.zeros((10, 2)),
    )
    bank = ReferenceBank.from_clips((("moving", clip, "train"),))

    def reference(frame, xy, yaw):
        sample = bank.sample_global_frame(frame)
        return align_reference(kin.from_qpos(sample.qpos, sample.qvel), xy, yaw)

    env = SimpleNamespace(
        reference_bank=bank,
        kinematics=kin,
        control_timestep=0.02,
        reference_state=reference,
    )
    xy, yaw = jnp.array([0.4, -0.3]), jnp.array(0.7)
    expected = kin.forward(reference(jnp.array(frame), xy, yaw))
    actual = jax.jit(lambda f: BelloSuperMimicEnv.reference_bodies(env, f, xy, yaw))(
        jnp.array(frame)
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=2e-6)


def test_heading_is_defined_and_differentiable_at_vertical_forward_axis():
    quaternion = rot.exp(jnp.array([0.0, jnp.pi / 2, 0.0]))
    np.testing.assert_allclose(rot.heading(quaternion), [1.0, 0.0, 0.0, 0.0])
    derivative = jax.jit(jax.jacfwd(rot.heading))(quaternion)
    assert np.isfinite(derivative).all()
    np.testing.assert_array_equal(derivative, 0.0)


def test_invalid_tree_contracts_fail_before_compilation(chain):
    model, _, _ = chain
    with pytest.raises(ValueError, match="every joint"):
        Kinematics.from_model(model, ("hinge",), ("root",), root_body_name="root")
    with pytest.raises(ValueError, match="outside"):
        Kinematics.from_model(
            model, ("hinge", "slide"), ("object",), root_body_name="root"
        )
    with pytest.raises(ValueError, match="unique"):
        Kinematics.from_model(
            model, ("hinge", "slide"), ("tip", "tip"), root_body_name="root"
        )


@pytest.mark.integration
def test_canonical_mini_kinematics_matches_native(bello_mini_asset):
    from bello_mujoco.sim import load_compiled_model
    from bello_mujoco.robot_tasks import motion_task_definition

    model = load_compiled_model(bello_mini_asset, load_mjx=False)
    names = motion_task_definition("bello_mini").beyondmimic_tracked_body_names
    kin = Kinematics.from_model(
        model.mj_model, model.actuator_joint_names, names, root_body_name="pelvis_link"
    )
    qpos = model.default_qpos.copy()
    qpos[model.actuator_qpos_indices] += np.random.default_rng(5).uniform(
        -0.02, 0.02, len(model.actuator_joint_names)
    )
    qvel = np.random.default_rng(8).normal(size=model.mj_model.nv)
    check_native(model.mj_model, kin, names, qpos, qvel)
