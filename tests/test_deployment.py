import jax
from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from onnx.reference import ReferenceEvaluator

from bello_mujoco import deployment
from bello_mujoco.deployment import actor_onnx
from bello_mujoco.envs.actions import bounded_motor_target
from bello_mujoco.policy import ActorCritic, ActorCriticConfig
from bello_mujoco.sim import hardware_imu_to_policy_rotation
from bello_mujoco.training.ppo_config import PPOConfig
from bello_mujoco.training.ppo_types import ObservationNormState
from bello_mujoco.training.normalization import normalize_observation


def test_export_imu_rotation_uses_fixed_body_frames_not_inertial_frames():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler fusestatic="false"/>
      <worldbody><body name="base"><freejoint/>
        <inertial pos=".2 0 0" quat=".7071067811865476 .7071067811865476 0 0"
          mass="1" diaginertia=".1 .2 .3"/>
        <body name="sensor" quat=".7071067811865476 0 0 .7071067811865476"/>
        <body name="moving"><joint/><geom size=".1"/></body>
      </body></worldbody></mujoco>""")
    description = SimpleNamespace(
        mj_model=model,
        default_qpos=model.qpos0.copy(),
        robot=SimpleNamespace(default_imu_body_name="base"),
        imu_body_name="sensor",
    )
    np.testing.assert_allclose(
        hardware_imu_to_policy_rotation(description),
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
        atol=1e-12,
    )
    description.imu_body_name = "moving"
    with pytest.raises(ValueError, match="rigidly attached"):
        hardware_imu_to_policy_rotation(description)
    description.imu_body_name = "missing"
    with pytest.raises(ValueError, match="must exist"):
        hardware_imu_to_policy_rotation(description)


@pytest.mark.parametrize("activation", ["tanh", "relu", "elu", "swish"])
@pytest.mark.parametrize("distribution", ["gaussian", "tanh_gaussian"])
@pytest.mark.parametrize(
    "normalization_mode", ["disabled", "variance_floor", "std_offset"]
)
def test_onnx_matches_actor_normalization_and_motor_mapping(
    activation, distribution, normalization_mode
):
    config = PPOConfig(
        policy=ActorCriticConfig(
            hidden_dims=(8, 6), activation=activation, action_distribution=distribution
        ),
        normalize_observations=normalization_mode != "disabled",
        observation_norm_epsilon_mode="std_offset"
        if normalization_mode == "std_offset"
        else "variance_floor",
    )
    actor = ActorCritic(5, 2, config.policy, key=jax.random.PRNGKey(1))
    norm = ObservationNormState(
        mean=jnp.arange(5) * 0.1, var=jnp.arange(5) * 0.2, count=jnp.asarray(10.0)
    )
    default = jnp.array([0.2, -0.3, 0.4])
    limits = jnp.array([[-1.0, 2.0], [-3.0, 1.0], [-1.0, 1.0]])
    indices = jnp.array([0, 1])
    metadata = {"git_commit": "a" * 40}
    graph = actor_onnx(
        actor,
        norm,
        config,
        default_ctrl=default,
        ctrl_limits=limits,
        policy_indices=indices,
        action_clip=1.0,
        metadata=metadata,
    )
    runtime = ReferenceEvaluator(graph)
    for observation in np.random.default_rng(5).normal(size=(3, 5)).astype(np.float32):
        action, motors = runtime.run(None, {"observation": observation[None]})
        expected_action = actor.action_mean(
            normalize_observation(jnp.asarray(observation), norm, config)
        )
        expected_motor = bounded_motor_target(
            expected_action, default, limits, indices, 1.0
        )
        np.testing.assert_allclose(action[0], expected_action, atol=1e-6)
        np.testing.assert_allclose(motors[0], expected_motor, atol=1e-6)


def test_onnx_effort_mapping_uses_applied_action_before_motor_limits():
    config = PPOConfig(policy=ActorCriticConfig(hidden_dims=(4,)))
    actor = ActorCritic(3, 2, config.policy, key=jax.random.PRNGKey(5))
    norm = ObservationNormState(jnp.zeros(3), jnp.ones(3), jnp.asarray(1.0))
    graph = actor_onnx(
        actor,
        norm,
        config,
        default_ctrl=[0.2, 0.3, 0.4],
        ctrl_limits=[[-1, 1]] * 3,
        policy_indices=[0, 2],
        action_clip=1.0,
        metadata={},
        effort_scale=[4.0, 6.0],
    )
    obs = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    action, target = ReferenceEvaluator(graph).run(None, {"observation": obs})
    expected = np.asarray([0.2, 0.3, 0.4])
    expected[[0, 2]] += np.clip(action[0], -1.0, 1.0) * [4.0, 6.0]
    np.testing.assert_allclose(target[0], np.clip(expected, -1.0, 1.0), atol=1e-6)


def test_policy_publication_does_not_overwrite_or_leave_staging_files(tmp_path):
    output = tmp_path / "policy.onnx"
    deployment._publish_policy(output, b"complete model")
    with pytest.raises(FileExistsError):
        deployment._publish_policy(output, b"replacement")
    assert output.read_bytes() == b"complete model"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("with_motion", [False, True])
def test_bundle_contains_policy_and_only_required_reference(with_motion):
    motion = SimpleNamespace(
        fps=50.0,
        joint_names=("hinge",),
        body_names=("body",),
        qpos=np.zeros((2, 8)),
        qvel=np.zeros((2, 7)),
        body_pos=np.zeros((2, 1, 3)),
        body_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1, 1)),
        body_lin_vel=np.zeros((2, 1, 3)),
        body_ang_vel=np.zeros((2, 1, 3)),
        foot_contacts=np.zeros((2, 2)),
    )
    content = deployment.deployment_bundle(
        b"actor", physics=b"physics", motion=motion if with_motion else None
    )
    with ZipFile(BytesIO(content)) as archive:
        assert archive.namelist() == ["policy.onnx", "physics.xml"] + (
            ["motion.npz"] if with_motion else []
        )
        assert archive.read("policy.onnx") == b"actor"
        assert archive.read("physics.xml") == b"physics"
        if with_motion:
            with np.load(
                BytesIO(archive.read("motion.npz")), allow_pickle=False
            ) as clip:
                np.testing.assert_array_equal(clip["qpos"], motion.qpos)
                assert float(clip["fps"]) == motion.fps


def test_sonic_bundle_contains_only_selected_named_clips(tmp_path):
    walk = tmp_path / "walk.npz"
    dance = tmp_path / "dance.npz"
    walk.write_bytes(b"walk")
    dance.write_bytes(b"dance")

    content = deployment.multi_clip_deployment_bundle(
        b"actor",
        physics=b"physics",
        clips=(("walk", walk), ("dance", dance)),
    )

    with ZipFile(BytesIO(content)) as archive:
        assert archive.namelist() == [
            "policy.onnx",
            "physics.xml",
            "clips.json",
            "clips/walk.npz",
            "clips/dance.npz",
        ]
        assert archive.read("clips.json") == b'{"clips": ["walk", "dance"]}'


@pytest.mark.parametrize("rough_terrain", [False, True])
def test_physics_snapshot_preserves_runtime_dynamics_and_needs_no_source(
    tmp_path, rough_terrain
):
    source = tmp_path / "source.xml"
    source.write_text("""<mujoco>
      <compiler angle="radian"/>
      <option integrator="implicitfast"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1"/>
        <body name="link" pos="0 0 1">
          <joint name="pitch" axis="0 1 0" range="-1 1"/>
          <inertial pos=".01 0 0" mass="1" diaginertia=".000012345678912 .000023456789123 .00003"/>
          <geom name="sole" type="box" size=".1 .05 .03"/>
          <site name="imu" pos=".02 0 0"/>
          <body name="child" pos="0 0 -.2">
            <joint name="roll" axis="1 0 0" range="-.5 .5"/>
            <inertial pos="0 0 0" mass=".2" diaginertia=".001 .002 .002"/>
            <geom type="sphere" size=".04"/>
          </body>
        </body>
      </worldbody>
      <tendon><fixed name="sum"><joint joint="pitch" coef="1"/><joint joint="roll" coef="1"/></fixed></tendon>
      <actuator><position name="motor" tendon="sum" kp="2" kv=".2" ctrllimited="true" ctrlrange="-1 1" forcelimited="true" forcerange="-3 3"/></actuator>
      <sensor><gyro name="gyro" site="imu"/><framequat name="orientation" objtype="site" objname="imu"/></sensor>
      <keyframe><key name="home" qpos=".1 .2" ctrl=".3"/></keyframe>
    </mujoco>""")
    if rough_terrain:
        tree = ET.parse(source)
        root = tree.getroot()
        asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "hfield",
            name="terrain",
            nrow="2",
            ncol="2",
            size="2 2 .1 .01",
            elevation="0 .25 .5 1",
        )
        floor = root.find("worldbody/geom")
        floor.set("type", "hfield")
        floor.set("hfield", "terrain")
        floor.attrib.pop("size")
        tree.write(source)
    native = mujoco.MjModel.from_xml_path(str(source))
    native.opt.timestep = 0.0015
    native.body_mass[1] = 1.23456789123
    native.body_ipos[1] = [0.012345678912, 0, 0]
    native.geom_friction[:, 0] = 0.73
    native.geom_margin[0] = 0.001
    native.geom_solref[:] = [0.012, 1.2]
    native.actuator_gainprm[0, 0] = 3
    native.actuator_biasprm[0, 1:3] = [-3, -0.4]
    mujoco.mj_setConst(native, mujoco.MjData(native))
    home = dict(
        default_qpos=np.array([0.15, 0.05]),
        default_qvel=np.array([0.1, -0.1]),
        default_ctrl=np.array([0.22]),
    )
    # A different model may have been loaded by checkpoint/actor setup.
    mujoco.MjModel.from_xml_string("<mujoco/>")
    content = deployment.physics_mjcf(
        SimpleNamespace(mj_model=native, xml_path=source, **home)
    )
    source.unlink()
    restored = mujoco.MjModel.from_xml_string(content.decode())
    assert restored.nmesh == restored.ntex == 0
    assert restored.opt.timestep == native.opt.timestep
    for name in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "geom_friction",
        "geom_margin",
        "geom_solref",
        "actuator_gainprm",
        "actuator_biasprm",
        "tendon_lengthspring",
        "sensor_type",
        "hfield_data",
    ):
        np.testing.assert_allclose(
            getattr(restored, name), getattr(native, name), rtol=1e-12, atol=1e-15
        )
    for attribute, values in home.items():
        np.testing.assert_array_equal(
            getattr(restored, "key_" + attribute.removeprefix("default_"))[0], values
        )
    original_data, restored_data = mujoco.MjData(native), mujoco.MjData(restored)
    for data in (original_data, restored_data):
        data.qpos[:] = home["default_qpos"]
        data.qvel[:] = home["default_qvel"]
        data.ctrl[:] = home["default_ctrl"]
    for _ in range(30):
        mujoco.mj_step(native, original_data)
        mujoco.mj_step(restored, restored_data)
    np.testing.assert_allclose(restored_data.qpos, original_data.qpos, atol=1e-10)
    np.testing.assert_allclose(
        restored_data.sensordata, original_data.sensordata, atol=1e-10
    )


@pytest.mark.parametrize("failure_call", [1, 2])
def test_policy_publication_reports_file_and_directory_sync_failures(
    tmp_path, monkeypatch, failure_call
):
    calls = 0

    def fail_sync(_descriptor):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError("durability failed")

    monkeypatch.setattr(deployment.os, "fsync", fail_sync)
    output = tmp_path / "policy.onnx"
    with pytest.raises(OSError, match="durability failed"):
        deployment._publish_policy(output, b"complete model")
    if failure_call == 1:
        assert not output.exists()
    else:
        assert output.read_bytes() == b"complete model"
    assert not list(tmp_path.glob(".policy.onnx.*"))
