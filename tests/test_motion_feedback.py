"""Motion feedback/timing checks on synthetic data, without robot-config assertions."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.feedback import (
    ObservationLowPassFilterConfig,
    feedback_filter_alpha,
    validate_command_delay,
)
from bello_mujoco.envs.motion_tracking import (
    BelloMotionTrackingEnv,
    MotionTrackingEnvConfig,
    MotionObservationNoiseConfig,
    MotionObservationLatencyConfig,
    MotionRewardConfig,
    squared_norm_gaussian_reward,
    scalar_gaussian_reward,
)
from bello_mujoco.evaluation.motion import (
    make_native_motion_step,
    reset_native_motion_rollout,
)
from bello_mujoco.motion.clip import MotionClip


@pytest.fixture
def env():
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.005" gravity="0 0 0"/>
      <worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
        <body name="anchor" pos=".2 0 0"><joint name="hinge"/>
          <geom type="sphere" size=".05" mass=".1"/>
        </body></body></worldbody>
      <actuator><position joint="hinge" kp="1"/></actuator>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [model.body(name).id for name in ("root", "anchor")]
    clip = MotionClip(
        fps=50.0,
        joint_names=("hinge",),
        body_names=("root", "anchor"),
        qpos=jnp.tile(data.qpos, (4, 1)),
        qvel=jnp.tile(data.qvel, (4, 1)),
        body_pos=jnp.tile(data.xpos[ids], (4, 1, 1)),
        body_quat=jnp.tile(data.xquat[ids], (4, 1, 1)),
        body_lin_vel=jnp.zeros((4, 2, 3)),
        body_ang_vel=jnp.zeros((4, 2, 3)),
        foot_contacts=jnp.zeros((4, 2)),
    )
    result = object.__new__(BelloMotionTrackingEnv)
    for name, value in dict(
        model=SimpleNamespace(
            mj_model=model,
            policy_ctrl_indices=np.array([0]),
            policy_qpos_indices=np.array([7]),
            policy_qvel_indices=np.array([6]),
            default_joint_target=np.zeros(1),
        ),
        motion=clip,
        config=MotionTrackingEnvConfig(
            anchor_body_name="anchor",
            command_delay_seconds=0.01,
            observation_low_pass_filter=ObservationLowPassFilterConfig(enabled=True),
            observation_latency=MotionObservationLatencyConfig(
                enabled=True, delay_steps=(1, 1)
            ),
        ),
        _initial_ctrl=lambda _: jnp.zeros(1),
    ).items():
        object.__setattr__(result, name, value)
    return result


@pytest.mark.parametrize("mode", ["full_state", "proprioceptive"])
def test_actor_state_dependency_and_full_critic_under_jit_vmap(env, mode, monkeypatch):
    object.__setattr__(env, "config", replace(env.config, observation_mode=mode))
    monkeypatch.setattr(
        BelloMotionTrackingEnv, "end_effector_names", property(lambda _: ("anchor",))
    )
    data = mujoco.MjData(env.model.mj_model)
    state, _ = reset_native_motion_rollout(env, data)
    shift = jnp.array([0.3, -0.2, 0.7])
    changed_data = replace(
        state.data,
        qpos=state.data.qpos.at[:3].add(shift),
        qvel=state.data.qvel.at[:3].set(jnp.array([0.4, -0.3, 0.2])),
        xpos=state.data.xpos + shift,
    )
    changed = state.replace(
        data=changed_data,
        actor_feedback_history=jnp.tile(
            env._actor_feedback(changed_data),
            (env.maximum_observation_delay_steps + 1, 1),
        ),
    )
    states = jax.tree.map(lambda a, b: jnp.stack([a, b]), state, changed)
    actors = jax.jit(env.observation_batch)(states)
    critics = jax.jit(env.critic_observation_batch)(states)
    assert actors.shape == (2, env.observation_size)
    assert critics.shape == (2, env.critic_observation_size)
    assert not np.allclose(critics[0], critics[1])
    if mode == "proprioceptive":
        np.testing.assert_array_equal(actors[0], actors[1])
        assert env.observation_size == env._full_observation_layout.size - 6
    else:
        assert not np.allclose(actors[0], actors[1])
        np.testing.assert_array_equal(actors, critics[:, : env.observation_size])
    # Height termination is still ground-truth-based in training, for both modes.
    flags = jax.jit(jax.vmap(env.termination_flags))(states)
    np.testing.assert_array_equal(flags["done/height"], [False, True])
    clean_critics = np.asarray(critics)
    object.__setattr__(
        env,
        "config",
        replace(
            env.config,
            observation_noise=MotionObservationNoiseConfig(
                enabled=True, anchor_orientation=0.1, base_angular_velocity=0.2
            ),
        ),
    )
    noisy = jax.jit(env.observation_batch)(states, jax.random.PRNGKey(42))
    assert noisy.shape == actors.shape
    assert not np.allclose(noisy, actors)
    np.testing.assert_array_equal(env.critic_observation_batch(states), clean_critics)


def test_unknown_motion_observation_mode_rejected():
    with pytest.raises(ValueError, match="observation_mode"):
        MotionTrackingEnvConfig(observation_mode="unknown")


@pytest.mark.parametrize("mode", ["full_state", "proprioceptive"])
def test_motion_observation_mode_survives_strict_config_roundtrip(mode):
    from bello_mujoco.configuration import dataclass_from_dict
    from bello_mujoco.training.tracking import to_builtin

    config = MotionTrackingEnvConfig(observation_mode=mode)
    restored = dataclass_from_dict(
        MotionTrackingEnvConfig,
        to_builtin(config),
        strict=True,
        require_all_fields=True,
    )
    assert restored == config


def test_filter_retains_raw_positions_filters_velocity_and_normalizes_pose(env):
    data = mujoco.MjData(env.model.mj_model)
    mujoco.mj_forward(env.model.mj_model, data)
    previous = env._actor_feedback(data)
    data.qvel[:] = 1.0
    data.qpos[7] = 0.3
    data.qpos[3:7] = [np.cos(0.4), 0, 0, np.sin(0.4)]
    mujoco.mj_forward(env.model.mj_model, data)
    raw = env._actor_feedback(data)
    # Data arrays, not MjData, are the JIT boundary.
    view = SimpleNamespace(
        qpos=jnp.array(data.qpos),
        qvel=jnp.array(data.qvel),
        xpos=jnp.array(data.xpos),
        xquat=jnp.array(data.xquat),
    )
    output = jax.jit(
        lambda history: env._advance_actor_feedback_history(history, view)
    )(jnp.stack([previous, previous]))
    qpos, qvel, anchor_pos, anchor_quat = env._unpack_actor_feedback(output[0])
    alpha = float(feedback_filter_alpha(0.02, 10.0))
    np.testing.assert_allclose(qvel, alpha, atol=1e-6)
    assert float(qpos[7]) == pytest.approx(0.3)
    np.testing.assert_allclose(
        qpos[3:7], [np.cos(0.4 * alpha), 0, 0, np.sin(0.4 * alpha)], atol=1e-6
    )
    assert np.linalg.norm(anchor_quat) == pytest.approx(1.0)
    assert np.linalg.norm(anchor_pos - qpos[:3]) == pytest.approx(0.2)
    np.testing.assert_array_equal(output[1], previous)
    state = SimpleNamespace(
        data=view,
        actor_feedback_history=output,
        actor_feedback_delay_steps=jnp.array(0),
    )
    np.testing.assert_array_equal(env._delayed_actor_feedback(state), output[0])
    # No smoothing is applied when explicitly disabled.
    object.__setattr__(
        env,
        "config",
        replace(
            env.config, observation_low_pass_filter=ObservationLowPassFilterConfig()
        ),
    )
    np.testing.assert_allclose(
        env._advance_actor_feedback_history(output, view)[0], raw
    )


def test_pre_filter_noise_is_seeded_and_does_not_modify_physics(env):
    object.__setattr__(
        env,
        "config",
        replace(
            env.config,
            observation_low_pass_filter=ObservationLowPassFilterConfig(
                enabled=True, noise_before_filter=True
            ),
            observation_noise=MotionObservationNoiseConfig(
                enabled=True,
                base_linear_velocity=0.2,
                base_angular_velocity=0.3,
                joint_position=0.1,
                joint_velocity=0.4,
            ),
        ),
    )
    data = mujoco.MjData(env.model.mj_model)
    mujoco.mj_forward(env.model.mj_model, data)
    raw = env._actor_feedback(data)
    key = jax.random.PRNGKey(3)
    noisy = env._noisy_actor_feedback(data, key)
    np.testing.assert_array_equal(noisy, env._noisy_actor_feedback(data, key))
    assert not np.array_equal(raw, noisy)
    np.testing.assert_array_equal(raw, env._actor_feedback(data))


def test_native_motion_keeps_feedback_history_and_obeys_substep_delay(env, monkeypatch):
    object.__setattr__(env, "action_target", lambda state, action: action)
    object.__setattr__(env, "_reward", lambda *args: (jnp.array(0.0), {}))
    object.__setattr__(
        env,
        "termination_flags",
        lambda _: {"done/nonfinite": jnp.array(False), "done/any": jnp.array(False)},
    )
    data = mujoco.MjData(env.model.mj_model)
    state, _ = reset_native_motion_rollout(env, data, rng=jax.random.PRNGKey(1))
    commands = []
    real_step = mujoco.mj_step

    def step(model, current):
        commands.append(current.ctrl.copy())
        real_step(model, current)

    monkeypatch.setattr(mujoco, "mj_step", step)
    stepped, _, _, _, _ = make_native_motion_step(env)(data, state, jnp.ones(1))
    np.testing.assert_allclose(np.asarray(commands).ravel(), [0, 0, 1, 1])
    np.testing.assert_array_equal(
        stepped.actor_feedback_history[1], state.actor_feedback_history[0]
    )
    assert int(stepped.actor_feedback_delay_steps) == 1
    np.testing.assert_array_equal(
        env._delayed_actor_feedback(stepped), state.actor_feedback_history[0]
    )
    with jax.disable_jit():
        eager_data = mujoco.MjData(env.model.mj_model)
        initial, _ = reset_native_motion_rollout(
            env, eager_data, rng=jax.random.PRNGKey(1)
        )
        eager, *_ = make_native_motion_step(env)(eager_data, initial, jnp.ones(1))
    for compiled_value, eager_value in zip(
        jax.tree.leaves(stepped), jax.tree.leaves(eager), strict=True
    ):
        np.testing.assert_allclose(compiled_value, eager_value, atol=1e-6)


def test_native_contact_view_pads_without_changing_active_contacts(env):
    from bello_mujoco.evaluation.motion import native_motion_state

    real = mujoco.MjData(env.model.mj_model)
    mujoco.mj_forward(env.model.mj_model, real)
    data = SimpleNamespace(
        **{
            name: getattr(real, name)
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "xpos",
                "xquat",
                "cvel",
                "subtree_com",
                "actuator_force",
            )
        },
        ncon=3,
        nefc=5,
        contact=SimpleNamespace(
            geom=np.array([[0, 1]] * 3),
            dim=np.array([1, 3, 4]),
            efc_address=np.array([0, 2, 4]),
        ),
        efc_force=np.arange(5, dtype=np.float32),
    )
    state = native_motion_state(
        env,
        data,
        last_action=np.zeros(1),
        reference_time=jnp.array(0.0),
        alignment_xy=jnp.zeros(2),
        alignment_yaw=jnp.array(0.0),
        joint_default_offset=jnp.zeros(1),
        step_count=0,
        episode_return=0.0,
    )
    contacts = jax.jit(lambda value: value)(state.data)._impl
    np.testing.assert_array_equal(contacts.contact__geom[:3], data.contact.geom)
    np.testing.assert_array_equal(contacts.contact__efc_address[:, 0], [0, 2, 4, -1])
    np.testing.assert_array_equal(contacts.efc__force, [0, 1, 2, 3, 4, 0, 0, 0])
    np.testing.assert_array_equal(contacts.contact__dim, [1, 3, 4, 0])
    assert int(contacts.nacon[0]) == 3


@pytest.mark.parametrize("delay", [-0.01, 0.003, 0.02, float("nan")])
def test_invalid_command_delays_are_rejected(delay):
    with pytest.raises(ValueError, match="command_delay_seconds"):
        validate_command_delay(delay, 0.005, 4)


def test_tracking_exponent_uses_mean_squared_body_error_not_mean_reward():
    errors = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
    assert float(jax.jit(squared_norm_gaussian_reward)(errors, 2.0)) == pytest.approx(
        np.exp(-4.5 / 4)
    )
    assert float(scalar_gaussian_reward(jnp.array([0.0, 2.0]), 2.0)) == pytest.approx(
        np.exp(-0.5)
    )


def test_joint_limit_cost_sums_linear_soft_limit_excess_under_jit_and_vmap():
    env = SimpleNamespace(
        model=SimpleNamespace(
            policy_qpos_indices=np.array([0, 1]),
            policy_ctrl_indices=np.array([0, 1]),
            joint_target_low=np.array([-1.0, -2.0]),
            joint_target_high=np.array([1.0, 2.0]),
        ),
        config=SimpleNamespace(reward=MotionRewardConfig(joint_limit_soft_factor=0.9)),
    )

    def cost(position):
        return BelloMotionTrackingEnv._joint_limit_cost(
            env, SimpleNamespace(data=SimpleNamespace(qpos=position))
        )

    positions = jnp.array([[0.0, 0.0], [-1.0, 2.0], [5.0, -5.0]])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(cost))(positions), [0.0, 0.3, 7.3], atol=1e-6
    )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_motion_model_loaders_forward_physics_and_training_randomization(
    monkeypatch, compiled, training
):
    from bello_mujoco.envs import motion_tracking as module
    from bello_mujoco.sim import ModelRandomizationConfig, SimulationPhysicsConfig

    seen = []

    def loader(*args, **kwargs):
        seen.append(kwargs)
        return object()

    monkeypatch.setattr(module, "load_model", loader)
    monkeypatch.setattr(module, "load_compiled_model", loader)
    monkeypatch.setattr(module, "load_motion_clip", lambda *args: object())
    monkeypatch.setattr(BelloMotionTrackingEnv, "__post_init__", lambda self: None)
    config = MotionTrackingEnvConfig(
        simulation_physics=SimulationPhysicsConfig(timestep=0.004),
        model_randomization=ModelRandomizationConfig(
            enabled=True, torso_com_offset_range=(-0.008, 0.008)
        ),
    )
    if compiled:
        BelloMotionTrackingEnv.from_compiled(
            object(), config, actuator_config=object(), load_mjx=training
        )
    else:
        BelloMotionTrackingEnv.load(
            SimpleNamespace(actuator_config=object()), config, load_mjx=training
        )
    assert seen[0]["simulation_physics"] == config.simulation_physics
    assert seen[0]["model_randomization"] == config.model_randomization


@pytest.mark.parametrize("enforce_effort_limits", [False, True])
def test_motion_asset_validation_checks_contract_not_default_config(
    enforce_effort_limits,
):
    from bello_mujoco.actuators import ActuatorConfig, JointActuatorConfig
    from bello_mujoco.training.motion import _validate_motion_asset

    config = SimpleNamespace(
        target_total_mass_kg=None,
        enforce_policy_effort_limits=enforce_effort_limits,
        actuator_config=ActuatorConfig(
            joints={"hinge": JointActuatorConfig(mode="policy", soft_limit=(-0.5, 0.5))}
        ),
    )
    _validate_motion_asset(config)
    config.target_total_mass_kg = 3.0
    with pytest.raises(ValueError, match="canonical robot mass"):
        _validate_motion_asset(config)
    config.target_total_mass_kg = None
    config.actuator_config = ActuatorConfig(joints={"hinge": JointActuatorConfig()})
    with pytest.raises(ValueError, match="all actuators in policy mode"):
        _validate_motion_asset(config)
