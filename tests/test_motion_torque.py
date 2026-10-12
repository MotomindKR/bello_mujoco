from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.motion_tracking import BelloMotionTrackingEnv


XML = """<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body><joint name="a"/><geom type="sphere" size=".1" mass="1"/></body>
    <body pos="1 0 0"><joint name="b"/><geom type="sphere" size=".1" mass="1"/></body>
  </worldbody>
  <tendon><fixed name="differential"><joint joint="a" coef=".5"/>
    <joint joint="b" coef="-.5"/></fixed></tendon>
  <actuator>
    <position joint="a" kp="25" kv="1.5" forcerange="-14 14"/>
    <position tendon="differential" kp="20" kv="2" forcerange="-10 10"/>
  </actuator>
</mujoco>"""


def environment(model, batched_model=None):
    env = object.__new__(BelloMotionTrackingEnv)
    object.__setattr__(
        env,
        "model",
        SimpleNamespace(
            mj_model=model,
            mjx_model=batched_model,
            policy_ctrl_indices=np.arange(model.nu),
            actuator_peak_force=np.max(np.abs(model.actuator_forcerange), axis=1),
        ),
    )
    return env


def costs(env, ctrl, length, velocity, force, world=0):
    state = SimpleNamespace(
        data=SimpleNamespace(
            ctrl=ctrl,
            actuator_length=length,
            actuator_force=force,
            _impl=SimpleNamespace(actuator_velocity=velocity),
        )
    )
    return env._torque_costs(state, world)


@pytest.mark.parametrize("target", [-2.0, -0.05, 0.05, 2.0])
@pytest.mark.parametrize("control_limited", [False, True])
def test_peak_cost_matches_unclamped_mujoco_with_clamped_applied_force(
    target,
    control_limited,
):
    model = mujoco.MjModel.from_xml_string(XML)
    model.actuator_biasprm[:, 0] = [1.7, -0.6]
    model.actuator_ctrllimited[:] = control_limited
    model.actuator_ctrlrange[:] = [-0.1, 0.1]
    data = mujoco.MjData(model)
    data.qpos[:] = [0.1, -0.2]
    data.qvel[:] = [0.7, -0.4]
    data.ctrl[:] = [target, -target]
    mujoco.mj_forward(model, data)
    applied = data.actuator_force.copy()
    env = environment(model)
    args = tuple(
        jnp.asarray(x.copy())
        for x in (
            data.ctrl,
            data.actuator_length,
            data.actuator_velocity,
            applied,
        )
    )
    measured = jax.jit(lambda *xs: costs(env, *xs))(*args)
    model.actuator_forcelimited[:] = False
    mujoco.mj_forward(model, data)
    peak = env.model.actuator_peak_force
    expected = np.mean(np.maximum(np.abs(data.actuator_force) / peak - 1, 0) ** 2)
    assert float(measured[0]) == pytest.approx(np.mean(np.abs(applied) / peak))
    assert float(measured[1]) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    if abs(target) == 2 and not control_limited:
        assert float(measured[0]) == pytest.approx(1)
        assert float(measured[1]) > 0


def test_peak_cost_uses_each_worlds_randomized_gains_under_jit_vmap():
    model = mujoco.MjModel.from_xml_string(XML)
    gains = np.stack([model.actuator_gainprm.copy()] * 2)
    biases = np.stack([model.actuator_biasprm.copy()] * 2)
    gains[1, :, 0] *= 2
    biases[1, :, 1] *= 2
    biases[1, :, 2] *= 0.5
    batch = SimpleNamespace(
        actuator_gainprm=jnp.asarray(gains),
        actuator_biasprm=jnp.asarray(biases),
        actuator_ctrllimited=model.actuator_ctrllimited,
        actuator_ctrlrange=model.actuator_ctrlrange,
    )
    env = environment(model, batch)
    inputs, expected = [], []
    for world in range(2):
        model.actuator_gainprm[:] = gains[world]
        model.actuator_biasprm[:] = biases[world]
        data = mujoco.MjData(model)
        data.qpos[:] = [0.1, -0.2]
        data.qvel[:] = [0.7, -0.4]
        data.ctrl[:] = [1, -1]
        model.actuator_forcelimited[:] = True
        mujoco.mj_forward(model, data)
        inputs.append(
            [
                x.copy()
                for x in (
                    data.ctrl,
                    data.actuator_length,
                    data.actuator_velocity,
                    data.actuator_force,
                )
            ]
        )
        model.actuator_forcelimited[:] = False
        mujoco.mj_forward(model, data)
        expected.append(
            np.mean(
                np.maximum(
                    np.abs(data.actuator_force) / env.model.actuator_peak_force - 1,
                    0,
                )
                ** 2
            )
        )
    arrays = [jnp.asarray(np.stack([row[i] for row in inputs])) for i in range(4)]
    _, peak = jax.jit(jax.vmap(lambda *xs: costs(env, *xs)))(
        *arrays,
        jnp.arange(2),
    )
    np.testing.assert_allclose(peak, expected, rtol=1e-6)
    assert peak[1] > peak[0]
