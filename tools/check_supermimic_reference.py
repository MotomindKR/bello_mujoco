"""Compare complete CPU learning updates against the pinned vibed implementation.

Uses recorded G1 poses, differentiated velocities and recomputed checkpoint actions.
This checks learning arithmetic on identical batches, not simulator equivalence.
"""

import argparse
import ast
import copy
from dataclasses import dataclass, replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch

from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.sonic.corpus import file_sha256
from bello_mujoco.supermimic.config import LearningConfig, ObservationConfig
from bello_mujoco.supermimic.kinematics import Kinematics
from bello_mujoco.supermimic.mdp import TrackingConfig, TrackingMDP, critic_features
from bello_mujoco.supermimic.model import (
    Actor,
    Critic,
    Moments,
    NetworkConfig,
    WorldModel,
    state_delta,
)
from bello_mujoco.supermimic.observations import ObservationCodec
from bello_mujoco.supermimic.optimization import (
    LearnerState,
    optimizers,
    learning_rate,
    update_actor_critic,
    update_world_model,
)
from bello_mujoco.supermimic.replay import ReplayStart, ReplayWindow, Transition
from bello_mujoco.supermimic.state import PhysicalState

REVISION = "a1508d42c9d1699a52e8233ab16d08c488441ab9"


@dataclass(frozen=True)
class Inputs:
    reference_checkout: Path
    recording: Path
    checkpoint: Path
    output: Path


def run(inputs):
    base = inputs.reference_checkout.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(base), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        revision != REVISION
        or subprocess.check_output(
            ["git", "-C", str(base), "status", "--porcelain"], text=True
        ).strip()
    ):
        raise ValueError("Reference checkout must be clean at the pinned revision")
    torch.set_num_threads(2)
    torch.manual_seed(4)
    checks = {}
    tolerances = {}
    parameter_differences = {}

    def tensor(value):
        return torch.tensor(np.asarray(value).copy(), dtype=torch.float32)

    def compare(name, ours, reference, *, atol=3e-5, rtol=3e-4):
        if isinstance(reference, torch.Tensor):
            reference = reference.detach().numpy()
        ours, reference = np.asarray(ours), np.asarray(reference)
        if not np.isfinite(ours).all() or not np.isfinite(reference).all():
            raise AssertionError(f"Nonfinite reference comparison: {name}")
        np.testing.assert_allclose(ours, reference, atol=atol, rtol=rtol, err_msg=name)
        checks[name] = float(np.max(np.abs(ours - reference)))
        tolerances[name] = {"atol": atol, "rtol": rtol}

    package = ModuleType("checked_reference")
    package.__path__ = [str(base / "supermimic")]
    sys.modules[package.__name__] = package
    robot = ModuleType("checked_reference.robot")
    tree = ast.parse((base / "supermimic/robot.py").read_text())
    constants = {
        "G1_JOINT_NAMES",
        "TRACKED_BODIES",
        "DEFAULT_JOINT_POS",
        "JOINT_MOTORS",
        "NATURAL_FREQ",
        "DAMPING_RATIO",
        "_M5020",
        "_M4010",
    }
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name)
            and (t.id in constants or t.id.startswith("ARMATURE_"))
            for t in node.targets
        ):
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    "reference_constants",
                    "exec",
                ),
                robot.__dict__,
            )
        if isinstance(node, ast.FunctionDef) and node.name in {
            "default_joint_pos_list",
            "motor_of",
            "action_scale",
            "_kp",
            "_kd",
        }:
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    "reference_defaults",
                    "exec",
                ),
                robot.__dict__,
            )
    robot.MJCF_PATH = str(base / "data/g1_description/g1_29dof.xml")
    sys.modules[robot.__name__] = robot
    tm = importlib.import_module("checked_reference.mdp")
    tn = importlib.import_module("checked_reference.networks")
    tr = importlib.import_module("checked_reference.rotations")
    kin = importlib.import_module("checked_reference.kinematics").G1Kinematics("cpu")
    motion_type = importlib.import_module("checked_reference.motion").MotionLib
    saved = torch.load(inputs.checkpoint, map_location="cpu", weights_only=True)
    k, hist = saved["args"]["goal_frames"], saved["args"]["hist"]
    n = len(robot.G1_JOINT_NAMES)
    with np.load(inputs.recording, allow_pickle=False) as stream:
        record = dict(stream)
    files = [
        str(base / "data/lafan1_g1" / (str(name) + ".csv")) for name in record["names"]
    ]
    motion = motion_type(files, kin, "cpu")
    compare("reference_joint_data", motion.q.numpy(), record["ref_q"], atol=1e-7)
    default = torch.tensor(robot.default_joint_pos_list())
    action_scales = tensor([robot.action_scale(name) for name in robot.G1_JOINT_NAMES])
    mdp = tm.MDP(kin, motion, default)
    mdp.cfg.body_angvel_std = np.pi
    mdp.obs_noise.zero_()
    native = mujoco.MjModel.from_xml_path(robot.MJCF_PATH)
    jkin = Kinematics.from_model(
        native, robot.G1_JOINT_NAMES, robot.TRACKED_BODIES, root_body_name="pelvis"
    )
    codec = ObservationCodec(
        ObservationConfig(history_frames=hist, future_frames=k),
        jnp.asarray(default.numpy()),
    )
    all_reference = PhysicalState(
        jnp.asarray(motion.root_p.numpy()),
        jnp.asarray(tr.mat_to_quat(motion.root_R).numpy()),
        *[
            jnp.asarray(x.numpy())
            for x in (motion.root_v, motion.root_w, motion.q, motion.qd)
        ],
    )

    def reference(frame, xy=None, yaw=None):
        return jax.tree.map(lambda x: x[frame.astype(jnp.int32)], all_reference)

    def goal(frame):
        index = jnp.minimum(
            frame.astype(jnp.int32) + jnp.arange(1, k + 1),
            jnp.asarray(motion.clip_end.numpy())[
                jnp.asarray(motion.clip_id.numpy())[frame.astype(jnp.int32)]
            ],
        )
        return codec.goal(
            reference(frame), jax.tree.map(lambda x: x[index], all_reference)
        )

    batch, start, length = 3, 32, 8
    if not record["alive"][: start + length + 1, :batch].all():
        raise ValueError("Recording needs three live initial trajectories")
    rotations = tr.quat_to_mat(tensor(record["root_quat"][:, :batch]))

    def measured(t):
        lo, hi = max(0, t - 1), t + 1
        dt = (hi - lo) * 0.02
        return (
            tensor(record["root_p"][t, :batch]),
            rotations[t],
            tensor((record["root_p"][hi, :batch] - record["root_p"][lo, :batch]) / dt),
            tr.so3_log(rotations[hi] @ rotations[lo].transpose(-1, -2)) / dt,
            tensor(record["q"][t, :batch]),
            tensor((record["q"][hi, :batch] - record["q"][lo, :batch]) / dt),
        )

    def physical(state):
        return PhysicalState(
            jnp.asarray(state[0].numpy()),
            jnp.asarray(tr.mat_to_quat(state[1]).numpy()),
            *[jnp.asarray(x.numpy()) for x in state[2:]],
        )

    pi, wm = tn.Policy(tm.OBS_DIM, codec.goal_size, hist=hist), tn.WorldModel()
    pi.load_state_dict(saved["pi"])
    wm.load_state_dict(saved["wm"])
    tn.AMP_DTYPE = None
    xi = torch.zeros(batch, n + 4)
    xi[:, 0] = 1
    context = jnp.asarray(
        torch.cat(
            (xi[:, 1 : 1 + n], xi[:, :1], xi[:, 1 + n :], torch.zeros(batch, 1)), -1
        ).numpy()
    )
    previous = torch.zeros(batch, n)
    state_history, action_history, obs_history = [], [], []
    history = None
    with torch.no_grad():
        for t in range(start + length + 1):
            state = measured(t)
            frame = torch.as_tensor(record["clip_start"][:batch] + t)
            obs = mdp.obs(state, previous, xi, frame)
            history = (
                obs[:, None].repeat(1, hist, 1)
                if history is None
                else torch.cat((history[:, 1:], obs[:, None]), 1)
            )
            state_history.append(state)
            obs_history.append(history.clone())
            previous = pi(history, motion.goal(frame, k))
            action_history.append(previous)
    frames = torch.as_tensor(record["clip_start"][:batch] + start)
    state = state_history[start]
    js = physical(state)
    history = obs_history[start]
    previous = action_history[start - 1]
    jref = jax.vmap(reference)(jnp.asarray(frames.numpy()))
    goals = motion.goal(frames, k)
    jfeatures = jax.vmap(lambda s, r, g, c: critic_features(s, r, g, c, jkin))(
        js, jref, jnp.asarray(goals.numpy()), context
    )
    before_context = 13 + 3 * n + 3 * len(robot.TRACKED_BODIES) + 9
    cmap = (
        list(range(before_context))
        + [before_context + n]
        + list(range(before_context, before_context + n))
        + list(range(before_context + n + 1, before_context + n + 4))
        + list(range(before_context + n + 5, jfeatures.shape[-1]))
    )
    before_context = 13 + 3 * n
    fmap = (
        list(range(before_context))
        + [before_context + n]
        + list(range(before_context, before_context + n))
        + list(range(before_context + n + 1, before_context + n + 4))
    )
    networks = NetworkConfig()
    ja = Actor(
        codec.sample_size,
        hist,
        codec.goal_size,
        n,
        config=networks,
        key=jax.random.key(0),
    )
    jw = WorldModel(n, n + 5, config=networks, key=jax.random.key(1))
    critic = tn.Critic(len(cmap))
    jc = Critic(jfeatures.shape[-1], config=networks, key=jax.random.key(2))

    def copy_mlp(ours, source, indices=None):
        linears = [m for m in source if isinstance(m, torch.nn.Linear)]
        layers = []
        for i, (layer, original) in enumerate(zip(ours.layers, linears, strict=True)):
            weight = original.weight.detach().numpy()
            if i == 0 and indices is not None:
                expanded = np.zeros(layer.weight.shape, np.float32)
                expanded[:, indices] = weight
                weight = expanded
            layers.append(
                eqx.tree_at(
                    lambda x: (x.weight, x.bias),
                    layer,
                    (jnp.asarray(weight), jnp.asarray(original.bias.detach().numpy())),
                )
            )
        result = eqx.tree_at(lambda x: x.layers, ours, tuple(layers))
        if ours.norms:
            norms = tuple(
                eqx.tree_at(
                    lambda x: (x.weight, x.bias),
                    layer,
                    (
                        jnp.asarray(original.weight.detach().numpy()),
                        jnp.asarray(original.bias.detach().numpy()),
                    ),
                )
                for layer, original in zip(
                    ours.norms,
                    [m for m in source if isinstance(m, torch.nn.LayerNorm)],
                    strict=True,
                )
            )
            result = eqx.tree_at(lambda x: x.norms, result, norms)
        return result

    def moments(source, size, indices=None):
        source.count.fill_(1024.0)
        indices = np.arange(size) if indices is None else indices
        mean, variance = np.zeros(size, np.float32), np.ones(size, np.float32)
        mean[indices], variance[indices] = source.mean.numpy(), source.var.numpy()
        return Moments(jnp.asarray(mean), jnp.asarray(variance), jnp.asarray(1024.0))

    ja = eqx.tree_at(
        lambda x: (x.encoder, x.decoder),
        ja,
        (copy_mlp(ja.encoder, pi.enc), copy_mlp(ja.decoder, pi.dec)),
    )
    jw = eqx.tree_at(lambda x: x.network, jw, copy_mlp(jw.network, wm.net, fmap))
    jc = eqx.tree_at(lambda x: x.network, jc, copy_mlp(jc.network, critic.net, cmap))
    cfg = LearningConfig(
        num_envs=batch,
        total_timesteps=300,
        warmup_steps=8,
        replay_steps=64,
        model_warmup_updates=0,
        model_batches=2,
        model_batch_size=batch,
        imagined_batch_size=batch,
        imagination_horizon=32,
        critic_epochs=5,
        critic_minibatches=4,
    )
    actor_opt, model_opt, critic_opt = optimizers(cfg)
    learner = LearnerState(
        ja,
        jw,
        jc,
        actor_opt.init(eqx.filter(ja, eqx.is_inexact_array)),
        model_opt.init(eqx.filter(jw, eqx.is_inexact_array)),
        critic_opt.init(eqx.filter(jc, eqx.is_inexact_array)),
        moments(wm.in_norm, jw.input_size, fmap),
        moments(wm.out_norm, 12 + 2 * n),
        moments(pi.obs_norm, codec.sample_size),
        moments(pi.goal_norm, codec.goal_size),
        moments(critic.norm, jfeatures.shape[-1], cmap),
    )
    ids = jnp.zeros(batch, jnp.int32)
    zeros = jnp.zeros((batch, n))
    transition = Transition(
        js,
        js,
        js,
        zeros,
        zeros,
        jnp.asarray(action_history[start].numpy()),
        jnp.asarray(previous.numpy()),
        jnp.asarray(history[:, -1].numpy()),
        context,
        jnp.zeros((batch, 2)),
        jnp.zeros(batch),
        jnp.asarray(frames.numpy(), dtype=jnp.float32),
        ids,
        ids,
        ids,
        jnp.full(batch, 100),
        jnp.zeros(batch),
        jnp.zeros(batch, bool),
        jnp.zeros(batch, bool),
    )
    states = jax.tree.map(
        lambda *x: jnp.stack(x, 1),
        *[physical(s) for s in state_history[start : start + length]],
    )
    successors = jax.tree.map(
        lambda *x: jnp.stack(x, 1),
        *[physical(s) for s in state_history[start + 1 : start + length + 1]],
    )
    actions = jnp.asarray(
        torch.stack(action_history[start : start + length], 1).numpy()
    )
    contexts = jnp.repeat(context[:, None], length, axis=1)
    window = ReplayWindow(
        replace(
            jax.tree.map(lambda x: jnp.repeat(x[:, None], length, axis=1), transition),
            state=states,
            successor=successors,
            action=actions,
            context=contexts,
        ),
        jnp.ones((batch, length)),
    )
    starts = ReplayStart(
        transition,
        jnp.asarray(history.numpy()),
        jax.tree.map(lambda x: x[:, None], js),
        jnp.asarray(True),
    )
    jmdp = TrackingMDP(
        jkin,
        jnp.asarray(kin.q_lo.numpy()),
        jnp.asarray(kin.q_hi.numpy()),
        0.02,
        TrackingConfig(),
    )
    feedback_config = MotionTrackingEnvConfig()
    feedback_config = replace(
        feedback_config,
        observation_noise=replace(feedback_config.observation_noise, enabled=False),
        observation_low_pass_filter=replace(
            feedback_config.observation_low_pass_filter, enabled=False
        ),
        observation_latency=replace(feedback_config.observation_latency, enabled=False),
    )
    env = SimpleNamespace(
        kinematics=jkin,
        codec=codec,
        config=feedback_config,
        mdp=jmdp,
        reference_state=reference,
        reference_bodies=lambda frame, xy, yaw: jkin.forward(reference(frame)),
        goal=goal,
        advance_reference_time=lambda f: f + 1,
        control_timestep=0.02,
        clip_id=lambda f: (
            jnp.searchsorted(jnp.asarray(motion.clip_start.numpy()), f, side="right")
            - 1
        ),
        reference_bank=SimpleNamespace(
            time_offsets=jnp.asarray(motion.clip_start.numpy()),
            lengths=jnp.asarray(motion.clip_len.numpy()),
        ),
    )
    reference_window = {
        "s": tensor(states.flatten()),
        "s1": tensor(successors.flatten()),
        "a": tensor(actions),
        "xi": xi[:, None].repeat(1, length, 1),
        "mask": tensor(window.mask),
    }
    hstates = jax.tree.map(
        lambda *x: jnp.stack(x, 1),
        *[physical(s) for s in state_history[start - hist + 1 : start + 1]],
    )
    hprevious = torch.stack(action_history[start - hist : start], 1)
    hframes = frames[:, None] + torch.arange(-hist + 1, 1)
    reference_start = {
        "s": tm.flatten_state(state),
        "a_prev": previous,
        "xi": xi,
        "f": frames,
    }
    reference_history = {
        "s": tensor(hstates.flatten()),
        "a_prev": hprevious,
        "xi": xi[:, None].repeat(1, hist, 1),
        "f": hframes,
    }
    definitions = {
        "_wm_rollout",
        "_wm_err",
        "_imag_step",
        "_reward_term",
        "_critic_input",
        "set_requires_grad",
        "wm_update",
        "policy_update",
    }
    nodes = [
        node
        for node in ast.parse((base / "supermimic/train.py").read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in definitions
    ]
    scope = dict(
        torch=torch,
        nn=torch.nn,
        wm=wm,
        pi=pi,
        critic=critic,
        mdp=mdp,
        motion=motion,
        K=k,
        Hh=hist,
        HZ=32,
        dev="cpu",
        PROFILE=False,
        unflatten_state=tm.unflatten_state,
        normalize_xi=tm.normalize_xi,
        args=SimpleNamespace(
            wm_window=length,
            pi_batch=batch,
            no_critic=False,
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
            critic_epochs=cfg.critic_epochs,
            critic_minibatches=cfg.critic_minibatches,
        ),
        buffer=SimpleNamespace(sample_windows=lambda *args: reference_window),
        sample_valid_starts=lambda *args: (reference_start, reference_history),
    )
    scope["args"].wm_batch = batch
    scope["ERR_FLOOR"] = tensor(
        np.concatenate(
            ([0.003] * 3, [0.005] * 3, [0.02] * 3, [0.05] * 3, [0.003] * n, [0.05] * n)
        )
    )
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(base / "supermimic/train.py"),
            "exec",
        ),
        scope,
    )
    scope.update(
        wm_rollout=scope["_wm_rollout"],
        wm_err=scope["_wm_err"],
        imag_step=scope["_imag_step"],
        reward_term=scope["_reward_term"],
        critic_input_c=scope["_critic_input"],
    )

    def record_parameter(name, ours, source):
        a, b = np.asarray(ours), source.detach().numpy()
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(f"Nonfinite parameters: {name}")
        parameter_differences[name] = {
            "max_abs": float(np.max(np.abs(a - b))),
            "rms": float(np.sqrt(np.mean((a - b) ** 2))),
        }

    def record_mlp(name, ours, source, indices=None):
        for i, (layer, original) in enumerate(
            zip(
                ours.layers,
                [m for m in source if isinstance(m, torch.nn.Linear)],
                strict=True,
            )
        ):
            weight = (
                layer.weight[:, indices]
                if i == 0 and indices is not None
                else layer.weight
            )
            record_parameter(f"{name}/weight_{i}", weight, original.weight)
            record_parameter(f"{name}/bias_{i}", layer.bias, original.bias)
        for i, (layer, original) in enumerate(
            zip(
                ours.norms,
                [m for m in source if isinstance(m, torch.nn.LayerNorm)],
                strict=True,
            )
        ):
            record_parameter(f"{name}/norm_weight_{i}", layer.weight, original.weight)
            record_parameter(f"{name}/norm_bias_{i}", layer.bias, original.bias)

    fit = eqx.filter_jit(update_world_model)
    improve = eqx.filter_jit(
        lambda learner, starts, update, key: update_actor_critic(
            learner, starts, env, cfg, update, key
        )
    )

    class MatchedTorch:
        def __init__(self, permutations):
            self.permutations = iter(permutations)

        def __getattr__(self, name):
            return getattr(torch, name)

        def randperm(self, size, *, device):
            result = next(self.permutations)
            assert len(result) == size
            return torch.as_tensor(np.asarray(result).copy(), device=device)

    def compare_optimizer(name, ours, original, pairs):
        adam = next(state for state in ours if hasattr(state, "mu"))
        for label, jmu, jnu, source, indices in pairs(adam):
            modules = [m for m in source if isinstance(m, torch.nn.Linear)]
            norms = [m for m in source if isinstance(m, torch.nn.LayerNorm)]
            for i, (mu, nu, module) in enumerate(
                zip(jmu.layers, jnu.layers, modules, strict=True)
            ):
                for parameter in ("weight", "bias"):
                    a, b = getattr(mu, parameter), getattr(nu, parameter)
                    if i == 0 and parameter == "weight" and indices is not None:
                        a, b = a[:, indices], b[:, indices]
                    state = original.state[getattr(module, parameter)]
                    compare(
                        f"{name}/{label}/{i}/{parameter}_mu",
                        a,
                        state["exp_avg"],
                        atol=1e-5,
                        rtol=5e-3,
                    )
                    compare(
                        f"{name}/{label}/{i}/{parameter}_nu",
                        b,
                        state["exp_avg_sq"],
                        atol=1e-7,
                        rtol=1e-2,
                    )
                    compare(f"{name}/step", adam.count, state["step"])
            for i, (mu, nu, module) in enumerate(
                zip(jmu.norms, jnu.norms, norms, strict=True)
            ):
                for parameter in ("weight", "bias"):
                    state = original.state[getattr(module, parameter)]
                    compare(
                        f"{name}/{label}/norm_{i}/{parameter}_mu",
                        getattr(mu, parameter),
                        state["exp_avg"],
                        atol=1e-5,
                        rtol=5e-3,
                    )
                    compare(
                        f"{name}/{label}/norm_{i}/{parameter}_nu",
                        getattr(nu, parameter),
                        state["exp_avg_sq"],
                        atol=1e-7,
                        rtol=1e-2,
                    )

    initial_learner = learner
    initial_reference = copy.deepcopy(
        {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict()}
    )
    for cycle in range(3):
        # Each case starts from identical state. Independent floating-point
        # trajectories are not identical inputs for an optimizer parity check.
        learner = initial_learner
        for name, module in (("wm", wm), ("pi", pi), ("critic", critic)):
            module.load_state_dict(initial_reference[name])
        scope.update(
            opt_wm=torch.optim.Adam(wm.parameters(), lr=cfg.learning_rate),
            opt_pi=torch.optim.Adam(
                pi.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95)
            ),
            opt_v=torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate),
        )
        # Match the once-per-collected-transition normalization phase explicitly.
        obs = mdp.obs(state, previous, xi, frames)
        pi.obs_norm.update(obs)
        pi.goal_norm.update(goals)
        wm.in_norm.update(wm.inputs(state, tensor(actions[:, 0]), tm.normalize_xi(xi)))
        wm.out_norm.update(wm.targets(state, state_history[start + 1]))
        learner = replace(
            learner,
            observation_norm=learner.observation_norm.update(jnp.asarray(obs.numpy())),
            goal_norm=learner.goal_norm.update(jnp.asarray(goals.numpy())),
            input_norm=learner.input_norm.update(
                learner.model.features(js, actions[:, 0], context)
            ),
            output_norm=learner.output_norm.update(
                state_delta(js, physical(state_history[start + 1]))
            ),
        )
        for opt in (scope["opt_wm"], scope["opt_pi"], scope["opt_v"]):
            for group in opt.param_groups:
                group["lr"] = float(learning_rate(cfg, cycle))
        for batch_index in range(cfg.model_batches):
            expected, _ = scope["wm_update"]()
            learner, metrics = fit(learner, window, cfg, jnp.asarray(cycle))
            compare(
                f"cycle_{cycle}/model_loss_{batch_index}",
                metrics["model/loss"],
                expected,
                atol=2e-4,
            )
        key = jax.random.key(cycle)
        _, critic_key = jax.random.split(key)
        permutations = jax.vmap(lambda rng: jax.random.permutation(rng, 32 * batch))(
            jax.random.split(critic_key, cfg.critic_epochs)
        )
        scope["torch"] = MatchedTorch(permutations)
        expected = scope["policy_update"]()
        learner, metrics = improve(learner, starts, jnp.asarray(cycle), key)
        compare(
            f"cycle_{cycle}/actor_loss", metrics["actor/loss"], expected[0], atol=2e-4
        )
        for name, ours, original, indices in [
            ("model_input", learner.input_norm, wm.in_norm, fmap),
            ("model_output", learner.output_norm, wm.out_norm, None),
            ("observation", learner.observation_norm, pi.obs_norm, None),
            ("goal", learner.goal_norm, pi.goal_norm, None),
            ("critic", learner.critic_norm, critic.norm, cmap),
        ]:
            compare(
                f"cycle_{cycle}/{name}_mean",
                ours.mean if indices is None else ours.mean[jnp.asarray(indices)],
                original.mean,
            )
            compare(
                f"cycle_{cycle}/{name}_var",
                ours.variance
                if indices is None
                else ours.variance[jnp.asarray(indices)],
                original.var,
            )
        record_mlp(f"cycle_{cycle}/wm", learner.model.network, wm.net, fmap)
        record_mlp(f"cycle_{cycle}/actor_encoder", learner.actor.encoder, pi.enc)
        record_mlp(f"cycle_{cycle}/actor_decoder", learner.actor.decoder, pi.dec)
        record_mlp(f"cycle_{cycle}/critic", learner.critic.network, critic.net, cmap)
        with torch.no_grad():
            predicted = wm.step(state, tensor(actions[:, 0]), tm.normalize_xi(xi))
            actor_output = pi(history, goals)
            critic_output = critic(
                torch.cat(
                    (mdp.critic_features(state, frames, tm.normalize_xi(xi)), goals), -1
                )
            )
        compare(
            f"cycle_{cycle}/model_output",
            learner.model.step(
                js, actions[:, 0], context, learner.input_norm, learner.output_norm
            ).flatten(),
            tm.flatten_state(predicted),
        )
        compare(
            f"cycle_{cycle}/actor_pd_target_rad",
            learner.actor(
                jnp.asarray(history.numpy()),
                jnp.asarray(goals.numpy()),
                learner.observation_norm,
                learner.goal_norm,
            )
            * jnp.asarray(action_scales.numpy()),
            actor_output * action_scales,
            atol=2e-3,
            rtol=0,
        )
        compare(
            f"cycle_{cycle}/critic_output",
            learner.critic(jfeatures, learner.critic_norm),
            critic_output,
            atol=5e-4,
            rtol=1e-3,
        )
        compare(
            f"cycle_{cycle}/actor_gradient_norm",
            metrics["actor/gradient_norm"],
            expected[3],
            atol=1e-4,
            rtol=1e-3,
        )
        compare_optimizer(
            f"cycle_{cycle}/wm_adam",
            learner.model_optimizer,
            scope["opt_wm"],
            lambda s: [("model", s.mu.network, s.nu.network, wm.net, fmap)],
        )
        compare_optimizer(
            f"cycle_{cycle}/actor_adam",
            learner.actor_optimizer,
            scope["opt_pi"],
            lambda s: [
                ("encoder", s.mu.encoder, s.nu.encoder, pi.enc, None),
                ("decoder", s.mu.decoder, s.nu.decoder, pi.dec, None),
            ],
        )
        compare_optimizer(
            f"cycle_{cycle}/critic_adam",
            learner.critic_optimizer,
            scope["opt_v"],
            lambda s: [("critic", s.mu.network, s.nu.network, critic.net, cmap)],
        )
        print(
            json.dumps(
                {
                    "case": cycle,
                    "actor_loss": float(metrics["actor/loss"]),
                    "checks": len(checks),
                }
            ),
            flush=True,
        )
    report = {
        "status": "passed_numerical_checks",
        "reference_commit": revision,
        "checks": checks,
        "tolerances": tolerances,
        "parameter_differences": parameter_differences,
        "scope": "Three independent full-width CPU update cases on recorded G1 poses with centrally differentiated velocities and recomputed FP32 checkpoint actions. Each case starts from the same copied network/normalizer state and fresh Adam states, and runs two model batches, a 32-step actor update and five critic epochs of four matched minibatches. Critic permutations differ between cases but match between frameworks. Noise disabled. Asserts numerical agreement for losses, network outputs, Adam moments/counts and normalization statistics. Actor outputs are compared in G1 PD-target radians with a 0.002rad absolute tolerance. Parameter differences are reported separately; near-zero FP32 gradients can be amplified by Adam epsilon. Initial normalization counts are 1024 so collected data measurably changes statistics. This is not a free-running multi-update training comparison, simulator equivalence test or learning-quality result.",
        "checkpoint_sha256": file_sha256(inputs.checkpoint),
        "recording_sha256": file_sha256(inputs.recording),
        "versions": {"jax": jax.__version__, "torch": torch.__version__},
        "tracking_config": vars(jmdp.config),
        "reference_reward_override": {"body_angvel_std": float(np.pi)},
        "ours_sources_sha256": {
            name: file_sha256(Path(__file__).resolve().parents[1] / name)
            for name in (
                "src/bello_mujoco/supermimic/model.py",
                "src/bello_mujoco/supermimic/optimization.py",
                "src/bello_mujoco/supermimic/imagination.py",
                "src/bello_mujoco/supermimic/mdp.py",
                "src/bello_mujoco/supermimic/control.py",
                "tools/check_supermimic_reference.py",
            )
        },
    }
    inputs.output.parent.mkdir(parents=True, exist_ok=True)
    inputs.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-checkout", "recording", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(Inputs(**vars(parser.parse_args())))
