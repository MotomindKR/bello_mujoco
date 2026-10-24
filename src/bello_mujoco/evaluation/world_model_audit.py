"""Contact-stratified open-loop model validation against native policy rollouts."""

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.configuration import command_config, config_path
from bello_mujoco.evaluation.motion import (
    make_native_motion_step,
    reset_native_motion_rollout,
)
from bello_mujoco.evaluation.reference_audit import ground_contact_bodies
from bello_mujoco.sonic.corpus import MANIFEST
from bello_mujoco.sonic.reference import ReferenceBank
from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.checkpoint import (
    checkpoint_metadata,
    load_actor,
    load_dynamics,
)
from bello_mujoco.supermimic.control import motor_limit_flags
from bello_mujoco.supermimic.sampling import ReturnSampler
from bello_mujoco.training.tracking import to_builtin


@dataclass(frozen=True)
class MotionWindow:
    clip: str
    start_frame: int = 0
    max_steps: int = 400

    def __post_init__(self):
        if not self.clip or Path(self.clip).name != self.clip or Path(self.clip).suffix:
            raise ValueError("window clip must be a flat extensionless corpus name")
        if type(self.start_frame) is not int or self.start_frame < 0:
            raise ValueError("start_frame must be a nonnegative integer")
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")


@dataclass(frozen=True)
class WorldModelAuditConfig:
    checkpoint_path: Path
    motion_dataset: Path
    output_path: Path
    windows: tuple[MotionWindow, ...]
    horizons: tuple[int, ...] = (1, 8, 32)
    anchor_stride: int = 32
    max_windows_per_horizon: int = 32

    def __post_init__(self):
        identities = [(w.clip, w.start_frame) for w in self.windows]
        if not identities or len(set(identities)) != len(identities):
            raise ValueError(
                "windows must be nonempty with unique clip/start_frame pairs"
            )
        if (
            not self.horizons
            or len(set(self.horizons)) != len(self.horizons)
            or any(type(h) is not int or h <= 0 for h in self.horizons)
        ):
            raise ValueError("horizons must be unique positive integers")
        for name in ("anchor_stride", "max_windows_per_horizon"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@eqx.filter_jit
def rollout_predictions(
    model, initial, previous_motor, motor, context, input_norm, output_norm
):
    controls = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), (previous_motor, motor, context)
    )

    def step(state, values):
        predicted = model.step(state, *values, input_norm, output_norm)
        return predicted, None

    predicted, _ = jax.lax.scan(step, initial, controls)
    return predicted


def prediction_errors(predicted, actual):
    return {
        "root_position_m": jnp.linalg.norm(
            predicted.position - actual.position, axis=-1
        ),
        "root_orientation_rad": jnp.linalg.norm(
            rot.log(
                rot.multiply(rot.conjugate(actual.quaternion), predicted.quaternion)
            ),
            axis=-1,
        ),
        "root_linear_velocity_m_s": jnp.linalg.norm(
            predicted.linear_velocity - actual.linear_velocity, axis=-1
        ),
        "root_angular_velocity_rad_s": jnp.linalg.norm(
            predicted.angular_velocity - actual.angular_velocity, axis=-1
        ),
        "joint_position_rms_rad": jnp.sqrt(
            jnp.mean((predicted.joint_position - actual.joint_position) ** 2, axis=-1)
        ),
        "joint_velocity_rms_rad_s": jnp.sqrt(
            jnp.mean((predicted.joint_velocity - actual.joint_velocity) ** 2, axis=-1)
        ),
    }


def contact_groups(signatures, anchors, horizon):
    changed = np.array(
        [
            any(signatures[i] != signatures[a] for i in range(a + 1, a + horizon + 1))
            for a in anchors
        ],
        dtype=bool,
    )
    airborne = np.array(
        [not any(signatures[i] for i in range(a, a + horizon + 1)) for a in anchors],
        dtype=bool,
    )
    return {
        "all": np.ones(len(anchors), dtype=bool),
        "ground_contact_change": changed,
        "ground_contact_stable": ~changed,
        "airborne": airborne,
    }


def summarize_prediction_errors(errors, mask):
    finite = np.logical_and.reduce([np.isfinite(value) for value in errors.values()])
    selected = mask & finite
    return {
        "windows": int(mask.sum()),
        "nonfinite_windows": int((mask & ~finite).sum()),
        "finite_windows": int(selected.sum()),
        "errors": {
            name: {
                "mean": float(value[selected].mean()),
                "p95": float(np.quantile(value[selected], 0.95)),
                "max": float(value[selected].max()),
            }
            if selected.any()
            else None
            for name, value in errors.items()
        },
    }


def window_environment(env, window):
    source = env.motion
    if window.start_frame >= source.num_frames - 1:
        raise ValueError(f"window start is outside {window.clip}")
    motion = replace(
        source,
        **{
            name: getattr(source, name)[window.start_frame :]
            for name in (
                "qpos",
                "qvel",
                "body_pos",
                "body_quat",
                "body_lin_vel",
                "body_ang_vel",
                "foot_contacts",
            )
        },
    )
    bank = ReferenceBank.from_clips(((window.clip, motion, "train"),))
    sampler = ReturnSampler.from_bank(bank)
    return replace(
        env,
        motion=motion,
        reference_bank=bank,
        sampler=sampler,
        reset_probabilities=sampler.probabilities(sampler.initial_state(), 10.0),
    )


def collect_validation_trace(env, actor, observation_norm, goal_norm, window):
    from bello_mujoco.evaluation.supermimic import _tracking_errors

    env = window_environment(env, window)
    native = env.model.mj_model
    data = mujoco.MjData(native)
    capacities = {"contact_capacity": 4096, "force_capacity": 65536}
    state, observation = reset_native_motion_rollout(env, data, **capacities)
    history = env.codec.initial_history(observation)
    step = make_native_motion_step(env, **capacities)
    infer = jax.jit(lambda h, g: actor(h, g, observation_norm, goal_norm))
    goal = jax.jit(env.goal)
    command_at = jax.jit(env.mapping.command)
    flags_at = jax.jit(motor_limit_flags)
    measure = jax.jit(lambda state: _tracking_errors(env, state))
    states = [env.physical_state(state)]
    previous, motors, contexts = [], [], []
    signatures = [ground_contact_bodies(native, data, env.model.terrain_geom_ids)]
    planned = min(
        window.max_steps,
        math.ceil(
            (env.motion.num_frames - 1) / (env.motion.fps * env.control_timestep)
        ),
    )
    errors, rewards = [], []
    flags_count = dict.fromkeys(
        ("position_projected", "rate_limited", "torque_saturated"), 0
    )
    motor_counts = {name: np.zeros(env.action_size, dtype=int) for name in flags_count}
    peak_effort, failure_event = 0.0, None
    failed = timed_out = False
    for _ in range(planned):
        previous.append(state.data.ctrl)
        contexts.append(env.context(state, 0))
        action = infer(history, goal(state.reference_time))
        command = command_at(action, state.data.ctrl, state.joint_default_offset)
        state, observation, reward, done, info = step(data, state, action)
        if bool(info["done/nonfinite"]):
            raise RuntimeError("nonfinite native state in model validation")
        motors.append(state.data.ctrl)
        states.append(env.physical_state(state))
        signatures.append(
            ground_contact_bodies(native, data, env.model.terrain_geom_ids)
        )
        errors.append(tuple(np.asarray(x) for x in measure(state)))
        rewards.append(float(reward))
        for name, values in flags_at(
            command, data.actuator_force, env.model.actuator_peak_force
        ).items():
            values = np.asarray(values)
            flags_count[name] += int(values.any())
            motor_counts[name] += values
        peak_effort = max(
            peak_effort,
            float(np.max(np.abs(data.actuator_force) / env.model.actuator_peak_force)),
        )
        history = env.codec.advance_history(history, observation)
        failed = bool(info["done/terminal"])
        timed_out = bool(info["done/timeout"])
        if failed:
            reference = env.reference_state(
                state.reference_time, state.alignment_xy, state.alignment_yaw
            )
            height, tilt = env.mdp.errors(env.physical_state(state), reference)
            failure_event = {
                "step": len(motors),
                "source_frame": window.start_frame + float(state.reference_time),
                "height_error_m": float(height),
                "gravity_projection_error": float(tilt),
                "height_failed": bool(info["done/height"]),
                "tilt_failed": bool(info["done/orientation"]),
            }
        if bool(done):
            break
    root_error, orientation_error, joint_error = map(
        np.asarray, zip(*errors, strict=True)
    )
    tracking = {
        "evaluated_steps": len(motors),
        "planned_steps": planned,
        "episode_return": sum(rewards),
        "normalized_return": sum(rewards) / (5 * env.control_timestep * planned),
        "completion_fraction": len(motors) / planned,
        "terminal": failed,
        "timeout": timed_out,
        "root_position_rms_m": float(np.sqrt(np.mean(root_error**2))),
        "root_orientation_rms_deg": float(
            np.rad2deg(np.sqrt(np.mean(orientation_error**2)))
        ),
        "joint_position_rms_rad": float(np.sqrt(np.mean(joint_error**2))),
        "motor_peak_effort_ratio": peak_effort,
        **{
            f"{name}_step_fraction": count / len(motors)
            for name, count in flags_count.items()
        },
        "motor_diagnostics": {
            motor: {
                f"{name}_step_fraction": float(count[index] / len(motors))
                for name, count in motor_counts.items()
            }
            for index, motor in enumerate(env.model.actuator_coordinate_names)
        },
        "failure_event": failure_event,
    }
    return (
        jax.tree.map(lambda *x: jnp.stack(x), *states),
        jnp.stack(previous),
        jnp.stack(motors),
        jnp.stack(contexts),
        signatures,
        tracking,
    )


def audit_world_model(config):
    if config.output_path.exists():
        raise FileExistsError(config.output_path)
    metadata = checkpoint_metadata(config.checkpoint_path)
    selected = tuple(dict.fromkeys(w.clip for w in config.windows))
    _, model, input_norm, output_norm, _ = load_dynamics(
        config.checkpoint_path,
        dataset_directory=config.motion_dataset,
        reference_clips=selected,
    )
    manifest = json.loads((config.motion_dataset / MANIFEST).read_text())
    splits = {Path(e["file"]).stem: e["split"] for e in manifest["clips"]}
    report = {
        "audit_config": to_builtin(config),
        "source_commit": metadata["git_commit"],
        "checkpoint_update": metadata["update"],
        "dynamics_sha256": metadata["dynamics_sha256"],
        "dataset_manifest_sha256": metadata["dataset_manifest_sha256"],
        "scope": "Native deterministic frozen-policy trajectories. Autoregressive world-model predictions use recorded executed controls, not actions recomputed from predicted states. Ground-contact strata use active native contact-body signatures. Zero samples are reported explicitly; nonfinite predictions are counted separately. Window resets and training/held-out membership are labelled.",
        "windows": {},
    }
    for window in config.windows:
        env, actor, obs_norm, goal_norm, _ = load_actor(
            config.checkpoint_path,
            dataset_directory=config.motion_dataset,
            reference_clips=(window.clip,),
        )
        states, previous, motor, context, signatures, tracking = (
            collect_validation_trace(env, actor, obs_norm, goal_norm, window)
        )
        result = {
            "clip": window.clip,
            "start_frame": window.start_frame,
            "recorded_split": splits[window.clip],
            "observed_steps": len(motor),
            "native_failed": tracking["terminal"],
            "tracking": tracking,
            "horizons": {},
        }
        for horizon in config.horizons:
            anchors = np.arange(0, len(motor) - horizon + 1, config.anchor_stride)
            if len(anchors) > config.max_windows_per_horizon:
                anchors = anchors[
                    np.linspace(
                        0, len(anchors) - 1, config.max_windows_per_horizon, dtype=int
                    )
                ]
            if len(anchors):
                indices = anchors[:, None] + np.arange(horizon)
                predicted = rollout_predictions(
                    model,
                    jax.tree.map(lambda x, indices=anchors: x[indices], states),
                    previous[indices],
                    motor[indices],
                    context[indices],
                    input_norm,
                    output_norm,
                )
                errors = {
                    name: np.asarray(value)
                    for name, value in prediction_errors(
                        predicted,
                        jax.tree.map(
                            lambda x, indices=anchors + horizon: x[indices], states
                        ),
                    ).items()
                }
            else:
                empty = jax.tree.map(lambda x: x[:0], states)
                errors = {
                    name: np.asarray(value)
                    for name, value in prediction_errors(empty, empty).items()
                }
            result["horizons"][str(horizon)] = {
                group: summarize_prediction_errors(errors, mask)
                for group, mask in contact_groups(signatures, anchors, horizon).items()
            }
        report["windows"][f"{window.clip}:{window.start_frame}"] = result
        print(
            json.dumps(
                {
                    "model_audit_window": window.clip,
                    "start_frame": window.start_frame,
                    "observed_steps": len(motor),
                    "native_failed": tracking["terminal"],
                }
            ),
            flush=True,
        )
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@hydra.main(
    version_base=None,
    config_path=config_path("commands"),
    config_name="audit_world_model",
)
def main(config):
    audit_world_model(command_config(WorldModelAuditConfig, config))


if __name__ == "__main__":
    main()
