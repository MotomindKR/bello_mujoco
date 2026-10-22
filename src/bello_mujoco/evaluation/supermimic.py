"""Deterministic full-clip Mini evaluation in native CPU MuJoCo."""

import json
import math
from pathlib import Path
from statistics import mean

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from bello_mujoco.evaluation.motion import (
    make_native_motion_step,
    reset_native_motion_rollout,
)
from bello_mujoco.evaluation.render import (
    CameraConfig,
    open_rollout_video_resources,
    write_video_frame,
)
from bello_mujoco.supermimic import rotations as rot
from bello_mujoco.supermimic.checkpoint import load_actor
from bello_mujoco.training.supermimic import load_environment


def render_supermimic_checkpoint(
    checkpoint,
    dataset,
    selected_clips,
    output_directory,
    *,
    steps=None,
    stop_on_terminal=True,
):
    if not selected_clips or len(set(selected_clips)) != len(selected_clips):
        raise ValueError("SuperMimic evaluation requires unique held-out clip names")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("steps must be a positive integer or null")
    output = Path(output_directory)
    if output.suffix:
        raise ValueError("SuperMimic output_path must be a directory")
    env, actor, obs_norm, goal_norm, config = load_actor(
        checkpoint,
        dataset_directory=dataset,
        reference_clips=(selected_clips[0],),
        use_viewer_model=True,
    )
    if any(name not in config.dataset.evaluation_clips for name in selected_clips):
        raise ValueError("selected clips must belong to the recorded held-out split")
    infer = jax.jit(lambda h, goal: actor(h, goal, obs_norm, goal_norm))
    results = {}
    for index, name in enumerate(selected_clips):
        if index:
            env = load_environment(
                config,
                load_mjx=False,
                paths=(config.dataset.directory / f"{name}.npz",),
                evaluation_names=(),
                use_viewer_model=True,
            )
        output.mkdir(parents=True, exist_ok=True)
        data = mujoco.MjData(env.model.mj_model)
        state, observation = reset_native_motion_rollout(env, data)
        history = env.codec.initial_history(observation)
        native_step = make_native_motion_step(env)
        planned_steps = math.ceil(
            (env.reference_bank.max_clip_frames - 1)
            / (env.reference_bank.fps * env.control_timestep)
        )
        limit = planned_steps if steps is None else min(steps, planned_steps)
        camera_config = CameraConfig(
            mode="follow", distance=3.0, azimuth=135.0, elevation=-16.0
        )
        resources, renderer, camera, writer = open_rollout_video_resources(
            env.model.mj_model,
            output / f"{name}.mp4",
            width=640,
            height=480,
            fps=round(1 / env.control_timestep),
            camera_config=camera_config,
        )
        total, failed, timed_out, completed = 0.0, False, False, 0
        root_errors, orientation_errors, joint_errors = [], [], []
        peak_effort, limited = 0.0, 0
        try:
            for _ in range(limit):
                action = infer(history, env.goal(state.reference_time))
                proposal = env.mapping.command(
                    action, state.data.ctrl, state.joint_default_offset
                )
                limited += int(
                    np.any(
                        np.abs(np.asarray(proposal.accepted - proposal.requested))
                        > 1e-6
                    )
                )
                state, observation, reward, done, info = native_step(
                    data, state, action
                )
                if bool(info["done/nonfinite"]):
                    raise RuntimeError(
                        "SuperMimic native evaluation encountered nonfinite state"
                    )
                physical = env.physical_state(state)
                reference = env.reference_state(
                    state.reference_time, state.alignment_xy, state.alignment_yaw
                )
                root_errors.append(
                    float(jnp.linalg.norm(physical.position - reference.position))
                )
                orientation_errors.append(
                    float(
                        jnp.linalg.norm(
                            rot.log(
                                rot.multiply(
                                    rot.conjugate(physical.quaternion),
                                    reference.quaternion,
                                )
                            )
                        )
                    )
                )
                joint_errors.append(
                    np.asarray(physical.joint_position - reference.joint_position)
                )
                peak_effort = max(
                    peak_effort,
                    float(
                        np.max(
                            np.abs(data.actuator_force) / env.model.actuator_peak_force
                        )
                    ),
                )
                total += float(reward)
                completed += 1
                history = env.codec.advance_history(history, observation)
                failed |= bool(info["done/terminal"])
                timed_out |= bool(info["done/timeout"])
                write_video_frame(
                    renderer,
                    camera,
                    writer,
                    env.model.mj_model,
                    data,
                    camera_config,
                    root_body_name=env.model.robot.root_body_name,
                )
                if bool(done) and (timed_out or (stop_on_terminal and failed)):
                    break
        finally:
            resources.close()
        results[name] = {
            "rendered_steps": completed,
            "planned_steps": planned_steps,
            "episode_return": total,
            "normalized_return": total / (5 * env.control_timestep * planned_steps),
            "completion_fraction": completed / planned_steps,
            "terminal": failed,
            "timeout": timed_out,
            "root_position_rms_m": float(np.sqrt(np.mean(np.square(root_errors)))),
            "root_orientation_rms_deg": float(
                np.rad2deg(np.sqrt(np.mean(np.square(orientation_errors))))
            ),
            "joint_position_rms_rad": float(np.sqrt(np.mean(np.square(joint_errors)))),
            "motor_peak_effort_ratio": peak_effort,
            "limited_step_fraction": limited / completed,
        }
    fields = (
        "episode_return",
        "normalized_return",
        "completion_fraction",
        "root_position_rms_m",
        "root_orientation_rms_deg",
        "joint_position_rms_rad",
        "motor_peak_effort_ratio",
        "limited_step_fraction",
    )
    if any(
        not math.isfinite(row[field]) for row in results.values() for field in fields
    ):
        raise RuntimeError("nonfinite SuperMimic evaluation results")
    summary = {
        "clip_count": len(results),
        "weighting": "equal_per_clip",
        "start": "first_frame",
        "actions": "deterministic",
        "failure_fraction": mean(float(row["terminal"]) for row in results.values()),
        **{
            f"mean_{field}": mean(row[field] for row in results.values())
            for field in fields
        },
    }
    report = {"summary": summary, "clips": results}
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report
