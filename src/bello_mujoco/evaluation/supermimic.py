"""Deterministic full-clip Mini evaluation in native CPU MuJoCo."""

import json
import math
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
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
    video_clips=None,
):
    if not selected_clips or len(set(selected_clips)) != len(selected_clips):
        raise ValueError("SuperMimic evaluation requires unique held-out clip names")
    if steps is not None and (type(steps) is not int or steps <= 0):
        raise ValueError("steps must be a positive integer or null")
    videos = set(selected_clips if video_clips is None else video_clips)
    if not videos.issubset(selected_clips) or (
        video_clips is not None and len(videos) != len(video_clips)
    ):
        raise ValueError("video_clips must be a unique subset of selected_clips")
    output = Path(output_directory)
    if output.suffix:
        raise ValueError("SuperMimic output_path must be a directory")
    env, actor, obs_norm, goal_norm, config = load_actor(
        checkpoint,
        dataset_directory=dataset,
        reference_clips=(selected_clips[0],),
        use_viewer_model=bool(videos),
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
                use_viewer_model=bool(videos),
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
        resources = None
        if name in videos:
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
                if name in videos:
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
            if resources is not None:
                resources.close()
        results[name] = {
            "evaluated_steps": completed,
            "rendered_steps": completed if name in videos else 0,
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
        print(
            json.dumps(
                {
                    "evaluated_clip": name,
                    "completion_fraction": results[name]["completion_fraction"],
                }
            ),
            flush=True,
        )
    report = _evaluation_report(results)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report


def _evaluation_report(results):
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
    return {"summary": summary, "clips": results}


def _evaluate_partition(arguments):
    checkpoint, dataset, selected, output, videos, index, workers = arguments
    subprocess.run(
        [sys.executable, "-m", "bello_mujoco.evaluation.supermimic_worker"],
        input=json.dumps(
            {
                "checkpoint": str(checkpoint),
                "dataset": str(dataset),
                "selected": selected,
                "output": str(output),
                "videos": videos,
                "index": index,
                "workers": workers,
            }
        ),
        text=True,
        check=True,
    )
    return json.loads((output / "summary.json").read_text())


def evaluate_supermimic_corpus(
    checkpoint,
    dataset,
    selected_clips,
    output_directory,
    *,
    video_clips=None,
    workers=1,
):
    """Evaluate every selected held-out clip, rendering only the explicit video subset."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not selected_clips or len(set(selected_clips)) != len(selected_clips):
        raise ValueError("evaluation requires unique selected_clips")
    videos = tuple(selected_clips if video_clips is None else video_clips)
    if len(set(videos)) != len(videos) or not set(videos).issubset(selected_clips):
        raise ValueError("video_clips must be a unique subset of selected_clips")
    if workers == 1:
        return render_supermimic_checkpoint(
            checkpoint, dataset, selected_clips, output_directory, video_clips=videos
        )
    output = Path(output_directory)
    count = min(workers, len(selected_clips))
    arguments = []
    for index in range(count):
        selected = tuple(selected_clips[index::count])
        arguments.append(
            (
                checkpoint,
                dataset,
                selected,
                output / f"worker-{index:02d}",
                tuple(name for name in videos if name in selected),
                index,
                count,
            )
        )
    results = {}
    with ThreadPoolExecutor(max_workers=count) as pool:
        for report, argument in zip(
            pool.map(_evaluate_partition, arguments), arguments, strict=True
        ):
            results.update(report["clips"])
            partition = argument[3]
            for video in partition.glob("*.mp4"):
                video.replace(output / video.name)
            (partition / "summary.json").unlink()
            partition.rmdir()
    report = _evaluation_report(results)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return report
