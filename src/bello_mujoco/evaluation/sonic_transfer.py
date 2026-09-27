"""Per-clip held-out evaluation for SONIC transfer checkpoints."""

from __future__ import annotations

from pathlib import Path

from bello_mujoco.evaluation.motion import render_motion_policy
from bello_mujoco.sonic.runtime import load_sonic_policy_runtime
from bello_mujoco.training.normalization import init_observation_norm_state


def render_sonic_checkpoint(
    checkpoint: Path,
    dataset: Path,
    selected_clips: tuple[str, ...],
    output_directory: Path,
    *,
    steps: int | None = None,
    stop_on_terminal: bool = True,
) -> dict[str, dict]:
    if not selected_clips:
        raise ValueError("sonic_transfer evaluation requires selected held-out clips")
    if len(set(selected_clips)) != len(selected_clips):
        raise ValueError("sonic_transfer evaluation clip names must be unique")
    if any(
        Path(name).name != name or Path(name).suffix or not name
        for name in selected_clips
    ):
        raise ValueError("sonic_transfer evaluation clips must be plain clip stems")
    dataset = Path(dataset).expanduser().resolve()
    paths = tuple(dataset / f"{name}.npz" for name in selected_clips)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"held-out SONIC clips are missing: {missing}")
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory.suffix:
        raise ValueError("sonic_transfer output_path must be a directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, path in zip(selected_clips, paths, strict=True):
        runtime = load_sonic_policy_runtime(
            checkpoint,
            (path,),
            evaluation_names=(),
            use_viewer_model=True,
        )
        recorded_held_out = set(runtime.run_config.dataset.evaluation_clips)
        if name not in recorded_held_out:
            raise ValueError(
                f"clip {name!r} was not recorded in the checkpoint evaluation split"
            )
        env = runtime.env
        ppo = runtime.run_config.config
        stats = render_motion_policy(
            env,
            runtime.policy,
            output_directory / f"{name}.mp4",
            steps=env.max_episode_steps if steps is None else steps,
            stop_on_terminal=stop_on_terminal,
            width=640,
            height=480,
            fps=50,
            obs_stack_size=ppo.policy.obs_stack_size,
            ppo_config=ppo,
            observation_norm_state=init_observation_norm_state(
                runtime.policy.actor.observation_size
            ),
        )
        results[name] = dict(stats)
    return results
