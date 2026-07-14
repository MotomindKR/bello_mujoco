"""Run a pinned upstream MuJoCo Playground locomotion baseline on Vast.ai."""

from __future__ import annotations

import functools
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from bello_mujoco.vast import (
    PLAYGROUND_REMOTE_TRAIN_SCRIPT,
    VastOfferConfig,
    VastTrainConfig,
    launch_pipeline,
    run_remote_training,
    safe_exception_message,
)


PLAYGROUND_REPOSITORY = "https://github.com/google-deepmind/mujoco_playground.git"
PLAYGROUND_COMMIT = "87a4bf98f1806adefd240d72dd53a2c3ceeb2f0d"
BERKELEY_ENVIRONMENT = "BerkeleyHumanoidJoystickFlatTerrain"


@dataclass(frozen=True)
class VastPlaygroundConfig:
    """Configuration for an unchanged pinned MuJoCo Playground training run."""

    seed: int = 0
    num_timesteps: int = 150_000_000
    max_price: float = 1.25
    branch: str = "main"
    run_name: str | None = None
    yes: bool = False
    dry_run: bool = False
    ready_timeout_seconds: int = 900
    ssh_timeout_seconds: int = 900
    destroy_retries: int = 3
    offer: VastOfferConfig = field(default_factory=VastOfferConfig)


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return Path(result.stdout.strip()).resolve()


def launch_playground_pipeline(repo_root: Path, config: VastPlaygroundConfig) -> int:
    if config.num_timesteps <= 0:
        raise ValueError("--num-timesteps must be positive")
    run_name = config.run_name or f"upstream-berkeley-seed{config.seed}"
    vast_config = VastTrainConfig(
        train_config=Path("experiments/train/locomotion_clean.yaml"),
        max_price=config.max_price,
        branch=config.branch,
        run_name=run_name,
        yes=config.yes,
        dry_run=config.dry_run,
        ready_timeout_seconds=config.ready_timeout_seconds,
        ssh_timeout_seconds=config.ssh_timeout_seconds,
        destroy_retries=config.destroy_retries,
        offer=config.offer,
    )
    remote_runner = functools.partial(
        run_remote_training,
        script=PLAYGROUND_REMOTE_TRAIN_SCRIPT,
        remote_env_extra={
            "PLAYGROUND_REPOSITORY": PLAYGROUND_REPOSITORY,
            "PLAYGROUND_COMMIT": PLAYGROUND_COMMIT,
            "PLAYGROUND_ENV_NAME": BERKELEY_ENVIRONMENT,
            "PLAYGROUND_SEED": str(config.seed),
            "PLAYGROUND_NUM_TIMESTEPS": str(config.num_timesteps),
        },
    )
    return launch_pipeline(repo_root, vast_config, remote_runner=remote_runner)


def entrypoint() -> None:
    config = tyro.cli(VastPlaygroundConfig)
    try:
        status = launch_playground_pipeline(repository_root(), config)
    except KeyboardInterrupt:
        print("error: interrupted; instance teardown was attempted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {safe_exception_message(exc)}", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(status)


if __name__ == "__main__":
    entrypoint()
