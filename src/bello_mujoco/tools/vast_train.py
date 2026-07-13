"""CLI entry point for one-shot Vast.ai PPO training."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import tyro

from bello_mujoco.vast import VastTrainConfig, launch_pipeline


def repository_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return Path(result.stdout.strip()).resolve()


def parse_args(args: list[str]) -> VastTrainConfig:
    if "--" in args:
        separator = args.index("--")
        cli_args = args[:separator]
        hydra_overrides = tuple(args[separator + 1 :])
    else:
        cli_args = args
        hydra_overrides = ()
    config = tyro.cli(VastTrainConfig, args=cli_args)
    return replace(config, hydra_overrides=hydra_overrides)


def entrypoint() -> None:
    config = parse_args(sys.argv[1:])
    try:
        status = launch_pipeline(repository_root(), config)
    except KeyboardInterrupt:
        print("error: interrupted; instance teardown was attempted", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    raise SystemExit(status)


if __name__ == "__main__":
    entrypoint()
