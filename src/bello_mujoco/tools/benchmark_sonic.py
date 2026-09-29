"""Measured single-GPU sizing; not a production training or artifact workflow."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import resource
import subprocess
import threading
import time

import jax
import numpy as np

from bello_mujoco.configuration import sonic_transfer_training_profile
from bello_mujoco.envs.sonic_transfer import BelloSonicTransferEnv
from bello_mujoco.sonic.checkpoint import import_sonic_checkpoint
from bello_mujoco.sonic.corpus import MANIFEST, verify_corpus
from bello_mujoco.training.ppo import init_runner_state_with_policy
from bello_mujoco.training.runtime import raise_on_invalid_training_metrics
from bello_mujoco.training.sonic_transfer import compile_sonic_update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--updates", type=int, default=21)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.updates < 3 or args.num_envs < 4 or args.num_envs % 4:
        parser.error("use at least 3 updates and a positive multiple of 4 environments")
    if args.output.exists():
        raise FileExistsError(args.output)
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("benchmark requires exactly one visible GPU")
    config = sonic_transfer_training_profile(
        args.dataset, args.checkpoint, args.revision
    )
    manifest = json.loads((args.dataset / MANIFEST).read_text())
    evaluation_names = tuple(
        Path(entry["file"]).stem
        for entry in manifest["clips"]
        if entry["split"] == "evaluation"
    )
    dataset_digest = verify_corpus(args.dataset, evaluation_names)
    ppo = replace(
        config.config,
        seed=args.seed,
        num_envs=args.num_envs,
        total_timesteps=args.num_envs * 24 * args.updates,
    )
    stop = threading.Event()
    samples: list[int] = []
    sampling_errors: list[str] = []

    def sample_memory():
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=10,
                )
                samples.append(sum(int(value) for value in output.splitlines()))
            except Exception as error:
                sampling_errors.append(str(error))
                return
            stop.wait(0.25)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    durations = []
    try:
        env = BelloSonicTransferEnv.load_corpus(
            config.asset_config,
            config.env_config,
            tuple(sorted(args.dataset.glob("*.npz"))),
            evaluation_names=evaluation_names,
            warp=config.warp,
            batch_size=ppo.num_envs,
        )
        print(
            json.dumps(
                {
                    "stage": "corpus_loaded",
                    "clips": env.reference_bank.num_clips,
                    "jax_memory": jax.devices()[0].memory_stats(),
                }
            ),
            flush=True,
        )
        imported = import_sonic_checkpoint(
            args.checkpoint,
            revision=args.revision,
            adapter=env.embodiment,
            key=jax.random.PRNGKey(ppo.seed),
        )
        state, static, optimizer = init_runner_state_with_policy(
            jax.random.PRNGKey(ppo.seed), env, ppo, imported.policy
        )
        print(
            json.dumps(
                {"stage": "initialized", "jax_memory": jax.devices()[0].memory_stats()}
            ),
            flush=True,
        )
        step = compile_sonic_update(env, ppo, static, optimizer)
        for update in range(args.updates):
            started = time.monotonic()
            state, metrics = step(state)
            jax.block_until_ready((state, metrics))
            durations.append(time.monotonic() - started)
            try:
                raise_on_invalid_training_metrics(metrics, update + 1)
            except RuntimeError:
                print(
                    json.dumps(
                        {
                            "failed_update": update + 1,
                            "metrics": {
                                name: np.asarray(value).tolist()
                                for name, value in metrics.items()
                            },
                            "whole_device_peak_used_mib": max(samples),
                            "jax_memory": jax.devices()[0].memory_stats(),
                        }
                    ),
                    flush=True,
                )
                raise
            print(
                json.dumps(
                    {
                        "update": update + 1,
                        "seconds": durations[-1],
                        "reward": float(metrics["mean_reward"]),
                        "learning_rate": float(metrics["next_learning_rate"]),
                        "gradient_norm": float(metrics["gradient_norm"]),
                        "actor_gradient_norm": float(metrics["actor_gradient_norm"]),
                        "critic_gradient_norm": float(metrics["critic_gradient_norm"]),
                        "exploration_gradient_norm": float(
                            metrics["exploration_gradient_norm"]
                        ),
                    }
                ),
                flush=True,
            )
    finally:
        stop.set()
        monitor.join(timeout=12)
    if sampling_errors or not samples:
        raise RuntimeError(f"GPU memory monitoring failed: {sampling_errors}")
    result = {
        "num_envs": ppo.num_envs,
        "updates": args.updates,
        "first_update_seconds_including_compile": durations[0],
        "seed": args.seed,
        "dataset_manifest_sha256": dataset_digest,
        "warmup_updates": 2,
        "warm_update_seconds_median": float(np.median(durations[2:])),
        "warm_environment_steps_per_second": ppo.batch_size
        / float(np.mean(durations[2:])),
        "update_seconds": durations,
        "whole_device_peak_used_mib": max(samples),
        "host_max_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "jax_memory": jax.devices()[0].memory_stats(),
        "pretrained_sha256": imported.sha256,
        "clips": env.reference_bank.clip_names,
        "total_reference_frames": int(np.sum(env.reference_bank.lengths)),
        "metrics": {
            name: np.asarray(value).tolist() for name, value in metrics.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
