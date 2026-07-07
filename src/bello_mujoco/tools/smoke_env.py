from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjx-impl", choices=("jax", "warp"), default="warp")
    parser.add_argument("--warp-naconmax", type=int, default=8192)
    parser.add_argument("--warp-naccdmax", type=int, default=None)
    parser.add_argument("--warp-njmax", type=int, default=256)
    parser.add_argument("--step", action="store_true", help="Also time MJX steps through a jitted step function.")
    parser.add_argument("--step-count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    env = BelloLocomotionEnv.load(
        mjx_impl=args.mjx_impl,
        warp_naconmax=args.warp_naconmax,
        warp_naccdmax=args.warp_naccdmax,
        warp_njmax=args.warp_njmax,
    )
    if args.batch_size == 1:
        state, obs = env.reset()
    else:
        state, obs = env.reset_batch(args.batch_size)
    obs.block_until_ready()
    reward = jnp.asarray(0.0)
    done = jnp.asarray(False)
    step_seconds: list[float] = []
    if args.step:
        if args.batch_size == 1:
            step = jax.jit(lambda s, a: env.step(s, a))
            action = jnp.zeros(env.action_size, dtype=jnp.float32)
        else:
            step = jax.jit(lambda s, a: env.step_batch(s, a))
            action = jnp.zeros((args.batch_size, env.action_size), dtype=jnp.float32)
        for _ in range(args.step_count):
            start = time.perf_counter()
            state, obs, reward, done, _ = step(state, action)
            obs.block_until_ready()
            reward.block_until_ready()
            done.block_until_ready()
            step_seconds.append(time.perf_counter() - start)

    result = {
        "backend": jax.default_backend(),
        "mjx_impl": env.model.mjx_impl,
        "batch_size": args.batch_size,
        "nq": env.model.mj_model.nq,
        "nv": env.model.mj_model.nv,
        "nu": env.model.mj_model.nu,
        "action_size": env.action_size,
        "observation_size": env.observation_size,
        "observation_shape": tuple(obs.shape),
        "reward_mean": float(jnp.mean(reward)),
        "done_mean": float(jnp.mean(done.astype(jnp.float32))),
    }
    if step_seconds:
        result["step_seconds"] = step_seconds
        result["env_steps_per_second"] = [args.batch_size / seconds for seconds in step_seconds]
    print(result)


if __name__ == "__main__":
    main()
