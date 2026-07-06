from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

from bello_mujoco.envs.locomotion import BelloLocomotionEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", action="store_true", help="Also run one MJX step. Mesh collision may compile slowly.")
    args = parser.parse_args()

    env = BelloLocomotionEnv.load()
    state, obs = env.reset()
    reward = jnp.asarray(0.0)
    done = jnp.asarray(False)
    if args.step:
        state, obs, reward, done, _ = env.step(state, jnp.zeros(env.action_size, dtype=jnp.float32))
    print(
        {
            "backend": jax.default_backend(),
            "nq": env.model.mj_model.nq,
            "nv": env.model.mj_model.nv,
            "nu": env.model.mj_model.nu,
            "action_size": env.action_size,
            "observation_size": int(obs.shape[0]),
            "reward": float(reward),
            "done": bool(done),
        }
    )


if __name__ == "__main__":
    main()
