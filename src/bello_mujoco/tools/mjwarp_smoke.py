from __future__ import annotations

import argparse
import time

import jax.numpy as jnp
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from bello_mujoco.assets import generated_model_path
from bello_mujoco.contracts import CONTROL_DECIMATION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nworld", type=int, default=512)
    parser.add_argument("--step-count", type=int, default=20)
    parser.add_argument("--decimation", type=int, default=CONTROL_DECIMATION)
    parser.add_argument("--nconmax", type=int, default=16)
    parser.add_argument("--njmax", type=int, default=256)
    parser.add_argument("--zero-copy-check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(generated_model_path("full_body")))
    data = home_data(model)
    warp_model = mjw.put_model(model)
    warp_data = mjw.make_data(model, nworld=args.nworld, nconmax=args.nconmax, njmax=args.njmax)
    init_data(warp_data, data, args.nworld)

    for _ in range(args.decimation):
        mjw.step(warp_model, warp_data)
    wp.synchronize()

    direct_seconds = []
    for _ in range(min(args.step_count, 5)):
        start = time.perf_counter()
        for _ in range(args.decimation):
            mjw.step(warp_model, warp_data)
        wp.synchronize()
        direct_seconds.append(time.perf_counter() - start)

    with wp.ScopedCapture() as capture:
        for _ in range(args.decimation):
            mjw.step(warp_model, warp_data)
    wp.capture_launch(capture.graph)
    wp.synchronize()

    graph_seconds = []
    for _ in range(args.step_count):
        start = time.perf_counter()
        wp.capture_launch(capture.graph)
        wp.synchronize()
        graph_seconds.append(time.perf_counter() - start)

    result = {
        "backend": "mjwarp",
        "nworld": args.nworld,
        "decimation": args.decimation,
        "nconmax": args.nconmax,
        "njmax": args.njmax,
        "direct_policy_step_seconds": direct_seconds,
        "direct_env_steps_per_second": [args.nworld / seconds for seconds in direct_seconds],
        "graph_policy_step_seconds_first5": graph_seconds[:5],
        "graph_policy_step_seconds_mean": float(np.mean(graph_seconds)),
        "graph_env_steps_per_second_mean": float(args.nworld / np.mean(graph_seconds)),
        "qpos_shape": tuple(warp_data.qpos.shape),
        "qvel_shape": tuple(warp_data.qvel.shape),
        "ctrl_shape": tuple(warp_data.ctrl.shape),
    }
    if args.zero_copy_check:
        result["zero_copy"] = zero_copy_check()
    print(result)


def home_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    return data


def init_data(warp_data: mjw.Data, data: mujoco.MjData, nworld: int) -> None:
    qpos = np.tile(data.qpos, (nworld, 1)).astype(np.float32)
    qvel = np.tile(data.qvel, (nworld, 1)).astype(np.float32)
    ctrl = np.tile(data.ctrl, (nworld, 1)).astype(np.float32)
    wp.copy(warp_data.qpos, wp.array(qpos, dtype=wp.float32, device=warp_data.qpos.device))
    wp.copy(warp_data.qvel, wp.array(qvel, dtype=wp.float32, device=warp_data.qvel.device))
    wp.copy(warp_data.ctrl, wp.array(ctrl, dtype=wp.float32, device=warp_data.ctrl.device))
    wp.synchronize()


def zero_copy_check() -> dict[str, bool]:
    jax_array = jnp.arange(8, dtype=jnp.float32)
    jax_array.block_until_ready()
    warp_view = wp.from_jax(jax_array)
    jax_to_warp_ptr_equal = int(jax_array.unsafe_buffer_pointer()) == int(warp_view.ptr)
    wp.copy(
        warp_view,
        wp.array(np.full((8,), 7.0, dtype=np.float32), dtype=wp.float32, device=warp_view.device),
    )
    wp.synchronize()
    jax_observed_warp_write = bool(np.all(np.asarray(jax_array) == 7.0))

    owned_warp_array = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device="cuda:0")
    jax_view = wp.to_jax(owned_warp_array)
    jax_view.block_until_ready()
    warp_to_jax_ptr_equal = int(owned_warp_array.ptr) == int(jax_view.unsafe_buffer_pointer())
    return {
        "jax_to_warp_ptr_equal": jax_to_warp_ptr_equal,
        "jax_observed_warp_write": jax_observed_warp_write,
        "warp_to_jax_ptr_equal": warp_to_jax_ptr_equal,
    }


if __name__ == "__main__":
    main()
