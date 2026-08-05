#!/usr/bin/env python3
"""Upgrade retargeted TWIST archives with the released head key body."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import mujoco
import numpy as np


RETARGET_FORMAT_VERSION = 6
_MODEL: mujoco.MjModel | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--robot-xml", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    root = args.input_root.expanduser().resolve()
    robot_xml = args.robot_xml.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    if args.jobs <= 0:
        raise ValueError("jobs must be positive")
    paths = sorted(root.glob("*.npz"))
    upgraded = 0
    failed: list[dict[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_initialize_worker,
        initargs=(str(robot_xml),),
    ) as executor:
        futures = {executor.submit(_upgrade, str(path)): path for path in paths}
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                changed = future.result()
                upgraded += int(changed)
                status = "upgraded" if changed else "current"
            except Exception as error:
                status = "failed"
                failed.append({"path": str(path), "error": str(error)})
            print(f"[{completed}/{len(paths)}] {status}: {path.name}", flush=True)
    report = {
        "schema_version": 1,
        "retarget_format_version": RETARGET_FORMAT_VERSION,
        "selected": len(paths),
        "upgraded": upgraded,
        "failed": failed,
    }
    report_path = root / "upgrade_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


def _initialize_worker(robot_xml: str) -> None:
    global _MODEL
    _MODEL = mujoco.MjModel.from_xml_path(robot_xml)


def _upgrade(path_text: str) -> bool:
    path = Path(path_text)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    body_names = tuple(str(name) for name in arrays["body_names"].tolist())
    version = int(np.asarray(arrays["retarget_format_version"]).item())
    if "head_link" in body_names and version == RETARGET_FORMAT_VERSION:
        return False
    if "head_link" not in body_names:
        position, quaternion, linear, angular = _head_state(
            arrays["qpos"], arrays["qvel"]
        )
        arrays["body_pos"] = np.concatenate(
            (arrays["body_pos"], position[:, None, :]), axis=1
        )
        arrays["body_quat_wxyz"] = np.concatenate(
            (arrays["body_quat_wxyz"], quaternion[:, None, :]), axis=1
        )
        arrays["body_lin_vel"] = np.concatenate(
            (arrays["body_lin_vel"], linear[:, None, :]), axis=1
        )
        arrays["body_ang_vel"] = np.concatenate(
            (arrays["body_ang_vel"], angular[:, None, :]), axis=1
        )
        arrays["body_names"] = np.asarray((*body_names, "head_link"))
    arrays["retarget_format_version"] = np.asarray(
        RETARGET_FORMAT_VERSION, dtype=np.int32
    )
    temporary = path.with_name(f".{path.stem}.upgrade.partial.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _head_state(qpos: np.ndarray, qvel: np.ndarray):
    if _MODEL is None:
        raise RuntimeError("worker model is not initialized")
    head_id = _MODEL.body("head_link").id
    data = mujoco.MjData(_MODEL)
    position = np.empty((len(qpos), 3), dtype=np.float32)
    quaternion = np.empty((len(qpos), 4), dtype=np.float32)
    linear = np.empty((len(qpos), 3), dtype=np.float32)
    angular = np.empty((len(qpos), 3), dtype=np.float32)
    velocity = np.empty(6, dtype=np.float64)
    for frame in range(len(qpos)):
        data.qpos[:] = qpos[frame]
        data.qvel[:] = qvel[frame]
        mujoco.mj_forward(_MODEL, data)
        position[frame] = data.xpos[head_id]
        quaternion[frame] = data.xquat[head_id]
        mujoco.mj_objectVelocity(
            _MODEL,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            head_id,
            velocity,
            0,
        )
        angular[frame] = velocity[:3]
        linear[frame] = velocity[3:]
    return position, quaternion, linear, angular


if __name__ == "__main__":
    main()
