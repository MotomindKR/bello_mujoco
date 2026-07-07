"""MuJoCo/MJX model loading and index metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    canonical_asset,
    prepare_asset,
)
from bello_mujoco.contracts import (
    DEFAULT_JOINT_POS,
    DEFAULT_ROOT_POS,
    DEFAULT_ROOT_QUAT,
    FIXED_JOINT_DEFAULT_POS,
    POLICY_JOINT_ORDER,
)

DEFAULT_CONTACT_BUDGET_PER_ENV = 24
DEFAULT_WARP_NJMAX = 256


@dataclass(frozen=True)
class WarpConfig:
    contact_budget_per_env: int = DEFAULT_CONTACT_BUDGET_PER_ENV
    naconmax: int | None = None
    naccdmax: int | None = None
    njmax: int | None = DEFAULT_WARP_NJMAX

    def resolved_naconmax(self, batch_size: int = 1) -> int | None:
        if self.naconmax is not None:
            return self.naconmax
        return batch_size * self.contact_budget_per_env


@dataclass(frozen=True)
class BelloModel:
    mj_model: mujoco.MjModel
    mjx_model: mjx.Model
    xml_path: Path
    warp_naconmax: int | None
    warp_naccdmax: int | None
    warp_njmax: int | None
    policy_qpos_indices: np.ndarray
    policy_qvel_indices: np.ndarray
    policy_ctrl_indices: np.ndarray
    actuator_qpos_indices: np.ndarray
    actuator_qvel_indices: np.ndarray
    fixed_ctrl_indices: np.ndarray
    default_qpos: np.ndarray
    default_qvel: np.ndarray
    default_ctrl: np.ndarray
    root_qpos_adr: int
    root_qvel_adr: int

    def make_data(self) -> mjx.Data:
        return mjx.make_data(
            self.mj_model,
            impl="warp",
            naconmax=self.warp_naconmax,
            naccdmax=self.warp_naccdmax,
            njmax=self.warp_njmax,
        )


def load_bello_model(
    config: AssetPrepConfig = AssetPrepConfig(),
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
) -> BelloModel:
    if canonical_asset(config.asset) == "upper_body":
        raise ValueError(
            "load_bello_model currently requires full_body or lower_body policy joints"
        )
    prepared = prepare_asset(config)
    return load_prepared_bello_model(prepared, warp=warp, batch_size=batch_size)


def load_prepared_bello_model(
    prepared: PreparedAsset,
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
) -> BelloModel:
    return load_bello_model_from_path(
        prepared.mjcf_path, warp=warp, batch_size=batch_size
    )


def load_bello_model_from_path(
    xml_path: Path,
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
) -> BelloModel:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(path)

    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mjx_model = mjx.put_model(mj_model, impl="warp")

    policy_qpos_indices = np.asarray(
        [joint_qpos_index(mj_model, name) for name in POLICY_JOINT_ORDER],
        dtype=np.int32,
    )
    policy_qvel_indices = np.asarray(
        [joint_qvel_index(mj_model, name) for name in POLICY_JOINT_ORDER],
        dtype=np.int32,
    )
    policy_ctrl_indices = np.asarray(
        [actuator_index(mj_model, f"{name}_pos") for name in POLICY_JOINT_ORDER],
        dtype=np.int32,
    )
    actuator_joint_ids = np.asarray(mj_model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_qpos_indices = np.asarray(
        [int(mj_model.jnt_qposadr[joint_id]) for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )
    actuator_qvel_indices = np.asarray(
        [int(mj_model.jnt_dofadr[joint_id]) for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )
    fixed_ctrl_indices = np.asarray(
        [i for i in range(mj_model.nu) if i not in set(policy_ctrl_indices.tolist())],
        dtype=np.int32,
    )

    default_qpos = np.zeros(mj_model.nq, dtype=np.float32)
    default_qvel = np.zeros(mj_model.nv, dtype=np.float32)
    default_ctrl = np.zeros(mj_model.nu, dtype=np.float32)

    root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    if root_id < 0:
        raise ValueError("generated full-body MJCF must include root_free")
    root_qpos_adr = int(mj_model.jnt_qposadr[root_id])
    root_qvel_adr = int(mj_model.jnt_dofadr[root_id])
    default_qpos[root_qpos_adr : root_qpos_adr + 3] = np.asarray(
        DEFAULT_ROOT_POS, dtype=np.float32
    )
    default_qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = np.asarray(
        DEFAULT_ROOT_QUAT, dtype=np.float32
    )

    for joint_name, value in DEFAULT_JOINT_POS.items():
        default_qpos[joint_qpos_index(mj_model, joint_name)] = value
    for joint_name, value in FIXED_JOINT_DEFAULT_POS.items():
        if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0:
            default_qpos[joint_qpos_index(mj_model, joint_name)] = value

    for actuator_id in range(mj_model.nu):
        joint_id = int(mj_model.actuator_trnid[actuator_id, 0])
        joint_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name in DEFAULT_JOINT_POS:
            default_ctrl[actuator_id] = DEFAULT_JOINT_POS[joint_name]
        elif joint_name in FIXED_JOINT_DEFAULT_POS:
            default_ctrl[actuator_id] = FIXED_JOINT_DEFAULT_POS[joint_name]
        else:
            raise ValueError(f"no default target for actuator joint {joint_name!r}")

    if mj_model.nu < len(POLICY_JOINT_ORDER):
        raise ValueError(
            f"expected at least {len(POLICY_JOINT_ORDER)} actuators, got {mj_model.nu}"
        )
    return BelloModel(
        mj_model=mj_model,
        mjx_model=mjx_model,
        xml_path=path,
        warp_naconmax=warp.resolved_naconmax(batch_size),
        warp_naccdmax=warp.naccdmax,
        warp_njmax=warp.njmax,
        policy_qpos_indices=policy_qpos_indices,
        policy_qvel_indices=policy_qvel_indices,
        policy_ctrl_indices=policy_ctrl_indices,
        actuator_qpos_indices=actuator_qpos_indices,
        actuator_qvel_indices=actuator_qvel_indices,
        fixed_ctrl_indices=fixed_ctrl_indices,
        default_qpos=default_qpos,
        default_qvel=default_qvel,
        default_ctrl=default_ctrl,
        root_qpos_adr=root_qpos_adr,
        root_qvel_adr=root_qvel_adr,
    )


def require_full_body_config(config: AssetPrepConfig) -> None:
    if canonical_asset(config.asset) != "full_body":
        raise ValueError("BelloLocomotionEnv currently requires asset='full_body'")


def joint_qpos_index(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"joint not found: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def joint_qvel_index(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"joint not found: {joint_name}")
    return int(model.jnt_dofadr[joint_id])


def actuator_index(model: mujoco.MjModel, actuator_name: str) -> int:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if actuator_id < 0:
        raise ValueError(f"actuator not found: {actuator_name}")
    return int(actuator_id)
