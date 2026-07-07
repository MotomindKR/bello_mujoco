"""MuJoCo/MJX model loading and index metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    canonical_asset,
    prepare_asset,
)
from bello_mujoco.contracts import (
    DEFAULT_ROOT_POS,
    DEFAULT_ROOT_QUAT,
)
from bello_mujoco.terrain import (
    ROBOT_ONLY_TERRAIN,
    TerrainConfig,
    runtime_mjcf_path,
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
    actuator_joint_names: tuple[str, ...]
    policy_qpos_indices: np.ndarray
    policy_qvel_indices: np.ndarray
    policy_ctrl_indices: np.ndarray
    actuator_qpos_indices: np.ndarray
    actuator_qvel_indices: np.ndarray
    hold_ctrl_indices: np.ndarray
    policy_action_scale: np.ndarray
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
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
) -> BelloModel:
    if canonical_asset(config.asset) == "upper_body":
        raise ValueError(
            "load_bello_model currently requires full_body or lower_body locomotion joints"
        )
    prepared = prepare_asset(config)
    return load_prepared_bello_model(
        prepared,
        warp=warp,
        batch_size=batch_size,
        terrain=terrain,
        actuator_config=config.actuator_config,
    )


def load_prepared_bello_model(
    prepared: PreparedAsset,
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
    actuator_config: ActuatorConfig = ActuatorConfig(),
) -> BelloModel:
    return _load_bello_model_from_path(
        prepared.mjcf_path,
        warp=warp,
        batch_size=batch_size,
        terrain=terrain,
        actuator_config=actuator_config,
    )


def _load_bello_model_from_path(
    xml_path: Path,
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
    actuator_config: ActuatorConfig = ActuatorConfig(),
) -> BelloModel:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(path)
    path = runtime_mjcf_path(path, terrain)

    mj_model = mujoco.MjModel.from_xml_path(str(path))
    mjx_model = mjx.put_model(mj_model, impl="warp")

    actuator_joint_ids = np.asarray(mj_model.actuator_trnid[:, 0], dtype=np.int32)
    actuator_joint_names = tuple(
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id))
        for joint_id in actuator_joint_ids
    )
    policy_joint_names = actuator_config.policy_joint_names
    hold_joint_names = actuator_config.hold_joint_names
    policy_ctrl_indices = np.asarray(
        [
            actuator_id
            for actuator_id, joint_name in enumerate(actuator_joint_names)
            if joint_name in policy_joint_names
        ],
        dtype=np.int32,
    )
    policy_joint_ids = actuator_joint_ids[policy_ctrl_indices]
    policy_qpos_indices = np.asarray(
        [int(mj_model.jnt_qposadr[joint_id]) for joint_id in policy_joint_ids],
        dtype=np.int32,
    )
    policy_qvel_indices = np.asarray(
        [int(mj_model.jnt_dofadr[joint_id]) for joint_id in policy_joint_ids],
        dtype=np.int32,
    )
    actuator_qpos_indices = np.asarray(
        [int(mj_model.jnt_qposadr[joint_id]) for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )
    actuator_qvel_indices = np.asarray(
        [int(mj_model.jnt_dofadr[joint_id]) for joint_id in actuator_joint_ids],
        dtype=np.int32,
    )
    hold_ctrl_indices = np.asarray(
        [
            actuator_id
            for actuator_id, joint_name in enumerate(actuator_joint_names)
            if joint_name in hold_joint_names
        ],
        dtype=np.int32,
    )

    root_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "root_free")
    if root_id < 0:
        raise ValueError("generated full-body MJCF must include root_free")
    root_qpos_adr = int(mj_model.jnt_qposadr[root_id])
    root_qvel_adr = int(mj_model.jnt_dofadr[root_id])

    default_qpos, default_qvel, default_ctrl = model_home_state(mj_model)
    default_qpos[root_qpos_adr : root_qpos_adr + 3] = np.asarray(
        DEFAULT_ROOT_POS, dtype=np.float32
    )
    default_qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = np.asarray(
        DEFAULT_ROOT_QUAT, dtype=np.float32
    )
    policy_action_scale = actuator_action_scale(
        mj_model, actuator_joint_names, policy_ctrl_indices, actuator_config
    )

    return BelloModel(
        mj_model=mj_model,
        mjx_model=mjx_model,
        xml_path=path,
        warp_naconmax=warp.resolved_naconmax(batch_size),
        warp_naccdmax=warp.naccdmax,
        warp_njmax=warp.njmax,
        actuator_joint_names=actuator_joint_names,
        policy_qpos_indices=policy_qpos_indices,
        policy_qvel_indices=policy_qvel_indices,
        policy_ctrl_indices=policy_ctrl_indices,
        actuator_qpos_indices=actuator_qpos_indices,
        actuator_qvel_indices=actuator_qvel_indices,
        hold_ctrl_indices=hold_ctrl_indices,
        policy_action_scale=policy_action_scale,
        default_qpos=default_qpos,
        default_qvel=default_qvel,
        default_ctrl=default_ctrl,
        root_qpos_adr=root_qpos_adr,
        root_qvel_adr=root_qvel_adr,
    )


def model_home_state(
    mj_model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        return (
            np.asarray(mj_model.key_qpos[key_id], dtype=np.float32).copy(),
            np.zeros(mj_model.nv, dtype=np.float32),
            np.asarray(mj_model.key_ctrl[key_id], dtype=np.float32).copy(),
        )
    return (
        np.zeros(mj_model.nq, dtype=np.float32),
        np.zeros(mj_model.nv, dtype=np.float32),
        np.zeros(mj_model.nu, dtype=np.float32),
    )


def actuator_action_scale(
    mj_model: mujoco.MjModel,
    actuator_joint_names: tuple[str, ...],
    policy_ctrl_indices: np.ndarray,
    config: ActuatorConfig,
) -> np.ndarray:
    ctrlrange = np.asarray(mj_model.actuator_ctrlrange, dtype=np.float32)
    all_scale = config.default_action_scale_ratio * (ctrlrange[:, 1] - ctrlrange[:, 0])
    joint_configs = config.by_name
    for actuator_id, joint_name in enumerate(actuator_joint_names):
        joint_config = joint_configs.get(joint_name)
        if joint_config is not None and joint_config.action_scale is not None:
            all_scale[actuator_id] = joint_config.action_scale
    scale = all_scale[policy_ctrl_indices]
    return scale.astype(np.float32)


def require_full_body_config(config: AssetPrepConfig) -> None:
    if canonical_asset(config.asset) != "full_body":
        raise ValueError("BelloLocomotionEnv currently requires asset='full_body'")
