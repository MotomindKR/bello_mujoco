"""MuJoCo/MJX model loading and index metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Literal

import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from bello_mujoco.actuators import ActuatorConfig
from bello_mujoco.assets_pipeline.prepare import (
    AssetPrepConfig,
    PreparedAsset,
    prepare_asset,
)
from bello_mujoco.contracts import (
    DEFAULT_ROOT_POS,
    DEFAULT_ROOT_QUAT,
    DEFAULT_SPAWN_CLEARANCE,
    SIM_TIMESTEP,
)
from bello_mujoco.terrain import (
    ROBOT_ONLY_TERRAIN,
    TerrainConfig,
    random_slope_angles_batch,
    runtime_mjcf_path,
    slope_quat,
    terrain_surface_height,
)

DEFAULT_CONTACT_BUDGET_PER_ENV = 32
DEFAULT_WARP_NJMAX = 256
ANKLE_TENDON_LOGICAL_JOINT = {
    "left_ankle_motor_1": "left_ankle_pitch_joint",
    "left_ankle_motor_2": "left_ankle_roll_joint",
    "right_ankle_motor_1": "right_ankle_pitch_joint",
    "right_ankle_motor_2": "right_ankle_roll_joint",
}


@dataclass(frozen=True)
class SimulationPhysicsConfig:
    """Runtime physics overrides shared by native MuJoCo and MJX-Warp."""

    timestep: float = SIM_TIMESTEP
    foot_terrain_margin: float | None = None
    foot_terrain_solref: tuple[float, float] | None = None
    foot_terrain_solimp: tuple[float, float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestep) or self.timestep <= 0.0:
            raise ValueError(
                "simulation_physics.timestep must be positive and finite, got "
                f"{self.timestep}"
            )
        if self.foot_terrain_margin is not None and (
            not math.isfinite(self.foot_terrain_margin)
            or self.foot_terrain_margin < 0.0
        ):
            raise ValueError(
                "simulation_physics.foot_terrain_margin must be finite and "
                f"nonnegative, got {self.foot_terrain_margin}"
            )
        if self.foot_terrain_solref is not None:
            time_constant, damping_ratio = self.foot_terrain_solref
            if (
                not math.isfinite(time_constant)
                or not math.isfinite(damping_ratio)
                or time_constant <= 0.0
                or damping_ratio <= 0.0
            ):
                raise ValueError(
                    "simulation_physics.foot_terrain_solref must contain positive "
                    f"finite values, got {self.foot_terrain_solref}"
                )
            if time_constant < 2.0 * self.timestep:
                raise ValueError(
                    "simulation_physics.foot_terrain_solref time constant must be "
                    "at least twice the simulation timestep, got "
                    f"{time_constant} for timestep {self.timestep}"
                )
        if self.foot_terrain_solimp is not None:
            dmin, dmax, width, midpoint, power = self.foot_terrain_solimp
            if not all(math.isfinite(value) for value in self.foot_terrain_solimp):
                raise ValueError(
                    "simulation_physics.foot_terrain_solimp must be finite, got "
                    f"{self.foot_terrain_solimp}"
                )
            if not 0.0 <= dmin <= dmax <= 1.0:
                raise ValueError(
                    "simulation_physics.foot_terrain_solimp impedance values must "
                    f"satisfy 0 <= dmin <= dmax <= 1, got {(dmin, dmax)}"
                )
            if width <= 0.0 or not 0.0 < midpoint < 1.0 or power < 1.0:
                raise ValueError(
                    "simulation_physics.foot_terrain_solimp requires width > 0, "
                    f"0 < midpoint < 1, and power >= 1, got {self.foot_terrain_solimp}"
                )


def _validate_randomization_range(
    name: str, bounds: tuple[float, float], *, positive: bool = False
) -> None:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain two values, got {bounds}")
    low, high = bounds
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        raise ValueError(f"{name} must be finite and ordered, got {bounds}")
    if positive and low <= 0.0:
        raise ValueError(f"{name} must contain positive values, got {bounds}")


@dataclass(frozen=True)
class ModelRandomizationConfig:
    """Fixed per-world physical model variation for vectorized MJX training."""

    enabled: bool = False
    seed: int = 0
    joint_origin_range: tuple[float, float] = (0.0, 0.0)
    joint_angle_offset_range: tuple[float, float] = (0.0, 0.0)
    joint_angle_offset_scope: Literal["all", "policy"] = "all"
    joint_inertia_scale_range: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(
                f"model_randomization.seed must be nonnegative, got {self.seed}"
            )
        _validate_randomization_range(
            "model_randomization.joint_origin_range", self.joint_origin_range
        )
        _validate_randomization_range(
            "model_randomization.joint_angle_offset_range",
            self.joint_angle_offset_range,
        )
        if self.joint_angle_offset_scope not in {"all", "policy"}:
            raise ValueError(
                "model_randomization.joint_angle_offset_scope must be 'all' or "
                f"'policy', got {self.joint_angle_offset_scope!r}"
            )
        _validate_randomization_range(
            "model_randomization.joint_inertia_scale_range",
            self.joint_inertia_scale_range,
            positive=True,
        )


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
    mjx_model: mjx.Model | None
    xml_path: Path
    viewer_xml_path: Path | None
    warp_naconmax: int | None
    warp_naccdmax: int | None
    warp_njmax: int | None
    actuator_names: tuple[str, ...]
    actuator_joint_names: tuple[str, ...]
    actuator_joint_ids: np.ndarray
    policy_qpos_indices: np.ndarray
    policy_qvel_indices: np.ndarray
    policy_ctrl_indices: np.ndarray
    actuator_qpos_indices: np.ndarray
    actuator_qvel_indices: np.ndarray
    hold_ctrl_indices: np.ndarray
    policy_action_scale: np.ndarray
    actuator_coordinate_transform: np.ndarray
    joint_angle_offset: np.ndarray
    geom_body_ids: np.ndarray
    foot_body_ids: np.ndarray
    robot_geom_ids: np.ndarray
    foot_geom_ids: np.ndarray
    terrain_geom_ids: np.ndarray
    terrain_geom_quat: np.ndarray
    nonfoot_robot_geom_ids: np.ndarray
    nonfoot_robot_body_ids: np.ndarray
    default_qpos: np.ndarray
    default_qvel: np.ndarray
    default_ctrl: np.ndarray
    root_qpos_adr: int
    root_qvel_adr: int

    def make_data(self) -> mjx.Data:
        if self.mjx_model is None:
            raise RuntimeError(
                "this BelloModel was loaded without MJX; load with load_mjx=True "
                "before using MJX reset/step APIs"
            )
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
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig(),
    model_randomization: ModelRandomizationConfig = ModelRandomizationConfig(),
    spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE,
    use_viewer_model: bool = False,
    load_mjx: bool = True,
) -> BelloModel:
    prepared = prepare_asset(config)
    return load_prepared_bello_model(
        prepared,
        warp=warp,
        batch_size=batch_size,
        terrain=terrain,
        simulation_physics=simulation_physics,
        model_randomization=model_randomization,
        actuator_config=config.actuator_config,
        spawn_clearance=spawn_clearance,
        use_viewer_model=use_viewer_model,
        load_mjx=load_mjx,
    )


def load_prepared_bello_model(
    prepared: PreparedAsset,
    *,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig(),
    model_randomization: ModelRandomizationConfig = ModelRandomizationConfig(),
    actuator_config: ActuatorConfig = ActuatorConfig(),
    spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE,
    use_viewer_model: bool = False,
    load_mjx: bool = True,
) -> BelloModel:
    xml_path = prepared.viewer_mjcf_path if use_viewer_model else prepared.mjcf_path
    return _load_bello_model_from_path(
        xml_path,
        viewer_xml_path=prepared.viewer_mjcf_path,
        warp=warp,
        batch_size=batch_size,
        terrain=terrain,
        simulation_physics=simulation_physics,
        model_randomization=model_randomization,
        actuator_config=actuator_config,
        spawn_clearance=spawn_clearance,
        load_mjx=load_mjx,
    )


def _load_bello_model_from_path(
    xml_path: Path,
    *,
    viewer_xml_path: Path | None = None,
    warp: WarpConfig = WarpConfig(),
    batch_size: int = 1,
    terrain: TerrainConfig = ROBOT_ONLY_TERRAIN,
    simulation_physics: SimulationPhysicsConfig = SimulationPhysicsConfig(),
    model_randomization: ModelRandomizationConfig = ModelRandomizationConfig(),
    actuator_config: ActuatorConfig = ActuatorConfig(),
    spawn_clearance: float = DEFAULT_SPAWN_CLEARANCE,
    load_mjx: bool = True,
) -> BelloModel:
    if not math.isfinite(spawn_clearance) or spawn_clearance < 0.0:
        raise ValueError(
            f"spawn_clearance must be a finite nonnegative value, got {spawn_clearance}"
        )
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(path)
    path = runtime_mjcf_path(path, terrain)
    viewer_path = None
    if viewer_xml_path is not None:
        viewer_path = Path(viewer_xml_path)
        if not viewer_path.exists():
            raise FileNotFoundError(viewer_path)
        viewer_path = runtime_mjcf_path(viewer_path, terrain)

    mj_model = mujoco.MjModel.from_xml_path(str(path))

    actuator_names = tuple(
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        for actuator_id in range(mj_model.nu)
    )
    if any(name is None for name in actuator_names):
        raise ValueError("every generated actuator must have a name")
    actuator_joint_names = logical_actuator_joint_names(mj_model)
    actuator_joint_ids = np.asarray(
        [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in actuator_joint_names
        ],
        dtype=np.int32,
    )
    if np.any(actuator_joint_ids < 0):
        raise ValueError("generated actuator maps to a missing logical joint")
    actuator_coordinate_transform_matrix = actuator_coordinate_transform(
        mj_model, actuator_joint_names
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
        raise ValueError("generated Bello MJCF must include root_free")
    root_qpos_adr = int(mj_model.jnt_qposadr[root_id])
    root_qvel_adr = int(mj_model.jnt_dofadr[root_id])

    terrain_geom_ids = named_geom_indices(mj_model, terrain.name)
    foot_body_ids = foot_body_indices(mj_model)
    robot_geom_ids = robot_collision_geom_indices(mj_model)
    foot_geom_ids = robot_collision_geom_indices(
        mj_model, body_ids=set(foot_body_ids.tolist())
    )
    nonfoot_robot_geom_ids = robot_collision_geom_indices(
        mj_model, excluded_body_ids=set(foot_body_ids.tolist())
    )
    nonfoot_robot_body_ids = np.unique(
        np.asarray(mj_model.geom_bodyid[nonfoot_robot_geom_ids], dtype=np.int32)
    ).astype(np.int32)
    randomized_fields = _randomized_model_fields(
        mj_model,
        model_randomization,
        batch_size=batch_size,
        actuator_joint_ids=actuator_joint_ids,
        policy_ctrl_indices=policy_ctrl_indices,
    )
    if randomized_fields is not None:
        randomized_fields["joint_angle_offset"] = (
            actuator_coordinate_transform_matrix
            @ randomized_fields["joint_angle_offset"].T
        ).T
        _apply_native_model_sample(mj_model, randomized_fields, sample=0)
    terrain_geom_quat, randomized_geom_quat = _terrain_geom_quat_samples(
        mj_model,
        terrain,
        terrain_geom_ids,
        batch_size=batch_size,
    )
    if randomized_geom_quat is not None:
        mj_model.geom_quat[:] = randomized_geom_quat[0]
        mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))

    _apply_simulation_physics(
        mj_model,
        simulation_physics,
        foot_geom_ids=foot_geom_ids,
        terrain_geom_ids=terrain_geom_ids,
    )

    mjx_model = mjx.put_model(mj_model, impl="warp") if load_mjx else None
    if mjx_model is not None:
        model_updates = {}
        if randomized_fields is not None:
            model_updates.update(
                jnt_pos=jnp.asarray(randomized_fields["jnt_pos"]),
                body_inertia=jnp.asarray(randomized_fields["body_inertia"]),
            )
        if randomized_geom_quat is not None:
            model_updates["geom_quat"] = jnp.asarray(randomized_geom_quat)
        if model_updates:
            mjx_model = mjx_model.replace(**model_updates)

    default_qpos, default_qvel, default_ctrl = model_home_state(mj_model)
    default_qpos[root_qpos_adr : root_qpos_adr + 3] = np.asarray(
        DEFAULT_ROOT_POS, dtype=np.float32
    )
    default_qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = np.asarray(
        DEFAULT_ROOT_QUAT, dtype=np.float32
    )
    default_qpos = place_root_above_terrain(
        mj_model,
        default_qpos,
        root_qpos_adr=root_qpos_adr,
        terrain=terrain,
        clearance=spawn_clearance,
    )
    expected_default_ctrl = (
        actuator_coordinate_transform_matrix @ default_qpos[actuator_qpos_indices]
    )
    if not np.allclose(default_ctrl, expected_default_ctrl, atol=1.0e-6):
        raise ValueError(
            "generated home controls do not match the actuator coordinate transform"
        )
    policy_action_scale = actuator_action_scale(
        mj_model,
        actuator_joint_names,
        policy_ctrl_indices,
        actuator_config,
    )
    return BelloModel(
        mj_model=mj_model,
        mjx_model=mjx_model,
        xml_path=path,
        viewer_xml_path=viewer_path,
        warp_naconmax=warp.resolved_naconmax(batch_size),
        warp_naccdmax=warp.naccdmax,
        warp_njmax=warp.njmax,
        actuator_names=actuator_names,
        actuator_joint_names=actuator_joint_names,
        actuator_joint_ids=actuator_joint_ids,
        policy_qpos_indices=policy_qpos_indices,
        policy_qvel_indices=policy_qvel_indices,
        policy_ctrl_indices=policy_ctrl_indices,
        actuator_qpos_indices=actuator_qpos_indices,
        actuator_qvel_indices=actuator_qvel_indices,
        hold_ctrl_indices=hold_ctrl_indices,
        policy_action_scale=policy_action_scale,
        actuator_coordinate_transform=actuator_coordinate_transform_matrix,
        joint_angle_offset=(
            randomized_fields["joint_angle_offset"]
            if randomized_fields is not None
            else np.zeros((1, mj_model.nu), dtype=np.float32)
        ),
        geom_body_ids=np.asarray(mj_model.geom_bodyid, dtype=np.int32),
        foot_body_ids=foot_body_ids,
        robot_geom_ids=robot_geom_ids,
        foot_geom_ids=foot_geom_ids,
        terrain_geom_ids=terrain_geom_ids,
        terrain_geom_quat=terrain_geom_quat,
        nonfoot_robot_geom_ids=nonfoot_robot_geom_ids,
        nonfoot_robot_body_ids=nonfoot_robot_body_ids,
        default_qpos=default_qpos,
        default_qvel=default_qvel,
        default_ctrl=default_ctrl,
        root_qpos_adr=root_qpos_adr,
        root_qvel_adr=root_qvel_adr,
    )


def _randomized_model_fields(
    mj_model: mujoco.MjModel,
    config: ModelRandomizationConfig,
    *,
    batch_size: int,
    actuator_joint_ids: np.ndarray,
    policy_ctrl_indices: np.ndarray,
) -> dict[str, np.ndarray] | None:
    if not config.enabled:
        return None
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    rng = np.random.default_rng(config.seed)
    joint_ids = np.unique(np.asarray(actuator_joint_ids, dtype=np.int32))
    joint_body_ids = np.unique(
        np.asarray(mj_model.jnt_bodyid[joint_ids], dtype=np.int32)
    )

    jnt_pos = np.broadcast_to(
        np.asarray(mj_model.jnt_pos, dtype=np.float32),
        (batch_size, mj_model.njnt, 3),
    ).copy()
    jnt_pos[:, joint_ids] += rng.uniform(
        *config.joint_origin_range,
        size=(batch_size, joint_ids.size, 3),
    ).astype(np.float32)

    body_inertia = np.broadcast_to(
        np.asarray(mj_model.body_inertia, dtype=np.float32),
        (batch_size, mj_model.nbody, 3),
    ).copy()
    inertia_scale = rng.uniform(
        *config.joint_inertia_scale_range,
        size=(batch_size, joint_body_ids.size, 1),
    ).astype(np.float32)
    body_inertia[:, joint_body_ids] *= inertia_scale

    offset_indices = (
        np.arange(mj_model.nu, dtype=np.int32)
        if config.joint_angle_offset_scope == "all"
        else np.asarray(policy_ctrl_indices, dtype=np.int32)
    )
    joint_angle_offset = np.zeros((batch_size, mj_model.nu), dtype=np.float32)
    joint_angle_offset[:, offset_indices] = rng.uniform(
        *config.joint_angle_offset_range,
        size=(batch_size, offset_indices.size),
    ).astype(np.float32)

    return {
        "jnt_pos": jnt_pos,
        "body_inertia": body_inertia,
        "joint_angle_offset": joint_angle_offset,
    }


def _apply_native_model_sample(
    mj_model: mujoco.MjModel,
    fields: dict[str, np.ndarray],
    *,
    sample: int,
) -> None:
    mj_model.jnt_pos[:] = fields["jnt_pos"][sample]
    mj_model.body_inertia[:] = fields["body_inertia"][sample]
    mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))


def _terrain_geom_quat_samples(
    mj_model: mujoco.MjModel,
    config: TerrainConfig,
    terrain_geom_ids: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if terrain_geom_ids.size == 0:
        return np.zeros((1, 4), dtype=np.float32), None
    terrain_geom_id = int(terrain_geom_ids[0])
    if not config.randomize_slope_per_world:
        return (
            np.asarray(mj_model.geom_quat[terrain_geom_id], dtype=np.float32)[None],
            None,
        )
    if config.mode == "none":
        raise ValueError("per-world terrain slope randomization requires terrain")

    angles = random_slope_angles_batch(config, batch_size)
    terrain_quat = np.asarray(
        [slope_quat(roll, pitch) for roll, pitch in angles], dtype=np.float32
    )
    geom_quat = np.broadcast_to(
        np.asarray(mj_model.geom_quat, dtype=np.float32),
        (batch_size, mj_model.ngeom, 4),
    ).copy()
    geom_quat[:, terrain_geom_id] = terrain_quat
    return terrain_quat, geom_quat


def _apply_simulation_physics(
    mj_model: mujoco.MjModel,
    config: SimulationPhysicsConfig,
    *,
    foot_geom_ids: np.ndarray,
    terrain_geom_ids: np.ndarray,
) -> None:
    mj_model.opt.timestep = config.timestep
    contact_geom_ids = np.unique(
        np.concatenate((foot_geom_ids, terrain_geom_ids))
    ).astype(np.int32)
    if contact_geom_ids.size == 0:
        return
    if config.foot_terrain_margin is not None:
        # MuJoCo takes the maximum of the two geom margins. Put the activation
        # margin on the terrain side so foot/self-collision pairs stay unchanged.
        mj_model.geom_margin[terrain_geom_ids] = config.foot_terrain_margin
    if config.foot_terrain_solref is not None:
        mj_model.geom_solref[contact_geom_ids] = config.foot_terrain_solref
    if config.foot_terrain_solimp is not None:
        mj_model.geom_solimp[contact_geom_ids] = config.foot_terrain_solimp


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


def place_root_above_terrain(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    root_qpos_adr: int,
    terrain: TerrainConfig,
    clearance: float,
) -> np.ndarray:
    """Place the lowest robot geom at the requested terrain clearance."""

    min_robot_z = robot_geom_min_z(mj_model, qpos)
    if min_robot_z is None:
        return qpos
    if terrain.mode == "none":
        return qpos
    terrain_geom_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_GEOM, terrain.name
    )
    if terrain_geom_id < 0:
        raise ValueError(f"terrain geom {terrain.name!r} is missing from the model")
    root_xy = qpos[root_qpos_adr : root_qpos_adr + 2]
    target_min_z = float(
        terrain_surface_height(mj_model, terrain_geom_id, root_xy)
    ) + float(clearance)
    if not math.isfinite(target_min_z):
        return qpos
    adjusted = qpos.copy()
    adjusted[root_qpos_adr + 2] += target_min_z - min_robot_z
    return adjusted


def robot_geom_min_z(
    mj_model: mujoco.MjModel,
    qpos: np.ndarray,
) -> float | None:
    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, data)
    min_z: float | None = None
    for geom_id in range(mj_model.ngeom):
        if mj_model.geom_bodyid[geom_id] == 0:
            continue
        if (
            mj_model.geom_contype[geom_id] == 0
            and mj_model.geom_conaffinity[geom_id] == 0
        ):
            continue
        geom_min_z = geom_lower_z(mj_model, data, geom_id)
        min_z = geom_min_z if min_z is None else min(min_z, geom_min_z)
    return min_z


def geom_lower_z(
    mj_model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> float:
    geom_type = mj_model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(mj_model.geom_dataid[geom_id])
        vert_adr = int(mj_model.mesh_vertadr[mesh_id])
        vert_num = int(mj_model.mesh_vertnum[mesh_id])
        vertices = mj_model.mesh_vert[vert_adr : vert_adr + vert_num]
        return transformed_vertices_min_z(data, geom_id, vertices)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        sx, sy, sz = mj_model.geom_size[geom_id]
        corners = np.asarray(
            [(x, y, z) for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
            dtype=np.float64,
        )
        return transformed_vertices_min_z(data, geom_id, corners)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(data.geom_xpos[geom_id, 2] - mj_model.geom_size[geom_id, 0])
    return float(data.geom_xpos[geom_id, 2] - mj_model.geom_rbound[geom_id])


def transformed_vertices_min_z(
    data: mujoco.MjData,
    geom_id: int,
    vertices: np.ndarray,
) -> float:
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = data.geom_xpos[geom_id] + vertices @ geom_xmat.T
    return float(np.min(world_vertices[:, 2]))


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
        if mj_model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_TENDON:
            side = joint_name.split("_", maxsplit=1)[0]
            ankle_scales = (
                joint_configs[f"{side}_ankle_pitch_joint"].action_scale,
                joint_configs[f"{side}_ankle_roll_joint"].action_scale,
            )
            configured_scales = [value for value in ankle_scales if value is not None]
            if configured_scales:
                all_scale[actuator_id] = max(configured_scales)
            continue
        joint_config = joint_configs.get(joint_name)
        if joint_config is not None and joint_config.action_scale is not None:
            all_scale[actuator_id] = joint_config.action_scale
    scale = all_scale[policy_ctrl_indices]
    return scale.astype(np.float32)


def logical_actuator_joint_names(mj_model: mujoco.MjModel) -> tuple[str, ...]:
    names: list[str] = []
    for actuator_id in range(mj_model.nu):
        transmission = mj_model.actuator_trntype[actuator_id]
        target_id = int(mj_model.actuator_trnid[actuator_id, 0])
        if transmission in {
            mujoco.mjtTrn.mjTRN_JOINT,
            mujoco.mjtTrn.mjTRN_JOINTINPARENT,
        }:
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, target_id)
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
            tendon_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_TENDON, target_id
            )
            name = ANKLE_TENDON_LOGICAL_JOINT.get(tendon_name or "")
        else:
            name = None
        if name is None:
            actuator_name = mujoco.mj_id2name(
                mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            raise ValueError(
                f"actuator {actuator_name!r} has no supported logical joint mapping"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("logical actuator joint mappings must be unique")
    return tuple(names)


def actuator_coordinate_transform(
    mj_model: mujoco.MjModel, actuator_joint_names: tuple[str, ...]
) -> np.ndarray:
    transform = np.eye(mj_model.nu, dtype=np.float32)
    index_by_joint = {name: index for index, name in enumerate(actuator_joint_names)}
    for actuator_id in range(mj_model.nu):
        if mj_model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        tendon_id = int(mj_model.actuator_trnid[actuator_id, 0])
        tendon_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_TENDON, tendon_id)
        if tendon_name not in ANKLE_TENDON_LOGICAL_JOINT:
            raise ValueError(f"unsupported policy tendon actuator {tendon_name!r}")
        side, _, _, motor_index = tendon_name.split("_")
        pitch_index = index_by_joint[f"{side}_ankle_pitch_joint"]
        roll_index = index_by_joint[f"{side}_ankle_roll_joint"]
        transform[actuator_id] = 0.0
        transform[actuator_id, pitch_index] = -1.0 if motor_index == "1" else 1.0
        transform[actuator_id, roll_index] = 1.0
    return transform


def foot_body_indices(mj_model: mujoco.MjModel) -> np.ndarray:
    return np.asarray(
        [
            body_id
            for body_id in range(mj_model.nbody)
            if is_foot_body_name(
                mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            )
        ],
        dtype=np.int32,
    )


def is_foot_body_name(name: str | None) -> bool:
    return bool(name and name.endswith("ankle_roll_link"))


def named_geom_indices(mj_model: mujoco.MjModel, name: str) -> np.ndarray:
    return np.asarray(
        [
            geom_id
            for geom_id in range(mj_model.ngeom)
            if mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) == name
        ],
        dtype=np.int32,
    )


def robot_collision_geom_indices(
    mj_model: mujoco.MjModel,
    *,
    body_ids: set[int] | None = None,
    excluded_body_ids: set[int] | None = None,
) -> np.ndarray:
    body_ids = set() if body_ids is None else body_ids
    excluded_body_ids = set() if excluded_body_ids is None else excluded_body_ids
    ids: list[int] = []
    for geom_id in range(mj_model.ngeom):
        body_id = int(mj_model.geom_bodyid[geom_id])
        if body_id == 0:
            continue
        if body_ids and body_id not in body_ids:
            continue
        if body_id in excluded_body_ids:
            continue
        if (
            mj_model.geom_contype[geom_id] == 0
            and mj_model.geom_conaffinity[geom_id] == 0
        ):
            continue
        ids.append(geom_id)
    return np.asarray(ids, dtype=np.int32)
