"""Whole-sequence LAFAN1 preparation for the canonical Mini tracking model."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import math
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import hydra
import mujoco
import numpy as np

from bello_mujoco.configuration import command_config, config_path
from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware.gmr_retarget import (
    GmrLiveRetargetConfig,
    GmrRetargeter,
)
from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.bvh import lafan1_targets, read_bvh
from bello_mujoco.motion.calibration import (
    CalibratedRetargeter,
    KinematicRetargetConfig,
    planted_feet,
)
from bello_mujoco.motion.clip import (
    MotionClip,
    load_motion_clip,
    motion_tracking_asset_config,
    save_motion_clip,
)
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import geom_lower_z, load_model
from bello_mujoco.sonic.corpus import file_sha256, prepare_corpus, verify_corpus
from bello_mujoco.sonic.amass import SmplxMotionReader
from bello_mujoco.terrain import FLAT_TERRAIN


@dataclass(frozen=True)
class LafanPreparationConfig:
    source_directory: Path
    output_directory: Path
    training_sequences: tuple[str, ...]
    evaluation_sequences: tuple[str, ...]
    fps: int = 50
    human_height_m: float = 1.75
    workers: int = 4
    human_source: Literal["bvh", "smplx"] = "bvh"
    smplx_directory: Path | None = None
    smplx_model_directory: Path | None = None
    penetration_tolerance_m: float = 0.001
    contact_height_m: float = 0.03
    retargeting: KinematicRetargetConfig = field(
        default_factory=KinematicRetargetConfig
    )

    def __post_init__(self):
        for name in (
            "fps",
            "workers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("human_height_m", "penetration_tolerance_m", "contact_height_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.human_source not in ("bvh", "smplx"):
            raise ValueError("LAFAN human_source must be bvh or smplx")
        paths = (self.smplx_directory, self.smplx_model_directory)
        if self.human_source == "smplx" and any(path is None for path in paths):
            raise ValueError("SMPL-X sources require motion and body-model directories")
        if self.human_source == "bvh" and any(path is not None for path in paths):
            raise ValueError("SMPL-X directories are unused with a BVH human source")
        sequences = self.training_sequences + self.evaluation_sequences
        if not self.training_sequences or not self.evaluation_sequences:
            raise ValueError(
                "LAFAN requires explicit training and evaluation sequences"
            )
        if len(sequences) != len(set(sequences)) or any(
            Path(name).name != name or Path(name).suffix or not name
            for name in sequences
        ):
            raise ValueError(
                "LAFAN sequence names must be unique, flat extensionless names"
            )


def pose_velocities(model, qpos, fps):
    """Central pose differences, with free-joint angular velocities at the current pose."""
    qvel = np.zeros((len(qpos), model.nv))
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    for frame in range(len(qpos)):
        left, right = max(0, frame - 1), min(len(qpos) - 1, frame + 1)
        mujoco.mj_differentiatePos(
            model, qvel[frame], (right - left) / fps, qpos[left], qpos[right]
        )
        for joint in free_joints:
            p, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            world, inverse, local = np.empty(3), np.empty(4), np.empty(3)
            mujoco.mju_rotVecQuat(
                world, qvel[frame, v + 3 : v + 6], qpos[left, p + 3 : p + 7]
            )
            mujoco.mju_negQuat(inverse, qpos[frame, p + 3 : p + 7])
            mujoco.mju_rotVecQuat(local, world, inverse)
            qvel[frame, v + 3 : v + 6] = local
    return qvel


def clip_from_poses(model, body_names, qpos, fps, contact_height_m):
    """Differentiate poses and cache link-origin FK/twists, never COM twists."""
    mj_model = model.mj_model
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape != (len(qpos), mj_model.nq) or len(qpos) < 2:
        raise ValueError("offline references require at least two complete poses")
    if not np.isfinite(qpos).all():
        raise ValueError("offline references contain nonfinite poses")
    logical = qpos[:, model.actuator_qpos_indices]
    constrained = (
        logical @ model.actuator_coordinate_transform.T @ model.target_limits.matrix.T
    )
    bounds = model.target_limits.bounds
    if np.any(constrained < bounds[:, 0] - 1e-4) or np.any(
        constrained > bounds[:, 1] + 1e-4
    ):
        raise ValueError("retarget violates canonical motor/joint limits")
    qvel = pose_velocities(mj_model, qpos, fps)
    data = mujoco.MjData(mj_model)
    body_ids = [mj_model.body(name).id for name in body_names]
    foot_ids = [mj_model.body(name).id for name in model.robot.foot_body_names]
    foot_geoms = [
        [int(g) for g in model.foot_geom_ids if mj_model.geom_bodyid[g] == body]
        for body in foot_ids
    ]
    positions, quaternions, linear, angular, contacts = [], [], [], [], []
    worst_floor, worst_self = 0.0, 0.0
    plane = mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    for pose, velocity in zip(qpos, qvel, strict=True):
        data.qpos[:], data.qvel[:] = pose, velocity
        mujoco.mj_forward(mj_model, data)
        positions.append(data.xpos[body_ids].copy())
        quaternions.append(data.xquat[body_ids].copy())
        twists = np.empty((len(body_ids), 6))
        for index, body in enumerate(body_ids):
            mujoco.mj_objectVelocity(
                mj_model, data, mujoco.mjtObj.mjOBJ_XBODY, body, twists[index], 0
            )
        angular.append(twists[:, :3].copy())
        linear.append(twists[:, 3:].copy())
        contacts.append(
            [
                min(geom_lower_z(mj_model, data, geom) for geom in geoms)
                <= contact_height_m
                for geoms in foot_geoms
            ]
        )
        for contact in data.contact:
            if any(plane[g] for g in contact.geom):
                worst_floor = min(worst_floor, float(contact.dist))
            else:
                worst_self = min(worst_self, float(contact.dist))
    clip = MotionClip(
        fps=float(fps),
        joint_names=model.actuator_joint_names,
        body_names=tuple(body_names),
        qpos=qpos.astype(np.float32),
        qvel=qvel.astype(np.float32),
        body_pos=np.asarray(positions, dtype=np.float32),
        body_quat=np.asarray(quaternions, dtype=np.float32),
        body_lin_vel=np.asarray(linear, dtype=np.float32),
        body_ang_vel=np.asarray(angular, dtype=np.float32),
        foot_contacts=np.asarray(contacts, dtype=np.float32),
    )
    return clip, {
        "frames": len(qpos),
        "seconds": clip.duration,
        "worst_floor_penetration_m": worst_floor,
        "worst_self_penetration_m": worst_self,
        "peak_joint_speed_rad_s": float(
            np.abs(qvel[:, model.actuator_qvel_indices]).max()
        ),
        "root_height_range_m": [float(qpos[:, 2].min()), float(qpos[:, 2].max())],
        "contact_fraction": np.mean(contacts, axis=0).tolist(),
    }


def retarget_sequence(config: LafanPreparationConfig, name: str):
    path = config.source_directory / f"{name}.bvh"
    source_hash = file_sha256(path)
    motion = read_bvh(path).resample(config.fps)
    source_targets = lafan1_targets(motion)
    feet = ("left_foot", "right_foot")
    labels = planted_feet(
        np.stack([source_targets[n][0] for n in feet], axis=1),
        np.stack([source_targets[n][1] for n in feet], axis=1),
        config.fps,
        config.retargeting.source_contacts,
    )
    targets = source_targets
    provenance = {"human_source": config.human_source}
    if config.human_source == "smplx":
        native_path = config.smplx_directory / f"{name}.npz"
        positions, quaternions, _ = SmplxMotionReader(
            config.smplx_model_directory
        ).read(native_path, config.fps)
        if len(positions) != len(motion.position):
            raise ValueError(
                "native SMPL-X and BVH source clocks have different lengths"
            )
        targets = {
            name: (positions[:, index], quaternions[:, index])
            for index, name in enumerate(SMPLX_BODY_JOINT_NAMES)
        }
        with np.load(native_path, allow_pickle=False) as data:
            gender = str(data["gender"])
        provenance.update(
            smplx_source_sha256=file_sha256(native_path),
            smplx_model_sha256=file_sha256(
                config.smplx_model_directory / f"SMPLX_{gender.upper()}.npz"
            ),
        )
    model = load_model(
        motion_tracking_asset_config(asset="bello_mini"),
        terrain=FLAT_TERRAIN,
        load_mjx=False,
    )
    wrapper = GmrRetargeter(
        SimpleNamespace(model=model, command_config=CommandConfig()),
        GmrLiveRetargetConfig(human_height_m=config.human_height_m),
    )
    calibrated = CalibratedRetargeter(
        model, wrapper.retargeter, config.fps, config.retargeting
    )
    qpos, calibration = calibrated.retarget(targets, labels)
    clip, report = clip_from_poses(
        model,
        motion_task_definition("bello_mini").beyondmimic_tracked_body_names,
        qpos,
        config.fps,
        config.contact_height_m,
    )
    speed_ratio = float(
        np.max(
            np.abs(clip.qvel[:, model.actuator_qvel_indices])
            / (calibrated.bounds.step * config.fps)
        )
    )
    report["peak_joint_speed_limit_ratio"] = speed_ratio
    if speed_ratio > 1.0001:
        raise ValueError(
            f"reference violates canonical joint speed limits: {speed_ratio}"
        )
    if report["worst_floor_penetration_m"] < -config.penetration_tolerance_m:
        raise ValueError("reference violates floor penetration tolerance")
    if report["worst_self_penetration_m"] < -(
        config.retargeting.contacts.max_self_penetration_m
        + config.penetration_tolerance_m
    ):
        raise ValueError("reference violates self penetration tolerance")
    destination = config.output_directory / "corpus" / f"{name}.npz"
    with destination.open("xb") as stream:
        save_motion_clip(stream, clip)
    load_motion_clip(destination, model, as_numpy=True)
    return {
        "sequence": name,
        "source_sha256": source_hash,
        **provenance,
        **calibration,
        **report,
    }


def _retarget_job(args):
    config, name = args
    try:
        return retarget_sequence(config, name)
    except Exception as error:
        return {"sequence": name, "error": f"{type(error).__name__}: {error}"}


def prepare_lafan(config: LafanPreparationConfig):
    names = config.training_sequences + config.evaluation_sequences
    for name in names:
        if not (config.source_directory / f"{name}.bvh").is_file():
            raise FileNotFoundError(config.source_directory / f"{name}.bvh")
        if config.human_source == "smplx":
            native = config.smplx_directory / f"{name}.npz"
            if not native.is_file():
                raise FileNotFoundError(native)
    config.output_directory.mkdir(parents=True, exist_ok=False)
    (config.output_directory / "corpus").mkdir()
    settings = asdict(config)
    for key in (
        "source_directory",
        "output_directory",
        "smplx_directory",
        "smplx_model_directory",
    ):
        value = getattr(config, key)
        settings[key] = str(value.resolve()) if value is not None else None
    (config.output_directory / "settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
    reports = []
    with (
        ProcessPoolExecutor(
            max_workers=config.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool,
        (config.output_directory / "preparation.jsonl").open("x") as journal,
    ):
        for report in pool.map(_retarget_job, [(config, name) for name in names]):
            reports.append(report)
            row = json.dumps(report, allow_nan=False)
            journal.write(row + "\n")
            journal.flush()
            print(row, flush=True)
    failures = [report for report in reports if "error" in report]
    if failures:
        raise ValueError(
            f"LAFAN preparation failed for {len(failures)}/{len(names)} sequences; see preparation.jsonl"
        )
    splits = {
        name: {
            "split": "train" if name in config.training_sequences else "evaluation",
            "source_sequence": f"lafan1/{name}",
        }
        for name in names
    }
    prepare_corpus(
        config.output_directory / "corpus",
        splits,
        provenance={
            "dataset": "Ubisoft LAFAN1",
            "preparation": settings,
            "sequences": reports,
        },
    )
    return verify_corpus(
        config.output_directory / "corpus", config.evaluation_sequences
    )


@hydra.main(
    version_base=None, config_path=config_path("commands"), config_name="prepare_lafan"
)
def main(config):
    print(prepare_lafan(command_config(LafanPreparationConfig, config)))


if __name__ == "__main__":
    main()
