"""Offline AMASS SMPL-X to canonical Mini clips using the unmodified pinned GMR."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from bello_mujoco.envs.rewards import CommandConfig
from bello_mujoco.hardware.gmr_retarget import (
    GmrRetargeter,
    GmrLiveRetargetConfig,
    HumanTargetFrame,
)
from bello_mujoco.hardware.smplx_contract import (
    SMPLX_BODY_JOINT_NAMES,
    SMPLX_BODY_JOINT_PARENTS,
)
from bello_mujoco.motion.clip import (
    MotionClip,
    load_motion_clip,
    motion_tracking_asset_config,
    save_motion_clip,
)
from bello_mujoco.robot_tasks import motion_task_definition
from bello_mujoco.sim import geom_lower_z, load_model
from bello_mujoco.sonic.corpus import (
    MANIFEST,
    file_sha256,
    prepare_corpus,
    verify_corpus,
)
from bello_mujoco.terrain import FLAT_TERRAIN
from bello_mujoco.repository import repository_root


@dataclass(frozen=True)
class AmassPreparationConfig:
    fps: int = 50
    train_hours: float = 8.0
    evaluation_hours: float = 0.8
    seed: int = 20260928
    min_seconds: float = 2.0
    max_seconds: float = 30.0
    max_joint_speed: float = 20.0
    max_penetration_m: float = 0.03
    max_ground_offset_m: float = 0.15
    max_floor_clearance_m: float = 0.10
    contact_height_m: float = 0.03

    def __post_init__(self):
        if any(
            not math.isfinite(value) or value <= 0 for value in asdict(self).values()
        ):
            raise ValueError("AMASS preparation settings must be positive and finite")
        if self.min_seconds > self.max_seconds:
            raise ValueError("AMASS minimum duration exceeds maximum duration")


def select_candidates(manifest: Path, config: AmassPreparationConfig) -> list[dict]:
    """Select whole sequences; hold out subjects and deduplicate source fingerprints."""
    document = yaml.safe_load(manifest.read_text())
    excluded = re.compile(
        r"stairs?|ladder|slope|chair|table|bench|treadmill|craw[l]?|lie|lying|sitting|kneel|jump|hopping|kick|cartwheel|backflip|handstand",
        re.I,
    )
    groups = defaultdict(list)
    seen = set()
    for item in document["sequences"]:
        if item["source_kind"] != "amass" or item["pose_representation"] != "smpl-x":
            continue
        # These collections concentrate on object interaction or extreme poses.
        if item["subdataset"] in {"GRAB", "PosePrior"}:
            continue
        relative = Path(item["metadata"]["relative_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or excluded.search(str(relative))
        ):
            continue
        seconds = (int(item["metadata"]["frame_count"]) - 1) / float(item["frame_rate"])
        if not config.min_seconds <= seconds <= config.max_seconds:
            continue
        identity = item.get("near_duplicate_fingerprint") or item["source_sha256"]
        if identity in seen:
            continue
        seen.add(identity)
        subject = source_subject(item["subdataset"], item["subject_id"])
        held_out = (
            int(hashlib.sha256(f"{config.seed}/{subject}".encode()).hexdigest()[:8], 16)
            % 10
            == 0
        )
        groups[item["subdataset"]].append(
            {
                "path": relative.as_posix(),
                "source_sha256": item["source_sha256"],
                "source_sequence": identity,
                "subject": subject,
                "split": "evaluation" if held_out else "train",
                "seconds": seconds,
            }
        )
    random = np.random.default_rng(config.seed)
    for entries in groups.values():
        random.shuffle(entries)
    # Round-robin subdatasets avoids allowing the largest archive to dominate.
    selected = []
    while any(groups.values()):
        for name in sorted(groups):
            if groups[name]:
                selected.append(groups[name].pop())
    return selected


def source_subject(subdataset: str, subject: str) -> str:
    if subdataset == "ACCAD":
        match = re.match(r"(?:Female|Male)\d+", subject, re.I)
        if match:
            subject = match.group().lower()
    elif subdataset == "DanceDB":
        subject = re.sub(r"^\d{8}_", "", subject).lower()
    elif subdataset == "HUMAN4D":
        subject = subject.rsplit("_", 1)[-1].lower()
    elif subdataset in {"DFaust", "MoSh", "SSM"}:
        # These MPI collections can reuse the same registered subject IDs.
        return f"MPI/{subject.rsplit('_', 1)[-1]}"
    return f"{subdataset}/{subject}"


def ground_translation(foot_heights: np.ndarray, config: AmassPreparationConfig):
    lowest = np.min(foot_heights, axis=1)
    if not np.isfinite(lowest).all():
        raise ValueError("nonfinite foot geometry")
    offset = float(np.quantile(lowest, 0.05))
    if abs(offset) > config.max_ground_offset_m:
        raise ValueError(f"excessive ground correction: {offset:.4f} m")
    if float(np.max(lowest - offset)) > config.max_floor_clearance_m:
        raise ValueError("unsupported airborne/elevated motion")
    return offset, lowest - offset


def resample_rotations(rotvecs: np.ndarray, frame: np.ndarray) -> np.ndarray:
    rotations = (
        Rotation.from_rotvec(rotvecs.reshape(-1, 3))
        .as_quat()
        .reshape(rotvecs.shape[:-1] + (4,))
    )
    lower = np.floor(frame).astype(int)
    upper = np.minimum(lower + 1, len(rotvecs) - 1)
    alpha = (frame - lower)[:, None, None]
    a, b = rotations[lower], rotations[upper]
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1))
    sine = np.sin(angle)
    result = np.where(
        sine > 1e-7,
        (np.sin((1 - alpha) * angle) * a + np.sin(alpha * angle) * b)
        / np.maximum(sine, 1e-7),
        (1 - alpha) * a + alpha * b,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def skeleton_forward(
    rest: np.ndarray, local_quaternions: np.ndarray, translation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X shape-dependent joint FK, before skinning; AMASS is already Z-up."""
    frames, joints = local_quaternions.shape[:2]
    positions = np.empty((frames, joints, 3))
    rotations = np.empty((frames, joints, 3, 3))
    local = (
        Rotation.from_quat(local_quaternions.reshape(-1, 4))
        .as_matrix()
        .reshape(frames, joints, 3, 3)
    )
    for joint, parent in enumerate(SMPLX_BODY_JOINT_PARENTS):
        if parent == -1:
            positions[:, joint] = rest[joint] + translation
            rotations[:, joint] = local[:, joint]
        else:
            positions[:, joint] = positions[:, parent] + np.einsum(
                "fij,j->fi", rotations[:, parent], rest[joint] - rest[parent]
            )
            rotations[:, joint] = rotations[:, parent] @ local[:, joint]
    quaternions = (
        Rotation.from_matrix(rotations.reshape(-1, 3, 3))
        .as_quat(scalar_first=True)
        .reshape(frames, joints, 4)
    )
    return positions, quaternions


class AmassRetargeter:
    def __init__(self, model_directory: Path, config: AmassPreparationConfig):
        self.config = config
        self.model_directory = model_directory
        self.body_models = {}
        self.model = load_model(
            motion_tracking_asset_config(asset="bello_mini"),
            terrain=FLAT_TERRAIN,
            load_mjx=False,
        )
        self.task = motion_task_definition("bello_mini")
        self.env = SimpleNamespace(model=self.model, command_config=CommandConfig())

    def human_motion(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["surface_model_type"]) != "smplx":
                raise ValueError("requires AMASS SMPL-X, not SMPL/SMPL-H")
            gender = str(data["gender"])
            fps = float(data["mocap_frame_rate"])
            root = np.asarray(data["root_orient"])
            pose = np.asarray(data["pose_body"]).reshape(len(root), 21, 3)
            trans = np.asarray(data["trans"])
            betas = np.asarray(data["betas"]).reshape(-1)
        if gender not in {"male", "female", "neutral"} or fps <= 0:
            raise ValueError("invalid AMASS gender or frame rate")
        if not all(np.isfinite(v).all() for v in (root, pose, trans, betas)):
            raise ValueError("nonfinite AMASS motion")
        if gender not in self.body_models:
            with np.load(
                self.model_directory / f"SMPLX_{gender.upper()}.npz", allow_pickle=False
            ) as body:
                regressor = np.asarray(body["J_regressor"])[:22]
                parents = np.asarray(body["kintree_table"])[0, :22].astype(int)
                parents[0] = -1
                if not np.array_equal(parents, SMPLX_BODY_JOINT_PARENTS):
                    raise ValueError("unexpected SMPL-X hierarchy")
                self.body_models[gender] = (
                    regressor @ body["v_template"],
                    np.einsum("jv,vck->jck", regressor, body["shapedirs"][:, :, :16]),
                )
        rest, directions = self.body_models[gender]
        rest = rest + np.einsum("jck,k->jc", directions[:, :, : len(betas)], betas)
        frame = (
            np.arange(int(np.floor((len(root) - 1) / fps * self.config.fps)) + 1)
            * fps
            / self.config.fps
        )
        local = resample_rotations(np.concatenate((root[:, None], pose), axis=1), frame)
        translation = np.stack(
            [
                np.interp(frame, np.arange(len(trans)), trans[:, axis])
                for axis in range(3)
            ],
            axis=-1,
        )
        positions, orientations = skeleton_forward(rest, local, translation)
        return positions, orientations, float(1.66 + 0.1 * betas[0])

    def retarget(self, source: Path, output: Path) -> dict:
        if output.exists():
            raise FileExistsError(output)
        positions, orientations, height = self.human_motion(source)
        retargeter = GmrRetargeter(
            self.env, GmrLiveRetargetConfig(human_height_m=height)
        )
        qpos = []
        names = retargeter.profile.source_body_names
        indices = [SMPLX_BODY_JOINT_NAMES.index(name) for name in names]
        for frame in range(len(positions)):
            human = HumanTargetFrame(
                transforms={
                    name: (positions[frame, index], orientations[frame, index])
                    for name, index in zip(names, indices, strict=True)
                },
                sequence=frame,
                source_timestamp_ns=1 + round(frame * 1e9 / self.config.fps),
                source_age_ms=0.0,
                received_at=0.0,
            )
            qpos.append(retargeter.retarget(human).qpos)
        qpos = np.asarray(qpos)
        model = self.model.mj_model
        logical = qpos[:, self.model.actuator_qpos_indices]
        motors = logical @ self.model.actuator_coordinate_transform.T
        constrained = motors @ self.model.target_limits.matrix.T
        bounds = self.model.target_limits.bounds
        if np.any(constrained < bounds[:, 0] - 1e-4) or np.any(
            constrained > bounds[:, 1] + 1e-4
        ):
            raise ValueError("retarget violates canonical motor/joint limits")
        data = mujoco.MjData(model)
        foot_heights = []
        for position in qpos:
            data.qpos[:] = position
            mujoco.mj_kinematics(model, data)
            foot_heights.append(
                [geom_lower_z(model, data, int(g)) for g in self.model.foot_geom_ids]
            )
        # A single clip-wide translation preserves all motion derivatives and
        # flight phases. Do not pull the root to the floor independently per frame.
        ground_offset, clearance = ground_translation(
            np.asarray(foot_heights), self.config
        )
        qpos[:, 2] -= ground_offset
        qvel = np.zeros((len(qpos), model.nv))
        for frame in range(len(qpos) - 1):
            mujoco.mj_differentiatePos(
                model, qvel[frame], 1 / self.config.fps, qpos[frame], qpos[frame + 1]
            )
        qvel[-1] = qvel[-2]
        peak_speed = float(np.max(np.abs(qvel[:, self.model.actuator_qvel_indices])))
        if peak_speed > self.config.max_joint_speed:
            raise ValueError(
                f"retarget joint speed exceeds threshold: {peak_speed:.3f} rad/s"
            )
        body_ids = [
            model.body(name).id for name in self.task.beyondmimic_tracked_body_names
        ]
        foot_ids = [model.body(name).id for name in self.model.robot.foot_body_names]
        positions, quaternions, linear, angular, contacts = [], [], [], [], []
        worst_penetration = 0.0
        foot_geoms = [
            [int(g) for g in self.model.foot_geom_ids if model.geom_bodyid[g] == body]
            for body in foot_ids
        ]
        for position, velocity in zip(qpos, qvel, strict=True):
            data.qpos[:], data.qvel[:] = position, velocity
            mujoco.mj_forward(model, data)
            positions.append(data.xpos[body_ids].copy())
            quaternions.append(data.xquat[body_ids].copy())
            body_velocity = np.empty((len(body_ids), 6))
            for index, body in enumerate(body_ids):
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_BODY, body, body_velocity[index], 0
                )
            linear.append(body_velocity[:, 3:].copy())
            angular.append(body_velocity[:, :3].copy())
            touching = np.asarray(
                [
                    min(geom_lower_z(model, data, g) for g in geoms)
                    <= self.config.contact_height_m
                    for geoms in foot_geoms
                ],
                dtype=np.float32,
            )
            for contact in data.contact:
                worst_penetration = min(worst_penetration, float(contact.dist))
            contacts.append(touching)
        if worst_penetration < -self.config.max_penetration_m:
            raise ValueError(
                f"retarget penetration exceeds threshold: {worst_penetration:.4f} m"
            )
        clip = MotionClip(
            fps=float(self.config.fps),
            joint_names=self.model.actuator_joint_names,
            body_names=self.task.beyondmimic_tracked_body_names,
            qpos=qpos.astype(np.float32),
            qvel=qvel.astype(np.float32),
            body_pos=np.asarray(positions, dtype=np.float32),
            body_quat=np.asarray(quaternions, dtype=np.float32),
            body_lin_vel=np.asarray(linear, dtype=np.float32),
            body_ang_vel=np.asarray(angular, dtype=np.float32),
            foot_contacts=np.asarray(contacts, dtype=np.float32),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        publish_motion_clip(output, clip, self.model)
        return {
            "frames": clip.num_frames,
            "seconds": clip.duration,
            "peak_joint_speed": peak_speed,
            "worst_penetration_m": worst_penetration,
            "ground_offset_m": ground_offset,
            "max_floor_clearance_m": float(np.max(clearance)),
            "contact_fraction": np.mean(contacts, axis=0).tolist(),
        }


def publish_motion_clip(output: Path, clip: MotionClip, model):
    """Validate before publication; support SMB without requiring hard links."""
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".pending-"
    ) as temporary:
        pending = Path(temporary) / "clip.npz"
        save_motion_clip(pending, clip)
        load_motion_clip(pending, model)
        # Reserve this exact destination exclusively before replacing it. A crash
        # leaves an unjournaled placeholder, which resume preserves and recomputes.
        with output.open("xb"):
            pass
        os.replace(pending, output)
        remove_generated_macos_sidecar(output)


def remove_generated_macos_sidecar(output: Path):
    """Discard only AppleDouble metadata for a file we just generated on SMB."""
    sidecar = output.with_name("._" + output.name)
    if sidecar.exists():
        with sidecar.open("rb") as stream:
            if stream.read(4) != b"\x00\x05\x16\x07":
                raise RuntimeError(f"unexpected non-AppleDouble sidecar: {sidecar}")
        sidecar.unlink()


_worker_retargeter: AmassRetargeter | None = None


def _initialize_worker(model_directory: Path, config: AmassPreparationConfig):
    global _worker_retargeter
    _worker_retargeter = AmassRetargeter(model_directory, config)


def _process_candidate(arguments: tuple[Path, Path, dict]) -> dict:
    root, output, item = arguments
    path = root / item["path"]
    if file_sha256(path) != item["source_sha256"]:
        raise ValueError(f"AMASS source checksum differs from source manifest: {path}")
    name = "amass__" + "__".join(Path(item["path"]).with_suffix("").parts)
    assert _worker_retargeter is not None
    try:
        quality = _worker_retargeter.retarget(path, output / "clips" / f"{name}.npz")
    except ValueError as error:
        return {**item, "reason": str(error)}
    return {
        **item,
        "clip": name,
        "quality": quality,
        "sha256": file_sha256(output / "clips" / f"{name}.npz"),
    }


def preparation_identity(root: Path, manifest: Path, config: AmassPreparationConfig):
    repository = repository_root()
    return {
        "schema": 1,
        "config": asdict(config),
        "source_manifest_sha256": file_sha256(manifest),
        "body_models": {
            gender: file_sha256(root / f"body_models/smplx/SMPLX_{gender}.npz")
            for gender in ("MALE", "FEMALE", "NEUTRAL")
        },
        "dependencies": {
            name: subprocess.check_output(
                ["git", "-C", str(repository / "deps" / name), "rev-parse", "HEAD"],
                text=True,
            ).strip()
            for name in ("GMR", "bello-stack")
        },
        "preparation_sources": {
            str(path.relative_to(repository)): file_sha256(path)
            for path in sorted((repository / "src/bello_mujoco").rglob("*.py"))
        },
        "lock_sha256": file_sha256(repository / "uv.lock"),
    }


def summarize_preparation(records: list[dict]) -> dict:
    accepted = [item for item in records if "clip" in item]
    return {
        "accepted": accepted,
        "rejected": [item for item in records if "reason" in item],
        "seconds": {
            split: sum(
                item["quality"]["seconds"]
                for item in accepted
                if item["split"] == split
            )
            for split in ("train", "evaluation")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="0 processes candidates until both duration budgets are met",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.max_clips < 0:
        parser.error("workers must be positive and max-clips nonnegative")
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("derived output must be outside the read-only AMASS source root")
    config = AmassPreparationConfig()
    candidates = select_candidates(args.manifest, config)
    identity = preparation_identity(args.root, args.manifest, config)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    identity_path = args.output / "identity.json"
    journal_path = args.output / "progress.jsonl"
    if args.resume:
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "preparation inputs/code/config changed; start a new corpus"
            )
        records = [json.loads(line) for line in journal_path.read_text().splitlines()]
        for item in records:
            if (
                "clip" in item
                and file_sha256(args.output / "clips" / f"{item['clip']}.npz")
                != item["sha256"]
            ):
                raise ValueError(f"prepared clip changed: {item['clip']}")
        recorded_clips = {f"{item['clip']}.npz" for item in records if "clip" in item}
        for path in (args.output / "clips").glob("*.npz"):
            if path.name not in recorded_clips:
                recovery = Path(
                    tempfile.mkdtemp(prefix="interrupted-", dir=args.output)
                )
                path.rename(recovery / path.name)
                print(f"Preserved unjournaled clip in {recovery}", flush=True)
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
        records = []
    summary = summarize_preparation(records)
    totals = summary["seconds"]
    budgets = {
        "train": config.train_hours * 3600,
        "evaluation": config.evaluation_hours * 3600,
    }
    if (args.output / "clips" / MANIFEST).exists():
        if not all(totals[key] >= budgets[key] for key in budgets):
            raise ValueError(
                "completed pilot corpus is immutable; use a new output directory"
            )
        verify_corpus(
            args.output / "clips",
            tuple(
                item["clip"]
                for item in summary["accepted"]
                if item["split"] == "evaluation"
            ),
        )
        print(json.dumps({"complete": True, "seconds": totals}), flush=True)
        return
    processed = {item["path"] for item in records}
    remaining = iter(item for item in candidates if item["path"] not in processed)
    # Bound outstanding work and consume results in selection order: worker count
    # must not change which clips make it into the immutable corpus.
    with (
        ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(args.root / "body_models/smplx", config),
        ) as pool,
        journal_path.open("a") as journal,
    ):
        while not all(totals[key] >= budgets[key] for key in budgets):
            count = 4
            if any(
                0 < budgets[key] - totals[key] < 4 * config.max_seconds
                for key in budgets
            ):
                count = 1
            if args.max_clips:
                count = min(count, args.max_clips - len(summary["accepted"]))
            if count <= 0:
                break
            batch = []
            for item in remaining:
                if totals[item["split"]] < budgets[item["split"]]:
                    batch.append((args.root, args.output, item))
                if len(batch) == count:
                    break
            if not batch:
                break
            for record in pool.map(_process_candidate, batch):
                journal.write(json.dumps(record) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                records.append(record)
                summary = summarize_preparation(records)
                totals = summary["seconds"]
                print(
                    json.dumps(
                        {
                            "processed": len(records),
                            "accepted": len(summary["accepted"]),
                            "seconds": totals,
                            "result": record,
                        }
                    ),
                    flush=True,
                )
            temporary = args.output / "preparation.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n")
            temporary.replace(args.output / "preparation.json")
    if not args.max_clips and any(totals[key] < budgets[key] for key in budgets):
        raise RuntimeError(f"insufficient accepted AMASS motion: {totals}")
    splits = {
        item["clip"]: {
            "source_sequence": item["source_sequence"],
            "split": item["split"],
        }
        for item in summary["accepted"]
    }
    prepare_corpus(
        args.output / "clips",
        splits,
        provenance={
            "identity": identity,
            "accepted": summary["accepted"],
            "seconds": totals,
            "rejected": len(summary["rejected"]),
            "rejection_categories": dict(
                Counter(item["reason"].split(":")[0] for item in summary["rejected"])
            ),
            "use": "Non-commercial research; AMASS and SMPL-X source terms apply",
            "storage_migration": (
                json.loads((args.output / "storage-migration.json").read_text())
                if (args.output / "storage-migration.json").exists()
                else None
            ),
        },
    )
    remove_generated_macos_sidecar(args.output / "clips" / MANIFEST)


if __name__ == "__main__":
    main()
