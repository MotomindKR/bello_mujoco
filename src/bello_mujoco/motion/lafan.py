"""Whole-sequence LAFAN1 preparation for the canonical Mini tracking model."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing
from pathlib import Path
from typing import Literal

import hydra
import numpy as np

from bello_mujoco.configuration import command_config, config_path
from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_NAMES
from bello_mujoco.motion.bvh import lafan1_targets, read_bvh
from bello_mujoco.motion.retarget import retarget_gmr
from bello_mujoco.motion.smplx import SmplxMotionReader
from bello_mujoco.motion.clip import (
    motion_tracking_asset_config,
    save_motion_clip,
)
from bello_mujoco.sim import load_model
from bello_mujoco.sonic.corpus import file_sha256, prepare_corpus, verify_corpus
from bello_mujoco.terrain import FLAT_TERRAIN


@dataclass(frozen=True)
class LafanPreparationConfig:
    source_directory: Path
    output_directory: Path
    training_sequences: tuple[str, ...]
    evaluation_sequences: tuple[str, ...]
    fps: int
    human_height_m: float
    workers: int
    human_source: Literal["bvh", "smplx"]
    smplx_directory: Path | None
    smplx_model_directory: Path | None
    contact_height_m: float

    def __post_init__(self):
        for name in ("fps", "workers"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("human_height_m", "contact_height_m"):
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


def retarget_sequence(config: LafanPreparationConfig, name: str):
    path = config.source_directory / f"{name}.bvh"
    source_hash = file_sha256(path)
    motion = read_bvh(path).resample(config.fps)
    targets = lafan1_targets(motion)
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
    clip, report = retarget_gmr(
        model,
        targets,
        fps=config.fps,
        human_height_m=config.human_height_m,
        contact_height_m=config.contact_height_m,
    )
    destination = config.output_directory / "corpus" / f"{name}.npz"
    with destination.open("xb") as stream:
        save_motion_clip(stream, clip)
    return {
        "sequence": name,
        "source_sha256": source_hash,
        **provenance,
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
