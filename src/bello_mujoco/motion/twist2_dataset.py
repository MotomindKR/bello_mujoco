"""Auditable dataset-composition checks for the public TWIST2 release."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from bello_mujoco.envs.twist2 import TWIST2_RELEASED_MOTION_COUNTS


@dataclass(frozen=True)
class Twist2DatasetAudit:
    clip_count: int
    train_clip_count: int
    evaluation_clip_count: int
    source_counts: tuple[tuple[str, int], ...]
    train_source_counts: tuple[tuple[str, int], ...]
    unit_manifest_weights: bool
    train_unit_manifest_weights: bool
    matches_released_training_composition: bool

    def __post_init__(self) -> None:
        if self.clip_count <= 0 or self.train_clip_count <= 0:
            raise ValueError("TWIST2 dataset audit requires training clips")
        if self.evaluation_clip_count < 0:
            raise ValueError("TWIST2 evaluation clip count must be nonnegative")
        if self.train_clip_count + self.evaluation_clip_count != self.clip_count:
            raise ValueError("TWIST2 dataset split counts do not sum to clip_count")
        for name, counts, expected in (
            ("source_counts", self.source_counts, self.clip_count),
            ("train_source_counts", self.train_source_counts, self.train_clip_count),
        ):
            if tuple(sorted(counts)) != counts:
                raise ValueError(f"TWIST2 {name} must be sorted")
            if len({source for source, _ in counts}) != len(counts):
                raise ValueError(f"TWIST2 {name} contains duplicate sources")
            if any(not source or count <= 0 for source, count in counts):
                raise ValueError(f"TWIST2 {name} entries must be nonempty and positive")
            if sum(count for _, count in counts) != expected:
                raise ValueError(f"TWIST2 {name} does not sum to its clip count")
        released = tuple(sorted(TWIST2_RELEASED_MOTION_COUNTS))
        matches = (
            self.train_unit_manifest_weights and self.train_source_counts == released
        )
        if self.matches_released_training_composition != matches:
            raise ValueError(
                "TWIST2 released training-composition result is inconsistent"
            )


def audit_twist2_dataset(index_path: Path) -> Twist2DatasetAudit:
    path = Path(index_path).expanduser().resolve()
    index = json.loads(path.read_text(encoding="utf-8"))
    clips = index.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("TWIST2 dataset index must contain a nonempty clips list")
    counts: dict[str, int] = {}
    train_counts: dict[str, int] = {}
    unit_weights = True
    train_unit_weights = True
    train_clip_count = 0
    evaluation_clip_count = 0
    for clip in clips:
        if not isinstance(clip, dict):
            raise TypeError("TWIST2 dataset clip entries must be mappings")
        source_id = clip.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("TWIST2 dataset clips require source_id")
        split = clip.get("split")
        if split not in {"train", "evaluation"}:
            raise ValueError("TWIST2 dataset clips require a train/evaluation split")
        counts[source_id] = counts.get(source_id, 0) + 1
        unit_weight = clip.get("sampling_weight") == 1.0
        unit_weights &= unit_weight
        if split == "train":
            train_clip_count += 1
            train_counts[source_id] = train_counts.get(source_id, 0) + 1
            train_unit_weights &= unit_weight
        else:
            evaluation_clip_count += 1
    source_counts = tuple(sorted(counts.items()))
    train_source_counts = tuple(sorted(train_counts.items()))
    released = tuple(sorted(TWIST2_RELEASED_MOTION_COUNTS))
    return Twist2DatasetAudit(
        clip_count=len(clips),
        train_clip_count=train_clip_count,
        evaluation_clip_count=evaluation_clip_count,
        source_counts=source_counts,
        train_source_counts=train_source_counts,
        unit_manifest_weights=unit_weights,
        train_unit_manifest_weights=train_unit_weights,
        matches_released_training_composition=(
            train_unit_weights and train_source_counts == released
        ),
    )


def require_released_twist2_dataset(index_path: Path) -> Twist2DatasetAudit:
    audit = audit_twist2_dataset(index_path)
    if not audit.matches_released_training_composition:
        expected_count = sum(count for _, count in TWIST2_RELEASED_MOTION_COUNTS)
        raise ValueError(
            "TWIST2 released-dataset parity was requested, but the index does not "
            f"use the exact {expected_count}-clip public composition as its "
            "training split with unit manifest weights"
        )
    return audit
