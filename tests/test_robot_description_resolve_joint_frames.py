from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "resolve_joint_frame_review.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "resolve_joint_frame_review", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


@pytest.fixture
def manifest() -> dict:
    return {
        "schema_version": 1,
        "review_contract": "frames_v1",
        "review_stage": "joint_frames",
        "source": {"sha256": "step"},
        "approved_joint_link_review": {"sha256": "topology"},
        "summary": {},
        "joints": [
            {"name": "joint_a", "axis_cad": [1.0, 0.0, 0.0]},
            {"name": "joint_b", "axis_cad": [0.0, 0.0, 1.0]},
        ],
    }


def review(decisions: dict[str, str]) -> dict:
    return {
        "review_contract": "frames_v1",
        "source": {"sha256": "step"},
        "approved_joint_link_review": {"sha256": "topology"},
        "joint_frame_decisions": decisions,
    }


def test_resolve_frames_applies_only_requested_axis_flips(manifest: dict) -> None:
    resolved = MODULE.resolve_frames(
        manifest,
        review({"joint_a": "confirmed", "joint_b": "flip"}),
        review_sha256="review",
    )

    assert resolved["joints"][0]["axis_cad"] == [1.0, 0.0, 0.0]
    assert resolved["joints"][1]["axis_cad"] == [0.0, 0.0, -1.0]
    assert resolved["summary"]["axis_flip_count"] == 1
    assert resolved["review_contract"] == MODULE.RESOLVED_CONTRACT
    assert manifest["joints"][1]["axis_cad"] == [0.0, 0.0, 1.0]


def test_resolve_frames_rejects_missing_decisions(manifest: dict) -> None:
    with pytest.raises(ValueError, match="inventory mismatch"):
        MODULE.resolve_frames(
            manifest,
            review({"joint_a": "confirmed"}),
            review_sha256="review",
        )


def test_resolve_frames_rejects_unresolved_corrections(manifest: dict) -> None:
    with pytest.raises(ValueError, match="unresolved"):
        MODULE.resolve_frames(
            manifest,
            review({"joint_a": "confirmed", "joint_b": "correction"}),
            review_sha256="review",
        )
