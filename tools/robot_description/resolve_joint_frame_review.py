"""Apply an approved Bello Mini joint-frame review to its source manifest."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
RESOLVED_CONTRACT = "bello_mini_resolved_joint_frames_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_review(manifest: dict[str, Any], review: dict[str, Any]) -> None:
    if review.get("review_contract") != manifest["review_contract"]:
        raise ValueError("joint-frame review contract does not match manifest")
    if review.get("source", {}).get("sha256") != manifest["source"]["sha256"]:
        raise ValueError("joint-frame review STEP checksum does not match manifest")
    if review.get("approved_joint_link_review") != manifest.get(
        "approved_joint_link_review"
    ):
        raise ValueError("joint-frame review is bound to a different topology review")
    decisions = review.get("joint_frame_decisions")
    if not isinstance(decisions, dict):
        raise ValueError("joint-frame decisions must be a mapping")
    expected = {joint["name"] for joint in manifest["joints"]}
    if set(decisions) != expected:
        missing = sorted(expected - set(decisions))
        unexpected = sorted(set(decisions) - expected)
        raise ValueError(
            f"joint-frame decision inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    invalid = {
        name: decision
        for name, decision in decisions.items()
        if decision not in {"confirmed", "flip"}
    }
    if invalid:
        raise ValueError(f"unresolved joint-frame decisions: {invalid}")


def resolve_frames(
    manifest: dict[str, Any], review: dict[str, Any], *, review_sha256: str
) -> dict[str, Any]:
    validate_review(manifest, review)
    resolved = deepcopy(manifest)
    decisions = review["joint_frame_decisions"]
    flip_count = 0
    for joint in resolved["joints"]:
        decision = decisions[joint["name"]]
        if decision == "flip":
            joint["axis_cad"] = [-float(value) for value in joint["axis_cad"]]
            flip_count += 1
        joint["frame_review_decision"] = decision
        joint["axis_sign_status"] = (
            "reviewed_flipped" if decision == "flip" else "reviewed_confirmed"
        )
    resolved["schema_version"] = SCHEMA_VERSION
    resolved["review_contract"] = RESOLVED_CONTRACT
    resolved["review_stage"] = "resolved_joint_frames"
    resolved["approved_frame_review"] = {
        "review_contract": review["review_contract"],
        "sha256": review_sha256,
    }
    resolved["summary"]["axis_flip_count"] = flip_count
    resolved["axis_sign_policy"] = (
        "every semantic positive axis direction is source-bound and human-reviewed"
    )
    return resolved


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    review_path = args.review.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    resolved = resolve_frames(manifest, review, review_sha256=file_sha256(review_path))
    atomic_write_json(args.output, resolved)
    print(f"wrote resolved joint frames: {args.output}")
    print(json.dumps(resolved["summary"], indent=2))


if __name__ == "__main__":
    main()
