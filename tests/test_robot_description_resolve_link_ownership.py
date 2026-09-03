from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "resolve_link_ownership_review.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "resolve_link_ownership_review", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = MODULE
MODULE_SPEC.loader.exec_module(MODULE)


def test_world_translation_preserves_orientation() -> None:
    transform = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 129.5, -800, 45, 1]

    result = MODULE.translated_transform(transform, [-259.0, 0.0, 0.0])

    assert result[:12] == transform[:12]
    assert result[12:] == [-129.5, -800.0, 45.0, 1.0]


def test_world_x_reflection_changes_handedness_and_position() -> None:
    transform = [1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 1]

    result = MODULE.mirrored_x_transform(transform)

    assert result == [-1, 2, 3, 0, -4, 5, 6, 0, -7, 8, 9, 0, -10, 11, 12, 1]


def test_review_validation_rejects_bilateral_owner_mismatch() -> None:
    manifest = {
        "review_contract": MODULE.INPUT_CONTRACT,
        "source": {"sha256": "step"},
        "source_manifests": {"frames": "hash"},
        "links": [
            {
                "name": "left_link",
                "physical_geometry_policy": "reviewed_cad_ownership",
            },
            {
                "name": "right_link",
                "physical_geometry_policy": "reviewed_cad_ownership",
            },
        ],
        "groups": [{"id": "left"}, {"id": "right"}],
        "bilateral_group_pairs": [{"left_group_id": "left", "right_group_id": "right"}],
    }
    review = {
        "review_contract": MODULE.INPUT_CONTRACT,
        "source": {"sha256": "step"},
        "source_manifests": {"frames": "hash"},
        "ownership_decisions": {
            "left": {"status": "confirmed", "link": "left_link"},
            "right": {"status": "confirmed", "link": "left_link"},
        },
    }

    with pytest.raises(ValueError, match="bilateral ownership"):
        MODULE.validate_review(manifest, review)


def test_completion_source_validation_rejects_shifted_group_identity() -> None:
    groups = {
        group_id: {
            "id": group_id,
            "kind": "structure",
            "label": label,
            "reviewed_link": "left_knee_link",
            "side": "left",
        }
        for group_id, label in MODULE.EXPECTED_COMPLETION_GROUP_LABELS.items()
    }
    groups["ownership-010"]["label"] = "a different future CAD group"

    with pytest.raises(ValueError, match="expected label 'Solid038'"):
        MODULE.validate_completion_sources(groups)


def test_camera_catalog_override_replaces_material_mass() -> None:
    override = MODULE.CATALOG_COMPONENT_OVERRIDES[0]
    group = {
        "path_prefix": override["path_prefix"],
        "label": override["label"],
        "kind": "structure",
        "reviewed_link": override["reviewed_link"],
        "estimated_material_mass_kg": 0.011357,
        "occurrences": [
            {"source_name": name}
            for name in sorted(override["occurrence_source_names"])
        ],
    }

    MODULE.apply_catalog_component_overrides([group])

    assert group["kind"] == "catalog_component"
    assert group["catalog_model"] == "ORBBEC_ASTRA_STEREO_S_U3"
    assert MODULE.group_mass_kg(group) == 0.030
    assert group["replaced_material_mass_kg"] == 0.011357
