from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).parents[1]
    / "tools"
    / "robot_description"
    / "build_review_manifest.py"
)
MODULE_SPEC = importlib.util.spec_from_file_location(
    "build_review_manifest", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)

AssemblyWalker = MODULE.AssemblyWalker
flattened_column_major = MODULE.flattened_column_major
motor_model = MODULE.motor_model
is_anonymous_end_effector_source = MODULE.is_anonymous_end_effector_source


def placement(x: float) -> dict[str, object]:
    return {
        "matrix4": [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }


def shape(minimum: float, maximum: float) -> dict[str, object]:
    return {
        "center_of_mass_mm": [(minimum + maximum) / 2.0, 0.5, 0.5],
        "bounding_box_mm": {
            "min": [minimum, 0.0, 0.0],
            "max": [maximum, 1.0, 1.0],
        },
    }


def obj(
    name: str,
    type_id: str,
    *,
    x: float,
    children: list[str] | None = None,
    linked_object: str | None = None,
    label: str | None = None,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> dict[str, object]:
    return {
        "name": name,
        "label": label or name,
        "type_id": type_id,
        "children": children or [],
        "linked_object": linked_object,
        "local_placement": placement(x),
        "shape": shape(*bounds),
    }


def test_walker_expands_group_and_link_placements() -> None:
    inventory = {
        "summary": {"root_objects": ["root"]},
        "objects": [
            obj(
                "root",
                "App::LinkGroup",
                x=0.0,
                children=["part", "part_link", "motor_group"],
                bounds=(1.0, 21.0),
            ),
            obj("part", "Part::Feature", x=1.0, bounds=(1.0, 2.0)),
            obj(
                "part_link",
                "App::Link",
                x=10.0,
                linked_object="part",
                bounds=(10.0, 11.0),
            ),
            obj(
                "motor_group",
                "App::LinkGroup",
                x=20.0,
                children=["motor_part"],
                label="RS03 v1",
                bounds=(20.0, 21.0),
            ),
            obj("motor_part", "Part::Feature", x=0.0),
        ],
    }
    walker = AssemblyWalker(inventory)

    walker.walk()

    assert len(walker.occurrences) == 3
    part_occurrences = [
        occurrence
        for occurrence in walker.occurrences
        if occurrence["source_name"] == "part"
    ]
    assert len(part_occurrences) == 2
    translations = sorted(
        np.asarray(occurrence["transform"]).reshape(4, 4).T[0, 3]
        for occurrence in part_occurrences
    )
    assert translations == [0.0, 9.0]
    assert len(walker.motor_assemblies) == 1
    assert walker.motor_assemblies[0]["suggested_model"] == "ROBSTRIDE_03"
    assert walker.motor_assemblies[0]["frame_transform"][12:15] == [20.0, 0.0, 0.0]
    motor_occurrence = next(
        occurrence
        for occurrence in walker.occurrences
        if occurrence["source_name"] == "motor_part"
    )
    assert motor_occurrence["motor_assembly"] == 0
    assert motor_occurrence["bounds_mm"]["min"] == [20.0, 0.0, 0.0]


def test_column_major_serialization_matches_webgl() -> None:
    value = np.eye(4)
    value[:3, 3] = [1.0, 2.0, 3.0]

    serialized = flattened_column_major(value)

    assert serialized[12:15] == [1.0, 2.0, 3.0]


def test_motor_labels_do_not_treat_spacers_as_actuators() -> None:
    assert motor_model("RS06 v2") == "ROBSTRIDE_06"
    assert motor_model("RS06_Spacer v2") is None
    assert motor_model("RS03_Spacer v1") is None


def test_human_hand_assemblies_are_anonymous_end_effectors() -> None:
    assert is_anonymous_end_effector_source("ASM_HUMAN_HAND_V2_0802 v2")
    assert is_anonymous_end_effector_source("ASM_HUMAN_HAND_V2_0802 v2 R v1")
    assert not is_anonymous_end_effector_source("Waist_Outer_Frame v3")
