import io
import tarfile
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.motion.bones import (
    BUCKETS,
    BonesPreparationConfig,
    bucket,
    extract_sources,
    family,
    safe_member,
    select_bones,
    validate_quarantine,
)
from bello_mujoco.motion.bvh import BvhMotion, soma_targets


def metadata_row(name, **overrides):
    return {
        "filename": name,
        "move_soma_uniform_path": f"soma_uniform/bvh/date/{name}.bvh",
        "move_duration_frames": "1201",
        "content_name": name,
        "is_mirror": "False",
        "package": "Communication",
        "category": "Gestures",
        "content_body_position": "standing",
        "content_type_of_movement": "gesture",
        "content_props": "0",
        **overrides,
    }


def test_soma_bind_removes_independent_limb_axes_and_preserves_global_turns():
    names = (
        "Hips",
        "Spine2",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftToe",
        "RightToe",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    )
    rename = {
        "Spine2": "Chest",
        "LeftUpLeg": "LeftLeg",
        "RightUpLeg": "RightLeg",
        "LeftLeg": "LeftShin",
        "RightLeg": "RightShin",
        "LeftToe": "LeftToeBase",
        "RightToe": "RightToeBase",
    }
    rng = np.random.default_rng(8)
    position = np.repeat(rng.normal(size=(1, len(names), 3)), 3, axis=0)
    q = np.repeat(rng.normal(size=(1, len(names), 4)), 3, axis=0)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    motion = BvhMotion(names, (-1,) + (0,) * (len(names) - 1), position, q, 0.02)
    bind = replace(motion, names=tuple(rename.get(n, n) for n in names))
    up = Rotation.from_euler("x", 90, degrees=True)
    for _, quaternion in soma_targets(bind, bind).values():
        np.testing.assert_allclose(
            Rotation.from_quat(quaternion, scalar_first=True).as_matrix(),
            np.repeat(up.as_matrix()[None], 3, axis=0),
            atol=1e-12,
        )
    turn = Rotation.from_euler("y", 35, degrees=True)
    turned = q.copy()
    turned[:, 0] = (turn * Rotation.from_quat(q[:, 0], scalar_first=True)).as_quat(
        scalar_first=True
    )
    for _, quaternion in soma_targets(replace(bind, quaternion=turned), bind).values():
        np.testing.assert_allclose(
            Rotation.from_quat(quaternion, scalar_first=True).as_matrix(),
            np.repeat((up * turn).as_matrix()[None], 3, axis=0),
            atol=1e-12,
        )
    with pytest.raises(ValueError, match="SOMA skeleton is missing"):
        soma_targets(motion, bind)
    dynamic_bind = bind.position.copy()
    dynamic_bind[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="constant rest pose"):
        soma_targets(bind, replace(bind, position=dynamic_bind))


@pytest.mark.parametrize("name", ["../x", "/root/x", "a/../../b", "a\\b", ""])
def test_source_paths_cannot_escape_the_archive(name):
    with pytest.raises(ValueError, match="confined"):
        safe_member(name)


@pytest.mark.parametrize(
    "position", ["sitting on floor", "lying on the floor", "crawling", "kneeling"]
)
def test_floor_actions_remain_candidates(position):
    assert bucket(metadata_row("move", content_body_position=position)) == "ground"


def test_skill_labels_do_not_match_substrings_or_treat_all_gaming_as_sport():
    assert bucket(metadata_row("dance_hips_sites", package="Dances")) == "dance"
    assert bucket(metadata_row("looking_around", package="Gaming")) == "other"
    assert bucket(metadata_row("punch_right", package="Gaming")) == "sport"


def test_mirrors_and_different_takes_share_the_content_family():
    assert family(metadata_row("take_001__A1", content_name="dance_M")) == family(
        metadata_row("take_002__A2", content_name="dance")
    )


def test_selection_is_balanced_deterministic_and_family_disjoint():
    rows = []
    categories = {
        "ground": {"content_body_position": "sitting on floor"},
        "jump": {"content_type_of_movement": "jumping"},
        "dance": {"package": "Dances"},
        "sport": {"package": "Sport"},
        "locomotion": {"package": "Locomotion"},
        "other": {},
    }
    for kind, fields in categories.items():
        for index in range(200):
            for mirror in (False, True):
                name = f"{kind}_{index}" + ("_M" if mirror else "")
                rows.append(metadata_row(name, is_mirror=str(mirror), **fields))
    rows.append(metadata_row("box", content_props="50cm box"))
    report = select_bones(rows, 0.04, 0.02, 0)
    assert report == select_bones(list(reversed(rows)), 0.04, 0.02, 0)
    selected = report["clips"]
    groups = {
        s: {e["source_sequence"] for e in selected if e["split"] == s}
        for s in ("train", "evaluation")
    }
    assert not groups["train"] & groups["evaluation"]
    assert {e["bucket"] for e in selected} == set(BUCKETS)
    assert report["estimated_hours"]["train"] >= 0.04
    assert report["estimated_hours"]["evaluation"] >= 0.02
    assert report["excluded"] == [
        {"sequence": "box", "reason": "unsupported_support", "term": "box"}
    ]


def test_archive_extraction_copies_only_requested_regular_sources(tmp_path):
    archive = tmp_path / "sources.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        for name in ("selected", "unused"):
            info = tarfile.TarInfo(f"soma_uniform/bvh/date/{name}.bvh")
            data = name.encode()
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))
    output = tmp_path / "output"
    entries = [
        {"sequence": "selected", "source_path": "soma_uniform/bvh/date/selected.bvh"}
    ]
    extract_sources(archive, output, entries)
    assert [p.name for p in output.iterdir()] == ["selected.bvh"]
    assert (output / "selected.bvh").read_bytes() == b"selected"
    with pytest.raises(ValueError, match="archive omits"):
        extract_sources(
            archive, output, [{"sequence": "missing", "source_path": "missing.bvh"}]
        )


def test_config_rejects_invalid_preparation_settings(tmp_path):
    config = BonesPreparationConfig(tmp_path, tmp_path / "out", tmp_path / "lafan")
    for values in (
        {"workers": 0},
        {"training_hours": -1},
        {"metadata_file": "../metadata.csv"},
        {"max_retarget_failure_fraction": 0.2},
    ):
        with pytest.raises(ValueError):
            replace(config, **values)


def test_quarantine_is_explicit_and_limited_by_both_clip_count_and_duration():
    entries = [
        {"sequence": f"{split}/{i}", "split": split, "estimated_seconds": 10.0}
        for split in ("train", "evaluation")
        for i in range(100)
    ]
    rejected = [{**entries[0], "failure_kind": "retargeting"}]
    assert validate_quarantine(entries, rejected, 0.01)["train"]["quarantined"] == 1
    with pytest.raises(ValueError, match="quarantine exceeds"):
        validate_quarantine(entries, rejected, 0.0)
    with pytest.raises(ValueError, match="unexpected preparation failure"):
        validate_quarantine(
            entries, [{**rejected[0], "failure_kind": "unexpected"}], 0.01
        )
    entries[0]["estimated_seconds"] = 1000.0
    with pytest.raises(ValueError, match="duration="):
        validate_quarantine(
            entries, [{**entries[0], "failure_kind": "retargeting"}], 0.01
        )
