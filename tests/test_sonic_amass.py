from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from bello_mujoco.hardware.smplx_contract import SMPLX_BODY_JOINT_PARENTS
from bello_mujoco.sonic.amass import (
    AmassPreparationConfig,
    ground_translation,
    resample_rotations,
    select_candidates,
    skeleton_forward,
    source_subject,
    summarize_preparation,
)
from bello_mujoco.sonic.embodiment import validate_named_joint_axes
from bello_mujoco.sonic import amass


def test_amass_resamples_on_shortest_rotation_arc():
    poses = np.zeros((2, 1, 3))
    poses[:, 0, 2] = np.deg2rad([170, -170])
    q = resample_rotations(poses, np.array([0, 0.5, 1]))[:, 0]
    np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1)
    np.testing.assert_allclose(
        Rotation.from_quat(q[1]).apply([1, 0, 0]), [-1, 0, 0], atol=1e-12
    )


def test_amass_shaped_joint_fk_preserves_root_and_hierarchy():
    count = len(SMPLX_BODY_JOINT_PARENTS)
    rest = np.zeros((count, 3))
    rest[:, 0] = np.arange(count) * 0.01
    rest += [0.2, 0.3, 0.4]
    q = np.zeros((1, count, 4))
    q[..., 3] = 1
    q[0, 0] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    positions, orientations = skeleton_forward(rest, q, np.array([[1, 2, 3]]))
    expected = np.tile(rest[0] + [1, 2, 3], (count, 1))
    expected[:, 1] += np.arange(count) * 0.01
    np.testing.assert_allclose(positions[0], expected, atol=1e-12)
    np.testing.assert_allclose(
        orientations[0], np.tile(q[0, 0, [3, 0, 1, 2]], (count, 1))
    )


def test_ground_offset_is_constant_not_per_frame():
    config = AmassPreparationConfig()
    heights = np.array([[-0.02, 0.04], [0.01, 0.08], [-0.01, 0.03]])
    offset, clearance = ground_translation(heights, config)
    np.testing.assert_allclose(np.diff(clearance), np.diff(heights.min(axis=1)))
    assert offset == pytest.approx(np.quantile(heights.min(axis=1), 0.05))
    with pytest.raises(ValueError, match="ground correction"):
        ground_translation(heights + 1, config)
    with pytest.raises(ValueError, match="airborne"):
        ground_translation(np.array([[0, 0], [0.3, 0.4]]), config)


def test_amass_selection_is_deduplicated_and_subject_disjoint(tmp_path):
    entries = []
    for number in range(100):
        for take in range(2):
            entries.append(
                {
                    "source_kind": "amass",
                    "pose_representation": "smpl-x",
                    "metadata": {
                        "relative_path": f"CMU/{number}/motion{take}.npz",
                        "frame_count": 501,
                    },
                    "frame_rate": 50,
                    "source_sha256": f"hash-{number}-{take}",
                    "near_duplicate_fingerprint": f"motion-{number}-{take}",
                    "subdataset": "CMU",
                    "subject_id": str(number),
                }
            )
    entries.append(entries[0])
    bad = {
        **entries[0],
        "metadata": {"relative_path": "../escape.npz", "frame_count": 501},
    }
    entries.append(bad)
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"sequences": entries}))
    candidates = select_candidates(path, AmassPreparationConfig())
    assert candidates == select_candidates(path, AmassPreparationConfig())
    assert len(candidates) == 200
    subjects = {
        split: {item["subject"] for item in candidates if item["split"] == split}
        for split in ("train", "evaluation")
    }
    assert subjects["evaluation"]
    assert not subjects["train"] & subjects["evaluation"]
    assert source_subject("ACCAD", "Female1Walking_c3d") == source_subject(
        "ACCAD", "Female1General_c3d"
    )
    assert source_subject("DanceDB", "20150901_Alice") == source_subject(
        "DanceDB", "20151001_Alice"
    )


def test_amass_progress_counts_only_accepted_motion():
    result = summarize_preparation(
        [
            {"clip": "walk", "split": "train", "quality": {"seconds": 4}},
            {"clip": "turn", "split": "evaluation", "quality": {"seconds": 2}},
            {"reason": "speed", "split": "train", "seconds": 100},
        ]
    )
    assert result["seconds"] == {"train": 4, "evaluation": 2}
    with pytest.raises(ValueError, match="positive"):
        replace(AmassPreparationConfig(), train_hours=-1)


def test_repeated_capture_sessions_share_subject_identity():
    assert source_subject("HUMAN4D", "Subject1_2_Aude") == source_subject(
        "HUMAN4D", "Subject2_Aude"
    )
    assert source_subject("SSM", "20160930_50032") == source_subject("MoSh", "50032")
    assert source_subject("DFaust", "50004") == source_subject("MoSh", "50004")


def test_sonic_axis_validation_uses_named_directions():
    names = ("a_pitch_joint", "b_roll_joint", "c_yaw_joint", "left_knee_joint")
    axes = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
    validate_named_joint_axes(names, axes)
    with pytest.raises(ValueError, match="review signs"):
        validate_named_joint_axes(names, -axes)
    with pytest.raises(ValueError, match="no declared"):
        validate_named_joint_axes(("unknown_joint",), np.array([[1, 0, 0]]))


def test_motion_publication_needs_no_hard_links_and_preserves_existing(
    tmp_path, monkeypatch
):
    def save(path, clip):
        np.savez(path, values=clip)

    def validate(path, model):
        with np.load(path) as data:
            np.testing.assert_array_equal(data["values"], [1, 2])

    monkeypatch.setattr(amass, "save_motion_clip", save)
    monkeypatch.setattr(amass, "load_motion_clip", validate)
    output = tmp_path / "clip.npz"
    amass.publish_motion_clip(output, np.array([1, 2]), None)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        amass.publish_motion_clip(output, np.array([1, 2]), None)
    assert output.read_bytes() == original
    assert list(tmp_path.iterdir()) == [output]


def test_failed_clip_validation_does_not_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(
        amass, "save_motion_clip", lambda path, clip: np.savez(path, values=clip)
    )

    def reject(path, model):
        raise ValueError("invalid clip")

    monkeypatch.setattr(amass, "load_motion_clip", reject)
    with pytest.raises(ValueError, match="invalid clip"):
        amass.publish_motion_clip(tmp_path / "clip.npz", np.array([1]), None)
    assert not list(tmp_path.iterdir())


def test_only_generated_appledouble_sidecars_are_removed(tmp_path):
    output = tmp_path / "clip.npz"
    sidecar = tmp_path / "._clip.npz"
    sidecar.write_bytes(b"\x00\x05\x16\x07metadata")
    amass.remove_generated_macos_sidecar(output)
    assert not sidecar.exists()
    sidecar.write_bytes(b"not metadata")
    with pytest.raises(RuntimeError, match="non-AppleDouble"):
        amass.remove_generated_macos_sidecar(output)
    assert sidecar.read_bytes() == b"not metadata"
