from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.motion.bvh import BvhMotion, lafan1_targets, read_bvh


def fixture_bvh(tmp_path, *, rotations="0 0 0\n90 0 0"):
    frames = rotations.splitlines()
    path = tmp_path / "motion.bvh"
    path.write_text(
        "HIERARCHY\nROOT Hips { OFFSET 1000 0 0\n"
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n"
        "JOINT child { OFFSET 100 0 0 CHANNELS 3 Zrotation Yrotation Xrotation\n"
        "End Site { OFFSET 10 0 0 } } }\nMOTION\nFrames: 2\nFrame Time: 0.1\n"
        + "\n".join(f"100 0 0 {angles} 0 0 0" for angles in frames)
        + "\n"
    )
    return path


def test_bvh_channel_order_units_world_axes_and_rigid_resampling(tmp_path):
    motion = read_bvh(fixture_bvh(tmp_path))
    assert motion.names == ("Hips", "child")
    assert motion.parents == (-1, 0)
    positions, _ = motion.world_transforms()
    np.testing.assert_allclose(positions[0], [[1, 0, 0], [2, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(positions[1], [[1, 0, 0], [1, 0, 1]], atol=1e-12)
    sampled = motion.resample(20)
    p, _ = sampled.world_transforms()
    np.testing.assert_allclose(p[1, 1], [1 + np.sqrt(0.5), 0, np.sqrt(0.5)], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(p[:, 1] - p[:, 0], axis=-1), 1)
    assert sampled.duration == pytest.approx(motion.duration)
    assert not sampled.position.flags.writeable


def test_bvh_composes_declared_intrinsic_rotation_order(tmp_path):
    motion = read_bvh(fixture_bvh(tmp_path, rotations="90 90 0\n90 90 0"))
    p, _ = motion.world_transforms()
    np.testing.assert_allclose(p[:, 1] - p[:, 0], [[0, 1, 0], [0, 1, 0]], atol=1e-12)


def test_bvh_preserves_a_translated_child_without_adding_its_offset(tmp_path):
    path = fixture_bvh(tmp_path)
    text = (
        path.read_text()
        .replace(
            "CHANNELS 3 Zrotation Yrotation Xrotation",
            "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation",
        )
        .replace("100 0 0 0 0 0 0 0 0", "100 0 0 0 0 0 0 200 0 0 0 0")
        .replace("100 0 0 90 0 0 0 0 0", "100 0 0 90 0 0 0 200 0 0 0 0")
    )
    path.write_text(text)
    motion = read_bvh(path)
    np.testing.assert_allclose(motion.position[:, 1], [[0, 2, 0], [0, 2, 0]])


@pytest.mark.parametrize("damage", ["truncate", "count", "nonfinite", "channel"])
def test_bvh_rejects_malformed_input(tmp_path, damage):
    path = fixture_bvh(tmp_path)
    text = path.read_text()
    if damage == "truncate":
        text = text[: text.index("End Site")]
    elif damage == "count":
        text = text.replace("Frames: 2", "Frames: 3")
    elif damage == "nonfinite":
        text = text.replace("100 0 0 90", "nan 0 0 90")
    else:
        text = text.replace("Zrotation", "Zscale")
    path.write_text(text)
    with pytest.raises(ValueError):
        read_bvh(path)


def test_lafan_conversion_uses_toe_centres_and_upright_smplx_axes():
    bones = (
        "Hips",
        "Spine2",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFoot",
        "RightFoot",
        "LeftToe",
        "RightToe",
        "LeftArm",
        "RightArm",
        "LeftForeArm",
        "RightForeArm",
        "LeftHand",
        "RightHand",
    )
    p = np.zeros((1, len(bones), 3))
    p[0, bones.index("LeftFoot")] = [0, 0, 0]
    p[0, bones.index("LeftToe")] = [0.17, 0, 0]
    q = np.zeros((1, len(bones), 4))
    q[..., 0] = 1
    up = Rotation.from_euler("x", 90, degrees=True)
    raw = Rotation.from_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    q[0, 0] = (up.inv() * raw).as_quat(scalar_first=True)
    motion = BvhMotion(bones, (-1,) + (0,) * (len(bones) - 1), p, q, 0.02)
    target = lafan1_targets(motion)
    world_p, _ = motion.world_transforms()
    np.testing.assert_allclose(
        target["left_foot"][0], world_p[:, bones.index("LeftToe")]
    )
    pelvis = Rotation.from_quat(target["pelvis"][1], scalar_first=True)
    np.testing.assert_allclose(pelvis.apply([0, 1, 0]), [[0, 0, 1]], atol=1e-12)
    with pytest.raises(ValueError, match="missing joints"):
        lafan1_targets(replace(motion, names=("root",) + bones[1:]))
