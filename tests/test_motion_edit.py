from pathlib import Path

import numpy as np

from bello_mujoco.motion.edit import (
    SYMMETRIC_DOWNWARD_ARM_EDIT_VERSION,
    SYMMETRIC_DOWNWARD_ARM_POSE,
    edit_symmetric_downward_arm_pose,
)


def test_deep_lunge_arm_edit_is_reproducible_and_collision_clean(
    tmp_path: Path,
) -> None:
    tracked = Path("motions/amass_cmu_20260818/cmu_144_17_deep_lunge.npz")
    source = tmp_path / "source.npz"
    output = tmp_path / "edited.npz"
    with np.load(tracked, allow_pickle=False) as archive:
        fields = {
            name: np.asarray(archive[name]).copy()
            for name in archive.files
            if not name.startswith("arm_pose_edit_") and name != "arm_pose_edit_version"
        }
    np.savez_compressed(source, **fields)

    edit_symmetric_downward_arm_pose(source, output, asset="bello")

    with np.load(output, allow_pickle=False) as edited:
        names = tuple(str(name) for name in edited["joint_names"])
        assert str(edited["arm_pose_edit_version"]) == (
            SYMMETRIC_DOWNWARD_ARM_EDIT_VERSION
        )
        assert int(edited["self_collision_frames"]) == 0
        assert float(edited["max_self_collision_penetration_m"]) == 0.0
        assert tuple(str(name) for name in edited["deepest_self_collision_pair"]) == (
            "",
            "",
        )
        for side in ("left", "right"):
            for suffix, target in SYMMETRIC_DOWNWARD_ARM_POSE.items():
                column = 7 + names.index(f"{side}_{suffix}")
                np.testing.assert_allclose(edited["qpos"][:, column], target)
                np.testing.assert_allclose(
                    edited["qvel"][:, column - 1], 0.0, atol=1.0e-6
                )
