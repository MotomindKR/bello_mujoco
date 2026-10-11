"""Human input geometry, synchronization, and corpus provenance."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.sonic.corpus import (
    MANIFEST,
    file_sha256,
    prepare_corpus,
    verify_corpus,
)
from bello_mujoco.sonic.pair_smpl import pair_corpus
from bello_mujoco.sonic.smpl import (
    SMPL_BASE_ROTATION,
    SMPL_GEOMETRY_FORMAT,
    SMPL_JOINT_NAMES,
    SMPL_PARENTS,
    SmplSkeleton,
    validate_smpl_motion,
)
from bello_mujoco.sonic.smpl_model import prepare_model
from test_sonic_recipe_reference import reference_function


@pytest.fixture
def skeleton(tmp_path):
    rest = np.zeros((24, 3))
    for joint, parent in enumerate(SMPL_PARENTS):
        if parent >= 0:
            rest[joint] = rest[parent] + [0.03 * (-1) ** joint, 0.1, 0.02]
    path = tmp_path / "models"
    path.mkdir()
    np.savez(
        path / "SMPL_NEUTRAL.npz",
        format=SMPL_GEOMETRY_FORMAT,
        J_regressor=np.eye(24),
        v_template=rest,
        shapedirs=np.zeros((24, 3, 0)),
        kintree_table=[SMPL_PARENTS, range(24)],
    )
    return path, SmplSkeleton.load(path / "SMPL_NEUTRAL.npz")


def test_live_global_rotations_match_offline_local_fk(skeleton):
    _, body = skeleton
    pose = Rotation.from_rotvec(np.random.default_rng(12).normal(0, 0.2, (24, 3)))
    local = pose.as_quat()
    local[22:] = [0, 0, 0, 1]
    world = []
    transforms = {}
    for joint, parent in enumerate(SMPL_PARENTS):
        rotation = Rotation.from_quat(local[joint])
        world.append(rotation if parent < 0 else world[parent] * rotation)
        name = "_".join(
            part.capitalize() for part in SMPL_JOINT_NAMES[joint].split("_")
        )
        transforms[name] = (
            np.zeros(3),
            (world[joint] * Rotation.from_euler("y", -180, degrees=True)).as_quat(
                scalar_first=True
            ),
        )
    offline = body.forward(local[None], np.zeros(0))
    live = body.from_xrobot(transforms)
    np.testing.assert_allclose(live[0], offline[0], atol=1e-6)
    np.testing.assert_allclose(live[1], offline[1], atol=1e-6)
    np.testing.assert_allclose(
        offline[1][0],
        (world[0] * SMPL_BASE_ROTATION.inv()).as_quat(scalar_first=True),
        atol=1e-6,
    )


def test_pairing_preserves_robot_bytes_split_and_exact_clip_clock(tmp_path, skeleton):
    models, body = skeleton
    corpus, sources = tmp_path / "robot", tmp_path / "sources"
    corpus.mkdir()
    sources.mkdir()
    splits, mapping = {}, {}
    for name, split in (("walk", "train"), ("wave", "evaluation")):
        np.savez(
            corpus / f"{name}.npz", fps=50.0, qpos=np.arange(4 * 35).reshape(4, 35)
        )
        rotations = np.zeros((12, 3))
        rotations[:, 2] = np.arange(12) * 0.1
        path = sources / f"{name}.npz"
        np.savez(
            path,
            surface_model_type="smplx",
            mocap_frame_rate=100.0,
            root_orient=rotations,
            pose_body=np.zeros((12, 63)),
        )
        splits[name] = {"split": split, "source_sequence": name}
        mapping[name] = {
            "path": path.name,
            "start_seconds": 0.01,
            "sha256": file_sha256(path),
        }
    prepare_corpus(corpus, splits)
    source_map = tmp_path / "sources.json"
    source_map.write_text(json.dumps(mapping))
    output = tmp_path / "paired"
    pair_corpus(corpus, sources, source_map, models, output)
    verify_corpus(output, ("wave",))
    document = json.loads((output / MANIFEST).read_text())
    assert document["provenance"]["robot_corpus_manifest_sha256"] == file_sha256(
        corpus / MANIFEST
    )
    for entry in document["clips"]:
        with (
            np.load(output / entry["file"]) as paired,
            np.load(corpus / entry["file"]) as original,
        ):
            for key in original.files:
                np.testing.assert_array_equal(paired[key], original[key])
            # 50 Hz Mini frames start 10 ms into the 100 Hz source: 1, 3, 5, 7.
            pose = np.tile([0.0, 0.0, 0.0, 1.0], (4, 24, 1))
            pose[:, 0] = Rotation.from_euler(
                "z", np.array([0.1, 0.3, 0.5, 0.7])[:, None]
            ).as_quat()
            expected_joints, expected_roots = body.forward(pose, np.zeros(0))
            np.testing.assert_allclose(
                paired["smpl_joints"], expected_joints, atol=1e-6
            )
            np.testing.assert_allclose(
                paired["smpl_root_quat"], expected_roots, atol=1e-6
            )
            assert entry["split"] == splits[entry["source_sequence"]]["split"]
    # A changed source fails without publishing a partially paired corpus.
    (sources / "walk.npz").write_bytes(b"changed")
    bad = tmp_path / "bad"
    with pytest.raises(ValueError, match="checksum mismatch"):
        pair_corpus(corpus, sources, source_map, models, bad)
    assert not bad.exists()


def test_human_reference_rejects_placeholder_geometry():
    with pytest.raises(ValueError, match="placeholder"):
        validate_smpl_motion(np.zeros((1, 24, 3)), np.array([[1, 0, 0, 0]]), 1)


def test_compact_skeleton_matches_pinned_55_joint_fk(tmp_path):
    torch = pytest.importorskip("torch")
    parents = np.array([*SMPL_PARENTS[:22], 15, 15, 15, *([20] * 15), *([21] * 15)])
    parents[38:40] = [37, 38]
    parents[53:55] = [52, 53]
    rng = np.random.default_rng(4)
    rest = rng.normal(0, 0.3, (55, 3)).astype(np.float32)
    vertices = np.zeros((10475, 3), dtype=np.float32)
    vertices[:55] = rest
    regressor = np.eye(55, 10475, dtype=np.float32)
    source, compact = tmp_path / "smplx.npz", tmp_path / "compact.npz"
    np.savez(
        source,
        v_template=vertices,
        J_regressor=regressor,
        kintree_table=[parents, range(55)],
    )
    prepare_model(source, compact)
    skeleton = SmplSkeleton.load(compact)
    pose = rng.normal(0, 0.2, (5, 24, 3)).astype(np.float32)
    pose[:, 22:] = 0
    actual, _ = skeleton.forward(
        Rotation.from_rotvec(pose.reshape(-1, 3)).as_quat().reshape(5, 24, 4),
        np.zeros(0),
    )
    function = reference_function(
        "trl/utils/torch_transform.py",
        "compute_human_joints",
        {
            "torch": torch,
            "np": np,
            "F": torch.nn.functional,
            "human_joints_info": {
                "J": torch.tensor(rest),
                "parents_list": parents.tolist(),
                "rot_mats": None,
            },
            "angle_axis_to_rotation_matrix": lambda p: torch.tensor(
                Rotation.from_rotvec(p.numpy().reshape(-1, 3))
                .as_matrix()
                .reshape(*p.shape[:-1], 3, 3),
                dtype=p.dtype,
            ),
        },
    )
    expected = function(
        torch.tensor(pose[:, 1:22].reshape(5, 63)), torch.tensor(pose[:, 0])
    ).numpy()
    # The wire contract explicitly removes the template pelvis offset.
    expected -= expected[:, :1].copy()
    np.testing.assert_allclose(actual, expected, atol=2e-6)
