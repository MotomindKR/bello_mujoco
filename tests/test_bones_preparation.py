import io
import tarfile
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bello_mujoco.motion.bones import (
    archive_sources,
    clip_directory,
    completed_sequences,
    family,
    safe_member,
    select_bones,
)
from bello_mujoco.motion.bvh import BvhMotion, soma_targets


def metadata_row(name, **overrides):
    return {
        "filename": name,
        "move_soma_uniform_path": f"soma_uniform/bvh/date/{name}.bvh",
        "move_duration_frames": "1201",
        "content_name": name,
        "is_mirror": "False",
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


def test_full_selection_retains_hard_floor_and_acrobatic_motions_and_records_exclusions():
    rows = [
        metadata_row(n)
        for n in (
            "crawl",
            "handstand",
            "cartwheel",
            "stair_walk",
            "jump_off_50cm",
            "dance",
            "dance_M",
        )
    ]
    report = select_bones(rows, ("stair", "jump off 50cm"))
    assert [e["sequence"] for e in report["clips"]] == [
        "crawl",
        "handstand",
        "cartwheel",
        "dance",
        "dance_M",
    ]
    assert [e["sequence"] for e in report["excluded"]] == [
        "stair_walk",
        "jump_off_50cm",
    ]
    assert report["estimated_hours"] == pytest.approx(50 / 3600)
    assert family(rows[-1]) == family(rows[-2])
    with pytest.raises(ValueError, match="unique"):
        select_bones(rows + rows[:1], ("chair",))


def test_archive_streams_only_requested_regular_sources(tmp_path):
    path = tmp_path / "sources.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in ("selected", "unused"):
            info = tarfile.TarInfo(f"soma_uniform/bvh/date/{name}.bvh")
            data = name.encode()
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    entries = [
        {"sequence": "selected", "source_path": "soma_uniform/bvh/date/selected.bvh"}
    ]
    assert list(archive_sources(path, entries)) == [(entries[0], b"selected")]
    with pytest.raises(ValueError, match="archive omits"):
        list(
            archive_sources(
                path, [{"sequence": "missing", "source_path": "missing.bvh"}]
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        list(archive_sources(path, entries * 2))


def test_archive_rejects_selected_symlinks(tmp_path):
    path = tmp_path / "sources.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("source.bvh")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(ValueError, match="regular file"):
        list(
            archive_sources(path, [{"sequence": "source", "source_path": "source.bvh"}])
        )


def test_resume_only_discovers_atomically_completed_directories(tmp_path):
    complete = clip_directory(tmp_path, "recording")
    complete.mkdir(parents=True)
    (complete.parent / ".unfinished").mkdir()
    (complete.parent / "not-a-directory").touch()
    assert completed_sequences(tmp_path) == {"recording"}


@pytest.mark.parametrize("failure", [None, "physical", "unexpected"])
def test_cache_commits_complete_receipt_or_leaves_no_resumable_partial(
    tmp_path, monkeypatch, failure
):
    import json
    from types import SimpleNamespace
    from bello_mujoco.motion import bones
    from bello_mujoco.motion.retarget import RetargetValidationError

    config = SimpleNamespace(output_directory=tmp_path, fps=50, contact_height_m=0.03)
    motion = SimpleNamespace(position=np.zeros((3, 1, 3)))
    motion.resample = lambda fps: motion
    monkeypatch.setattr(bones, "parse_bvh", lambda text: motion)

    def convert(motion, path):
        path.write_bytes(b"native")
        return {"soma_sha256": "soma"}

    converter = SimpleNamespace(
        convert=convert,
        reader=SimpleNamespace(
            read=lambda *args: (np.zeros((3, 22, 3)), np.zeros((3, 22, 4)), 1.66)
        ),
    )
    monkeypatch.setattr(bones, "_worker", (config, converter, object()))

    def retarget(*args, **kwargs):
        if failure == "physical":
            raise RetargetValidationError("limits")
        if failure == "unexpected":
            raise RuntimeError("broken converter")
        return object(), {"frames": 3}

    monkeypatch.setattr(bones, "retarget_gmr", retarget)
    monkeypatch.setattr(
        bones, "save_motion_clip", lambda stream, clip: stream.write(b"mini")
    )
    entry = {"sequence": "take", "source_frames": 3}
    if failure == "unexpected":
        with pytest.raises(RuntimeError, match="broken converter"):
            bones._prepare_job(entry, b"source")
        assert completed_sequences(tmp_path) == set()
        return
    report = bones._prepare_job(entry, b"source")
    directory = clip_directory(tmp_path, "take")
    assert completed_sequences(tmp_path) == {"take"}
    assert json.loads((directory / "preparation.json").read_text()) == report
    assert report["status"] == ("rejected" if failure else "prepared")
    assert (directory / "mini.npz").exists() is (failure is None)
    assert list(directory.parent.iterdir()) == [directory]
