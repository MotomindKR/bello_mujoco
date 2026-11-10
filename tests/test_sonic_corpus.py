from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bello_mujoco.sonic.corpus import (
    MANIFEST,
    prepare_corpus,
    prepare_preflight_corpus,
    verify_corpus,
    verify_artifact,
)


def corpus(tmp_path):
    for name in ("walk", "dance"):
        np.savez(tmp_path / f"{name}.npz", qpos=np.zeros((3, 35)))
    splits = {
        "walk": {"source_sequence": "bones/walk", "split": "train"},
        "dance": {"source_sequence": "bones/dance", "split": "evaluation"},
    }
    prepare_corpus(tmp_path, splits)
    return tmp_path


def test_corpus_hash_and_split_are_immutable(tmp_path):
    root = corpus(tmp_path)
    assert len(verify_corpus(root, ("dance",))) == 64
    with pytest.raises(ValueError, match="immutable corpus split"):
        verify_corpus(root, ("walk",))
    np.savez(root / "walk.npz", qpos=np.ones((3, 35)))
    with pytest.raises(ValueError, match="content changed"):
        verify_corpus(root, ("dance",))


def test_corpus_rejects_source_sequence_split_leakage(tmp_path):
    for name in ("walk", "walk_mirrored"):
        np.savez(tmp_path / f"{name}.npz", qpos=np.zeros((3, 35)))
    with pytest.raises(ValueError, match="both splits"):
        prepare_corpus(
            tmp_path,
            {
                "walk": {"source_sequence": "bones/walk", "split": "train"},
                "walk_mirrored": {
                    "source_sequence": "bones/walk",
                    "split": "evaluation",
                },
            },
        )


def test_preflight_subset_preserves_sources_and_original_corpus(tmp_path):
    import json

    original = tmp_path / "original"
    original.mkdir()
    splits = {}
    for index, name in enumerate(("short", "long", "eval_a", "eval_b", "eval_c")):
        np.savez(original / f"{name}.npz", qpos=np.zeros((index + 2, 35)))
        splits[name] = {
            "source_sequence": f"motion/{name}",
            "split": "evaluation" if name.startswith("eval") else "train",
        }
    prepare_corpus(original, splits)
    before = verify_corpus(original, ("eval_a", "eval_b", "eval_c"))
    destination = tmp_path / "preflight"
    evaluation = prepare_preflight_corpus(original, destination)
    assert evaluation == ("eval_a", "eval_b")
    verify_corpus(destination, evaluation)
    subset = json.loads((destination / MANIFEST).read_text())
    assert {entry["file"] for entry in subset["clips"]} == {
        "long.npz",
        "eval_a.npz",
        "eval_b.npz",
    }
    assert subset["provenance"]["preflight_source_manifest_sha256"] == before
    for entry in subset["clips"]:
        assert (destination / entry["file"]).samefile(original / entry["file"])
        assert (
            entry["source_sequence"]
            == splits[Path(entry["file"]).stem]["source_sequence"]
        )
    assert verify_corpus(original, ("eval_a", "eval_b", "eval_c")) == before


def test_artifact_verifies_bytes_and_rejects_references(tmp_path):
    root = corpus(tmp_path)
    calls = []
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(entries={}), verify=lambda **kw: calls.append(kw)
    )
    run = SimpleNamespace(use_artifact=lambda *args, **kwargs: artifact)
    verify_artifact(root, "entity/project/corpus:v1", ("dance",), run=run)
    assert calls == [{"root": str(root)}]
    artifact.manifest.entries["walk.npz"] = SimpleNamespace(ref="s3://mutable/path")
    with pytest.raises(ValueError, match="mutable external"):
        verify_artifact(root, "entity/project/corpus:v1", ("dance",), run=run)
    with pytest.raises(ValueError, match="immutable digest"):
        verify_artifact(
            Path("missing"), "entity/project/corpus:latest", ("dance",), run=run
        )
