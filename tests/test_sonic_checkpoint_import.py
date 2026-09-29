"""Optional PyTorch checks for the original release's serialization boundary."""

from collections import deque
from pathlib import Path
import sys
from types import ModuleType

import pytest

from bello_mujoco.sonic.checkpoint import load_source_checkpoint


def test_release_metadata_loads_without_importing_training_stack(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    modules = {}
    for name in ("trl", "trl.trainer", "trl.trainer.utils"):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
    modules["trl"].trainer = modules["trl.trainer"]
    modules["trl.trainer"].utils = modules["trl.trainer.utils"]
    metadata_type = type("OnlineTrainerState", (), {})
    metadata_type.__module__ = "trl.trainer.utils"
    modules["trl.trainer.utils"].OnlineTrainerState = metadata_type
    metadata = metadata_type()
    metadata.history = deque([1.0, 2.0], maxlen=10)
    path = tmp_path / "release.pt"
    torch.save({"trainer_state": metadata, "weight": torch.ones(2)}, path)
    for name in modules:
        monkeypatch.delitem(sys.modules, name)

    loaded = load_source_checkpoint(path)

    assert loaded["trainer_state"].history == [1.0, 2.0]
    torch.testing.assert_close(loaded["weight"], torch.ones(2))
    assert "trl.trainer.utils" not in sys.modules


def test_checkpoint_rejects_unrecognized_pickle_globals(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "unsupported.pt"
    torch.save({"metadata": Path("unsupported")}, path)

    with pytest.raises(RuntimeError, match="could not safely load SONIC checkpoint"):
        load_source_checkpoint(path)
