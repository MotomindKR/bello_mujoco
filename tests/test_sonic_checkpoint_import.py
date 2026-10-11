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


def test_dual_encoder_import_requires_released_smpl_tensors(tmp_path, monkeypatch):
    import jax
    import numpy as np
    from bello_mujoco.sonic import checkpoint
    from bello_mujoco.sonic.model import SonicNetworkConfig
    from test_sonic_transfer import synthetic_adapter

    torch = pytest.importorskip("torch")
    actor, critic = {"std": torch.ones(29)}, {}
    for state, prefix, dims in (
        (actor, "actor_module.encoders.g1.module", [640, 4, 4, 4, 4, 64]),
        (actor, "actor_module.encoders.smpl.module", [840, 4, 4, 4, 4, 64]),
        (actor, "actor_module.decoders.g1_dyn.module", [994, 4, 4, 4, 4, 4, 4, 29]),
        (critic, "critic_module.module", [1645, 4, 4, 4, 4, 4, 4, 1]),
    ):
        for i, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            state[f"{prefix}.{2 * i}.weight"] = torch.full((n_out, n_in), 0.01)
            state[f"{prefix}.{2 * i}.bias"] = torch.zeros(n_out)
    monkeypatch.setattr(
        checkpoint,
        "load_source_checkpoint",
        lambda _: {"policy_state_dict": actor, "value_state_dict": critic},
    )
    path = tmp_path / "source.pt"
    path.write_bytes(b"synthetic import contract")
    arguments = dict(
        revision="a" * 40,
        adapter=synthetic_adapter(),
        key=jax.random.PRNGKey(0),
        config=SonicNetworkConfig(smpl_enabled=True),
    )
    imported = checkpoint.import_sonic_checkpoint(path, **arguments)
    assert imported.policy.actor.observation_size == 15710
    for layer in imported.policy.actor.smpl_encoder.layers:
        assert layer.rank == 0
        np.testing.assert_array_equal(layer.weight, np.float32(0.01))
    del actor["actor_module.encoders.smpl.module.0.weight"]
    with pytest.raises(ValueError, match="encoders.smpl.module.0.weight"):
        checkpoint.import_sonic_checkpoint(path, **arguments)
