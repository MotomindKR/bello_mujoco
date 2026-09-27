"""Strict import of NVIDIA SONIC training checkpoints into JAX modules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.sonic.embodiment import MiniEmbodimentAdapter
from bello_mujoco.sonic.model import (
    SonicActor,
    SonicActorCritic,
    SonicCritic,
    SonicMLP,
    SonicNetworkConfig,
)

SONIC_HUGGINGFACE_REPOSITORY = "nvidia/GEAR-SONIC"
SONIC_CHECKPOINT_FILE = "sonic_release/last.pt"
SONIC_CONFIG_FILE = "sonic_release/config.yaml"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class ImportedSonicCheckpoint:
    policy: SonicActorCritic
    sha256: str
    revision: str


def import_sonic_checkpoint(
    path: Path,
    *,
    revision: str,
    adapter: MiniEmbodimentAdapter,
    key: jax.Array,
    config: SonicNetworkConfig = SonicNetworkConfig(),
) -> ImportedSonicCheckpoint:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError(
            "pretrained revision must be a 40-character lowercase Hugging Face commit"
        )
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "SONIC checkpoint import requires the optional reference-validation "
            "dependency group"
        ) from error
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise RuntimeError(
            f"could not safely load SONIC checkpoint {path}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError("SONIC checkpoint root must be a mapping")
    policy_state = _required_mapping(checkpoint, "policy_state_dict")
    value_state = _required_mapping(checkpoint, "value_state_dict")

    encoder_tensors = _linear_tensors(
        policy_state,
        "actor_module.encoders.g1.module",
        linear_indices=(0, 2, 4, 6, 8),
    )
    decoder_tensors = _linear_tensors(
        policy_state,
        "actor_module.decoders.g1_dyn.module",
        linear_indices=(0, 2, 4, 6, 8, 10, 12),
    )
    critic_tensors = _linear_tensors(
        value_state,
        "critic_module.module",
        linear_indices=(0, 2, 4, 6, 8, 10, 12),
    )
    source_std = _required_tensor(policy_state, "std")
    if source_std.shape != (config.source_action_size,):
        raise ValueError(
            "SONIC checkpoint std must contain one value per source action"
        )
    if np.any(source_std <= 0.0):
        raise ValueError("SONIC checkpoint std must be strictly positive")
    log_std = np.log(source_std)
    encoder_key, decoder_key, critic_key = jax.random.split(key, 3)
    encoder = SonicMLP(encoder_tensors, rank=0, alpha=0.0, key=encoder_key)
    decoder = SonicMLP(
        decoder_tensors,
        rank=config.decoder_lora_rank,
        alpha=config.decoder_lora_alpha,
        key=decoder_key,
    )
    critic = SonicMLP(
        critic_tensors,
        rank=config.critic_lora_rank,
        alpha=config.critic_lora_alpha,
        key=critic_key,
    )
    actor = SonicActor(encoder, decoder, adapter, config)
    policy = SonicActorCritic(
        actor,
        SonicCritic(critic, adapter, body_count=14, history_steps=config.history_steps),
        jnp.asarray(log_std),
    )
    return ImportedSonicCheckpoint(
        policy=policy,
        sha256=_sha256(path),
        revision=revision,
    )


def _required_mapping(mapping: Mapping, key: str) -> Mapping:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SONIC checkpoint is missing required mapping {key!r}")
    return value


def _required_tensor(mapping: Mapping, key: str) -> np.ndarray:
    if key not in mapping:
        raise ValueError(f"SONIC checkpoint is missing required tensor {key!r}")
    value = mapping[key]
    if not hasattr(value, "detach"):
        raise TypeError(f"SONIC checkpoint value {key!r} is not a tensor")
    array = value.detach().cpu().float().numpy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"SONIC checkpoint tensor {key!r} contains nonfinite values")
    return array


def _linear_tensors(
    state: Mapping,
    prefix: str,
    *,
    linear_indices: tuple[int, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    tensors = []
    for index in linear_indices:
        weight = _required_tensor(state, f"{prefix}.{index}.weight")
        bias = _required_tensor(state, f"{prefix}.{index}.bias")
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(
                f"SONIC layer {prefix}.{index} has invalid shapes "
                f"weight={weight.shape}, bias={bias.shape}"
            )
        tensors.append((weight, bias))
    return tuple(tensors)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
