#!/usr/bin/env python3
"""Export NVIDIA's SONIC ``last.pt`` to a safe, JAX-readable artifact.

PyTorch is intentionally not a core Bello dependency.  Run this one-time tool
inside a trusted Torch environment; the resulting NPZ/JSON files contain only
plain arrays and are kept below the ignored ``artifacts`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np

OFFICIAL_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
OFFICIAL_CONFIG_SHA256 = (
    "f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"
)
SCHEMA_VERSION = 3
ENCODER_DIMS = (640, 2048, 1024, 512, 512, 64)
DECODER_DIMS = (994, 2048, 2048, 1024, 1024, 512, 512, 29)
CRITIC_DIMS = (1645, 2048, 2048, 1024, 1024, 512, 512, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=OFFICIAL_CHECKPOINT_SHA256,
        help="Exact trusted checkpoint digest; pickle loading is refused on mismatch.",
    )
    parser.add_argument(
        "--expected-config-sha256",
        default=OFFICIAL_CONFIG_SHA256,
        help="Exact trusted release-config digest.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for name, path in (("checkpoint", checkpoint), ("config", config)):
        if not path.is_file():
            raise FileNotFoundError(f"SONIC {name} not found: {path}")
    expected = args.expected_checkpoint_sha256.removeprefix("sha256:")
    _require_sha256(expected, "expected checkpoint SHA-256")
    checkpoint_sha256 = _file_sha256(checkpoint)
    if checkpoint_sha256 != expected:
        raise ValueError(
            "refusing to unpickle a SONIC checkpoint whose digest was not explicitly "
            f"trusted: actual={checkpoint_sha256}, expected={expected}"
        )
    expected_config = args.expected_config_sha256.removeprefix("sha256:")
    _require_sha256(expected_config, "expected config SHA-256")
    config_sha256 = _file_sha256(config)
    if config_sha256 != expected_config:
        raise ValueError(
            "SONIC config differs from the explicitly trusted release config: "
            f"actual={config_sha256}, expected={expected_config}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"SONIC artifact output is not empty: {output_dir}; exports never overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    import torch

    checkpoint_document = _load_trusted_checkpoint(torch, checkpoint)
    policy = checkpoint_document.get("policy_state_dict")
    if not isinstance(policy, dict):
        raise ValueError("SONIC checkpoint has no policy_state_dict")
    value = checkpoint_document.get("value_state_dict")
    if not isinstance(value, dict):
        raise ValueError("SONIC checkpoint has no value_state_dict")
    arrays = _extract_arrays(policy, value)
    _validate_arrays(arrays)
    golden = _golden_vectors(torch, arrays)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        weights_path = staging / "weights.npz"
        golden_path = staging / "golden.npz"
        np.savez_compressed(weights_path, **arrays)
        np.savez_compressed(golden_path, **golden)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_config_sha256": config_sha256,
            "weights_file_sha256": _file_sha256(weights_path),
            "golden_file_sha256": _file_sha256(golden_path),
            "artifact_sha256": _artifact_sha256(arrays),
            "array_sha256": {
                name: _array_sha256(value) for name, value in sorted(arrays.items())
            },
            "golden_array_sha256": {
                name: _array_sha256(value) for name, value in sorted(golden.items())
            },
            "architecture": _architecture_document(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _load_trusted_checkpoint(torch, path: Path):
    """Load a digest-approved checkpoint without installing its trainer stack."""

    stub_names = (
        "trl.trainer.utils.OnlineTrainerState",
        "accelerate.state.PartialState",
        "trl.trainer.ppo_config.PPOConfig",
        "accelerate.utils.dataclasses.DistributedType",
        "transformers.trainer_utils.SaveStrategy",
        "transformers.trainer_utils.SchedulerType",
        "transformers.trainer_utils.IntervalStrategy",
        "transformers.trainer_utils.HubStrategy",
        "transformers.training_args.OptimizerNames",
        "transformers.trainer_pt_utils.AcceleratorConfig",
    )

    class PickleStub:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.state = state

    for qualified_name in stub_names:
        module_name, class_name = qualified_name.rsplit(".", 1)
        parts = module_name.split(".")
        for count in range(1, len(parts) + 1):
            name = ".".join(parts[:count])
            sys.modules.setdefault(name, types.ModuleType(name))
            if count > 1:
                parent = sys.modules[".".join(parts[: count - 1])]
                setattr(parent, parts[count - 1], sys.modules[name])
        stub = type(class_name, (PickleStub,), {"__module__": module_name})
        setattr(sys.modules[module_name], class_name, stub)
    # ``weights_only=False`` is safe here only because main verified the exact
    # user-supplied trusted digest before reaching this function.
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_arrays(policy: dict, value: dict) -> dict[str, np.ndarray]:
    arrays = {}
    encoder_prefix = "actor_module.encoders.g1.module"
    decoder_prefix = "actor_module.decoders.g1_dyn.module"
    for output_index, module_index in enumerate((0, 2, 4, 6, 8)):
        arrays[f"robot_encoder_weight_{output_index}"] = _tensor_array(
            policy[f"{encoder_prefix}.{module_index}.weight"]
        )
        arrays[f"robot_encoder_bias_{output_index}"] = _tensor_array(
            policy[f"{encoder_prefix}.{module_index}.bias"]
        )
    for output_index, module_index in enumerate((0, 2, 4, 6, 8, 10, 12)):
        arrays[f"decoder_weight_{output_index}"] = _tensor_array(
            policy[f"{decoder_prefix}.{module_index}.weight"]
        )
        arrays[f"decoder_bias_{output_index}"] = _tensor_array(
            policy[f"{decoder_prefix}.{module_index}.bias"]
        )
        arrays[f"critic_weight_{output_index}"] = _tensor_array(
            value[f"critic_module.module.{module_index}.weight"]
        )
        arrays[f"critic_bias_{output_index}"] = _tensor_array(
            value[f"critic_module.module.{module_index}.bias"]
        )
    arrays["critic_observation_mean"] = _tensor_array(
        value["running_mean_std.running_mean"]
    )
    arrays["critic_observation_variance"] = _tensor_array(
        value["running_mean_std.running_var"]
    )
    arrays["released_std"] = _tensor_array(policy["std"])
    return arrays


def _tensor_array(value) -> np.ndarray:
    return (
        value.detach().cpu().to(dtype=value.new_zeros(()).float().dtype).numpy().copy()
    )


def _validate_arrays(arrays: dict[str, np.ndarray]) -> None:
    expected_names = {
        *(f"robot_encoder_weight_{index}" for index in range(5)),
        *(f"robot_encoder_bias_{index}" for index in range(5)),
        *(f"decoder_weight_{index}" for index in range(7)),
        *(f"decoder_bias_{index}" for index in range(7)),
        *(f"critic_weight_{index}" for index in range(7)),
        *(f"critic_bias_{index}" for index in range(7)),
        "critic_observation_mean",
        "critic_observation_variance",
        "released_std",
    }
    if set(arrays) != expected_names:
        raise ValueError("extracted SONIC arrays differ from schema")
    for prefix, dims in (("robot_encoder", ENCODER_DIMS), ("decoder", DECODER_DIMS)):
        for index, (input_size, output_size) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            weight = arrays[f"{prefix}_weight_{index}"]
            bias = arrays[f"{prefix}_bias_{index}"]
            if weight.shape != (output_size, input_size) or bias.shape != (
                output_size,
            ):
                raise ValueError(f"{prefix} layer {index} shape differs from release")
            if weight.dtype != np.float32 or bias.dtype != np.float32:
                raise ValueError(f"{prefix} layer {index} is not float32")
    for index, (input_size, output_size) in enumerate(
        zip(CRITIC_DIMS[:-1], CRITIC_DIMS[1:], strict=True)
    ):
        weight = arrays[f"critic_weight_{index}"]
        bias = arrays[f"critic_bias_{index}"]
        if weight.shape != (output_size, input_size) or bias.shape != (output_size,):
            raise ValueError(f"critic layer {index} shape differs from release")
        if weight.dtype != np.float32 or bias.dtype != np.float32:
            raise ValueError(f"critic layer {index} is not float32")
    for name in ("critic_observation_mean", "critic_observation_variance"):
        if arrays[name].shape != (CRITIC_DIMS[0],):
            raise ValueError(f"{name} shape differs from release")
    if arrays["released_std"].shape != (29,):
        raise ValueError("released SONIC standard deviation shape differs from release")


def _golden_vectors(torch, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20251107)
    robot_encoder_input = (
        torch.randn((8, 640), generator=generator, dtype=torch.float32) * 0.25
    )
    g1_proprio = torch.randn((8, 930), generator=generator, dtype=torch.float32) * 0.2
    critic_input = torch.randn(
        (8, CRITIC_DIMS[0]), generator=generator, dtype=torch.float32
    )
    with torch.inference_mode():
        latent = robot_encoder_input
        for index in range(5):
            latent = torch.nn.functional.linear(
                latent,
                torch.from_numpy(arrays[f"robot_encoder_weight_{index}"]),
                torch.from_numpy(arrays[f"robot_encoder_bias_{index}"]),
            )
            if index != 4:
                latent = torch.nn.functional.silu(latent)
        latent_tokens = latent.reshape(-1, 2, 32)
        half_l = torch.tensor((32.0 - 1.0) * (1.0 + 1.0e-3) / 2.0)
        offset = torch.tensor(0.5)
        shift = torch.atanh(offset / half_l)
        tokens = torch.round(torch.tanh(latent_tokens + shift) * half_l - offset) / 16.0
        decoder = torch.cat((tokens.reshape(-1, 64), g1_proprio), dim=-1)
        for index in range(7):
            decoder = torch.nn.functional.linear(
                decoder,
                torch.from_numpy(arrays[f"decoder_weight_{index}"]),
                torch.from_numpy(arrays[f"decoder_bias_{index}"]),
            )
            if index != 6:
                decoder = torch.nn.functional.silu(decoder)
        critic = torch.clamp(
            (critic_input - torch.from_numpy(arrays["critic_observation_mean"]))
            / torch.sqrt(
                torch.from_numpy(arrays["critic_observation_variance"]) + 1.0e-5
            ),
            min=-5.0,
            max=5.0,
        )
        critic_normalized = critic
        for index in range(7):
            critic = torch.nn.functional.linear(
                critic,
                torch.from_numpy(arrays[f"critic_weight_{index}"]),
                torch.from_numpy(arrays[f"critic_bias_{index}"]),
            )
            if index != 6:
                critic = torch.nn.functional.silu(critic)
    return {
        "robot_encoder_input": robot_encoder_input.numpy(),
        "g1_proprio": g1_proprio.numpy(),
        "encoder_latent": latent.numpy(),
        "quantized_tokens": tokens.numpy(),
        "g1_action": decoder.numpy(),
        "critic_input": critic_input.numpy(),
        "critic_normalized_input": critic_normalized.numpy(),
        "critic_value": critic.numpy(),
    }


def _architecture_document() -> dict:
    return {
        "robot_encoder_dims": list(ENCODER_DIMS),
        "robot_frame_count": 10,
        "robot_frame_size": 64,
        "robot_future_offsets": list(range(0, 50, 5)),
        "command_temporal_layout": "qpos_major_then_qvel_major_before_reshape",
        "fsq_levels": [32] * 32,
        "token_count": 2,
        "decoder_dims": list(DECODER_DIMS),
        "critic_dims": list(CRITIC_DIMS),
        "critic_observation_normalization": {
            "epsilon": 1.0e-5,
            "clip": 5.0,
        },
        "activation": "SiLU",
    }


def _artifact_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(_array_sha256(arrays[name])))
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


if __name__ == "__main__":
    main()
