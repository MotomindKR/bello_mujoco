"""Released SONIC weights and deterministic Bello edge initialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bello_mujoco.motion.sonic_dataset import (
    G1_ISAACLAB_JOINT_NAMES,
    SonicJointCalibration,
)

SONIC_ARTIFACT_SCHEMA_VERSION = 1
SONIC_SMPL_FRAME_COUNT = 10
SONIC_SMPL_FRAME_SIZE = 84
SONIC_SMPL_INPUT_SIZE = SONIC_SMPL_FRAME_COUNT * SONIC_SMPL_FRAME_SIZE
SONIC_TOKEN_COUNT = 2
SONIC_TOKEN_SIZE = 32
SONIC_TOKEN_FLAT_SIZE = SONIC_TOKEN_COUNT * SONIC_TOKEN_SIZE
SONIC_G1_ACTION_SIZE = 29
SONIC_BELLO_ACTION_SIZE = 25
SONIC_G1_PROPRIO_SIZE = 930
SONIC_BELLO_PROPRIO_SIZE = 810
SONIC_ROOT_MOTION_COMMAND_SIZE = 2 * SONIC_SMPL_FRAME_COUNT
SONIC_ACTOR_OBSERVATION_SIZE = (
    SONIC_SMPL_INPUT_SIZE
    + SONIC_ROOT_MOTION_COMMAND_SIZE
    + SONIC_BELLO_PROPRIO_SIZE
)
SONIC_ENCODER_DIMS = (840, 2048, 1024, 512, 512, 64)
SONIC_DECODER_DIMS = (994, 2048, 2048, 1024, 1024, 512, 512, 29)
SONIC_OFFICIAL_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
SONIC_OFFICIAL_CONFIG_SHA256 = (
    "f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"
)
SONIC_OFFICIAL_ARTIFACT_SHA256 = (
    "bd8251d1c6c2dea85c67f27e22485bdd2b6872298373913626f8314ad748381d"
)

BELLO_TO_G1_VECTOR_BASIS = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)

_G1_NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
_G1_STIFFNESS_5020 = 0.003609725 * _G1_NATURAL_FREQUENCY**2
_G1_STIFFNESS_7520_14 = 0.010177520 * _G1_NATURAL_FREQUENCY**2
_G1_STIFFNESS_7520_22 = 0.025101925 * _G1_NATURAL_FREQUENCY**2
_G1_STIFFNESS_4010 = 0.00425 * _G1_NATURAL_FREQUENCY**2


def _g1_action_scale(joint_name: str) -> float:
    """Mirror NVIDIA's pinned ``0.25 * effort / stiffness`` G1 contract."""

    if "hip_yaw" in joint_name or joint_name == "waist_yaw_joint":
        effort, stiffness = 88.0, _G1_STIFFNESS_7520_14
    elif any(part in joint_name for part in ("hip_pitch", "hip_roll", "knee")):
        effort, stiffness = 139.0, _G1_STIFFNESS_7520_22
    elif "ankle" in joint_name or joint_name in {
        "waist_roll_joint",
        "waist_pitch_joint",
    }:
        effort, stiffness = 50.0, 2.0 * _G1_STIFFNESS_5020
    elif "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        effort, stiffness = 5.0, _G1_STIFFNESS_4010
    else:
        effort, stiffness = 25.0, _G1_STIFFNESS_5020
    return 0.25 * effort / stiffness


SONIC_G1_ACTION_SCALE = np.asarray(
    tuple(_g1_action_scale(name) for name in G1_ISAACLAB_JOINT_NAMES),
    dtype=np.float32,
)


@dataclass(frozen=True)
class SonicReleaseWeights:
    smpl_encoder_weights: tuple[np.ndarray, ...]
    smpl_encoder_biases: tuple[np.ndarray, ...]
    decoder_weights: tuple[np.ndarray, ...]
    decoder_biases: tuple[np.ndarray, ...]
    released_std: np.ndarray
    checkpoint_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _validate_linear_stack(
            self.smpl_encoder_weights,
            self.smpl_encoder_biases,
            SONIC_ENCODER_DIMS,
            "SMPL encoder",
        )
        _validate_linear_stack(
            self.decoder_weights,
            self.decoder_biases,
            SONIC_DECODER_DIMS,
            "dynamic decoder",
        )
        if self.released_std.shape != (SONIC_G1_ACTION_SIZE,):
            raise ValueError("released SONIC standard deviation must have shape (29,)")
        if not np.all(np.isfinite(self.released_std)) or np.any(
            self.released_std <= 0.0
        ):
            raise ValueError(
                "released SONIC standard deviations must be positive and finite"
            )
        for name, value in (
            ("checkpoint_sha256", self.checkpoint_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            _require_sha256(value, name)


@dataclass(frozen=True)
class SonicDecoderInitialization:
    token_projection_weight: np.ndarray
    root_motion_projection_weight: np.ndarray
    proprioception_projection_weight: np.ndarray
    first_bias: np.ndarray
    middle_weights: tuple[np.ndarray, ...]
    middle_biases: tuple[np.ndarray, ...]
    output_weight: np.ndarray
    output_bias: np.ndarray
    output_std: np.ndarray
    g1_from_bello_proprio: np.ndarray
    g1_from_bello_proprio_offset: np.ndarray

    def __post_init__(self) -> None:
        if self.token_projection_weight.shape != (2048, SONIC_TOKEN_FLAT_SIZE):
            raise ValueError("SONIC token projection shape is invalid")
        if self.root_motion_projection_weight.shape != (
            2048,
            SONIC_ROOT_MOTION_COMMAND_SIZE,
        ):
            raise ValueError("SONIC root-motion projection shape is invalid")
        if self.proprioception_projection_weight.shape != (
            2048,
            SONIC_BELLO_PROPRIO_SIZE,
        ):
            raise ValueError("SONIC Bello proprioception projection shape is invalid")
        if self.first_bias.shape != (2048,):
            raise ValueError("SONIC first-layer bias shape is invalid")
        if len(self.middle_weights) != 5 or len(self.middle_biases) != 5:
            raise ValueError("SONIC must retain exactly five frozen middle linears")
        if self.output_weight.shape != (SONIC_BELLO_ACTION_SIZE, 512):
            raise ValueError("SONIC Bello output projection shape is invalid")
        if self.output_bias.shape != (SONIC_BELLO_ACTION_SIZE,):
            raise ValueError("SONIC Bello output bias shape is invalid")
        if self.output_std.shape != (SONIC_BELLO_ACTION_SIZE,) or not np.all(
            np.isfinite(self.output_std) & (self.output_std > 0.0)
        ):
            raise ValueError("SONIC Bello output standard deviation is invalid")
        if self.g1_from_bello_proprio.shape != (
            SONIC_G1_PROPRIO_SIZE,
            SONIC_BELLO_PROPRIO_SIZE,
        ) or self.g1_from_bello_proprio_offset.shape != (SONIC_G1_PROPRIO_SIZE,):
            raise ValueError("SONIC proprioception affine shape is invalid")


def load_sonic_release_artifact(directory: Path) -> SonicReleaseWeights:
    """Load and hash-check the pickle-free artifact produced by the exporter."""

    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    weights_path = directory / "weights.npz"
    golden_path = directory / "golden.npz"
    if not all(path.is_file() for path in (manifest_path, weights_path, golden_path)):
        raise FileNotFoundError(
            "SONIC release artifact requires manifest.json, weights.npz, and "
            f"golden.npz: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "source_checkpoint_sha256",
        "source_config_sha256",
        "weights_file_sha256",
        "golden_file_sha256",
        "artifact_sha256",
        "array_sha256",
        "golden_array_sha256",
        "architecture",
    }
    if set(manifest) != expected_fields:
        raise ValueError("SONIC artifact manifest fields differ from schema version 1")
    if manifest["schema_version"] != SONIC_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported SONIC artifact schema version")
    for name in (
        "source_checkpoint_sha256",
        "source_config_sha256",
        "weights_file_sha256",
        "golden_file_sha256",
        "artifact_sha256",
    ):
        _require_sha256(manifest[name], name)
    if manifest["source_checkpoint_sha256"] != SONIC_OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError("SONIC artifact does not come from the pinned checkpoint")
    if manifest["source_config_sha256"] != SONIC_OFFICIAL_CONFIG_SHA256:
        raise ValueError("SONIC artifact does not come from the pinned release config")
    if manifest["architecture"] != _architecture_document():
        raise ValueError("SONIC artifact architecture differs from the released model")
    actual_weights_hash = _file_sha256(weights_path)
    if actual_weights_hash != manifest["weights_file_sha256"]:
        raise ValueError("SONIC weights.npz hash differs from manifest")
    if _file_sha256(golden_path) != manifest["golden_file_sha256"]:
        raise ValueError("SONIC golden.npz hash differs from manifest")
    with np.load(weights_path, allow_pickle=False) as archive:
        expected_names = _artifact_array_names()
        if set(archive.files) != expected_names:
            raise ValueError(
                "SONIC artifact arrays differ from schema: "
                f"missing={sorted(expected_names - set(archive.files))}, "
                f"extra={sorted(set(archive.files) - expected_names)}"
            )
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in archive.files
        }
    if set(manifest["array_sha256"]) != expected_names:
        raise ValueError("SONIC manifest array-hash fields differ from the schema")
    for name, array in arrays.items():
        if _array_sha256(array) != manifest["array_sha256"][name]:
            raise ValueError(f"SONIC array {name!r} differs from its manifest hash")
        array.flags.writeable = False
    artifact_hash = _artifact_sha256(arrays)
    if artifact_hash != manifest["artifact_sha256"]:
        raise ValueError("SONIC artifact content hash differs from manifest")
    if artifact_hash != SONIC_OFFICIAL_ARTIFACT_SHA256:
        raise ValueError("SONIC artifact arrays differ from the pinned release export")
    load_sonic_golden(directory, manifest=manifest)
    return SonicReleaseWeights(
        smpl_encoder_weights=tuple(
            arrays[f"smpl_encoder_weight_{index}"] for index in range(5)
        ),
        smpl_encoder_biases=tuple(
            arrays[f"smpl_encoder_bias_{index}"] for index in range(5)
        ),
        decoder_weights=tuple(arrays[f"decoder_weight_{index}"] for index in range(7)),
        decoder_biases=tuple(arrays[f"decoder_bias_{index}"] for index in range(7)),
        released_std=arrays["released_std"],
        checkpoint_sha256=manifest["source_checkpoint_sha256"],
        artifact_sha256=artifact_hash,
    )


def load_sonic_golden(
    directory: Path, *, manifest: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Load the exporter parity vectors and validate every array digest."""

    directory = Path(directory).expanduser().resolve()
    if manifest is None:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "golden.npz"
    if _file_sha256(path) != manifest.get("golden_file_sha256"):
        raise ValueError("SONIC golden.npz hash differs from manifest")
    expected_shapes = {
        "smpl_input": (8, SONIC_SMPL_INPUT_SIZE),
        "g1_proprio": (8, SONIC_G1_PROPRIO_SIZE),
        "encoder_latent": (8, SONIC_TOKEN_FLAT_SIZE),
        "quantized_tokens": (8, SONIC_TOKEN_COUNT, SONIC_TOKEN_SIZE),
        "g1_action": (8, SONIC_G1_ACTION_SIZE),
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected_shapes):
            raise ValueError("SONIC golden arrays differ from the artifact schema")
        arrays = {
            name: np.asarray(archive[name], dtype=np.float32).copy()
            for name in archive.files
        }
    recorded_hashes = manifest.get("golden_array_sha256")
    if not isinstance(recorded_hashes, dict) or set(recorded_hashes) != set(
        expected_shapes
    ):
        raise ValueError("SONIC golden array hashes differ from the artifact schema")
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"SONIC golden array {name!r} has an invalid shape")
        if _array_sha256(arrays[name]) != recorded_hashes[name]:
            raise ValueError(f"SONIC golden array {name!r} differs from its hash")
        arrays[name].flags.writeable = False
    return arrays


def build_bello_decoder_initialization(
    release: SonicReleaseWeights,
    calibration: SonicJointCalibration,
    *,
    bello_joint_names: tuple[str, ...],
    bello_action_scale: tuple[float, ...] | np.ndarray,
    minimum_action_std: float,
) -> SonicDecoderInitialization:
    """Compose the original G1 edge linears with coupled morphology maps."""

    if calibration.g1_joint_names != G1_ISAACLAB_JOINT_NAMES:
        raise ValueError("joint calibration does not use released SONIC G1 order")
    if calibration.bello_joint_names != bello_joint_names:
        raise ValueError("joint calibration and Bello model joint orders differ")
    if len(bello_joint_names) != SONIC_BELLO_ACTION_SIZE:
        raise ValueError("SONIC Bello embodiment must expose exactly 25 actions")
    bello_action_scale = _validated_bello_action_scale(bello_action_scale)
    affine, affine_offset = g1_from_bello_proprio_affine(
        calibration,
        bello_action_scale=bello_action_scale,
    )
    first_weight = release.decoder_weights[0]
    token_weight = first_weight[:, :SONIC_TOKEN_FLAT_SIZE].copy()
    original_prop_weight = first_weight[:, SONIC_TOKEN_FLAT_SIZE:]
    proprio_weight = original_prop_weight @ affine
    first_bias = release.decoder_biases[0] + original_prop_weight @ affine_offset

    output_weight = np.zeros((SONIC_BELLO_ACTION_SIZE, 512), dtype=np.float32)
    output_bias = np.zeros((SONIC_BELLO_ACTION_SIZE,), dtype=np.float32)
    output_std = bello_action_std_from_g1(
        release.released_std,
        calibration,
        bello_action_scale=bello_action_scale,
        minimum_action_std=minimum_action_std,
    )
    bello_from_g1 = _normalized_bello_from_g1(
        calibration,
        bello_action_scale,
    )
    output_weight[:] = bello_from_g1 @ release.decoder_weights[-1]
    output_bias[:] = bello_from_g1 @ release.decoder_biases[-1]
    return SonicDecoderInitialization(
        token_projection_weight=token_weight.astype(np.float32),
        root_motion_projection_weight=np.zeros(
            (2048, SONIC_ROOT_MOTION_COMMAND_SIZE), dtype=np.float32
        ),
        proprioception_projection_weight=proprio_weight.astype(np.float32),
        first_bias=first_bias.astype(np.float32),
        middle_weights=tuple(weight.copy() for weight in release.decoder_weights[1:-1]),
        middle_biases=tuple(bias.copy() for bias in release.decoder_biases[1:-1]),
        output_weight=output_weight,
        output_bias=output_bias,
        output_std=output_std,
        g1_from_bello_proprio=affine,
        g1_from_bello_proprio_offset=affine_offset,
    )


def bello_action_std_from_g1(
    released_std: np.ndarray,
    calibration: SonicJointCalibration,
    *,
    bello_action_scale: tuple[float, ...] | np.ndarray,
    minimum_action_std: float,
) -> np.ndarray:
    """Propagate diagonal G1 exploration variance through the dense edge map."""

    released_std = np.asarray(released_std, dtype=np.float32)
    if released_std.shape != (SONIC_G1_ACTION_SIZE,) or not np.all(
        np.isfinite(released_std) & (released_std > 0.0)
    ):
        raise ValueError("released SONIC standard deviation is invalid")
    if not np.isfinite(minimum_action_std) or minimum_action_std <= 0.0:
        raise ValueError("minimum Bello action standard deviation is invalid")
    bello_from_g1 = _normalized_bello_from_g1(
        calibration,
        _validated_bello_action_scale(bello_action_scale),
    )
    variance = (bello_from_g1 * bello_from_g1) @ (released_std * released_std)
    return np.maximum(np.sqrt(variance), minimum_action_std).astype(np.float32)


def g1_from_bello_proprio_affine(
    calibration: SonicJointCalibration,
    *,
    bello_action_scale: tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the default-anchored Bello-to-G1 proprioception edge map."""
    matrix = np.zeros(
        (SONIC_G1_PROPRIO_SIZE, SONIC_BELLO_PROPRIO_SIZE), dtype=np.float32
    )
    offset = np.zeros((SONIC_G1_PROPRIO_SIZE,), dtype=np.float32)
    joint_matrix = np.asarray(calibration.g1_from_bello_matrix, dtype=np.float32)
    bello_action_scale = _validated_bello_action_scale(bello_action_scale)
    normalized_action_matrix = (
        joint_matrix * bello_action_scale[None, :] / SONIC_G1_ACTION_SCALE[:, None]
    )
    # Term-major history: gravity, angular velocity, q, qdot, previous action.
    _repeat_vector_basis(matrix, 0, 0, 10)
    _repeat_vector_basis(matrix, 30, 30, 10)
    g1_joint_start = 60
    bello_joint_start = 60
    g1_velocity_start = g1_joint_start + 10 * SONIC_G1_ACTION_SIZE
    bello_velocity_start = bello_joint_start + 10 * SONIC_BELLO_ACTION_SIZE
    g1_action_start = g1_velocity_start + 10 * SONIC_G1_ACTION_SIZE
    bello_action_start = bello_velocity_start + 10 * SONIC_BELLO_ACTION_SIZE
    for history in range(10):
        g1_joint_slice = slice(
            g1_joint_start + history * SONIC_G1_ACTION_SIZE,
            g1_joint_start + (history + 1) * SONIC_G1_ACTION_SIZE,
        )
        bello_joint_slice = slice(
            bello_joint_start + history * SONIC_BELLO_ACTION_SIZE,
            bello_joint_start + (history + 1) * SONIC_BELLO_ACTION_SIZE,
        )
        g1_velocity_slice = slice(
            g1_velocity_start + history * SONIC_G1_ACTION_SIZE,
            g1_velocity_start + (history + 1) * SONIC_G1_ACTION_SIZE,
        )
        bello_velocity_slice = slice(
            bello_velocity_start + history * SONIC_BELLO_ACTION_SIZE,
            bello_velocity_start + (history + 1) * SONIC_BELLO_ACTION_SIZE,
        )
        g1_action_slice = slice(
            g1_action_start + history * SONIC_G1_ACTION_SIZE,
            g1_action_start + (history + 1) * SONIC_G1_ACTION_SIZE,
        )
        bello_action_slice = slice(
            bello_action_start + history * SONIC_BELLO_ACTION_SIZE,
            bello_action_start + (history + 1) * SONIC_BELLO_ACTION_SIZE,
        )
        matrix[g1_joint_slice, bello_joint_slice] = joint_matrix
        matrix[g1_velocity_slice, bello_velocity_slice] = joint_matrix
        matrix[g1_action_slice, bello_action_slice] = normalized_action_matrix
    return matrix, offset


def _validated_bello_action_scale(
    value: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    scale = np.asarray(value, dtype=np.float32)
    if scale.shape != (SONIC_BELLO_ACTION_SIZE,) or not np.all(
        np.isfinite(scale) & (scale > 0.0)
    ):
        raise ValueError("Bello SONIC action scale must contain 25 positive values")
    return scale


def _normalized_bello_from_g1(
    calibration: SonicJointCalibration,
    bello_action_scale: np.ndarray,
) -> np.ndarray:
    joint_matrix = np.asarray(calibration.bello_from_g1_matrix, dtype=np.float32)
    return (
        joint_matrix
        * SONIC_G1_ACTION_SCALE[None, :]
        / bello_action_scale[:, None]
    ).astype(np.float32)


def sonic_fsq(value: jax.Array) -> jax.Array:
    """Released 32-level finite scalar quantizer, including even-level shift."""

    value = jnp.asarray(value, dtype=jnp.float32)
    half_l = jnp.asarray((32.0 - 1.0) * (1.0 + 1.0e-3) / 2.0, value.dtype)
    offset = jnp.asarray(0.5, value.dtype)
    shift = jnp.arctanh(offset / half_l)
    bounded = jnp.tanh(value + shift) * half_l - offset
    # The forward value is round(bounded), with a straight-through derivative.
    rounded = bounded + jax.lax.stop_gradient(jnp.round(bounded) - bounded)
    return rounded / 16.0


def apply_frozen_mlp(
    value: jax.Array,
    weights: tuple[jax.Array, ...],
    biases: tuple[jax.Array, ...],
) -> jax.Array:
    for index, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
        value = jnp.matmul(
            value,
            jax.lax.stop_gradient(weight).T,
            precision=jax.lax.Precision.HIGHEST,
        ) + jax.lax.stop_gradient(bias)
        if index != len(weights) - 1:
            value = jax.nn.silu(value)
    return value


def _repeat_vector_basis(
    matrix: np.ndarray, output_start: int, input_start: int, history: int
) -> None:
    for frame in range(history):
        out = output_start + 3 * frame
        inp = input_start + 3 * frame
        matrix[out : out + 3, inp : inp + 3] = BELLO_TO_G1_VECTOR_BASIS


def _validate_linear_stack(
    weights: tuple[np.ndarray, ...],
    biases: tuple[np.ndarray, ...],
    dims: tuple[int, ...],
    name: str,
) -> None:
    if len(weights) != len(dims) - 1 or len(biases) != len(dims) - 1:
        raise ValueError(f"{name} layer count is invalid")
    for index, (weight, bias, input_size, output_size) in enumerate(
        zip(weights, biases, dims[:-1], dims[1:], strict=True)
    ):
        if weight.shape != (output_size, input_size) or bias.shape != (output_size,):
            raise ValueError(f"{name} layer {index} shape is invalid")
        if weight.dtype != np.float32 or bias.dtype != np.float32:
            raise ValueError(f"{name} layer {index} must use float32")


def _artifact_array_names() -> set[str]:
    return {
        *(f"smpl_encoder_weight_{index}" for index in range(5)),
        *(f"smpl_encoder_bias_{index}" for index in range(5)),
        *(f"decoder_weight_{index}" for index in range(7)),
        *(f"decoder_bias_{index}" for index in range(7)),
        "released_std",
    }


def _architecture_document() -> dict[str, Any]:
    return {
        "smpl_encoder_dims": list(SONIC_ENCODER_DIMS),
        "smpl_frame_count": SONIC_SMPL_FRAME_COUNT,
        "smpl_frame_size": SONIC_SMPL_FRAME_SIZE,
        "fsq_levels": [32] * SONIC_TOKEN_SIZE,
        "token_count": SONIC_TOKEN_COUNT,
        "decoder_dims": list(SONIC_DECODER_DIMS),
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
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
