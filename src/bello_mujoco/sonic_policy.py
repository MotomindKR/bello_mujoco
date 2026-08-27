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

SONIC_ARTIFACT_SCHEMA_VERSION = 3
SONIC_ROBOT_FRAME_COUNT = 10
SONIC_ROBOT_FRAME_SIZE = 64
SONIC_ROBOT_INPUT_SIZE = SONIC_ROBOT_FRAME_COUNT * SONIC_ROBOT_FRAME_SIZE
SONIC_TOKEN_COUNT = 2
SONIC_TOKEN_SIZE = 32
SONIC_TOKEN_FLAT_SIZE = SONIC_TOKEN_COUNT * SONIC_TOKEN_SIZE
SONIC_G1_ACTION_SIZE = 29
SONIC_BELLO_ACTION_SIZE = 25
SONIC_G1_PROPRIO_SIZE = 930
SONIC_BELLO_PROPRIO_SIZE = 810
SONIC_PROPRIO_HISTORY_FRAMES = 10
SONIC_ACTOR_OBSERVATION_SIZE = SONIC_ROBOT_INPUT_SIZE + SONIC_BELLO_PROPRIO_SIZE
SONIC_G1_CRITIC_OBSERVATION_SIZE = 1645
SONIC_ENCODER_DIMS = (640, 2048, 1024, 512, 512, 64)
SONIC_DECODER_DIMS = (994, 2048, 2048, 1024, 1024, 512, 512, 29)
SONIC_CRITIC_DIMS = (1645, 2048, 2048, 1024, 1024, 512, 512, 1)
SONIC_OFFICIAL_CHECKPOINT_SHA256 = (
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909"
)
SONIC_OFFICIAL_CONFIG_SHA256 = (
    "f08187795fa16a839a28bc1c18e0555d38d9420e03733744341cdcb56ab629c7"
)
SONIC_OFFICIAL_ARTIFACT_SHA256 = (
    "abfcabcae82fde61f873c87ef735cdf9d8b278f6140521d5c3e4df66bd3088bf"
)

BELLO_TO_G1_VECTOR_BASIS = np.asarray(
    (
        (0.0, 1.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)

# The pinned G1 model tilts its roll/yaw hip pair by 0.1749 rad.  Bello's
# canonical URDF has a mirrored hip-pitch axis 30 degrees from the sagittal
# pitch axis.  These small deterministic blocks are the joint-level
# decouplings from Any2Any, expressed in G1's pitch/roll/yaw semantic order.
_G1_HIP_ROLL_INCLINATION_RADIANS = 0.1749
_BELLO_HIP_PITCH_INCLINATION_RADIANS = np.pi / 6.0

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

SONIC_G1_DEFAULT_JOINT_POSITION = np.asarray(
    tuple(
        -0.312
        if "hip_pitch" in name
        else 0.669
        if "knee" in name
        else -0.363
        if "ankle_pitch" in name
        else 0.6
        if name.endswith("_elbow_joint")
        else 0.2
        if name
        in {
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
        }
        else -0.2
        if name == "right_shoulder_roll_joint"
        else 0.0
        for name in G1_ISAACLAB_JOINT_NAMES
    ),
    dtype=np.float32,
)


@dataclass(frozen=True)
class SonicReleaseWeights:
    robot_encoder_weights: tuple[np.ndarray, ...]
    robot_encoder_biases: tuple[np.ndarray, ...]
    decoder_weights: tuple[np.ndarray, ...]
    decoder_biases: tuple[np.ndarray, ...]
    critic_weights: tuple[np.ndarray, ...]
    critic_biases: tuple[np.ndarray, ...]
    critic_observation_mean: np.ndarray
    critic_observation_variance: np.ndarray
    released_std: np.ndarray
    checkpoint_sha256: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        _validate_linear_stack(
            self.robot_encoder_weights,
            self.robot_encoder_biases,
            SONIC_ENCODER_DIMS,
            "robot motion encoder",
        )
        _validate_linear_stack(
            self.decoder_weights,
            self.decoder_biases,
            SONIC_DECODER_DIMS,
            "dynamic decoder",
        )
        _validate_linear_stack(
            self.critic_weights,
            self.critic_biases,
            SONIC_CRITIC_DIMS,
            "critic",
        )
        for name, value in (
            ("mean", self.critic_observation_mean),
            ("variance", self.critic_observation_variance),
        ):
            if value.shape != (SONIC_G1_CRITIC_OBSERVATION_SIZE,) or not np.all(
                np.isfinite(value)
            ):
                raise ValueError(f"released SONIC critic observation {name} is invalid")
        if np.any(self.critic_observation_variance < 0.0):
            raise ValueError("released SONIC critic observation variance is negative")
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
    g1_from_bello_proprio: np.ndarray
    g1_from_bello_proprio_offset: np.ndarray
    bello_from_g1_action: np.ndarray

    def __post_init__(self) -> None:
        if self.g1_from_bello_proprio.shape != (
            SONIC_G1_PROPRIO_SIZE,
            SONIC_BELLO_PROPRIO_SIZE,
        ) or self.g1_from_bello_proprio_offset.shape != (SONIC_G1_PROPRIO_SIZE,):
            raise ValueError("SONIC proprioception affine shape is invalid")
        if self.bello_from_g1_action.shape != (
            SONIC_BELLO_ACTION_SIZE,
            SONIC_G1_ACTION_SIZE,
        ):
            raise ValueError("SONIC Bello-from-G1 action map shape is invalid")


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
        raise ValueError(
            "SONIC artifact manifest fields differ from the supported schema"
        )
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
        robot_encoder_weights=tuple(
            arrays[f"robot_encoder_weight_{index}"] for index in range(5)
        ),
        robot_encoder_biases=tuple(
            arrays[f"robot_encoder_bias_{index}"] for index in range(5)
        ),
        decoder_weights=tuple(arrays[f"decoder_weight_{index}"] for index in range(7)),
        decoder_biases=tuple(arrays[f"decoder_bias_{index}"] for index in range(7)),
        critic_weights=tuple(arrays[f"critic_weight_{index}"] for index in range(7)),
        critic_biases=tuple(arrays[f"critic_bias_{index}"] for index in range(7)),
        critic_observation_mean=arrays["critic_observation_mean"],
        critic_observation_variance=arrays["critic_observation_variance"],
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
        "robot_encoder_input": (8, SONIC_ROBOT_INPUT_SIZE),
        "g1_proprio": (8, SONIC_G1_PROPRIO_SIZE),
        "encoder_latent": (8, SONIC_TOKEN_FLAT_SIZE),
        "quantized_tokens": (8, SONIC_TOKEN_COUNT, SONIC_TOKEN_SIZE),
        "g1_action": (8, SONIC_G1_ACTION_SIZE),
        "critic_input": (8, SONIC_G1_CRITIC_OBSERVATION_SIZE),
        "critic_normalized_input": (8, SONIC_G1_CRITIC_OBSERVATION_SIZE),
        "critic_value": (8, 1),
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
    bello_default_joint_position: tuple[float, ...] | np.ndarray,
) -> SonicDecoderInitialization:
    """Build fixed source-semantic alignment around the released G1 policy."""

    if calibration.g1_joint_names != G1_ISAACLAB_JOINT_NAMES:
        raise ValueError("joint calibration does not use released SONIC G1 order")
    if calibration.bello_joint_names != bello_joint_names:
        raise ValueError("joint calibration and Bello model joint orders differ")
    if len(bello_joint_names) != SONIC_BELLO_ACTION_SIZE:
        raise ValueError("SONIC Bello embodiment must expose exactly 25 actions")
    bello_action_scale = _validated_bello_action_scale(bello_action_scale)
    bello_default_joint_position = _validated_bello_default_joint_position(
        bello_default_joint_position
    )
    affine, affine_offset = g1_from_bello_proprio_affine(
        calibration,
        bello_action_scale=bello_action_scale,
        bello_default_joint_position=bello_default_joint_position,
    )
    bello_from_g1 = _normalized_bello_from_g1(
        semantic_bello_from_g1_joint_map(calibration), bello_action_scale
    )
    return SonicDecoderInitialization(
        g1_from_bello_proprio=affine,
        g1_from_bello_proprio_offset=affine_offset,
        bello_from_g1_action=bello_from_g1,
    )


def semantic_g1_from_bello_joint_map(
    calibration: SonicJointCalibration,
) -> np.ndarray:
    """Return Any2Any-style sparse source semantics for Bello's 25 joints.

    Corresponding scalar axes are signed scatters.  The hip blocks compose the
    fixed G1 and Bello inclination decouplings.  No cross-side, cross-limb, or
    motion-correlation coefficients can enter the policy interface.
    """

    return _semantic_joint_maps(calibration)[0]


def semantic_bello_from_g1_joint_map(
    calibration: SonicJointCalibration,
) -> np.ndarray:
    """Return the paired G1-to-Bello action map for matched joints."""

    return _semantic_joint_maps(calibration)[1]


def _semantic_joint_maps(
    calibration: SonicJointCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Build paired Any2Any maps through a common canonical hip basis."""

    if calibration.g1_joint_names != G1_ISAACLAB_JOINT_NAMES:
        raise ValueError("joint calibration does not use released SONIC G1 order")
    g1_index = {name: index for index, name in enumerate(calibration.g1_joint_names)}
    bello_index = {
        name: index for index, name in enumerate(calibration.bello_joint_names)
    }
    dense = np.asarray(calibration.g1_from_bello_matrix, dtype=np.float64)
    scatter = np.zeros_like(dense)
    for g1_name in calibration.g1_joint_names:
        bello_name = _corresponding_bello_joint_name(g1_name, bello_index)
        if bello_name is None:
            continue
        coefficient = dense[g1_index[g1_name], bello_index[bello_name]]
        if abs(coefficient) <= 1.0e-8:
            raise ValueError(
                f"SONIC semantic alignment has no axis evidence for {g1_name}"
            )
        scatter[g1_index[g1_name], bello_index[bello_name]] = np.copysign(
            1.0, coefficient
        )
    if not np.allclose(
        scatter.T @ scatter,
        np.eye(SONIC_BELLO_ACTION_SIZE),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("SONIC semantic alignment does not cover every Bello joint")

    source_decoupling = np.eye(SONIC_G1_ACTION_SIZE, dtype=np.float64)
    target_decoupling = np.eye(SONIC_G1_ACTION_SIZE, dtype=np.float64)
    source_cos = np.cos(_G1_HIP_ROLL_INCLINATION_RADIANS)
    source_sin = np.sin(_G1_HIP_ROLL_INCLINATION_RADIANS)
    target_cos = np.cos(_BELLO_HIP_PITCH_INCLINATION_RADIANS)
    target_sin = np.sin(_BELLO_HIP_PITCH_INCLINATION_RADIANS)
    source_block = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, source_cos, -source_sin),
            (0.0, source_sin, source_cos),
        )
    )
    for side, mirrored_sine in (("left", -target_sin), ("right", target_sin)):
        names = tuple(f"{side}_hip_{axis}_joint" for axis in ("pitch", "roll", "yaw"))
        rows = [g1_index[name] for name in names]
        target_block = np.asarray(
            (
                (target_cos, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (mirrored_sine, 0.0, 1.0),
            )
        )
        source_decoupling[np.ix_(rows, rows)] = source_block
        target_decoupling[np.ix_(rows, rows)] = target_block

    g1_from_bello = np.linalg.solve(source_decoupling, target_decoupling @ scatter)
    bello_from_g1 = scatter.T @ np.linalg.solve(target_decoupling, source_decoupling)
    if not np.allclose(
        bello_from_g1 @ g1_from_bello,
        np.eye(SONIC_BELLO_ACTION_SIZE),
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise ValueError("SONIC paired semantic joint maps are not inverse on Bello")
    return g1_from_bello.astype(np.float32), bello_from_g1.astype(np.float32)


def _corresponding_bello_joint_name(
    g1_name: str, bello_index: dict[str, int]
) -> str | None:
    candidates = [g1_name]
    if g1_name.endswith("_elbow_joint"):
        candidates.insert(0, g1_name.replace("_elbow_joint", "_elbow_pitch_joint"))
    elif g1_name.endswith("_wrist_roll_joint"):
        candidates.insert(0, g1_name.replace("_wrist_roll_joint", "_elbow_yaw_joint"))
    return next((name for name in candidates if name in bello_index), None)


def g1_from_bello_proprio_affine(
    calibration: SonicJointCalibration,
    *,
    bello_action_scale: tuple[float, ...] | np.ndarray,
    bello_default_joint_position: tuple[float, ...] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the fixed source-semantic map in relative proprio coordinates."""
    matrix = np.zeros(
        (SONIC_G1_PROPRIO_SIZE, SONIC_BELLO_PROPRIO_SIZE), dtype=np.float32
    )
    offset = np.zeros((SONIC_G1_PROPRIO_SIZE,), dtype=np.float32)
    joint_matrix = semantic_g1_from_bello_joint_map(calibration)
    bello_action_scale = _validated_bello_action_scale(bello_action_scale)
    _validated_bello_default_joint_position(bello_default_joint_position)
    position_offset = np.zeros((SONIC_G1_ACTION_SIZE,), dtype=np.float32)
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
        offset[g1_joint_slice] = position_offset
        matrix[g1_velocity_slice, bello_velocity_slice] = joint_matrix
        matrix[g1_action_slice, bello_action_slice] = normalized_action_matrix
    return matrix, offset


def _validated_bello_default_joint_position(
    value: tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    position = np.asarray(value, dtype=np.float32)
    if position.shape != (SONIC_BELLO_ACTION_SIZE,) or not np.all(
        np.isfinite(position)
    ):
        raise ValueError("Bello SONIC default joint position must contain 25 values")
    return position


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
    joint_matrix: np.ndarray,
    bello_action_scale: np.ndarray,
) -> np.ndarray:
    joint_matrix = np.asarray(joint_matrix, dtype=np.float32)
    return (
        joint_matrix * SONIC_G1_ACTION_SCALE[None, :] / bello_action_scale[:, None]
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


def _architecture_document() -> dict[str, Any]:
    return {
        "robot_encoder_dims": list(SONIC_ENCODER_DIMS),
        "robot_frame_count": SONIC_ROBOT_FRAME_COUNT,
        "robot_frame_size": SONIC_ROBOT_FRAME_SIZE,
        "robot_future_offsets": list(range(0, 50, 5)),
        "command_temporal_layout": "qpos_major_then_qvel_major_before_reshape",
        "fsq_levels": [32] * SONIC_TOKEN_SIZE,
        "token_count": SONIC_TOKEN_COUNT,
        "decoder_dims": list(SONIC_DECODER_DIMS),
        "critic_dims": list(SONIC_CRITIC_DIMS),
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
    normalized = value.removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
