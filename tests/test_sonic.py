from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.envs.sonic import (
    BelloSonicEnv,
    SONIC_TRACKED_BODY_NAMES,
    SonicObservationContract,
    _build_sampling_bins,
)
from bello_mujoco.motion.sonic_dataset import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_MUJOCO_JOINT_NAMES,
    G1_WRIST_JOINT_NAMES,
    SONIC_CRITIC_FUTURE_OFFSETS,
    SONIC_DATASET_SCHEMA_VERSION,
    SONIC_G1_RETARGET_FORMAT_VERSION,
    SONIC_SMPL_COORDINATE_CONVENTION,
    SONIC_SMPL_FUTURE_OFFSETS,
    SONIC_SMPL_JOINT_NAMES,
    SONIC_TRAIN_SAMPLING,
    SonicCalibrationConfig,
    SonicJointCalibration,
    SonicReferenceBank,
    _JointCalibrationMoments,
    _align_smpl_root_to_bello_world,
    _json_sha256,
    build_padded_sonic_sampling_bins,
    fit_joint_calibration,
    load_sonic_g1_retarget,
    load_sonic_dataset_index,
    load_sonic_reference_bank,
    select_sonic_reference_clip,
)
from bello_mujoco.policy import (
    ActorCritic,
    SonicActor,
    SonicEdgeDecoder,
    SonicFrozenSmplEncoder,
    partition_policy_parameters,
)
from bello_mujoco.sonic_policy import (
    BELLO_TO_G1_VECTOR_BASIS,
    SONIC_BELLO_PROPRIO_SIZE,
    SONIC_G1_PROPRIO_SIZE,
    apply_frozen_mlp,
    bello_action_std_from_g1,
    g1_from_bello_proprio_affine,
    load_sonic_golden,
    load_sonic_release_artifact,
    sonic_fsq,
)
from bello_mujoco.training.sonic import (
    _tracking_dataset_summary,
    sonic_reference_shard_for_update,
)
from tools.sonic_gmr_retarget import (
    G1_QPOS_JOINT_NAMES,
    G1_RETARGET_FORMAT_VERSION,
    SMPL_JOINT_NAMES,
    SMPL_COORDINATE_CONVENTION,
    _canonicalize_smpl_joints,
    _resample_translation,
)


BELLO_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
    "right_wrist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "left_wrist_pitch_joint",
)
TEST_EDGE_PAIRS = tuple(
    (
        name.replace("_elbow_pitch_joint", "_elbow_joint"),
        name,
    )
    for name in BELLO_JOINT_NAMES
    if not name.endswith("_elbow_yaw_joint")
) + (
    ("right_wrist_yaw_joint", "right_elbow_yaw_joint"),
    ("left_wrist_yaw_joint", "left_elbow_yaw_joint"),
)


def test_sidecar_tool_and_runtime_schema_constants_match() -> None:
    assert G1_QPOS_JOINT_NAMES == G1_MUJOCO_JOINT_NAMES
    assert G1_RETARGET_FORMAT_VERSION == SONIC_G1_RETARGET_FORMAT_VERSION
    assert SMPL_COORDINATE_CONVENTION == SONIC_SMPL_COORDINATE_CONVENTION
    assert SMPL_JOINT_NAMES == SONIC_SMPL_JOINT_NAMES


def _calibration(*, scales: tuple[float, ...] | None = None) -> SonicJointCalibration:
    g1_index = {name: index for index, name in enumerate(G1_ISAACLAB_JOINT_NAMES)}
    bello_index = {name: index for index, name in enumerate(BELLO_JOINT_NAMES)}
    count = len(TEST_EDGE_PAIRS)
    scales = scales or (1.0,) * count
    g1_from_bello = np.zeros((29, 25), dtype=np.float64)
    bello_from_g1 = np.zeros((25, 29), dtype=np.float64)
    for (g1_name, bello_name), scale in zip(TEST_EDGE_PAIRS, scales, strict=True):
        g1_joint = g1_index[g1_name]
        bello_joint = bello_index[bello_name]
        g1_from_bello[g1_joint, bello_joint] = scale
        bello_from_g1[bello_joint, g1_joint] = 1.0 / scale
    g1_from_bello[g1_index["waist_roll_joint"], 0] = 0.25
    bello_from_g1[0, g1_index["waist_roll_joint"]] = 0.2
    return SonicJointCalibration(
        g1_joint_names=G1_ISAACLAB_JOINT_NAMES,
        bello_joint_names=BELLO_JOINT_NAMES,
        g1_from_bello_matrix=tuple(tuple(row) for row in g1_from_bello),
        bello_from_g1_matrix=tuple(tuple(row) for row in bello_from_g1),
        g1_explained_variance=(1.0,) * 29,
        g1_rms_residual_radians=(0.0,) * 29,
        bello_explained_variance=(1.0,) * 25,
        bello_rms_residual_radians=(0.0,) * 25,
        sample_count=1_000,
        training_content_sha256="a" * 64,
    )


def test_sonic_observation_contract_matches_bello_edge_dimensions() -> None:
    contract = SonicObservationContract(BELLO_JOINT_NAMES)

    assert contract.action_size == 25
    assert contract.history_frame_size == 84
    assert contract.actor_observation_size == 1650
    assert contract.critic_command_frame_size == 50
    assert contract.critic_observation_size == 1445
    assert contract.actor_layout.size == 1650
    assert contract.critic_layout.size == 1445


def _coupled_calibration_fixture(
    frames: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    bello = rng.normal(size=(frames, len(BELLO_JOINT_NAMES)))
    matrix = rng.normal(scale=0.04, size=(len(G1_ISAACLAB_JOINT_NAMES), 25))
    matrix[:25] += np.eye(25) * 0.8
    offset = np.linspace(-0.15, 0.15, 29)
    g1 = bello @ matrix.T + offset
    qpos = np.zeros((frames, 7 + len(BELLO_JOINT_NAMES)), dtype=np.float64)
    qpos[:, 7:] = bello
    clip = {
        "qpos": qpos,
        "qpos_joint_names": np.asarray(BELLO_JOINT_NAMES),
        "g1_joint_pos": g1,
    }
    return clip, matrix, offset


def test_joint_calibration_recovers_coupled_edge_maps() -> None:
    clip, matrix, offset = _coupled_calibration_fixture(600)

    fitted = fit_joint_calibration(
        (clip,),
        bello_joint_names=BELLO_JOINT_NAMES,
        content_hashes=("sha256:" + "b" * 64,),
        config=SonicCalibrationConfig(
            minimum_samples=100,
            ridge_regularization=1.0e-10,
            minimum_joint_standard_deviation=0.01,
            minimum_mean_explained_variance=0.99,
            maximum_rms_residual_radians=1.0e-4,
        ),
    )

    np.testing.assert_allclose(fitted.g1_from_bello_matrix, matrix, atol=1.0e-8)
    g1 = np.asarray(clip["g1_joint_pos"])
    bello = np.asarray(clip["qpos"][:, 7:])
    predicted_bello = (g1 - np.mean(g1, axis=0)) @ np.asarray(
        fitted.bello_from_g1_matrix
    ).T + np.mean(bello, axis=0)
    np.testing.assert_allclose(predicted_bello, bello, atol=1.0e-7)
    np.testing.assert_allclose(np.mean(g1 - bello @ matrix.T, axis=0), offset)
    assert fitted.training_content_sha256.startswith("sha256:")


def test_joint_calibration_streaming_matches_one_combined_update() -> None:
    frames = 240
    full_clip, _, _ = _coupled_calibration_fixture(frames)

    def clip(start: int, stop: int) -> dict[str, np.ndarray]:
        return {
            "qpos": full_clip["qpos"][start:stop],
            "qpos_joint_names": np.asarray(BELLO_JOINT_NAMES),
            "g1_joint_pos": full_clip["g1_joint_pos"][start:stop],
        }

    combined = _JointCalibrationMoments(BELLO_JOINT_NAMES)
    combined.update(clip(0, frames))
    streamed = _JointCalibrationMoments(BELLO_JOINT_NAMES)
    for start, stop in ((0, 37), (37, 151), (151, frames)):
        streamed.update(clip(start, stop))
    config = SonicCalibrationConfig(
        minimum_samples=100,
        ridge_regularization=1.0e-6,
        minimum_joint_standard_deviation=0.01,
        minimum_mean_explained_variance=0.99,
        maximum_rms_residual_radians=1.0e-3,
    )
    expected = combined.finalize(("sha256:" + "4" * 64,), config)
    actual = streamed.finalize(("sha256:" + "4" * 64,), config)

    np.testing.assert_allclose(
        actual.g1_from_bello_matrix, expected.g1_from_bello_matrix, atol=1.0e-12
    )
    np.testing.assert_allclose(
        actual.bello_from_g1_matrix, expected.bello_from_g1_matrix, atol=1.0e-10
    )
    np.testing.assert_allclose(
        actual.g1_explained_variance,
        expected.g1_explained_variance,
        atol=1.0e-12,
    )
    assert actual.sample_count == frames


def test_proprio_affine_maps_positions_velocities_actions_and_vector_basis() -> None:
    scales = tuple(0.8 + 0.01 * index for index in range(len(TEST_EDGE_PAIRS)))
    calibration = _calibration(scales=scales)
    matrix, bias = g1_from_bello_proprio_affine(calibration)

    np.testing.assert_array_equal(matrix[:3, :3], BELLO_TO_G1_VECTOR_BASIS)
    pair = 7
    history = 3
    g1_name, bello_name = TEST_EDGE_PAIRS[pair]
    g1_index = G1_ISAACLAB_JOINT_NAMES.index(g1_name)
    bello_index = BELLO_JOINT_NAMES.index(bello_name)
    scale = scales[pair]
    g1_q = 60 + history * 29 + g1_index
    bello_q = 60 + history * 25 + bello_index
    g1_qvel = 350 + history * 29 + g1_index
    bello_qvel = 310 + history * 25 + bello_index
    g1_action = 640 + history * 29 + g1_index
    bello_action = 560 + history * 25 + bello_index

    assert matrix[g1_q, bello_q] == pytest.approx(scale)
    assert bias[g1_q] == 0.0
    assert matrix[g1_qvel, bello_qvel] == pytest.approx(scale)
    assert bias[g1_qvel] == 0.0
    assert matrix[g1_action, bello_action] == pytest.approx(scale)
    assert bias[g1_action] == 0.0
    assert matrix.shape == (SONIC_G1_PROPRIO_SIZE, SONIC_BELLO_PROPRIO_SIZE)


def test_coupled_calibration_routes_complete_joint_blocks_to_the_edge() -> None:
    matrix, _ = g1_from_bello_proprio_affine(_calibration())
    expected = np.asarray(_calibration().g1_from_bello_matrix)

    for history in range(10):
        np.testing.assert_array_equal(
            matrix[
                60 + history * 29 : 60 + (history + 1) * 29,
                60 + history * 25 : 60 + (history + 1) * 25,
            ],
            expected,
        )


def test_released_action_std_is_propagated_through_dense_output_map() -> None:
    scales = tuple(
        (-1.0 if index % 2 else 1.0) * (0.5 + index / 20.0)
        for index in range(len(TEST_EDGE_PAIRS))
    )
    calibration = _calibration(scales=scales)
    released = np.linspace(0.25, 0.50, len(G1_ISAACLAB_JOINT_NAMES), dtype=np.float32)

    result = bello_action_std_from_g1(released, calibration, minimum_action_std=0.05)

    dense = np.asarray(calibration.bello_from_g1_matrix, dtype=np.float32)
    expected = np.sqrt((dense * dense) @ (released * released))
    np.testing.assert_allclose(result, np.maximum(expected, 0.05), atol=1.0e-7)


def test_sonic_action_target_applies_randomized_default_and_release_scale() -> None:
    fake = SimpleNamespace(
        action_size=2,
        model=SimpleNamespace(
            policy_ctrl_indices=np.asarray((0, 1), dtype=np.int32),
            default_joint_target=np.asarray((0.1, -0.2), dtype=np.float32),
            joint_angle_offset=np.asarray(((0.01, -0.02),), dtype=np.float32),
            actuator_coordinate_transform=np.eye(2, dtype=np.float32),
        ),
    )
    fake._action_target = MethodType(BelloSonicEnv._action_target, fake)

    target = fake._action_target(
        jnp.asarray((2.0, -4.0)),
        randomization=jnp.asarray((0.03, 0.04)),
    )

    np.testing.assert_allclose(target, np.asarray((1.13, -2.16)), atol=1.0e-6)


def test_sampling_bins_are_host_materialized_and_clip_local() -> None:
    clips, starts, widths, offsets = _build_sampling_bins(
        np.asarray((146, 96), dtype=np.int32), 50
    )

    np.testing.assert_array_equal(clips, np.asarray((0, 0, 0, 1, 1)))
    np.testing.assert_array_equal(starts, np.asarray((0, 50, 100, 0, 50)))
    np.testing.assert_array_equal(widths, np.asarray((50, 50, 1, 50, 1)))
    np.testing.assert_array_equal(offsets, np.asarray((0, 3, 5)))


def test_padded_sampling_bins_have_one_fixed_jit_shape() -> None:
    clips, starts, widths, valid, offsets = build_padded_sonic_sampling_bins(
        np.asarray((50, 100), dtype=np.int32),
        bin_size=50,
        frame_capacity=256,
        clip_capacity=4,
    )

    assert clips.shape == starts.shape == widths.shape == valid.shape == (9,)
    np.testing.assert_array_equal(clips[:3], np.asarray((0, 1, 1)))
    np.testing.assert_array_equal(starts[:3], np.asarray((0, 0, 50)))
    np.testing.assert_array_equal(widths[:3], np.asarray((5, 50, 5)))
    np.testing.assert_array_equal(valid[:3], np.ones(3, dtype=bool))
    assert not np.any(valid[3:])
    np.testing.assert_array_equal(offsets, np.asarray((0, 1, 3, 3, 3)))


def test_reference_shard_schedule_is_mass_weighted_and_sweeps_every_shard() -> None:
    index = {
        "shards": {
            "train": [
                {"sampling_mass": 4.0},
                {"sampling_mass": 2.0},
                {"sampling_mass": 1.0},
            ]
        }
    }
    schedule = [
        sonic_reference_shard_for_update(
            index, update=update, seed=0, full_shard_updates=8
        )
        for update in range(14)
    ]

    assert schedule.count(0) == 8
    assert schedule.count(1) == 4
    assert schedule.count(2) == 2
    assert set(schedule) == {0, 1, 2}


def test_tracking_dataset_summary_excludes_clip_and_fingerprint_payloads() -> None:
    index = {
        "schema_version": 3,
        "dataset_fingerprint": "sha256:" + "a" * 64,
        "portable_manifest_fingerprint": "sha256:" + "b" * 64,
        "policy_fps": 50.0,
        "train_sampling": "frame_count_times_manifest_weight",
        "shard_frame_capacity": 65536,
        "shard_clip_capacity": 256,
        "joint_names": list(BELLO_JOINT_NAMES),
        "body_names": ["pelvis_link"],
        "g1_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "robot_fingerprint": "sha256:" + "c" * 64,
        "gmr_revision": "d" * 40,
        "bello_gmr_config_sha256": "sha256:" + "e" * 64,
        "g1_gmr_revision": "f" * 40,
        "g1_gmr_config_sha256": "sha256:" + "1" * 64,
        "g1_robot_sha256": "sha256:" + "2" * 64,
        "joint_calibration": {"sample_count": 30},
        "shards": {
            "train": [{"sampling_mass": 20.0}],
            "evaluation": [{"sampling_mass": 10.0}],
        },
        "clips": [
            {"split": "train", "frames": 20},
            {"split": "evaluation", "frames": 10},
        ],
        "fingerprint_payload": {"portable_manifest": "large"},
    }

    summary = _tracking_dataset_summary(index)

    assert "clips" not in summary
    assert "fingerprint_payload" not in summary
    assert summary["splits"] == {
        "train": {"shards": 1, "clips": 1, "frames": 20, "sampling_mass": 20.0},
        "evaluation": {
            "shards": 1,
            "clips": 1,
            "frames": 10,
            "sampling_mass": 10.0,
        },
    }


def test_reference_clip_selection_rebases_packed_offsets() -> None:
    lengths = np.asarray((50, 60), dtype=np.int32)
    frames = int(lengths.sum())
    bank = _reference_bank(frames, lengths=lengths)

    selected = select_sonic_reference_clip(bank, 1)

    np.testing.assert_array_equal(selected.offsets, np.asarray((0,)))
    np.testing.assert_array_equal(selected.lengths, np.asarray((60,)))
    assert selected.source_ids == ("source_1",)
    assert selected.qpos.shape[0] == 60
    np.testing.assert_array_equal(selected.qpos, bank.qpos[50:])


def test_dataset_index_and_split_hashes_are_enforced(tmp_path: Path) -> None:
    index_path = _write_dataset_fixture(tmp_path)

    index = load_sonic_dataset_index(index_path)
    bank = load_sonic_reference_bank(index_path, split="train", shard_index=0)

    assert index["dataset_fingerprint"].startswith("sha256:")
    assert bank.valid_clip_count == 1
    assert bank.lengths[: bank.valid_clip_count].tolist() == [50]
    shard_path = tmp_path / "shards" / "train" / "00000.npz"
    shard_path.write_bytes(shard_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="shard 0 file differs"):
        load_sonic_reference_bank(index_path, split="train", shard_index=0)


def test_dataset_index_rejects_contract_tampering(tmp_path: Path) -> None:
    index_path = _write_dataset_fixture(tmp_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["joint_names"][0] = "renamed_joint"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint payload"):
        load_sonic_dataset_index(index_path)


def test_release_artifact_matches_exporter_golden() -> None:
    artifact = Path("artifacts/sonic/default_release")
    if not artifact.is_dir():
        pytest.skip("ignored SONIC release artifact is not installed")
    release = load_sonic_release_artifact(artifact)
    golden = load_sonic_golden(artifact)
    latent = apply_frozen_mlp(
        jnp.asarray(golden["smpl_input"]),
        tuple(jnp.asarray(value) for value in release.smpl_encoder_weights),
        tuple(jnp.asarray(value) for value in release.smpl_encoder_biases),
    )
    tokens = sonic_fsq(latent.reshape((-1, 2, 32)))
    g1_action = apply_frozen_mlp(
        jnp.concatenate((tokens.reshape((-1, 64)), golden["g1_proprio"]), axis=-1),
        tuple(jnp.asarray(value) for value in release.decoder_weights),
        tuple(jnp.asarray(value) for value in release.decoder_biases),
    )

    np.testing.assert_allclose(latent, golden["encoder_latent"], atol=2.0e-6)
    np.testing.assert_array_equal(tokens, golden["quantized_tokens"])
    np.testing.assert_allclose(g1_action, golden["g1_action"], atol=3.0e-6)


def test_sonic_parameter_partition_freezes_only_the_semantic_core() -> None:
    encoder = object.__new__(SonicFrozenSmplEncoder)
    object.__setattr__(encoder, "weights", (jnp.ones((1, 1)),))
    object.__setattr__(encoder, "biases", (jnp.ones((1,)),))
    decoder = object.__new__(SonicEdgeDecoder)
    object.__setattr__(decoder, "token_projection_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "proprioception_projection_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "first_bias", jnp.ones((1,)))
    object.__setattr__(decoder, "middle_weights", (jnp.ones((1, 1)),))
    object.__setattr__(decoder, "middle_biases", (jnp.ones((1,)),))
    object.__setattr__(decoder, "output_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "output_bias", jnp.ones((1,)))
    actor = object.__new__(SonicActor)
    object.__setattr__(actor, "smpl_encoder", encoder)
    object.__setattr__(actor, "decoder", decoder)
    policy = object.__new__(ActorCritic)
    object.__setattr__(policy, "actor", actor)
    object.__setattr__(policy, "critic", encoder)
    object.__setattr__(policy, "log_std", jnp.ones((1,)))
    object.__setattr__(policy, "min_log_std", -5.0)
    object.__setattr__(policy, "max_log_std", 1.0)
    object.__setattr__(policy, "action_distribution", "gaussian")
    object.__setattr__(policy, "action_limit", 20.0)
    object.__setattr__(policy, "stochastic_actor", False)
    trainable, frozen = partition_policy_parameters(policy)

    assert all(value is None for value in trainable.actor.smpl_encoder.weights)
    assert trainable.actor.decoder.token_projection_weight is None
    assert all(value is None for value in trainable.actor.decoder.middle_weights)
    assert trainable.actor.decoder.proprioception_projection_weight is not None
    assert trainable.actor.decoder.first_bias is not None
    assert trainable.actor.decoder.output_weight is not None
    assert all(value is not None for value in frozen.actor.smpl_encoder.weights)
    assert trainable.critic.weights[0] is not None
    assert trainable.log_std is not None


def test_smpl_translation_resampling_matches_gmr_frame_coordinates() -> None:
    translation = np.arange(15, dtype=np.float64).reshape(5, 3)

    result = _resample_translation(
        translation,
        source_fps=100.0,
        target_fps=50.0,
        target_frames=3,
    )

    np.testing.assert_array_equal(result, translation[[0, 2, 4]])


def test_smpl_canonicalization_retains_body_model_pelvis_offset() -> None:
    frames = 2
    translation = np.asarray(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    local = np.zeros((frames, len(SMPL_JOINT_NAMES), 3), dtype=np.float64)
    local[:, 0] = np.asarray((0.03, -0.02, 0.11))
    local[:, 1] = np.asarray((0.15, 0.04, -0.08))
    joint_world = local + translation[:, None, :]

    result = _canonicalize_smpl_joints(
        joint_world,
        translation,
        np.broadcast_to(np.eye(3), (frames, 3, 3)),
    )

    np.testing.assert_allclose(result, local, atol=1.0e-12)
    assert np.all(np.linalg.norm(result[:, 0], axis=-1) > 0.0)


def test_smpl_root_uses_the_same_initial_heading_removal_as_bello() -> None:
    half = np.sqrt(0.5)
    bello_initial = np.asarray((half, 0.0, 0.0, half))
    smpl_roots = np.asarray(
        (
            (half, 0.0, 0.0, half),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    aligned = _align_smpl_root_to_bello_world(smpl_roots, bello_initial)

    np.testing.assert_allclose(
        aligned[0], np.asarray((1.0, 0.0, 0.0, 0.0)), atol=1.0e-12
    )
    np.testing.assert_allclose(
        aligned[1], np.asarray((half, 0.0, 0.0, half)), atol=1.0e-12
    )


def test_g1_sidecar_enforces_shared_time_zero_and_root_quaternions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sidecar.npz"
    _write_g1_sidecar_fixture(path)

    loaded = load_sonic_g1_retarget(path)

    assert loaded["g1_qpos"].shape == (3, 36)

    _write_g1_sidecar_fixture(path, timestamp_offset=0.01)
    with pytest.raises(ValueError, match="timestamps must start at zero"):
        load_sonic_g1_retarget(path)

    _write_g1_sidecar_fixture(path, valid_g1_root=False)
    with pytest.raises(ValueError, match="G1 root quaternions are not normalized"):
        load_sonic_g1_retarget(path)


def _reference_bank(frames: int, *, lengths: np.ndarray) -> SonicReferenceBank:
    offsets = np.zeros(len(lengths), dtype=np.int32)
    if len(lengths) > 1:
        offsets[1:] = np.cumsum(lengths[:-1])
    qpos = np.zeros((frames, 32), dtype=np.float32)
    qpos[:, 0] = np.arange(frames, dtype=np.float32)
    qpos[:, 3] = 1.0
    bins = build_padded_sonic_sampling_bins(
        lengths,
        bin_size=50,
        frame_capacity=frames,
        clip_capacity=len(lengths),
    )
    return SonicReferenceBank(
        qpos=qpos,
        qvel=np.zeros((frames, 31), dtype=np.float32),
        body_pos=np.zeros((frames, len(SONIC_TRACKED_BODY_NAMES), 3), dtype=np.float32),
        body_quat_wxyz=np.broadcast_to(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frames, len(SONIC_TRACKED_BODY_NAMES), 4),
        ).copy(),
        body_lin_vel=np.zeros(
            (frames, len(SONIC_TRACKED_BODY_NAMES), 3), dtype=np.float32
        ),
        body_ang_vel=np.zeros(
            (frames, len(SONIC_TRACKED_BODY_NAMES), 3), dtype=np.float32
        ),
        foot_contacts=np.zeros((frames, 2), dtype=np.float32),
        smpl_joint_pos=np.zeros((frames, 24, 3), dtype=np.float32),
        smpl_root_quat_wxyz=np.broadcast_to(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (frames, 4)
        ).copy(),
        smpl_root_translation=np.zeros((frames, 3), dtype=np.float32),
        g1_joint_pos=np.zeros((frames, 29), dtype=np.float32),
        offsets=offsets,
        lengths=lengths,
        sampling_probabilities=np.ones(len(lengths), dtype=np.float32) / len(lengths),
        sampling_bin_clip_indices=bins[0],
        sampling_bin_starts=bins[1],
        sampling_bin_widths=bins[2],
        sampling_bin_valid=bins[3],
        sampling_clip_bin_offsets=bins[4],
        sampling_bin_size=50,
        valid_clip_count=len(lengths),
        source_ids=tuple(f"source_{index}" for index in range(len(lengths))),
        sequence_ids=tuple(f"sequence_{index}" for index in range(len(lengths))),
        joint_names=BELLO_JOINT_NAMES,
        body_names=SONIC_TRACKED_BODY_NAMES,
        calibration=_calibration(),
    )


def _write_g1_sidecar_fixture(
    path: Path,
    *,
    timestamp_offset: float = 0.0,
    valid_g1_root: bool = True,
) -> None:
    frames = 3
    timestamps = timestamp_offset + np.arange(frames, dtype=np.float64) / 50.0
    qpos = np.zeros((frames, 36), dtype=np.float32)
    if valid_g1_root:
        qpos[:, 3] = 1.0
    np.savez_compressed(
        path,
        timestamp_seconds=timestamps,
        source_timestamp_seconds=timestamps,
        g1_qpos=qpos,
        g1_qvel=np.zeros((frames, 35), dtype=np.float32),
        g1_qpos_joint_names=np.asarray(G1_MUJOCO_JOINT_NAMES),
        smpl_joint_pos=np.zeros((frames, 24, 3), dtype=np.float32),
        smpl_root_quat_wxyz=np.broadcast_to(
            np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
            (frames, 4),
        ),
        smpl_root_translation=np.zeros((frames, 3), dtype=np.float32),
        smpl_joint_names=np.asarray(SONIC_SMPL_JOINT_NAMES),
        source_sha256=np.asarray("a" * 64),
        retarget_format_version=np.asarray(
            SONIC_G1_RETARGET_FORMAT_VERSION, dtype=np.int32
        ),
        gmr_revision=np.asarray("b" * 40),
        gmr_config_sha256=np.asarray("c" * 64),
        g1_robot_sha256=np.asarray("d" * 64),
        coordinate_convention=np.asarray(SONIC_SMPL_COORDINATE_CONVENTION),
    )


def _write_dataset_fixture(root: Path) -> Path:
    frames = 50
    frame_capacity = 64
    clip_capacity = 2
    bank = _reference_bank(
        frame_capacity, lengths=np.asarray((frames,), dtype=np.int32)
    )
    packed = {
        name: getattr(bank, name)
        for name in (
            "qpos",
            "qvel",
            "body_pos",
            "body_quat_wxyz",
            "body_lin_vel",
            "body_ang_vel",
            "foot_contacts",
            "smpl_joint_pos",
            "smpl_root_quat_wxyz",
            "smpl_root_translation",
            "g1_joint_pos",
        )
    }
    shards = {}
    for split in ("train", "evaluation"):
        directory = root / "shards" / split
        directory.mkdir(parents=True)
        path = directory / "00000.npz"
        np.savez_compressed(path, **packed)
        shards[split] = [
            {
                "file": f"shards/{split}/00000.npz",
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "clip_count": 1,
                "data_frames": frames,
                "sampling_mass": float(frames),
            }
        ]
    clips = [
        {
            "source_id": f"source_{index}",
            "sequence_id": f"sequence_{index}",
            "split": split,
            "source_kind": "amass",
            "subject_id": f"subject_{index}",
            "sampling_weight": 1.0,
            "shard_index": 0,
            "offset": 0,
            "frames": frames,
            "duration_seconds": 0.98,
            "source_sha256": "sha256:" + "2" * 64,
            "synchronized_sha256": "sha256:" + "3" * 64,
            **(
                {
                    "sampling_probability_numerator": float(frames),
                    "sampling_probability_denominator": float(frames),
                }
                if split == "train"
                else {"evaluation_order": 0}
            ),
        }
        for index, split in enumerate(("train", "evaluation"))
    ]
    provenance = {
        "robot_fingerprint": "c" * 64,
        "gmr_revision": "d" * 40,
        "bello_gmr_config_sha256": "9" * 64,
        "g1_gmr_revision": "e" * 40,
        "g1_gmr_config_sha256": "f" * 64,
        "g1_robot_sha256": "1" * 64,
    }
    portable = {"schema_version": 1, "sequences": []}
    payload = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "portable_manifest": portable,
        "accepted_clip_hashes": [],
        "quality": {},
        "calibration_config": {},
        "sharding": {
            "frame_capacity": frame_capacity,
            "clip_capacity": clip_capacity,
        },
        "joint_calibration": _calibration().to_document(),
        "smpl_future_offsets": SONIC_SMPL_FUTURE_OFFSETS,
        "critic_future_offsets": SONIC_CRITIC_FUTURE_OFFSETS,
        "train_sampling": SONIC_TRAIN_SAMPLING,
        "shards": shards,
        "joint_names": BELLO_JOINT_NAMES,
        "body_names": SONIC_TRACKED_BODY_NAMES,
        "g1_joint_names": G1_ISAACLAB_JOINT_NAMES,
        "g1_wrist_joint_names": G1_WRIST_JOINT_NAMES,
        "provenance": provenance,
        "clips": clips,
    }
    index = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": _json_sha256(payload),
        "portable_manifest_fingerprint": _json_sha256(portable),
        "policy_fps": 50.0,
        "smpl_future_offsets": list(SONIC_SMPL_FUTURE_OFFSETS),
        "critic_future_offsets": list(SONIC_CRITIC_FUTURE_OFFSETS),
        "train_sampling": SONIC_TRAIN_SAMPLING,
        "shard_frame_capacity": frame_capacity,
        "shard_clip_capacity": clip_capacity,
        "shards": shards,
        "joint_names": list(BELLO_JOINT_NAMES),
        "body_names": list(SONIC_TRACKED_BODY_NAMES),
        "g1_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "g1_wrist_joint_names": list(G1_WRIST_JOINT_NAMES),
        **provenance,
        "joint_calibration": _calibration().to_document(),
        "clips": clips,
        "fingerprint_payload": payload,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path
