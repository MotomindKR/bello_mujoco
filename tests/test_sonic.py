from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
import bello_mujoco.evaluation.sonic as sonic_evaluation
import bello_mujoco.motion.sonic_dataset as sonic_dataset
from bello_mujoco.evaluation.sonic import (
    _REWARD_TERM_NAMES,
    _TERMINATION_NAMES,
    _build_compiled_clip_evaluator,
    _compiled_clip_result,
    _device_reference_bank,
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
    SonicSanitationConfig,
    _JointCalibrationMoments,
    _anatomy_prior_map,
    _align_smpl_root_to_bello_world,
    _json_sha256,
    build_sonic_sampling_bins,
    fit_joint_calibration,
    load_sonic_g1_retarget,
    load_sonic_dataset_index,
    load_sonic_reference_bank,
    select_sonic_reference_clip,
    select_sonic_reference_clip_in_bank,
    sonic_semantic_quality,
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
    SONIC_G1_ACTION_SCALE,
    SONIC_G1_DEFAULT_JOINT_POSITION,
    SONIC_G1_PROPRIO_SIZE,
    SONIC_ROOT_MOTION_COMMAND_SIZE,
    SONIC_SMPL_INPUT_SIZE,
    apply_frozen_mlp,
    g1_from_bello_proprio_affine,
    load_sonic_golden,
    load_sonic_release_artifact,
    sonic_fsq,
)
from bello_mujoco.training.sonic import (
    _device_reference_bank_arrays,
    _tracking_dataset_summary,
)
from bello_mujoco.training.sonic_pretraining import SonicOfflineBatchBuilder
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
    ("right_wrist_roll_joint", "right_elbow_yaw_joint"),
    ("left_wrist_roll_joint", "left_elbow_yaw_joint"),
)
BELLO_ACTION_SCALE = np.linspace(0.1, 0.34, 25, dtype=np.float32)
BELLO_QPOS_JOINT_NAMES = (
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


class _FakeEvaluationEnv:
    def __init__(self, reference_bank: SonicReferenceBank) -> None:
        self.reference_bank = reference_bank
        self.config = SimpleNamespace(max_episode_steps=5)

    def _unchecked_runtime_copy(
        self, *, reference_bank: SonicReferenceBank | None
    ) -> _FakeEvaluationEnv:
        copied = object.__new__(_FakeEvaluationEnv)
        copied.reference_bank = reference_bank
        copied.config = self.config
        return copied

    def reset(self):
        return jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0, dtype=jnp.float32)

    def step(self, state, action):
        del action
        next_state = state + 1
        value = next_state.astype(jnp.float32)
        done = next_state >= 3
        info = {name: jnp.asarray(False) for name in _TERMINATION_NAMES}
        info.update(
            {
                "done/terminal": done,
                "done/any": done,
                "tracking/root_position_error_m": value,
                "tracking/root_orientation_error_rad": value * 0.1,
                "tracking/joint_position_rms_rad": value * 0.2,
                "tracking/foot_slip_mean_mps": jnp.asarray(0.3),
                **{name: jnp.asarray(2.0) for name in _REWARD_TERM_NAMES},
            }
        )
        return next_state, value, jnp.asarray(1.0), done, info


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
        g1_joint_mean=(0.0,) * 29,
        bello_joint_mean=(0.0,) * 25,
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
    assert contract.actor_observation_size == 1670
    assert contract.critic_command_frame_size == 50
    assert contract.critic_observation_size == 1445
    assert contract.actor_layout.size == 1670
    assert dict(contract.actor_layout.components)["desired_root_xy_trajectory"] == 20
    assert contract.critic_layout.size == 1445


def test_g1_action_scales_match_the_pinned_release_actuator_groups() -> None:
    natural_frequency = 10.0 * 2.0 * 3.1415926535
    expected = {
        "left_hip_pitch_joint": 0.25 * 139.0 / (0.025101925 * natural_frequency**2),
        "waist_yaw_joint": 0.25 * 88.0 / (0.010177520 * natural_frequency**2),
        "left_ankle_pitch_joint": 0.25
        * 50.0
        / (2.0 * 0.003609725 * natural_frequency**2),
        "left_wrist_roll_joint": 0.25 * 25.0 / (0.003609725 * natural_frequency**2),
        "left_wrist_pitch_joint": 0.25 * 5.0 / (0.00425 * natural_frequency**2),
    }

    for name, value in expected.items():
        index = G1_ISAACLAB_JOINT_NAMES.index(name)
        assert SONIC_G1_ACTION_SCALE[index] == pytest.approx(value)


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
    np.testing.assert_allclose(
        fitted.g1_joint_mean, np.mean(clip["g1_joint_pos"], axis=0)
    )
    np.testing.assert_allclose(
        fitted.bello_joint_mean, np.mean(clip["qpos"][:, 7:], axis=0)
    )
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


def test_calibration_mean_shift_changes_rms_not_explained_variance() -> None:
    matrix = np.eye(2, dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64)

    rms, explained = sonic_dataset._regression_metrics(
        matrix,
        source_covariance=covariance,
        target_covariance=covariance,
        cross_covariance=covariance,
        residual_mean=np.asarray((0.3, -0.4)),
    )

    np.testing.assert_allclose(rms, (0.3, 0.4))
    np.testing.assert_allclose(explained, (1.0, 1.0))


def test_anatomy_prior_maps_forearm_twist_to_g1_wrist_roll() -> None:
    source_names = (
        "right_elbow_yaw_joint",
        "right_wrist_pitch_joint",
        "waist_yaw_joint",
    )
    target_names = (
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "waist_yaw_joint",
    )
    source_covariance = np.diag((4.0, 9.0, 16.0))
    cross_covariance = np.zeros((3, 4), dtype=np.float64)
    cross_covariance[0, 0] = 8.0
    cross_covariance[0, 1] = 20.0
    cross_covariance[1, 2] = -4.5
    cross_covariance[2, 3] = 4.0

    prior = _anatomy_prior_map(
        source_covariance,
        cross_covariance,
        source_joint_names=source_names,
        target_joint_names=target_names,
    )

    assert prior[0, 0] == 2.0
    assert np.all(prior[1] == 0.0)
    assert prior[2, 1] == -0.5
    assert prior[3, 2] == 0.25


def test_proprio_affine_maps_positions_velocities_actions_and_vector_basis() -> None:
    scales = tuple(0.8 + 0.01 * index for index in range(len(TEST_EDGE_PAIRS)))
    calibration = _calibration(scales=scales)
    matrix, bias = g1_from_bello_proprio_affine(
        calibration,
        bello_action_scale=BELLO_ACTION_SCALE,
        bello_default_joint_position=np.zeros(25, dtype=np.float32),
    )

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
    assert matrix[g1_action, bello_action] == pytest.approx(
        scale * BELLO_ACTION_SCALE[bello_index] / SONIC_G1_ACTION_SCALE[g1_index]
    )
    assert bias[g1_action] == 0.0
    assert matrix.shape == (SONIC_G1_PROPRIO_SIZE, SONIC_BELLO_PROPRIO_SIZE)


def test_proprio_affine_preserves_centered_fit_in_default_relative_coordinates() -> (
    None
):
    calibration = replace(
        _calibration(),
        g1_joint_mean=tuple(np.linspace(-0.4, 0.4, 29)),
        bello_joint_mean=tuple(np.linspace(0.3, -0.3, 25)),
    )
    bello_default = np.linspace(-0.1, 0.1, 25, dtype=np.float32)
    matrix, offset = g1_from_bello_proprio_affine(
        calibration,
        bello_action_scale=BELLO_ACTION_SCALE,
        bello_default_joint_position=bello_default,
    )
    joint_matrix = np.asarray(calibration.g1_from_bello_matrix)
    expected = (
        np.asarray(calibration.g1_joint_mean)
        - SONIC_G1_DEFAULT_JOINT_POSITION
        - joint_matrix @ (np.asarray(calibration.bello_joint_mean) - bello_default)
    )

    for history in range(10):
        np.testing.assert_allclose(
            offset[60 + history * 29 : 60 + (history + 1) * 29],
            expected,
            atol=1.0e-7,
        )
    np.testing.assert_array_equal(offset[:60], np.zeros(60, dtype=np.float32))
    np.testing.assert_array_equal(offset[350:], np.zeros(580, dtype=np.float32))


def test_coupled_calibration_routes_complete_joint_blocks_to_the_edge() -> None:
    matrix, _ = g1_from_bello_proprio_affine(
        _calibration(),
        bello_action_scale=BELLO_ACTION_SCALE,
        bello_default_joint_position=np.zeros(25, dtype=np.float32),
    )
    expected = np.asarray(_calibration().g1_from_bello_matrix)

    for history in range(10):
        np.testing.assert_array_equal(
            matrix[
                60 + history * 29 : 60 + (history + 1) * 29,
                60 + history * 25 : 60 + (history + 1) * 25,
            ],
            expected,
        )


def test_sonic_action_target_applies_randomized_default_and_release_scale() -> None:
    fake = SimpleNamespace(
        action_size=2,
        action_scale=(0.25, 0.1),
        model=SimpleNamespace(
            policy_ctrl_indices=np.asarray((0, 1), dtype=np.int32),
            default_joint_target=np.asarray((0.1, -0.2), dtype=np.float32),
            joint_angle_offset=np.asarray(((0.01, -0.02),), dtype=np.float32),
            joint_target_low=np.asarray((-0.5, -0.5), dtype=np.float32),
            joint_target_high=np.asarray((0.5, 0.5), dtype=np.float32),
            actuator_coordinate_transform=np.eye(2, dtype=np.float32),
        ),
    )
    fake._unbounded_policy_target = MethodType(
        BelloSonicEnv._unbounded_policy_target, fake
    )
    fake._action_target = MethodType(BelloSonicEnv._action_target, fake)

    target = fake._action_target(
        jnp.asarray((2.0, -4.0)),
        randomization=jnp.asarray((0.03, 0.04)),
    )

    np.testing.assert_allclose(target, np.asarray((0.5, -0.5)), atol=1.0e-6)


def test_sonic_semantic_sanitation_quarantines_treadmill_motion() -> None:
    frames = 100
    phase = np.arange(frames, dtype=np.float64) * 0.2
    qpos = np.zeros((frames, 7 + len(BELLO_JOINT_NAMES)), dtype=np.float64)
    qpos[:, 7:] = np.sin(phase)[:, None]
    contacts = np.stack(
        ((np.arange(frames) // 5) % 2 == 0, (np.arange(frames) // 5) % 2 == 1),
        axis=-1,
    )
    arrays = {
        "qpos": qpos,
        "qpos_joint_names": np.asarray(BELLO_JOINT_NAMES),
        "timestamp_seconds": np.arange(frames, dtype=np.float64) / 50.0,
        "foot_contacts": contacts,
        "body_names": np.asarray(("left_ankle_roll_link", "right_ankle_roll_link")),
        "body_lin_vel": np.zeros((frames, 2, 3), dtype=np.float64),
    }
    sequence = SimpleNamespace(
        source_id="amass",
        sequence_id="subject_treadmill_walk",
        original_sequence_id="treadmill_walk",
    )

    labels, reasons = sonic_semantic_quality(
        arrays,
        sequence,
        SonicSanitationConfig(),
    )

    assert labels == ("stationary_root", "gait_contact_cycles")
    assert any("quarantine pattern" in reason for reason in reasons)
    assert any("treadmill candidate" in reason for reason in reasons)


def test_root_motion_command_exposes_desired_travel_in_robot_frame() -> None:
    fake = SimpleNamespace()

    def aligned_reference(state, clip_index, offset):
        del state, clip_index
        return SimpleNamespace(
            qpos=jnp.asarray((0.1 * offset, 0.02 * offset, 0.0, 1.0, 0.0, 0.0, 0.0))
        )

    fake._aligned_reference = aligned_reference
    fake._root_motion_command = MethodType(BelloSonicEnv._root_motion_command, fake)
    state = SimpleNamespace(
        clip_index=jnp.asarray(0),
        motion_state=SimpleNamespace(
            data=SimpleNamespace(qpos=jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)))
        ),
    )

    command = np.asarray(fake._root_motion_command(state)).reshape((-1, 2))

    assert command.shape == (SONIC_ROOT_MOTION_COMMAND_SIZE // 2, 2)
    np.testing.assert_allclose(command[-1], np.asarray((0.9, 0.18)), atol=1.0e-6)


def test_sampling_bins_are_host_materialized_and_clip_local() -> None:
    clips, starts, widths, offsets = _build_sampling_bins(
        np.asarray((146, 96), dtype=np.int32), 50
    )

    np.testing.assert_array_equal(clips, np.asarray((0, 0, 0, 1, 1)))
    np.testing.assert_array_equal(starts, np.asarray((0, 50, 100, 0, 50)))
    np.testing.assert_array_equal(widths, np.asarray((50, 50, 1, 50, 1)))
    np.testing.assert_array_equal(offsets, np.asarray((0, 3, 5)))


def test_split_sampling_bins_cover_every_valid_start() -> None:
    clips, starts, widths, valid, offsets = build_sonic_sampling_bins(
        np.asarray((50, 100), dtype=np.int32),
        bin_size=50,
    )

    assert clips.shape == starts.shape == widths.shape == valid.shape == (3,)
    np.testing.assert_array_equal(clips, np.asarray((0, 1, 1)))
    np.testing.assert_array_equal(starts, np.asarray((0, 0, 50)))
    np.testing.assert_array_equal(widths, np.asarray((5, 50, 5)))
    np.testing.assert_array_equal(valid, np.ones(3, dtype=bool))
    np.testing.assert_array_equal(offsets, np.asarray((0, 1, 3)))


def test_tracking_dataset_summary_excludes_clip_and_fingerprint_payloads() -> None:
    index = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "dataset_fingerprint": "sha256:" + "a" * 64,
        "portable_manifest_fingerprint": "sha256:" + "b" * 64,
        "policy_fps": 50.0,
        "train_sampling": "frame_count_times_manifest_weight",
        "joint_names": list(BELLO_JOINT_NAMES),
        "qpos_joint_names": list(BELLO_QPOS_JOINT_NAMES),
        "body_names": ["pelvis_link"],
        "g1_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "robot_fingerprint": "sha256:" + "c" * 64,
        "gmr_revision": "d" * 40,
        "bello_gmr_config_sha256": "sha256:" + "e" * 64,
        "g1_gmr_revision": "f" * 40,
        "g1_gmr_config_sha256": "sha256:" + "1" * 64,
        "g1_robot_sha256": "sha256:" + "2" * 64,
        "joint_calibration": {"sample_count": 30},
        "heldout_calibration": {"sample_count": 10},
        "sanitation_config": {"maximum_fast_contact_fraction": 0.15},
        "split_files": {
            "train": {"file": "train.npz", "sampling_mass": 20.0},
            "evaluation": {
                "file": "evaluation.npz",
                "sampling_mass": 10.0,
            },
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
        "train": {
            "file": "train.npz",
            "clips": 1,
            "frames": 20,
            "sampling_mass": 20.0,
        },
        "evaluation": {
            "file": "evaluation.npz",
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


def test_offline_pretraining_batch_uses_synchronized_history_and_future() -> None:
    lengths = np.asarray((60, 60, 60), dtype=np.int32)
    bank = _reference_bank(int(np.sum(lengths)), lengths=lengths)
    frame = np.arange(len(bank.qpos), dtype=np.float32)
    bank.qpos[:, 3] = 1.0
    bank.qpos[:, 7:] = frame[:, None] * 0.001
    bank.qvel[:, 6:] = 0.05
    bank.g1_joint_pos[:] = frame[:, None] * 0.002
    bank.smpl_root_quat_wxyz[:, 0] = 1.0
    builder = SonicOfflineBatchBuilder(
        bank,
        bello_default_joint_position=np.zeros(25, dtype=np.float32),
        bello_action_scale=np.ones(25, dtype=np.float32),
        validation_clip_fraction=0.34,
        seed=7,
    )

    batch = builder.batch("train", 8)

    assert batch.observation.shape == (8, 1670)
    assert batch.next_observation.shape == (8, 1670)
    assert batch.g1_proprioception.shape == (8, 930)
    assert batch.action_target.shape == (8, 25)
    assert batch.next_action_target.shape == (8, 25)
    current_qpos_start = (
        SONIC_SMPL_INPUT_SIZE + SONIC_ROOT_MOTION_COMMAND_SIZE + 60 + 9 * 25
    )
    np.testing.assert_allclose(
        batch.action_target,
        batch.observation[:, current_qpos_start : current_qpos_start + 25] + 0.001,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        batch.next_action_target - batch.action_target,
        np.full((8, 25), 0.001, dtype=np.float32),
        atol=1.0e-7,
    )
    assert np.all(np.isfinite(batch.observation))


def test_in_bank_reference_clip_selection_preserves_full_array_shapes() -> None:
    lengths = np.asarray((50, 60), dtype=np.int32)
    bank = _reference_bank(int(lengths.sum()), lengths=lengths)

    selected = select_sonic_reference_clip_in_bank(bank, 1)

    assert selected.qpos is bank.qpos
    assert selected.qpos.shape == bank.qpos.shape
    assert selected.lengths.shape == bank.lengths.shape
    np.testing.assert_array_equal(selected.offsets, np.asarray((0, 50)))
    np.testing.assert_array_equal(selected.lengths, np.asarray((50, 60)))
    np.testing.assert_array_equal(
        selected.sampling_probabilities, np.asarray((0.0, 1.0))
    )
    assert selected.valid_clip_count == 2
    assert selected.source_ids == ("source_0", "source_1")
    absolute = BelloSonicEnv._absolute_reference_frame(
        SimpleNamespace(reference_bank=selected),
        SimpleNamespace(reference_time=jnp.asarray(0.02, dtype=jnp.float32)),
        jnp.asarray(1, dtype=jnp.int32),
        0,
    )
    assert int(absolute) == 51
    device_bank = _device_reference_bank(bank)
    device_selected = select_sonic_reference_clip_in_bank(device_bank, 1)
    assert device_selected.qpos is device_bank.qpos


def test_full_bank_sampler_selects_from_a_cdf_without_reindexing_clips() -> None:
    lengths = np.asarray((50, 60), dtype=np.int32)
    selected = select_sonic_reference_clip_in_bank(
        _reference_bank(int(lengths.sum()), lengths=lengths), 1
    )
    fake = SimpleNamespace(
        _sampling_bins=(
            selected.sampling_bin_clip_indices,
            selected.sampling_bin_starts,
            selected.sampling_bin_widths,
            selected.sampling_bin_valid,
            selected.sampling_clip_bin_offsets,
        ),
        sampling_bin_cdf=None,
        reference_bank=selected,
        sampling=SimpleNamespace(enabled=False, pre_failure_window=0),
    )

    clip_index, reference_time = BelloSonicEnv._sample_clip_start(fake, None)

    assert int(clip_index) == 1
    assert float(reference_time) == 0.0


def test_compiled_evaluation_stops_at_terminal_and_aggregates_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = _reference_bank(50, lengths=np.asarray((50,), dtype=np.int32))
    env = _FakeEvaluationEnv(bank)
    monkeypatch.setattr(
        sonic_evaluation,
        "normalize_observation",
        lambda observation, state, config: observation,
    )
    monkeypatch.setattr(
        sonic_evaluation,
        "apply_actor",
        lambda policy, observation: observation,
    )
    evaluate = _build_compiled_clip_evaluator(
        env,
        policy=None,
        observation_norm_state=None,
        metadata=SimpleNamespace(ppo=None),
    )

    result = _compiled_clip_result(evaluate(_device_reference_bank_arrays(bank)))

    assert result["steps"] == 3
    assert result["expected_steps"] == 5
    assert result["completion_fraction"] == pytest.approx(0.6)
    assert result["completed"] is False
    assert result["episode_return"] == pytest.approx(3.0)
    assert result["root_position_rms_m"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert result["root_orientation_rms_deg"] == pytest.approx(
        np.degrees(np.sqrt(0.14 / 3.0))
    )
    assert result["joint_position_rms_rad"] == pytest.approx(np.sqrt(0.56 / 3.0))
    assert result["foot_slip_mean_mps"] == pytest.approx(0.3)
    assert result["terminal"] is True
    assert result["timeout"] is False
    assert result["termination_counts"] == {
        "done/any": 1,
        "done/terminal": 1,
    }
    assert set(result["reward_terms"]) == set(_REWARD_TERM_NAMES)
    assert set(result["reward_terms"].values()) == {2.0}


def test_dataset_index_and_split_hashes_are_enforced(tmp_path: Path) -> None:
    index_path = _write_dataset_fixture(tmp_path)

    index = load_sonic_dataset_index(index_path)
    bank = load_sonic_reference_bank(index_path, split="train")

    assert index["dataset_fingerprint"].startswith("sha256:")
    assert bank.valid_clip_count == 1
    assert bank.lengths[: bank.valid_clip_count].tolist() == [50]
    split_path = tmp_path / "train.npz"
    split_path.write_bytes(split_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="train file differs"):
        load_sonic_reference_bank(index_path, split="train")


def test_reference_bank_memory_preflight_fails_before_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "train.npz"
    np.savez_compressed(archive_path, qpos=np.zeros((100, 32), dtype=np.float32))
    monkeypatch.setattr(sonic_dataset, "_available_memory_bytes", lambda: 1)

    with pytest.raises(MemoryError, match="available host memory"):
        sonic_dataset._require_reference_bank_memory(archive_path, split="train")


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
    object.__setattr__(decoder, "root_motion_projection_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "proprioception_projection_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "first_bias", jnp.ones((1,)))
    object.__setattr__(decoder, "middle_weights", (jnp.ones((1, 1)), jnp.ones((1, 1))))
    object.__setattr__(decoder, "middle_biases", (jnp.ones((1,)), jnp.ones((1,))))
    object.__setattr__(decoder, "output_weight", jnp.ones((1, 1)))
    object.__setattr__(decoder, "output_bias", jnp.ones((1,)))
    object.__setattr__(decoder, "trainable_middle_layers", (False, True))
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
    object.__setattr__(policy, "train_log_std", False)
    trainable, frozen = partition_policy_parameters(policy)

    assert all(value is None for value in trainable.actor.smpl_encoder.weights)
    assert trainable.actor.decoder.token_projection_weight is None
    assert trainable.actor.decoder.middle_weights[0] is None
    assert trainable.actor.decoder.middle_weights[1] is not None
    assert trainable.actor.decoder.root_motion_projection_weight is not None
    assert trainable.actor.decoder.proprioception_projection_weight is not None
    assert trainable.actor.decoder.first_bias is not None
    assert trainable.actor.decoder.output_weight is not None
    assert all(value is not None for value in frozen.actor.smpl_encoder.weights)
    assert trainable.critic.weights[0] is not None
    assert trainable.log_std is None
    assert frozen.log_std is not None


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
    bins = build_sonic_sampling_bins(
        lengths,
        bin_size=50,
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
        qpos_joint_names=BELLO_QPOS_JOINT_NAMES,
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
    bank = _reference_bank(frames, lengths=np.asarray((frames,), dtype=np.int32))
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
    split_files = {}
    for split in ("train", "evaluation"):
        path = root / f"{split}.npz"
        np.savez_compressed(path, **packed)
        split_files[split] = {
            "file": path.name,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "clip_count": 1,
            "frames": frames,
            "sampling_mass": float(frames),
        }
    clips = [
        {
            "source_id": f"source_{index}",
            "sequence_id": f"sequence_{index}",
            "split": split,
            "source_kind": "amass",
            "subject_id": f"subject_{index}",
            "sampling_weight": 1.0,
            "offset": 0,
            "frames": frames,
            "duration_seconds": 0.98,
            "source_sha256": "sha256:" + "2" * 64,
            "synchronized_sha256": "sha256:" + "3" * 64,
            "semantic_labels": ["traveling_root", "gait_contact_cycles"],
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
    sanitation = {
        **SonicSanitationConfig().__dict__,
        "quarantine_name_patterns": list(
            SonicSanitationConfig().quarantine_name_patterns
        ),
    }
    heldout_calibration = {
        "sample_count": frames,
        "g1_explained_variance": [1.0] * 29,
        "g1_rms_residual_radians": [0.0] * 29,
        "bello_explained_variance": [1.0] * 25,
        "bello_rms_residual_radians": [0.0] * 25,
    }
    payload = {
        "schema_version": SONIC_DATASET_SCHEMA_VERSION,
        "portable_manifest": portable,
        "accepted_clip_hashes": [],
        "quality": {},
        "calibration_config": {},
        "joint_calibration": _calibration().to_document(),
        "heldout_calibration": heldout_calibration,
        "sanitation_config": sanitation,
        "smpl_future_offsets": SONIC_SMPL_FUTURE_OFFSETS,
        "critic_future_offsets": SONIC_CRITIC_FUTURE_OFFSETS,
        "train_sampling": SONIC_TRAIN_SAMPLING,
        "split_files": split_files,
        "joint_names": BELLO_JOINT_NAMES,
        "qpos_joint_names": BELLO_QPOS_JOINT_NAMES,
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
        "split_files": split_files,
        "joint_names": list(BELLO_JOINT_NAMES),
        "qpos_joint_names": list(BELLO_QPOS_JOINT_NAMES),
        "body_names": list(SONIC_TRACKED_BODY_NAMES),
        "g1_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "g1_wrist_joint_names": list(G1_WRIST_JOINT_NAMES),
        **provenance,
        "joint_calibration": _calibration().to_document(),
        "heldout_calibration": heldout_calibration,
        "sanitation_config": sanitation,
        "clips": clips,
        "fingerprint_payload": payload,
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path
