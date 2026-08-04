from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

from bello_mujoco.envs.twist import (
    TWIST_HISTORY_FRAMES,
    TwistEnvConfig,
    TwistObservationContract,
    teacher_observation_from_student_critic,
)
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ActionLayout
from bello_mujoco.envs.motion_tracking import MotionTrackingEnvConfig
from bello_mujoco.motion.clip import (
    FIXED_TRACKING_JOINT_NAMES,
    GMR_BELLO_JOINT_NAMES,
)
from bello_mujoco.motion.twist_dataset import (
    AssignedSequence,
    QualityConfig,
    SourceManifest,
    SourceSequence,
    _arrays_fingerprint,
    assign_splits,
    build_packed_dataset,
    clip_quality_reasons,
    load_packed_reference_bank,
    load_source_manifest,
    portable_manifest,
    portable_manifest_fingerprint,
    sample_packed_start_frames,
    scan_amass_manifest,
    validate_source_manifest,
    validate_split_isolation,
)
from bello_mujoco.policy import ActorCriticConfig, init_actor_critic
from bello_mujoco.sim import WarpConfig
from bello_mujoco.training.checkpoint import (
    ObservationNormMetadata,
    TwistDatasetContract,
    TwistStudentCheckpointMetadata,
    TwistTeacherCheckpointMetadata,
    ValueNormMetadata,
    _validate_metadata_layouts,
    resumed_checkpoint_metadata,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    cosine_distillation_weight,
    gaussian_policy_kl_student_teacher,
)

JOINT_NAMES = tuple(
    name for name in GMR_BELLO_JOINT_NAMES if name not in FIXED_TRACKING_JOINT_NAMES
)


def source_sequence(
    index: int,
    *,
    subject: str | None = None,
    checksum: str | None = None,
    requested_split: str = "auto",
) -> SourceSequence:
    digest = checksum or hashlib.sha256(f"sequence-{index}".encode()).hexdigest()
    return SourceSequence(
        source_kind="amass",
        source_id="amass",
        sequence_id=f"sequence_{index}",
        dataset_release="AMASS-test",
        license_id="AMASS-terms",
        local_path=Path(f"/private/amass/sequence_{index}.npz"),
        subject_id=subject or f"subject_{index}",
        session_id="session",
        original_sequence_id=f"original_{index}",
        pose_representation="smpl-x",
        frame_rate=60.0,
        requested_split=requested_split,
        source_sha256=digest,
        subdataset="CMU",
    )


def test_manifest_split_is_grouped_deterministic_and_portable() -> None:
    sequences = tuple(
        source_sequence(index, subject=f"subject_{index // 2}") for index in range(20)
    )
    manifest = SourceManifest(1, 17, sequences)
    validate_source_manifest(manifest, verify_files=False)

    first = assign_splits(manifest)
    second = assign_splits(manifest)

    assert first == second
    by_subject = {}
    for assigned in first:
        by_subject.setdefault(assigned.sequence.subject_id, set()).add(assigned.split)
    assert all(len(splits) == 1 for splits in by_subject.values())
    assert sum(item.split == "evaluation" for item in first) == 2
    document = portable_manifest(first)
    assert all("local_path" not in item for item in document["sequences"])
    assert portable_manifest_fingerprint(first).startswith("sha256:")


def test_manifest_duplicate_and_split_leakage_are_rejected() -> None:
    digest = hashlib.sha256(b"duplicate").hexdigest()
    manifest = SourceManifest(
        1,
        0,
        (
            source_sequence(0, checksum=digest),
            source_sequence(1, checksum=digest),
        ),
    )
    with pytest.raises(ValueError, match="duplicate source content"):
        validate_source_manifest(manifest, verify_files=False)

    first = source_sequence(2)
    second = replace(first, sequence_id="other")
    with pytest.raises(ValueError, match="split leakage"):
        validate_split_isolation(
            (
                AssignedSequence(first, "train"),
                AssignedSequence(second, "evaluation"),
            )
        )


def test_amass_scanner_generates_reviewable_stageii_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "amass"
    for index, subject in enumerate(("Subject 01", "Subject 02")):
        source_dir = source_root / "CMU" / subject
        source_dir.mkdir(parents=True)
        frames = 4 + index
        np.savez(
            source_dir / f"walk {index}_stageii.npz",
            surface_model_type=np.asarray("smplx"),
            gender=np.asarray("neutral"),
            mocap_frame_rate=np.asarray(60.0),
            trans=np.full((frames, 3), index, dtype=np.float32),
            root_orient=np.full((frames, 3), index * 0.1, dtype=np.float32),
            pose_body=np.full((frames, 63), index * 0.2, dtype=np.float32),
        )
        np.savez(source_dir / "neutral_stagei.npz", betas=np.zeros(16))
    output_path = tmp_path / "amass_manifest.yaml"

    generated = scan_amass_manifest(
        source_root,
        output_path,
        dataset_release="AMASS-SMPL-X-G-test",
        license_id="AMASS-test-terms",
        split_seed=17,
    )
    loaded = load_source_manifest(output_path, verify_files=False)

    assert generated == loaded
    assert len(loaded.sequences) == 2
    assert {item.sequence_id for item in loaded.sequences} == {
        "CMU__Subject_01__walk_0_stageii",
        "CMU__Subject_02__walk_1_stageii",
    }
    assert all(item.pose_representation == "smpl-x" for item in loaded.sequences)
    assert all(item.subdataset == "CMU" for item in loaded.sequences)
    assert all(
        item.near_duplicate_fingerprint.startswith("amass-motion-v1:")
        for item in loaded.sequences
        if item.near_duplicate_fingerprint is not None
    )
    assert "neutral_stagei.npz" not in output_path.read_text(encoding="utf-8")


def test_twist_observation_contract_exact_dimensions_and_isolation() -> None:
    contract = TwistObservationContract(
        joint_names=JOINT_NAMES,
        randomization_size=len(JOINT_NAMES) + 2,
    )

    assert len(JOINT_NAMES) == 25
    assert contract.reference_frame_size == 63
    assert contract.proprioception_frame_size == 80
    assert contract.teacher_observation_size == 1340
    assert contract.student_observation_size == 943
    assert contract.critic_observation_size == 1400
    assert TWIST_HISTORY_FRAMES == 11
    assert contract.actor_layout("student").components == (
        ("current_reference", 63),
        ("proprioception_history", 880),
    )
    critic = jnp.arange(contract.critic_observation_size)
    teacher = teacher_observation_from_student_critic(critic, contract)
    assert teacher.shape == (contract.teacher_observation_size,)
    assert not any(
        name == "future_reference"
        for name, _ in contract.actor_layout("student").components
    )


@pytest.mark.parametrize(
    ("stage", "observation_size"),
    (("teacher", 1340), ("student", 943)),
)
def test_twist_policy_encoder_and_backbone_shapes(
    stage: str, observation_size: int
) -> None:
    config = ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        architecture=f"twist_{stage}",
        layer_norm_final_hidden=True,
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        observation_size,
        len(JOINT_NAMES),
        config,
        critic_obs_size=1400,
    )

    mean, log_std, value = eqx.filter_jit(policy)(
        jnp.zeros(observation_size), jnp.zeros(1400)
    )

    assert mean.shape == (25,)
    assert log_std.shape == (25,)
    assert value.shape == ()
    assert policy.actor.backbone.hidden[-1].out_features == 128


def test_corrected_cosine_schedule_has_both_endpoints() -> None:
    assert float(cosine_distillation_weight(0)) == pytest.approx(0.1)
    assert float(cosine_distillation_weight(30_000)) == pytest.approx(0.055)
    assert float(cosine_distillation_weight(60_000)) == pytest.approx(0.01)
    assert float(cosine_distillation_weight(90_000)) == pytest.approx(0.01)


def test_distillation_kl_direction_and_teacher_is_frozen() -> None:
    student_config = ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        architecture="twist_student",
        layer_norm_final_hidden=True,
    )
    teacher_config = replace(student_config, architecture="twist_teacher")
    student = init_actor_critic(
        jax.random.PRNGKey(1), 943, 25, student_config, critic_obs_size=1400
    )
    teacher = init_actor_critic(
        jax.random.PRNGKey(2), 1340, 25, teacher_config, critic_obs_size=1400
    )
    student = eqx.tree_at(lambda policy: policy.log_std, student, jnp.full((25,), -0.5))
    teacher = eqx.tree_at(lambda policy: policy.log_std, teacher, jnp.full((25,), 0.25))
    student_obs = jnp.zeros((2, 943))
    teacher_obs = jnp.zeros((2, 1340))

    forward = gaussian_policy_kl_student_teacher(
        student, teacher, student_obs, teacher_obs
    )
    reverse = gaussian_policy_kl_student_teacher(
        teacher, student, teacher_obs, student_obs
    )
    teacher_gradient = jax.grad(
        lambda teacher_log_std: gaussian_policy_kl_student_teacher(
            student,
            eqx.tree_at(lambda policy: policy.log_std, teacher, teacher_log_std),
            student_obs,
            teacher_obs,
        ).sum()
    )(teacher.log_std)

    assert np.all(np.asarray(forward) >= 0.0)
    assert not np.allclose(forward, reverse)
    np.testing.assert_array_equal(teacher_gradient, np.zeros(25))


def test_qc_rejects_short_and_nonfinite_clips_and_hashes_are_reproducible() -> None:
    frames = 100
    arrays = {
        "timestamp_seconds": np.arange(frames, dtype=np.float64) / 50.0,
        "source_timestamp_seconds": np.arange(frames, dtype=np.float64) / 50.0,
        "qpos": np.tile(
            np.concatenate(([0.0, 0.0, 0.9], [1.0, 0.0, 0.0, 0.0], np.zeros(25))),
            (frames, 1),
        ).astype(np.float32),
        "qvel": np.zeros((frames, 31), dtype=np.float32),
        "body_pos": np.zeros((frames, 14, 3), dtype=np.float32),
        "body_quat_wxyz": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 14, 1)),
        "body_lin_vel": np.zeros((frames, 14, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((frames, 14, 3), dtype=np.float32),
        "foot_contacts": np.zeros((frames, 2), dtype=np.float32),
    }
    quality = QualityConfig()

    assert clip_quality_reasons(arrays, quality) == []
    assert _arrays_fingerprint(arrays) == _arrays_fingerprint(
        {name: value.copy() for name, value in arrays.items()}
    )
    short = {name: value[:95] for name, value in arrays.items()}
    assert any(
        "future horizon" in reason for reason in clip_quality_reasons(short, quality)
    )
    invalid = {name: value.copy() for name, value in arrays.items()}
    invalid["qpos"][0, 0] = np.nan
    assert clip_quality_reasons(invalid, quality) == ["qpos contains non-finite values"]


def test_packed_sampler_is_jittable_and_never_crosses_clip_end() -> None:
    lengths = jnp.asarray([100, 180, 300])
    probabilities = jnp.asarray([0.2, 0.3, 0.5])
    sample = jax.jit(
        lambda key: sample_packed_start_frames(
            key,
            lengths,
            probabilities,
            batch_size=128,
        )
    )

    clip, start = sample(jax.random.PRNGKey(4))

    assert clip.shape == start.shape == (128,)
    assert np.all(np.asarray(start + 95) < np.asarray(lengths[clip]))


def test_reference_bank_concatenates_unequal_clips_without_padding(
    tmp_path: Path,
) -> None:
    lengths = (100, 137)
    total_frames = sum(lengths)
    qpos = np.arange(total_frames * 4, dtype=np.float32).reshape(total_frames, 4)
    arrays = {
        "qpos": qpos,
        "qvel": np.zeros((total_frames, 3), dtype=np.float32),
        "body_pos": np.zeros((total_frames, 2, 3), dtype=np.float32),
        "body_quat_wxyz": np.zeros((total_frames, 2, 4), dtype=np.float32),
        "body_lin_vel": np.zeros((total_frames, 2, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((total_frames, 2, 3), dtype=np.float32),
        "foot_contacts": np.zeros((total_frames, 2), dtype=np.float32),
    }
    np.savez(tmp_path / "shard_00000.npz", **arrays)
    clips = [
        {
            "source_id": "amass",
            "sequence_id": f"clip_{index}",
            "split": "train",
            "shard": "shard_00000.npz",
            "offset": sum(lengths[:index]),
            "frames": frames,
            "sampling_weight": 1.0,
        }
        for index, frames in enumerate(lengths)
    ]
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"clips": clips}), encoding="utf-8")

    bank = load_packed_reference_bank(index_path, split="train")

    assert bank.qpos.shape == (total_frames, 4)
    np.testing.assert_array_equal(bank.offsets, (0, lengths[0]))
    np.testing.assert_array_equal(bank.lengths, lengths)
    np.testing.assert_array_equal(bank.qpos, qpos)


def test_packed_build_fingerprint_and_heldout_manifest_are_reproducible(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    robot_path = (
        repository
        / "deps/bello-stack/descriptions/bello/full_body/urdf/bello_full_body.urdf"
    )
    gmr_root = repository / "deps/GMR"
    config_path = repository / "configs/twist/gmr_smplx_to_bello.json"
    revision = subprocess.run(
        ["git", "-C", str(gmr_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    robot_fingerprint = "sha256:" + hashlib.sha256(robot_path.read_bytes()).hexdigest()
    config_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest()
    retargeted = tmp_path / "retargeted"
    retargeted.mkdir()
    sequences = []
    body_names = (
        "pelvis_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "l_shoulder_roll_link",
        "l_elbow_link",
        "l_wrist_link",
        "r_shoulder_roll_link",
        "r_elbow_link",
        "r_wrist_link",
    )
    for index, split in enumerate(("train", "evaluation")):
        source = tmp_path / f"source_{index}.bvh"
        source.write_text(f"synthetic fixture {index}", encoding="utf-8")
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        sequence_id = f"fixture_{index}"
        sequences.append(
            {
                "source_kind": "in_house",
                "source_id": "fixture",
                "sequence_id": sequence_id,
                "dataset_release": "test-v1",
                "license_id": "repository-test-fixture",
                "local_path": str(source),
                "subject_id": f"subject_{index}",
                "session_id": "session",
                "original_sequence_id": sequence_id,
                "pose_representation": "bvh",
                "frame_rate": 50.0,
                "requested_split": split,
                "sampling_weight": 1.0,
                "source_sha256": source_digest,
                "subdataset": None,
                "near_duplicate_fingerprint": None,
                "metadata": {
                    "owner": "repository",
                    "ownership_basis": "generated test fixture",
                    "bvh_skeleton": "synthetic-v1",
                },
            }
        )
        frames = 120
        qpos = np.zeros((frames, 32), dtype=np.float32)
        qpos[:, 2] = 0.9
        qpos[:, 3] = 1.0
        body_quat = np.zeros((frames, len(body_names), 4), dtype=np.float32)
        body_quat[..., 0] = 1.0
        np.savez_compressed(
            retargeted / f"fixture__{sequence_id}.npz",
            timestamp_seconds=np.arange(frames, dtype=np.float64) / 50.0,
            source_timestamp_seconds=np.arange(frames, dtype=np.float64) / 50.0,
            source_sha256=np.asarray(source_digest),
            qpos=qpos,
            qvel=np.zeros((frames, 31), dtype=np.float32),
            body_pos=np.zeros((frames, len(body_names), 3), dtype=np.float32),
            body_quat_wxyz=body_quat,
            body_lin_vel=np.zeros((frames, len(body_names), 3), dtype=np.float32),
            body_ang_vel=np.zeros((frames, len(body_names), 3), dtype=np.float32),
            foot_contacts=np.zeros((frames, 2), dtype=np.float32),
            joint_names=np.asarray(JOINT_NAMES),
            body_names=np.asarray(body_names),
            joint_lower=np.full((25,), -2.0, dtype=np.float32),
            joint_upper=np.full((25,), 2.0, dtype=np.float32),
            gmr_revision=np.asarray(revision),
            gmr_config_sha256=np.asarray(config_fingerprint),
            robot_fingerprint=np.asarray(robot_fingerprint),
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "split_seed": 7, "sequences": sequences}),
        encoding="utf-8",
    )

    first = build_packed_dataset(
        manifest_path,
        retargeted,
        tmp_path / "packed_first",
        body_model_root=None,
        robot_model_path=robot_path,
        gmr_root=gmr_root,
    )
    second = build_packed_dataset(
        manifest_path,
        retargeted,
        tmp_path / "packed_second",
        body_model_root=None,
        robot_model_path=robot_path,
        gmr_root=gmr_root,
    )

    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert {clip["split"] for clip in first["clips"]} == {
        "train",
        "evaluation",
    }
    assert (tmp_path / "packed_first/heldout_manifest.json").read_bytes() == (
        tmp_path / "packed_second/heldout_manifest.json"
    ).read_bytes()


def test_typed_twist_checkpoints_validate_and_reject_cross_stage_resume() -> None:
    contract = TwistObservationContract(
        joint_names=JOINT_NAMES,
        randomization_size=27,
    )
    fingerprint = "sha256:" + "1" * 64
    dataset = TwistDatasetContract(
        index_path=Path("datasets/twist/index.json"),
        schema_version=1,
        dataset_fingerprint=fingerprint,
        robot_fingerprint=fingerprint,
        gmr_revision="2" * 40,
        joint_names=JOINT_NAMES,
        body_names=contract.key_body_names,
    )
    action_layout = ActionLayout(JOINT_NAMES)

    def ppo(stage: str) -> PPOConfig:
        return PPOConfig(
            total_timesteps=4,
            num_envs=2,
            num_steps=2,
            update_epochs=1,
            num_minibatches=1,
            learning_rate=2.0e-4,
            min_learning_rate=1.0e-5,
            max_learning_rate=2.0e-4,
            policy=ActorCriticConfig(
                hidden_dims=(512, 512, 256, 128),
                activation="swish",
                obs_stack_size=1,
                architecture=f"twist_{stage}",
                layer_norm_final_hidden=True,
            ),
        )

    def normalization(size: int) -> ObservationNormMetadata:
        return ObservationNormMetadata(
            mean=(0.0,) * size,
            var=(1.0,) * size,
            count=0.0,
        )

    common = {
        "asset": AssetPrepConfig(),
        "warp": WarpConfig(),
        "critic_obs_size": contract.critic_observation_size,
        "action_size": len(JOINT_NAMES),
        "critic_observation_layout": contract.critic_layout,
        "action_layout": action_layout,
        "motion_env": MotionTrackingEnvConfig(),
        "twist_env": TwistEnvConfig(),
        "observation_contract": contract,
        "dataset": dataset,
        "robot_fingerprint": fingerprint,
        "source_revision": "3" * 40,
        "value_norm": ValueNormMetadata(0.0, 1.0, 0.0),
        "critic_observation_norm": normalization(contract.critic_observation_size),
    }
    teacher = TwistTeacherCheckpointMetadata(
        **common,
        ppo=ppo("teacher"),
        obs_size=contract.teacher_observation_size,
        actor_observation_layout=contract.actor_layout("teacher"),
        observation_norm=normalization(contract.teacher_observation_size),
    )
    student = TwistStudentCheckpointMetadata(
        **common,
        ppo=ppo("student"),
        obs_size=contract.student_observation_size,
        actor_observation_layout=contract.actor_layout("student"),
        observation_norm=normalization(contract.student_observation_size),
        teacher_fingerprint="sha256:" + "4" * 64,
    )

    _validate_metadata_layouts(teacher)
    _validate_metadata_layouts(student)
    with pytest.raises(ValueError, match="resume checkpoint contract differs"):
        resumed_checkpoint_metadata(teacher, student, saved_update=0)
