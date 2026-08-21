from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import equinox as eqx
import mujoco
import numpy as np
import pytest

from bello_mujoco.envs.twist import (
    BelloTwistEnv,
    TWIST_HISTORY_FRAMES,
    TwistEnvConfig,
    TwistObservationContract,
    TwistSamplingConfig,
    TwistSimulationConfig,
    _batched_contact_max,
    _key_body_pose_error_exceeds,
    _validate_reference_bank_clip_bounds,
    teacher_observation_from_student_critic,
)
from bello_mujoco.assets_pipeline.prepare import AssetPrepConfig
from bello_mujoco.contracts import ActionLayout
from bello_mujoco.configuration import twist_training_profile
from bello_mujoco.motion.clip import (
    FIXED_TRACKING_JOINT_NAMES,
    GMR_BELLO_JOINT_NAMES,
)
from bello_mujoco.motion.twist_dataset import (
    AssignedSequence,
    PackedReferenceBank,
    QualityConfig,
    SourceCurationConfig,
    SourceExclusionRule,
    SourceManifest,
    SourceSequence,
    TWIST_DATASET_SCHEMA_VERSION,
    TWIST_TRAIN_SAMPLING,
    _align_floor_and_heading,
    _arrays_fingerprint,
    _knee_pole_quality_reasons,
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
from bello_mujoco.motion.twist_retarget import batch_retarget_manifest
from bello_mujoco.policy import ActorCriticConfig, init_actor_critic
from bello_mujoco.sim import WarpConfig, logical_actuator_joint_names
from bello_mujoco.training.checkpoint import (
    CheckpointConfig,
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
    physics_diagnostics,
)
from bello_mujoco.training.twist import (
    _bind_reference_bank,
    _device_reference_bank_arrays,
    upload_twist_checkpoint,
)

JOINT_NAMES = tuple(
    name for name in GMR_BELLO_JOINT_NAMES if name not in FIXED_TRACKING_JOINT_NAMES
)


def test_batched_contact_reduction_is_linear_in_flattened_contacts() -> None:
    world_ids = jnp.asarray([0, 1, 0, 2, 1], dtype=jnp.int32)
    values = jnp.asarray([[1.0, 8.0], [7.0, 2.0], [4.0, 3.0], [5.0, 6.0], [2.0, 9.0]])

    reduced = jax.jit(lambda: _batched_contact_max(world_ids, values, 4))()

    np.testing.assert_array_equal(
        reduced,
        np.asarray([[4.0, 8.0], [7.0, 9.0], [5.0, 6.0], [0.0, 0.0]]),
    )


def test_released_teacher_training_defaults_are_locked() -> None:
    config = twist_training_profile("teacher")

    assert config.config.num_updates == 30_002
    assert config.twist_env.action_clip == 5.0
    assert config.sampling == TwistSamplingConfig()
    assert config.asset_config.target_total_mass_kg is None


def test_twist_checkpoint_upload_honors_checkpoint_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    monkeypatch.setattr(
        "bello_mujoco.training.twist.track_checkpoint_artifact",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    checkpoint = tmp_path / "run" / "final"
    root = checkpoint.parent
    run = object()

    upload_twist_checkpoint(
        run,
        checkpoint,
        root,
        CheckpointConfig(
            upload_wandb_artifact=True,
            wandb_artifact_name="teacher",
            wandb_artifact_type="checkpoint",
        ),
    )
    upload_twist_checkpoint(
        run,
        checkpoint,
        root,
        CheckpointConfig(upload_wandb_artifact=False),
    )
    upload_twist_checkpoint(
        None,
        checkpoint,
        root,
        CheckpointConfig(upload_wandb_artifact=True),
    )

    assert calls == [
        (
            (run, checkpoint),
            {"name": "teacher", "artifact_type": "checkpoint"},
        )
    ]


def test_released_pose_termination_is_root_local_and_per_body() -> None:
    identity = jnp.asarray((1.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    reference_root = jnp.asarray((0.0, 0.0, 1.0), dtype=jnp.float32)
    reference = (
        jnp.tile(reference_root, (9, 1))
        .at[:, 0]
        .set(jnp.arange(9, dtype=jnp.float32) * 0.1)
    )
    translated_root = reference_root + jnp.asarray((3.0, -2.0, 0.5))
    translated = reference + (translated_root - reference_root)

    assert not bool(
        _key_body_pose_error_exceeds(
            translated,
            translated_root,
            identity,
            reference,
            reference_root,
            identity,
            0.7,
        )
    )
    coherent_error = reference + jnp.asarray((0.3, 0.0, 0.0))
    assert not bool(
        _key_body_pose_error_exceeds(
            coherent_error,
            reference_root,
            identity,
            reference,
            reference_root,
            identity,
            0.7,
        )
    )
    one_bad_body = reference.at[4, 0].add(0.71)
    assert bool(
        _key_body_pose_error_exceeds(
            one_bad_body,
            reference_root,
            identity,
            reference,
            reference_root,
            identity,
            0.7,
        )
    )


def test_twist_action_diagnostics_accept_rollout_and_environment_axes() -> None:
    fake_env = SimpleNamespace(
        action_size=2,
        model=SimpleNamespace(
            policy_ctrl_indices=np.asarray([0, 1]),
            actuator_coordinate_transform=np.eye(2, dtype=np.float32),
            default_joint_target=np.zeros(2, dtype=np.float32),
            joint_target_low=np.asarray([-1.0, -1.0], dtype=np.float32),
            joint_target_high=np.asarray([1.0, 1.0], dtype=np.float32),
        ),
    )
    fake_env._action_target = MethodType(BelloTwistEnv._action_target, fake_env)
    actions = jnp.zeros((2, 3, 2), dtype=jnp.float32).at[1, 2, 0].set(3.0)

    metrics = BelloTwistEnv.ppo_action_diagnostics(fake_env, actions)

    assert metrics["target/outside_joint_limit_fraction"] == pytest.approx(1.0 / 12.0)


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
    invalid_dir = source_root / "EKUT" / "265"
    invalid_dir.mkdir(parents=True)
    np.savez(
        invalid_dir / "empty_stageii.npz",
        surface_model_type=np.asarray("smplx"),
        mocap_frame_rate=np.asarray(60.0),
        trans=np.zeros((1, 3), dtype=np.float32),
        root_orient=np.zeros((1, 3), dtype=np.float32),
        pose_body=np.zeros((1, 63), dtype=np.float32),
    )
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
    report = json.loads(
        output_path.with_suffix(".scan_report.json").read_text(encoding="utf-8")
    )
    assert report["discovered"] == 3
    assert report["accepted"] == 2
    assert report["rejected"][0]["relative_path"] == "EKUT/265/empty_stageii.npz"


def test_amass_scanner_applies_flat_ground_curation_before_inspection(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "amass"

    def write_motion(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            surface_model_type=np.asarray("smplx"),
            gender=np.asarray("neutral"),
            mocap_frame_rate=np.asarray(60.0),
            trans=np.zeros((4, 3), dtype=np.float32),
            root_orient=np.zeros((4, 3), dtype=np.float32),
            pose_body=np.zeros((4, 63), dtype=np.float32),
        )

    write_motion(source_root / "CMU/01/walk_stageii.npz")
    write_motion(source_root / "KIT/167/upstairs01_stageii.npz")
    generated = source_root / "twist_work_v2/retargeted/copied_stageii.npz"
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"excluded before archive inspection")
    review_fixture = source_root / "twist_validation_20/normalized/copied_stageii.npz"
    review_fixture.parent.mkdir(parents=True)
    review_fixture.write_bytes(b"also excluded before archive inspection")
    output_path = tmp_path / "curated.yaml"
    curation = SourceCurationConfig(
        exclusion_rules=(
            SourceExclusionRule(
                "relative_path",
                r"^twist_[^/]*(?:/|$)",
                "generated TWIST pipeline artifact",
            ),
            SourceExclusionRule(
                "sequence_id",
                r"(?i)(?:upstairs|downstairs|stair(?:s|case)?|slope_(?:up|down)|ladder)",
                "motion requires non-flat terrain",
            ),
        )
    )

    manifest = scan_amass_manifest(
        source_root,
        output_path,
        dataset_release="test",
        license_id="test",
        split_seed=17,
        curation=curation,
    )
    report = json.loads(
        output_path.with_suffix(".scan_report.json").read_text(encoding="utf-8")
    )

    assert tuple(item.sequence_id for item in manifest.sequences) == (
        "CMU__01__walk_stageii",
    )
    assert report["accepted"] == 1
    assert {item["reason"] for item in report["excluded"]} == {
        "generated TWIST pipeline artifact",
        "motion requires non-flat terrain",
    }


def test_batch_retarget_is_atomic_and_resumes_matching_outputs(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "source.npz"
    source.write_bytes(b"licensed source fixture")
    sequence = replace(
        source_sequence(0),
        local_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "schema_version": 1,
        "split_seed": 17,
        "sequences": [
            {
                **sequence.__dict__,
                "local_path": str(sequence.local_path),
            }
        ],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    body_model_root = tmp_path / "body_models"
    for gender in ("NEUTRAL", "FEMALE", "MALE"):
        path = body_model_root / "smplx" / f"SMPLX_{gender}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test body model")
    robot_xml = tmp_path / "bello.xml"
    robot_xml.write_text("<mujoco/>", encoding="utf-8")
    commands = []

    def fake_runner(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        if command[1].endswith("twist_normalize_smpl.py"):
            output.write_bytes(b"normalized")
        else:
            config = repository / (
                "deps/GMR/general_motion_retargeting/ik_configs/smplx_to_bello.json"
            )
            robot = repository / (
                "deps/bello-stack/descriptions/bello/full_body/urdf/"
                "bello_full_body.urdf"
            )
            revision = subprocess.run(
                ["git", "-C", str(repository / "deps/GMR"), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            np.savez(
                output,
                source_sha256=np.asarray(sequence.source_sha256),
                retarget_format_version=np.asarray(9, dtype=np.int32),
                gmr_revision=np.asarray(revision),
                gmr_config_sha256=np.asarray(
                    hashlib.sha256(config.read_bytes()).hexdigest()
                ),
                robot_fingerprint=np.asarray(
                    "sha256:" + hashlib.sha256(robot.read_bytes()).hexdigest()
                ),
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    options = {
        "body_model_root": body_model_root,
        "robot_xml": robot_xml,
        "output_root": tmp_path / "retargeted",
        "normalized_root": tmp_path / "normalized",
        "gmr_python": Path(sys.executable),
    }
    first = batch_retarget_manifest(
        manifest_path,
        **options,
        command_runner=fake_runner,
        gmr_environment={},
    )
    second = batch_retarget_manifest(
        manifest_path,
        **options,
        command_runner=lambda *args, **kwargs: pytest.fail(
            "matching output should have been skipped"
        ),
        gmr_environment={},
    )

    assert (first.completed, first.skipped, first.failed) == (1, 0, 0)
    assert (second.completed, second.skipped, second.failed) == (0, 1, 0)
    assert len(commands) == 2
    assert not tuple((tmp_path / "retargeted").glob("*.partial.npz"))
    assert not tuple((tmp_path / "normalized").glob("*.npz"))
    assert (tmp_path / "retargeted/batch_retarget_report.json").is_file()


def test_gmr_retarget_metadata_separates_actuator_and_qpos_order(
    full_body_asset,
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))

    names = tool._logical_actuator_joint_names(model)
    qpos_names = tool._qpos_ordered_joint_names(model, names)
    ranges = tool._qpos_ordered_joint_ranges(model, names)
    qpos_joint_ids = sorted(
        range(1, model.njnt), key=lambda joint_id: int(model.jnt_qposadr[joint_id])
    )

    assert names == logical_actuator_joint_names(model)
    assert qpos_names == tuple(
        model.joint(joint_id).name for joint_id in qpos_joint_ids
    )
    assert names != qpos_names
    np.testing.assert_array_equal(ranges, model.jnt_range[qpos_joint_ids])


def test_gmr_offline_temporal_limiter_respects_rate_and_native_limits() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    config = tool.OfflineSolverConfig(
        passes_per_frame=5,
        max_joint_velocity_radians_per_second=3.0 * np.pi,
        max_joint_acceleration_radians_per_second_squared=200.0,
        knee_pole_reference_flexion_radians=np.deg2rad(50.0),
        knee_target_minimum_flexion_radians=np.deg2rad(5.0),
        knee_pole_sample_minimum_flexion_radians=np.deg2rad(20.0),
        knee_pole_sample_minimum_distance_meters=0.04,
        knee_pole_smoothing_frames=5,
    )
    previous = np.asarray([0.0, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    previous_velocity = np.zeros(2)
    ranges = np.asarray([[-0.25, 0.25], [-0.1, 0.1]])
    trajectory = [previous.copy()]

    for frame in range(30):
        candidate = previous.copy()
        candidate[0] = frame + 1.0
        candidate[7:] = (1.0, -1.0)
        previous, previous_velocity = tool._limit_scalar_joint_candidate(
            candidate,
            previous,
            previous_velocity,
            fps=50.0,
            joint_ranges=ranges,
            solver_config=config,
        )
        assert previous[0] == frame + 1.0
        trajectory.append(previous.copy())

    joints = np.asarray(trajectory)[:, 7:]
    velocity = np.diff(joints, axis=0) * 50.0
    acceleration = np.diff(velocity, axis=0) * 50.0
    assert np.max(np.abs(velocity)) <= 3.0 * np.pi + 1.0e-10
    assert np.max(np.abs(acceleration)) <= 200.0 + 1.0e-10
    assert np.all(joints >= ranges[:, 0] - 1.0e-10)
    assert np.all(joints <= ranges[:, 1] + 1.0e-10)


def test_gmr_knee_pole_uses_positive_flexion_through_hyperextension() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    config = tool.OfflineSolverConfig(
        passes_per_frame=5,
        max_joint_velocity_radians_per_second=3.0 * np.pi,
        max_joint_acceleration_radians_per_second_squared=200.0,
        knee_pole_reference_flexion_radians=np.deg2rad(50.0),
        knee_target_minimum_flexion_radians=np.deg2rad(5.0),
        knee_pole_sample_minimum_flexion_radians=np.deg2rad(20.0),
        knee_pole_sample_minimum_distance_meters=0.01,
        knee_pole_smoothing_frames=5,
    )
    thigh = np.asarray((0.0, -0.4, 0.0))
    shin = np.asarray((0.0, -0.4, 0.0))
    frames = []
    for yaw_degrees, flexion_degrees in zip(
        (0.0, 20.0, 40.0, 60.0),
        (-5.0, 0.0, 30.0, 50.0),
        strict=True,
    ):
        hip_rotation = tool.Rotation.from_euler("z", yaw_degrees, degrees=True)
        knee_rotation = hip_rotation * tool.Rotation.from_rotvec(
            (np.deg2rad(flexion_degrees), 0.0, 0.0)
        )
        hip = np.asarray((0.1, -0.2, 0.9))
        knee = hip + hip_rotation.apply(thigh)
        foot = knee + knee_rotation.apply(shin)
        frames.append(
            {
                "left_hip": [
                    hip,
                    hip_rotation.as_quat(scalar_first=True),
                ],
                "left_knee": [
                    knee,
                    knee_rotation.as_quat(scalar_first=True),
                ],
                "left_foot": [foot, np.asarray((1.0, 0.0, 0.0, 0.0))],
            }
        )

    trajectory = tool._infer_positive_knee_pole_trajectory(frames, "left", config)
    positive_reference_foot = thigh + tool.Rotation.from_rotvec(
        (config.knee_pole_reference_flexion_radians, 0.0, 0.0)
    ).apply(shin)
    positive_axis = tool._normalized_knee_pole(
        np.zeros(3), thigh, positive_reference_foot
    )
    negative_pole = tool._normalized_knee_pole(
        frames[0]["left_hip"][0],
        frames[0]["left_knee"][0],
        frames[0]["left_foot"][0],
    )
    negative_pole_local = (
        tool.Rotation.from_quat(frames[0]["left_hip"][1], scalar_first=True)
        .inv()
        .apply(negative_pole)
    )

    assert np.all(trajectory @ positive_axis > 0.95)
    assert np.all(trajectory @ negative_pole_local < 0.0)


def test_gmr_retarget_measures_canonical_sole_collision_height(
    full_body_asset,
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = mujoco.MjModel.from_xml_path(str(full_body_asset.mjcf_path))
    qpos = np.repeat(model.qpos0[None, :], 2, axis=0)
    qpos[1, 2] += 0.125

    sole_z = tool._foot_sole_z(model, qpos)

    assert sole_z.shape == (2, 2)
    np.testing.assert_allclose(sole_z[1] - sole_z[0], 0.125, atol=1.0e-8)


def test_gmr_grounding_preserves_flight_and_prevents_swing_foot_penetration() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    qpos = np.zeros((3, 7))
    qpos[:, 2] = 1.0
    sole_z = np.asarray([[0.1, 0.2], [0.05, 0.4], [0.3, 0.2]])
    contacts = np.asarray([[True, False], [False, False], [False, True]])

    grounded_qpos, grounded_sole = tool._ground_root_to_support(qpos, sole_z, contacts)

    np.testing.assert_allclose(grounded_sole, [[0.0, 0.1], [0.0, 0.35], [0.1, 0.0]])
    np.testing.assert_allclose(grounded_qpos[:, 2], [0.9, 0.95, 0.8])


def test_twist_observation_contract_exact_dimensions_and_isolation() -> None:
    contract = TwistObservationContract(
        joint_names=JOINT_NAMES,
        randomization_size=5 + 2 * len(JOINT_NAMES),
    )

    assert len(JOINT_NAMES) == 25
    assert contract.student_reference_frame_size == 33
    assert contract.teacher_reference_frame_size == 60
    assert contract.proprioception_frame_size == 80
    assert contract.teacher_observation_size == 1368
    assert contract.student_observation_size == 1243
    assert contract.critic_observation_size == 1368
    assert TWIST_HISTORY_FRAMES == 11
    assert contract.actor_layout("student").components == (
        ("reference_proprioception_history", 1243),
    )
    critic = jnp.arange(contract.critic_observation_size)
    teacher = teacher_observation_from_student_critic(critic, contract)
    assert teacher.shape == (contract.teacher_observation_size,)
    assert not any(
        name == "future_reference"
        for name, _ in contract.actor_layout("student").components
    )


def test_twist_runtime_does_not_depend_on_motion_tracking_pipeline() -> None:
    repository = Path(__file__).resolve().parents[1]
    for relative_path in (
        "src/bello_mujoco/envs/twist.py",
        "src/bello_mujoco/training/twist.py",
        "src/bello_mujoco/evaluation/twist.py",
        "configs/train_twist/default.yaml",
    ):
        source = (repository / relative_path).read_text(encoding="utf-8").lower()
        assert "beyondmimic" not in source
        assert "motiontrackingenv" not in source
        assert "motion_tracking_asset_config" not in source
        assert "motion_env" not in source


def test_twist_initial_step_counters_do_not_alias_donated_buffers() -> None:
    class Replaceable(SimpleNamespace):
        def replace(self, **changes):
            return Replaceable(**{**vars(self), **changes})

    step_count = jnp.arange(4, dtype=jnp.int32)
    state = Replaceable(
        motion_state=Replaceable(step_count=None, episode_length=None),
        total_step=None,
    )

    initialized = BelloTwistEnv.with_initial_step_counts(None, state, step_count)
    counters = (
        initialized.motion_state.step_count,
        initialized.motion_state.episode_length,
        initialized.total_step,
    )

    assert len({counter.unsafe_buffer_pointer() for counter in counters}) == 3
    for counter in counters:
        np.testing.assert_array_equal(counter, np.arange(4, dtype=np.int32))


@pytest.mark.parametrize(
    ("stage", "observation_size"),
    (("teacher", 1368), ("student", 1243)),
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
        action_distribution="gaussian",
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        observation_size,
        len(JOINT_NAMES),
        config,
        critic_obs_size=1368,
    )

    mean, log_std, value = eqx.filter_jit(policy)(
        jnp.zeros(observation_size), jnp.zeros(1368)
    )

    assert mean.shape == (25,)
    assert log_std.shape == (25,)
    assert value.shape == ()
    assert policy.actor.backbone.hidden[-1].out_features == 128


def test_twist_student_encoder_reads_newest_current_reference() -> None:
    config = ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        architecture="twist_student",
        layer_norm_final_hidden=True,
        action_distribution="gaussian",
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(3), 1243, len(JOINT_NAMES), config, critic_obs_size=1368
    )
    observation = jnp.arange(1243, dtype=jnp.float32)
    frame_size = 33 + 80

    reference = policy.actor._reference_observation(observation)

    np.testing.assert_array_equal(
        reference,
        observation[10 * frame_size : 10 * frame_size + 33],
    )


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
        action_distribution="gaussian",
    )
    teacher_config = replace(student_config, architecture="twist_teacher")
    student = init_actor_critic(
        jax.random.PRNGKey(1), 1243, 25, student_config, critic_obs_size=1368
    )
    teacher = init_actor_critic(
        jax.random.PRNGKey(2), 1368, 25, teacher_config, critic_obs_size=1368
    )
    student = eqx.tree_at(lambda policy: policy.log_std, student, jnp.full((25,), -0.5))
    teacher = eqx.tree_at(lambda policy: policy.log_std, teacher, jnp.full((25,), 0.25))
    student_obs = jnp.zeros((2, 1243))
    teacher_obs = jnp.zeros((2, 1368))

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
        "foot_sole_z": np.zeros((frames, 2), dtype=np.float32),
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
    penetrating = {name: value.copy() for name, value in arrays.items()}
    penetrating["foot_sole_z"][0, 0] = -0.011
    assert "foot sole penetration exceeds configured limit" in clip_quality_reasons(
        penetrating, quality
    )
    discontinuous = {name: value.copy() for name, value in arrays.items()}
    discontinuous["qpos"][50:, 7] = 0.3
    discontinuity_reasons = clip_quality_reasons(discontinuous, quality)
    assert "joint trajectory contains a discontinuous frame step" in (
        discontinuity_reasons
    )
    assert "joint velocity exceeds configured limit" in discontinuity_reasons
    assert "joint acceleration exceeds configured limit" in discontinuity_reasons


def test_qc_rejects_persistent_hip_and_waist_limit_saturation() -> None:
    frames = 100
    quality = QualityConfig()
    names = tuple(quality.persistent_limit_joint_names) + tuple(
        f"other_{index}"
        for index in range(25 - len(quality.persistent_limit_joint_names))
    )
    lower = np.full(25, -2.0, dtype=np.float32)
    upper = np.full(25, 2.0, dtype=np.float32)
    qpos = np.zeros((frames, 32), dtype=np.float32)
    qpos[:, 2] = 0.9
    qpos[:, 3] = 1.0
    qpos[:, 7:10] = lower[:3]
    arrays = {
        "timestamp_seconds": np.arange(frames, dtype=np.float64) / 50.0,
        "source_timestamp_seconds": np.arange(frames, dtype=np.float64) / 50.0,
        "qpos": qpos,
        "qvel": np.zeros((frames, 31), dtype=np.float32),
        "body_pos": np.zeros((frames, 14, 3), dtype=np.float32),
        "body_quat_wxyz": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 14, 1)),
        "body_lin_vel": np.zeros((frames, 14, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((frames, 14, 3), dtype=np.float32),
        "foot_contacts": np.zeros((frames, 2), dtype=np.float32),
        "foot_sole_z": np.zeros((frames, 2), dtype=np.float32),
        "qpos_joint_names": np.asarray(names),
        "joint_lower": lower,
        "joint_upper": upper,
    }

    reasons = clip_quality_reasons(arrays, quality)

    assert any(
        "persistent hip/waist joint-limit saturation" in item for item in reasons
    )


def test_knee_pole_qc_ignores_the_undefined_near_extension_plane() -> None:
    frames = 20
    body_names = (
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    )
    positions = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    for side, lateral in (("left", 0.1), ("right", -0.1)):
        positions[:, body_names.index(f"{side}_hip_roll_link")] = (lateral, 0.0, 0.8)
        positions[:, body_names.index(f"{side}_knee_link")] = (lateral, -0.05, 0.45)
        positions[:, body_names.index(f"{side}_ankle_roll_link")] = (
            lateral,
            0.0,
            0.1,
        )
    arrays = {
        "qpos": np.zeros((frames, 7 + len(JOINT_NAMES)), dtype=np.float32),
        "qpos_joint_names": np.asarray(JOINT_NAMES),
        "body_names": np.asarray(body_names),
        "body_pos": positions,
        "knee_pole_world_direction": np.tile(
            np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32),
            (frames, 1, 1),
        ),
    }
    quality = QualityConfig()

    assert _knee_pole_quality_reasons(arrays, quality) == []

    bent = {name: value.copy() for name, value in arrays.items()}
    for side in ("left", "right"):
        index = JOINT_NAMES.index(f"{side}_knee_joint")
        bent["qpos"][:, 7 + index] = np.deg2rad(20.0)
    reasons = _knee_pole_quality_reasons(bent, quality)
    assert "left knee follows the opposite bend branch" in reasons
    assert "right knee follows the opposite bend branch" in reasons


def test_floor_alignment_uses_lowest_canonical_sole_point() -> None:
    frames = 3
    arrays = {
        "qpos": np.tile(
            np.concatenate(([0.0, 0.0, 0.9], [1.0, 0.0, 0.0, 0.0], np.zeros(25))),
            (frames, 1),
        ),
        "qvel": np.zeros((frames, 31)),
        "body_pos": np.zeros((frames, 2, 3)),
        "body_quat_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 2, 1)),
        "body_lin_vel": np.zeros((frames, 2, 3)),
        "foot_sole_z": np.asarray([[0.18, 0.16], [0.12, 0.2], [0.14, 0.13]]),
    }

    aligned = _align_floor_and_heading(arrays)

    assert aligned["foot_sole_z"].min() == pytest.approx(0.0)
    np.testing.assert_allclose(aligned["qpos"][:, 2], 0.78)


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
    np.savez(tmp_path / "train.npz", **arrays)
    clips = [
        {
            "source_id": "amass",
            "sequence_id": f"clip_{index}",
            "split": "train",
            "offset": sum(lengths[:index]),
            "frames": frames,
            "sampling_weight": 1.0,
            "sampling_probability_numerator": float(frames),
        }
        for index, frames in enumerate(lengths)
    ]
    index_path = tmp_path / "index.json"
    index_path.write_text(
        json.dumps({"clips": clips, "split_files": {"train": "train.npz"}}),
        encoding="utf-8",
    )

    bank = load_packed_reference_bank(index_path, split="train")

    assert bank.qpos.shape == (total_frames, 4)
    np.testing.assert_array_equal(bank.offsets, (0, lengths[0]))
    np.testing.assert_array_equal(bank.lengths, lengths)
    np.testing.assert_array_equal(bank.qpos, qpos)
    np.testing.assert_allclose(
        bank.sampling_probabilities,
        np.asarray(lengths, dtype=np.float64) / sum(lengths),
    )
    release_style = load_packed_reference_bank(
        index_path, split="train", sampling_mode="manifest_weight"
    )
    np.testing.assert_allclose(release_style.sampling_probabilities, (0.5, 0.5))


def test_released_motion_curriculum_updates_weights_and_completion() -> None:
    bank = PackedReferenceBank(
        qpos=np.zeros((300, 1), dtype=np.float32),
        qvel=np.zeros((300, 1), dtype=np.float32),
        body_pos=np.zeros((300, 1, 3), dtype=np.float32),
        body_quat_wxyz=np.zeros((300, 1, 4), dtype=np.float32),
        body_lin_vel=np.zeros((300, 1, 3), dtype=np.float32),
        body_ang_vel=np.zeros((300, 1, 3), dtype=np.float32),
        foot_contacts=np.zeros((300, 2), dtype=np.float32),
        offsets=np.asarray((0, 100, 200), dtype=np.int32),
        lengths=np.asarray((100, 100, 100), dtype=np.int32),
        sampling_probabilities=np.asarray((0.8, 0.1, 0.1), dtype=np.float32),
        source_ids=("a", "b", "c"),
        sequence_ids=("a", "b", "c"),
    )
    env = object.__new__(BelloTwistEnv)
    object.__setattr__(env, "reference_bank", bank)
    object.__setattr__(env, "sampling", TwistSamplingConfig())
    for name in ("model", "motion", "stage", "config"):
        object.__setattr__(env, name, None)
    difficulty = jnp.asarray((5.0, 5.0, 5.0), dtype=jnp.float32)
    np.testing.assert_array_equal(env.initial_task_state(), (9.0, 9.0, 9.0))
    info = {
        "reference/clip_index": jnp.asarray((0.0, 1.0, 1.0)),
        "returned_episode": jnp.asarray((True, True, True)),
        "episode_length": jnp.asarray((40, 100, 90)),
    }
    updated = jax.jit(lambda value: env.update_task_state(value, info))(difficulty)

    np.testing.assert_allclose(updated, (5.05, 4.95, 5.0), rtol=1.0e-6)
    probabilities = jax.jit(
        lambda value: env.with_task_state(value).reference_bank.sampling_probabilities
    )(jnp.asarray((1.0, 8.0, 8.0)))
    np.testing.assert_allclose(
        probabilities,
        np.asarray((0.8, 0.8, 0.8)) / 2.4,
    )


def test_reference_bank_bounds_require_contiguous_positive_clips() -> None:
    offsets = np.asarray((0, 100), dtype=np.int32)
    lengths = np.asarray((100, 137), dtype=np.int32)
    probabilities = np.asarray((0.5, 0.5), dtype=np.float32)

    _validate_reference_bank_clip_bounds(offsets, lengths, probabilities, 237)

    with pytest.raises(ValueError, match="do not cover"):
        _validate_reference_bank_clip_bounds(
            np.asarray((0, 101)),
            lengths,
            probabilities,
            237,
        )


def test_reference_bank_arrays_remain_dynamic_jit_inputs() -> None:
    def bank(value: float) -> PackedReferenceBank:
        scalar = np.full((4, 1), value, dtype=np.float32)
        return PackedReferenceBank(
            qpos=scalar,
            qvel=scalar,
            body_pos=np.full((4, 1, 3), value, dtype=np.float32),
            body_quat_wxyz=np.full((4, 1, 4), value, dtype=np.float32),
            body_lin_vel=np.full((4, 1, 3), value, dtype=np.float32),
            body_ang_vel=np.full((4, 1, 3), value, dtype=np.float32),
            foot_contacts=np.full((4, 2), value, dtype=np.float32),
            offsets=np.asarray((0, 2), dtype=np.int32),
            lengths=np.asarray((2, 2), dtype=np.int32),
            sampling_probabilities=np.asarray((0.5, 0.5), dtype=np.float32),
            source_ids=("source-a", "source-b"),
            sequence_ids=("sequence-a", "sequence-b"),
        )

    static_env = object.__new__(BelloTwistEnv)
    object.__setattr__(static_env, "model", None)
    object.__setattr__(static_env, "motion", None)
    object.__setattr__(static_env, "stage", "teacher")
    object.__setattr__(static_env, "config", TwistEnvConfig())
    object.__setattr__(static_env, "reference_bank", None)
    object.__setattr__(static_env, "sampling", TwistSamplingConfig())

    @jax.jit
    def bank_sum(arrays):
        bound = _bind_reference_bank(static_env, arrays)
        return jnp.sum(bound.reference_bank.qpos)

    first = _device_reference_bank_arrays(bank(1.0))
    second = _device_reference_bank_arrays(bank(2.0))
    assert float(bank_sum(first)) == pytest.approx(4.0)
    assert float(bank_sum(second)) == pytest.approx(8.0)
    assert bank_sum._cache_size() == 1


def test_physics_diagnostics_report_peak_capacity_and_overflow_fraction() -> None:
    diagnostics = physics_diagnostics(
        {
            "physics/contact_capacity_fraction": jnp.asarray((0.25, 0.75)),
            "physics/contact_capacity_overflow": jnp.asarray((False, True)),
            "unrelated": jnp.asarray((100.0,)),
        }
    )

    assert diagnostics.keys() == {
        "max_physics/contact_capacity_fraction",
        "mean_physics/contact_capacity_overflow",
    }
    assert diagnostics["max_physics/contact_capacity_fraction"] == pytest.approx(0.75)
    assert diagnostics["mean_physics/contact_capacity_overflow"] == pytest.approx(0.5)


def test_packed_build_fingerprint_and_heldout_manifest_are_reproducible(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    robot_path = (
        repository
        / "deps/bello-stack/descriptions/bello/full_body/urdf/bello_full_body.urdf"
    )
    gmr_root = repository / "deps/GMR"
    config_path = repository / (
        "deps/GMR/general_motion_retargeting/ik_configs/smplx_to_bello.json"
    )
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
        body_position = np.zeros((frames, len(body_names), 3), dtype=np.float32)
        for side, lateral in (("left", 0.1), ("right", -0.1)):
            body_position[:, body_names.index(f"{side}_hip_roll_link")] = (
                lateral,
                0.0,
                0.8,
            )
            body_position[:, body_names.index(f"{side}_knee_link")] = (
                lateral,
                0.05,
                0.45,
            )
            body_position[:, body_names.index(f"{side}_ankle_roll_link")] = (
                lateral,
                0.0,
                0.1,
            )
        knee_pole = np.tile(
            np.asarray(((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))),
            (frames, 1, 1),
        ).astype(np.float32)
        np.savez_compressed(
            retargeted / f"fixture__{sequence_id}.npz",
            timestamp_seconds=np.arange(frames, dtype=np.float64) / 50.0,
            source_timestamp_seconds=np.arange(frames, dtype=np.float64) / 50.0,
            source_sha256=np.asarray(source_digest),
            retarget_format_version=np.asarray(9, dtype=np.int32),
            qpos=qpos,
            qvel=np.zeros((frames, 31), dtype=np.float32),
            body_pos=body_position,
            body_quat_wxyz=body_quat,
            body_lin_vel=np.zeros((frames, len(body_names), 3), dtype=np.float32),
            body_ang_vel=np.zeros((frames, len(body_names), 3), dtype=np.float32),
            foot_contacts=np.zeros((frames, 2), dtype=np.float32),
            foot_sole_z=np.zeros((frames, 2), dtype=np.float32),
            knee_pole_local_axis=knee_pole,
            knee_pole_world_direction=knee_pole,
            knee_pole_sides=np.asarray(("left", "right")),
            joint_names=np.asarray(JOINT_NAMES),
            qpos_joint_names=np.asarray(JOINT_NAMES),
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
    assert first["schema_version"] == TWIST_DATASET_SCHEMA_VERSION
    assert first["train_sampling"] == TWIST_TRAIN_SAMPLING
    assert first["split_files"] == {
        "train": "train.npz",
        "evaluation": "evaluation.npz",
    }
    assert {clip["split"] for clip in first["clips"]} == {
        "train",
        "evaluation",
    }
    with np.load(tmp_path / "packed_first/train.npz", allow_pickle=False) as packed:
        assert packed["qpos"].dtype == np.float32
    assert (tmp_path / "packed_first/heldout_manifest.json").read_bytes() == (
        tmp_path / "packed_second/heldout_manifest.json"
    ).read_bytes()


def test_typed_twist_checkpoints_validate_and_reject_cross_stage_resume() -> None:
    contract = TwistObservationContract(
        joint_names=JOINT_NAMES,
        randomization_size=55,
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
                action_distribution="gaussian",
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
        "simulation": TwistSimulationConfig(),
        "twist_env": TwistEnvConfig(),
        "sampling": TwistSamplingConfig(),
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
    revised_teacher = replace(teacher, source_revision="5" * 40)
    with pytest.raises(ValueError, match="source_revision"):
        resumed_checkpoint_metadata(teacher, revised_teacher, saved_update=0)
    assert (
        resumed_checkpoint_metadata(
            teacher,
            revised_teacher,
            saved_update=0,
            allow_source_revision_mismatch=True,
        )
        == revised_teacher
    )
    with pytest.raises(ValueError, match="warp"):
        resumed_checkpoint_metadata(
            teacher,
            replace(
                revised_teacher,
                warp=WarpConfig(contact_budget_per_env=64),
            ),
            saved_update=0,
            allow_source_revision_mismatch=True,
        )
