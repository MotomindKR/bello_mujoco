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
from bello_mujoco.robot import robot_definition
from bello_mujoco.robot_tasks import motion_task_definition
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

MOTION_PROFILE = motion_task_definition("bello")
JOINT_NAMES = MOTION_PROFILE.imported_actuator_joint_names
GMR_KNEE_POLE = MOTION_PROFILE.gmr.knee_pole


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
            robot = repository / ("deps/bello-stack/descriptions/bello/urdf/bello.urdf")
            revision = subprocess.run(
                ["git", "-C", str(repository / "deps/GMR"), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            np.savez(
                output,
                source_sha256=np.asarray(sequence.source_sha256),
                retarget_format_version=np.asarray(10, dtype=np.int32),
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
        "asset": "bello",
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
    bello_asset,
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = mujoco.MjModel.from_xml_path(str(bello_asset.mjcf_path))

    names = logical_actuator_joint_names(model, robot_definition("bello"))
    qpos_names = tool._qpos_ordered_joint_names(model, names)
    ranges = tool._qpos_ordered_joint_ranges(model, names)
    qpos_joint_ids = sorted(
        range(1, model.njnt), key=lambda joint_id: int(model.jnt_qposadr[joint_id])
    )

    assert names == logical_actuator_joint_names(model, robot_definition("bello"))
    assert qpos_names == tuple(
        model.joint(joint_id).name for joint_id in qpos_joint_ids
    )
    assert names != qpos_names
    np.testing.assert_array_equal(ranges, model.jnt_range[qpos_joint_ids])


@pytest.mark.parametrize(
    ("asset", "use_velocity_limit", "joint_smoothing_passes"),
    (
        pytest.param("bello", True, 3, id="bello"),
        pytest.param("bello_mini", False, 14, id="bello-mini"),
    ),
)
def test_gmr_profile_uses_current_pinned_offline_solver_contract(
    asset: str,
    use_velocity_limit: bool,
    joint_smoothing_passes: int,
) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = motion_task_definition(asset).gmr
    document = json.loads(profile.config_path(root / "deps/GMR").read_text())

    assert profile.task_profile == document["default_task_profile"] == "universal"
    assert document["offline_solver"]["use_velocity_limit"] is use_velocity_limit
    assert document["offline_solver"]["passes_per_frame"] == 1
    assert document["offline_solver"]["initial_settle_passes"] == 20
    assert (
        document["offline_solver"]["joint_smoothing_passes"] == joint_smoothing_passes
    )
    assert (
        "max_joint_acceleration_radians_per_second_squared"
        not in document["offline_solver"]
    )


def test_bello_mini_gmr_live_upper_body_profile_uses_observable_arm_frames() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = motion_task_definition("bello_mini").gmr
    document = json.loads(profile.config_path(root / "deps/GMR").read_text())

    assert profile.task_profile == document["default_task_profile"] == "universal"
    assert document["task_profiles"] == {
        "universal": {
            "ik_match_table2": {
                "left_wrist_roll_link": {
                    "position_cost": 30,
                    "orientation_cost": 10,
                },
                "right_wrist_roll_link": {
                    "position_cost": 30,
                    "orientation_cost": 10,
                },
            }
        },
        "live_upper_body": {
            "ik_match_table2": {
                "left_elbow_link": {
                    "position_cost": 30,
                },
                "left_wrist_roll_link": {
                    "position_cost": 50,
                    "orientation_cost": 10,
                },
                "right_elbow_link": {
                    "position_cost": 30,
                },
                "right_wrist_roll_link": {
                    "position_cost": 50,
                    "orientation_cost": 10,
                },
            },
            "posture_costs": {
                "left_shoulder_yaw_joint": 10,
                "left_wrist_yaw_joint": 10,
                "left_wrist_roll_joint": 10,
                "right_shoulder_yaw_joint": 10,
                "right_wrist_yaw_joint": 10,
                "right_wrist_roll_joint": 10,
            },
        },
    }


def test_bello_mini_live_upper_body_profile_loads_generated_model(
    bello_mini_asset,
) -> None:
    from bello_mujoco.hardware.smplx_retarget import (
        _configure_pinned_gmr_profile,
        import_pinned_gmr_retargeter,
    )

    root = Path(__file__).resolve().parents[1]
    profile = motion_task_definition("bello_mini").gmr
    _configure_pinned_gmr_profile(profile, root / "deps/GMR")
    retargeter_type = import_pinned_gmr_retargeter()
    retargeter = retargeter_type(
        src_human=profile.source_human,
        tgt_robot=profile.target_robot,
        actual_human_height=1.75,
        solver="daqp",
        damping=5.0e-1,
        verbose=False,
        use_velocity_limit=None,
        robot_xml_path=bello_mini_asset.mjcf_path,
        task_profile="live_upper_body",
    )

    assert retargeter.task_profile == "live_upper_body"
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            retargeter.human_body_to_task1[f"{side}_shoulder"].cost,
            [0, 0, 0, 5, 5, 5],
        )
        np.testing.assert_array_equal(
            retargeter.human_body_to_task2[f"{side}_shoulder"].cost,
            [10, 10, 10, 5, 5, 5],
        )
        np.testing.assert_array_equal(
            retargeter.human_body_to_task2[f"{side}_elbow"].cost,
            [30, 30, 30, 5, 5, 5],
        )
        np.testing.assert_array_equal(
            retargeter.human_body_to_task2[f"{side}_wrist"].cost,
            [50, 50, 50, 10, 10, 10],
        )
    np.testing.assert_array_equal(
        retargeter.profile_posture_task.target_q,
        retargeter.model.qpos0,
    )
    expected_posture_cost = np.zeros(retargeter.model.nv)
    for joint_name in (
        "left_shoulder_yaw_joint",
        "left_wrist_yaw_joint",
        "left_wrist_roll_joint",
        "right_shoulder_yaw_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
    ):
        expected_posture_cost[retargeter.model.joint(joint_name).dofadr[0]] = 10
    np.testing.assert_array_equal(
        retargeter.profile_posture_task.cost,
        expected_posture_cost,
    )


def test_bello_mini_gmr_follows_unitree_g1_task_schedule() -> None:
    root = Path(__file__).resolve().parents[1] / "deps/GMR"
    mini = json.loads(
        (
            root / "general_motion_retargeting/ik_configs/smplx_to_bello_mini.json"
        ).read_text()
    )
    g1 = json.loads(
        (root / "general_motion_retargeting/ik_configs/smplx_to_g1.json").read_text()
    )

    expected_scales = {
        "pelvis": 0.9,
        "spine3": 0.9,
        "left_hip": 0.9,
        "right_hip": 0.9,
        "left_knee": 0.9,
        "right_knee": 0.9,
        "left_foot": 0.9,
        "right_foot": 0.9,
        "left_shoulder": 0.8,
        "right_shoulder": 0.8,
        "left_elbow": 0.8,
        "right_elbow": 0.8,
        "left_wrist": 0.8,
        "right_wrist": 0.8,
    }
    assert mini["human_height_assumption"] == g1["human_height_assumption"] == 1.8
    assert mini["human_scale_table"] == g1["human_scale_table"] == expected_scales
    assert "initial_joint_positions" not in mini
    assert "velocity_limit_tendon_joint_names" not in mini

    corresponding_frames = {
        "bello_mini_root": "pelvis",
        "left_hip_roll_link": "left_hip_roll_link",
        "left_knee_link": "left_knee_link",
        "left_ankle_roll_link": "left_toe_link",
        "right_hip_roll_link": "right_hip_roll_link",
        "right_knee_link": "right_knee_link",
        "right_ankle_roll_link": "right_toe_link",
        "torso_link": "torso_link",
        "left_upper_arm_link": "left_shoulder_yaw_link",
        "left_elbow_link": "left_elbow_link",
        "left_wrist_roll_link": "left_wrist_yaw_link",
        "right_upper_arm_link": "right_shoulder_yaw_link",
        "right_elbow_link": "right_elbow_link",
        "right_wrist_roll_link": "right_wrist_yaw_link",
    }
    stage2 = mini["ik_match_table2"]
    g1_stage2 = g1["ik_match_table2"]
    assert set(stage2) == set(corresponding_frames)
    for mini_frame, g1_frame in corresponding_frames.items():
        assert stage2[mini_frame][:3] == g1_stage2[g1_frame][:3]

    for side in ("left", "right"):
        wrist_frame = f"{side}_wrist_roll_link"
        assert mini["ik_match_table1"][wrist_frame][0] == f"{side}_wrist"
        assert stage2[wrist_frame][0] == f"{side}_wrist"
        assert f"{side}_end_effector_sphere_link" not in mini["ik_match_table1"]
        assert f"{side}_end_effector_sphere_link" not in stage2

    stage1 = mini["ik_match_table1"]
    g1_stage1 = g1["ik_match_table1"]
    g1_aligned_stage1_frames = {
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    }
    branch_stability_exceptions = {
        "bello_mini_root": ["pelvis", 100, 20],
        "torso_link": ["spine3", 0, [25, 25, 0]],
        "left_upper_arm_link": ["left_shoulder", 0, 5],
        "left_elbow_link": ["left_elbow", 0, 0],
        "left_wrist_roll_link": ["left_wrist", 0, 0],
        "right_upper_arm_link": ["right_shoulder", 0, 5],
        "right_elbow_link": ["right_elbow", 0, 0],
        "right_wrist_roll_link": ["right_wrist", 0, 0],
    }
    assert set(stage1) == set(corresponding_frames)
    assert set(corresponding_frames) == (
        g1_aligned_stage1_frames | set(branch_stability_exceptions)
    )
    for mini_frame in g1_aligned_stage1_frames:
        g1_frame = corresponding_frames[mini_frame]
        assert stage1[mini_frame][:3] == g1_stage1[g1_frame][:3]
    for mini_frame, expected in branch_stability_exceptions.items():
        assert stage1[mini_frame][:3] == expected
        g1_frame = corresponding_frames[mini_frame]
        assert stage1[mini_frame][:3] != g1_stage1[g1_frame][:3]


def test_bello_mini_gmr_profile_references_generated_model(
    bello_mini_asset,
) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = motion_task_definition("bello_mini").gmr
    document = json.loads(profile.config_path(root / "deps/GMR").read_text())
    model = mujoco.MjModel.from_xml_path(str(bello_mini_asset.mjcf_path))
    robot = robot_definition("bello_mini")

    assert profile.target_robot == robot.asset == "bello_mini"
    assert document["robot_root_name"] == robot.root_body_name
    assert document["human_root_name"] in profile.source_body_names
    assert set(document["human_scale_table"]) == set(profile.source_body_names)

    referenced_body_names = {
        document["robot_root_name"],
        *profile.output_body_names,
    }
    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = document[table_name]
        assert {match[0] for match in table.values()} == set(profile.source_body_names)
        referenced_body_names.update(table)
        for match in table.values():
            assert len(match) == 5
            _, position_cost, orientation_cost, position_offset, quaternion = match
            for cost in (position_cost, orientation_cost):
                cost_array = np.asarray(cost, dtype=float)
                assert cost_array.shape in ((), (3,))
                assert np.all(np.isfinite(cost_array))
                assert np.all(cost_array >= 0.0)
            assert np.asarray(position_offset).shape == (3,)
            assert np.all(np.isfinite(position_offset))
            assert np.asarray(quaternion).shape == (4,)
            np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1.0e-6)

    protected_bodies = document["offline_solver"]["protected_bodies"]
    referenced_body_names.update(item["body_name"] for item in protected_bodies)

    collision = document["collision_avoidance"]
    assert collision["enabled"] is True
    for selector_pair in collision["geom_pairs"]:
        assert len(selector_pair) == 2
        for selectors in selector_pair:
            assert selectors
            for selector in selectors:
                selector_type, separator, body_name = selector.partition(":")
                assert (selector_type, separator) == ("body", ":")
                assert body_name
                referenced_body_names.add(body_name)

    assert "planar_relative_yaw_task" not in document

    for body_name in referenced_body_names:
        assert model.body(body_name).id >= 0

    actuator_by_joint = {actuator.joint_name: actuator for actuator in robot.actuators}
    referenced_joint_names = {
        *profile.standing_default_joint_names,
        *(leg.target_knee_joint_name for leg in profile.knee_pole.legs),
    }
    for task_profile in document["task_profiles"].values():
        referenced_joint_names.update(task_profile.get("posture_costs", ()))
    assert set(profile.standing_default_joint_names).issubset(actuator_by_joint)
    for joint_name in referenced_joint_names:
        assert model.joint(joint_name).id >= 0

    ground_geoms = tuple(document["ground_clearance_geoms"])
    assert len(ground_geoms) == len(robot.foot_support_geometries)
    for geom_name, support in zip(
        ground_geoms, robot.foot_support_geometries, strict=True
    ):
        assert geom_name in support.geom_names
        geom_id = model.geom(geom_name).id
        assert model.geom_bodyid[geom_id] == model.body(support.body_name).id


def test_bello_mini_gmr_default_stance_places_soles_on_ground(
    bello_mini_asset,
) -> None:
    root = Path(__file__).resolve().parents[1]
    profile = motion_task_definition("bello_mini").gmr
    document = json.loads(profile.config_path(root / "deps/GMR").read_text())
    model = mujoco.MjModel.from_xml_path(str(bello_mini_asset.mjcf_path))
    robot = robot_definition("bello_mini")
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    root_joint = model.joint(robot.root_joint_name)
    root_qpos_address = int(root_joint.qposadr[0])
    data.qpos[root_qpos_address : root_qpos_address + 3] = robot.default_root_position
    data.qpos[root_qpos_address + 3 : root_qpos_address + 7] = (
        robot.default_root_quaternion
    )
    for actuator in robot.actuators:
        joint = model.joint(actuator.joint_name)
        joint_id = joint.id
        data.qpos[int(joint.qposadr[0])] = actuator.default_pos
        if model.jnt_limited[joint_id]:
            lower, upper = model.jnt_range[joint_id]
            assert lower <= actuator.default_pos <= upper
    mujoco.mj_forward(model, data)
    sole_clearances = []
    sole_up_axes = []
    for geom_name in document["ground_clearance_geoms"]:
        geom_id = model.geom(geom_name).id
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        vertical_half_extent = np.abs(rotation[2]) @ model.geom_size[geom_id]
        sole_clearances.append(data.geom_xpos[geom_id, 2] - vertical_half_extent)
        sole_up_axes.append(rotation[:, 2])

    np.testing.assert_allclose(
        sole_clearances,
        document["ground_height"],
        atol=5.0e-6,
    )
    np.testing.assert_allclose(
        sole_up_axes,
        np.tile((0.0, 0.0, 1.0), (len(sole_up_axes), 1)),
        atol=1.0e-6,
    )


def test_gmr_knee_pole_uses_positive_flexion_through_hyperextension() -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    config = motion_task_definition("bello").gmr.knee_pole
    leg = config.legs[0]
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

    trajectory = tool._infer_positive_knee_pole_trajectory(frames, leg, config)
    positive_reference_foot = thigh + tool.Rotation.from_rotvec(
        (config.reference_flexion_radians, 0.0, 0.0)
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
    bello_asset,
) -> None:
    tool_path = Path(__file__).resolve().parents[1] / "tools/twist_gmr_retarget.py"
    spec = importlib.util.spec_from_file_location("twist_gmr_retarget", tool_path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    model = mujoco.MjModel.from_xml_path(str(bello_asset.mjcf_path))
    qpos = np.repeat(model.qpos0[None, :], 2, axis=0)
    qpos[1, 2] += 0.125

    profile = motion_task_definition("bello").gmr
    config = json.loads(
        profile.config_path(
            Path(__file__).resolve().parents[1] / "deps/GMR"
        ).read_text()
    )
    geom_ids = tuple(model.geom(name).id for name in config["ground_clearance_geoms"])
    sole_z = tool._foot_sole_z(model, qpos, geom_ids)

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

    assert clip_quality_reasons(arrays, quality, knee_pole=GMR_KNEE_POLE) == []
    assert _arrays_fingerprint(arrays) == _arrays_fingerprint(
        {name: value.copy() for name, value in arrays.items()}
    )
    short = {name: value[:95] for name, value in arrays.items()}
    assert any(
        "future horizon" in reason
        for reason in clip_quality_reasons(short, quality, knee_pole=GMR_KNEE_POLE)
    )
    invalid = {name: value.copy() for name, value in arrays.items()}
    invalid["qpos"][0, 0] = np.nan
    assert clip_quality_reasons(invalid, quality, knee_pole=GMR_KNEE_POLE) == [
        "qpos contains non-finite values"
    ]
    penetrating = {name: value.copy() for name, value in arrays.items()}
    penetrating["foot_sole_z"][0, 0] = -0.011
    assert "foot sole penetration exceeds configured limit" in clip_quality_reasons(
        penetrating, quality, knee_pole=GMR_KNEE_POLE
    )
    discontinuous = {name: value.copy() for name, value in arrays.items()}
    discontinuous["qpos"][50:, 7] = 0.3
    discontinuity_reasons = clip_quality_reasons(
        discontinuous, quality, knee_pole=GMR_KNEE_POLE
    )
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

    reasons = clip_quality_reasons(arrays, quality, knee_pole=GMR_KNEE_POLE)

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
        "knee_pole_sides": np.asarray(("left", "right")),
    }
    quality = QualityConfig()

    assert _knee_pole_quality_reasons(arrays, quality, GMR_KNEE_POLE) == []

    bent = {name: value.copy() for name, value in arrays.items()}
    for side in ("left", "right"):
        index = JOINT_NAMES.index(f"{side}_knee_joint")
        bent["qpos"][:, 7 + index] = np.deg2rad(20.0)
    reasons = _knee_pole_quality_reasons(bent, quality, GMR_KNEE_POLE)
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
    robot_path = repository / "deps/bello-stack/descriptions/bello/urdf/bello.urdf"
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
            retarget_format_version=np.asarray(10, dtype=np.int32),
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
        asset="bello",
        body_model_root=None,
        gmr_root=gmr_root,
    )
    second = build_packed_dataset(
        manifest_path,
        retargeted,
        tmp_path / "packed_second",
        asset="bello",
        body_model_root=None,
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
