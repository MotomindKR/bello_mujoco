from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from pathlib import Path
import hashlib
import subprocess
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bello_mujoco.contracts import ActionLayout
from bello_mujoco.envs.twist import TwistSamplingConfig, TwistSimulationConfig
from bello_mujoco.envs.twist2 import (
    Twist2EnvConfig,
    Twist2ObservationContract,
    Twist2ParityContract,
    default_twist2_asset_config,
)
from bello_mujoco.hardware.runtime import HardwareFrame
from bello_mujoco.hardware.twist2 import (
    HardwareTwist2Config,
    HardwareTwist2Environment,
    HardwareTwist2PolicyAdapter,
    ReferenceUnavailable,
    Twist2ReferenceFrame,
    control_physical_twist2,
    validate_hardware_twist2_config,
    validate_twist2_reference,
)
from bello_mujoco.hardware.smplx_retarget import (
    GmrBelloRetargeter,
    GmrLiveRetargetConfig,
    LiveGmrReferenceSource,
    SMPLX_GMR_BODY_NAMES,
    SmplxTargetFrame,
    import_pinned_gmr_retargeter,
    validate_smplx_target_frame,
)
from bello_mujoco.repository import repository_root
from bello_mujoco.motion.twist2_dataset import Twist2DatasetAudit
from bello_mujoco.policy import ActorCriticConfig, init_actor_critic
from bello_mujoco.sim import WarpConfig, load_bello_model
from bello_mujoco.training.checkpoint import (
    CheckpointNormalizationState,
    Twist2CheckpointMetadata,
    TwistDatasetContract,
)
from bello_mujoco.training.ppo import (
    PPOConfig,
    init_observation_norm_state,
    init_value_norm_state,
)


FINGERPRINT = "sha256:" + "a" * 64


@pytest.fixture(scope="module")
def twist2_hardware_adapter() -> HardwareTwist2PolicyAdapter:
    asset = default_twist2_asset_config()
    simulation = TwistSimulationConfig()
    model = load_bello_model(
        asset,
        terrain=simulation.terrain,
        load_mjx=False,
        simulation_timestep=1.0 / 500.0,
    )
    joint_names = tuple(
        model.actuator_joint_names[index] for index in model.policy_ctrl_indices
    )
    contract = Twist2ObservationContract(joint_names)
    policy_config = ActorCriticConfig(
        hidden_dims=(512, 512, 256, 128),
        activation="swish",
        obs_stack_size=1,
        action_distribution="gaussian",
        architecture="twist2",
        layer_norm_final_hidden=True,
    )
    ppo = PPOConfig(
        total_timesteps=4096 * 24,
        num_envs=4096,
        num_steps=24,
        normalize_observations=True,
        policy=policy_config,
    )
    metadata = Twist2CheckpointMetadata(
        ppo=ppo,
        asset=asset,
        warp=WarpConfig(),
        obs_size=contract.actor_observation_size,
        critic_obs_size=contract.critic_observation_size,
        action_size=25,
        actor_observation_layout=contract.actor_layout,
        critic_observation_layout=contract.critic_layout,
        action_layout=ActionLayout(joint_names),
        simulation=simulation,
        twist2_env=Twist2EnvConfig(),
        sampling=TwistSamplingConfig(
            base_distribution="manifest_weight",
            curriculum_max_difficulty=10.0,
        ),
        observation_contract=contract,
        parity_contract=Twist2ParityContract(),
        dataset=TwistDatasetContract(
            index_path=Path("dataset/index.json"),
            schema_version=3,
            dataset_fingerprint=FINGERPRINT,
            robot_fingerprint=FINGERPRINT,
            gmr_revision="b" * 40,
            joint_names=joint_names,
            body_names=contract.key_body_names,
        ),
        dataset_audit=Twist2DatasetAudit(
            clip_count=1,
            train_clip_count=1,
            evaluation_clip_count=0,
            source_counts=(("amass", 1),),
            train_source_counts=(("amass", 1),),
            unit_manifest_weights=True,
            train_unit_manifest_weights=True,
            matches_released_training_composition=False,
        ),
        robot_fingerprint=FINGERPRINT,
        source_revision="c" * 40,
    )
    policy = init_actor_critic(
        jax.random.PRNGKey(0),
        contract.actor_observation_size,
        25,
        policy_config,
        critic_obs_size=contract.critic_observation_size,
    )
    normalization = CheckpointNormalizationState(
        value=init_value_norm_state(),
        observation=init_observation_norm_state(contract.actor_observation_size),
        critic_observation=init_observation_norm_state(
            contract.critic_observation_size
        ),
    )
    env = HardwareTwist2Environment(model, metadata.twist2_env, contract)
    return HardwareTwist2PolicyAdapter(
        env, policy, metadata, normalization, HardwareTwist2Config()
    )


def hardware_frame(adapter: HardwareTwist2PolicyAdapter) -> HardwareFrame:
    target = adapter.nominal_physical_target
    return HardwareFrame(
        positions=target.copy(),
        velocities=np.zeros_like(target),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )


def reference_frame(
    adapter: HardwareTwist2PolicyAdapter,
    *,
    sequence: int = 0,
    first_joint_offset: float = 0.0,
    source_age_ms: int | None = None,
) -> Twist2ReferenceFrame:
    qpos = np.asarray(adapter.env.model.default_qpos, dtype=np.float64).copy()
    qpos[adapter.policy_qpos_indices[0]] += first_joint_offset
    return Twist2ReferenceFrame(
        qpos=qpos,
        qvel=np.zeros(adapter.env.model.mj_model.nv),
        sequence=sequence,
        source_age_ms=source_age_ms,
        received_at=time.monotonic(),
    )


def neutral_smplx_frame(*, sequence: int = 0) -> SmplxTargetFrame:
    positions = {
        "pelvis": (0.0, 0.0, 0.94),
        "spine3": (0.0, 0.0, 1.30),
        "left_hip": (0.0, 0.10, 0.90),
        "left_knee": (0.0, 0.10, 0.50),
        "left_foot": (0.08, 0.10, 0.06),
        "right_hip": (0.0, -0.10, 0.90),
        "right_knee": (0.0, -0.10, 0.50),
        "right_foot": (0.08, -0.10, 0.06),
        "left_shoulder": (0.0, 0.22, 1.42),
        "left_elbow": (0.0, 0.47, 1.42),
        "left_wrist": (0.0, 0.72, 1.42),
        "right_shoulder": (0.0, -0.22, 1.42),
        "right_elbow": (0.0, -0.47, 1.42),
        "right_wrist": (0.0, -0.72, 1.42),
    }
    identity = np.asarray((1.0, 0.0, 0.0, 0.0))
    return SmplxTargetFrame(
        transforms={
            name: (np.asarray(positions[name]), identity.copy())
            for name in SMPLX_GMR_BODY_NAMES
        },
        sequence=sequence,
        source_timestamp_ns=sequence * 33_333_333,
        source_age_ms=0,
        received_at=time.monotonic(),
    )


def test_twist2_hardware_reconstructs_exact_actor_history_contract(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    adapter.reset()

    first = adapter.step(frame, reference_frame(adapter, sequence=1))
    frame_size = adapter.env.contract.actor_frame_size
    command_size = adapter.env.contract.command_frame_size
    np.testing.assert_array_equal(
        first.observation[frame_size : frame_size * 11],
        np.zeros(frame_size * 10),
    )
    np.testing.assert_array_equal(first.observation[-command_size:], first.command)
    first_current = first.observation[:frame_size]

    second = adapter.step(
        frame,
        reference_frame(adapter, sequence=2, first_joint_offset=0.1),
    )
    current = second.observation[:frame_size]
    reset_history = np.repeat(first_current[None, :], 10, axis=0).reshape(-1)
    np.testing.assert_array_equal(
        second.observation[frame_size : frame_size * 11], reset_history
    )

    third = adapter.step(
        frame,
        reference_frame(adapter, sequence=3, first_joint_offset=0.2),
    )
    expected_history = np.repeat(current[None, :], 10, axis=0).reshape(-1)
    np.testing.assert_array_equal(
        third.observation[frame_size : frame_size * 11], expected_history
    )
    assert third.observation.shape == (1252,)
    assert third.policy_observation.shape == (1252,)
    assert third.action.shape == third.raw_action.shape == (25,)
    assert third.physical_target.shape == (25,)


def test_twist2_hardware_keeps_raw_previous_action_and_clips_motor_command(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    frame = hardware_frame(adapter)
    raw_action = np.zeros(25, dtype=np.float32)
    raw_action[0] = 11.0
    adapter._infer = lambda observation: (observation, jnp.asarray(raw_action))
    adapter.reset()

    step = adapter.step(frame, reference_frame(adapter))
    proprioception = adapter.proprioception(frame)

    assert step.raw_action[0] == pytest.approx(11.0)
    assert step.action[0] == pytest.approx(10.0)
    assert step.action_clipped
    np.testing.assert_array_equal(proprioception[-25:], raw_action)
    limits = np.asarray(adapter.env.model.mj_model.actuator_ctrlrange)
    assert np.all(step.physical_target >= limits[:, 0])
    assert np.all(step.physical_target <= limits[:, 1])


def test_twist2_reference_validation_is_fail_closed(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    config = HardwareTwist2Config(max_reference_age_ms=100)
    reference = reference_frame(adapter, source_age_ms=0)
    validate_twist2_reference(reference, adapter.env, config)

    with pytest.raises(ReferenceUnavailable, match="stale"):
        validate_twist2_reference(
            replace(reference, source_age_ms=101), adapter.env, config
        )
    with pytest.raises(ReferenceUnavailable, match="qpos"):
        validate_twist2_reference(
            replace(reference, qpos=np.zeros(2)), adapter.env, config
        )
    invalid_height = reference.qpos.copy()
    invalid_height[adapter.env.model.root_qpos_adr + 2] = 2.0
    with pytest.raises(ReferenceUnavailable, match="root height"):
        validate_twist2_reference(
            replace(reference, qpos=invalid_height), adapter.env, config
        )


def test_twist2_hardware_config_rejects_unsafe_ranges() -> None:
    validate_hardware_twist2_config(HardwareTwist2Config())
    with pytest.raises(ValueError, match="max_raw_action_abs"):
        validate_hardware_twist2_config(HardwareTwist2Config(max_raw_action_abs=9.0))
    with pytest.raises(ValueError, match="root-height bounds"):
        validate_hardware_twist2_config(
            HardwareTwist2Config(
                min_reference_root_height_m=1.0,
                max_reference_root_height_m=1.0,
            )
        )


def test_pinned_gmr_import_avoids_optional_ui_and_torch_dependencies() -> None:
    retargeter_type = import_pinned_gmr_retargeter()

    assert retargeter_type.__name__ == "GeneralMotionRetargeting"
    assert "torch" not in sys.modules
    assert "_bello_pinned_gmr.robot_motion_viewer" not in sys.modules


def test_smplx_target_contract_requires_complete_global_transforms() -> None:
    frame = neutral_smplx_frame()
    validated = validate_smplx_target_frame(frame, live=False)
    assert tuple(validated) == SMPLX_GMR_BODY_NAMES

    missing = dict(frame.transforms)
    missing.pop("left_wrist")
    with pytest.raises(ReferenceUnavailable, match="left_wrist"):
        validate_smplx_target_frame(replace(frame, transforms=missing), live=False)
    invalid = dict(frame.transforms)
    invalid["pelvis"] = (np.zeros(3), np.zeros(4))
    with pytest.raises(ReferenceUnavailable, match="pelvis.*norm"):
        validate_smplx_target_frame(replace(frame, transforms=invalid), live=False)


def test_pinned_gmr_produces_finite_standing_reference(
    twist2_hardware_adapter: HardwareTwist2PolicyAdapter,
) -> None:
    adapter = twist2_hardware_adapter
    root = repository_root()
    revision = subprocess.run(
        ("git", "-C", str(root / "deps/GMR"), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    robot_source = (
        root / "deps/bello-stack/descriptions/bello/full_body/urdf/bello_full_body.urdf"
    )
    fingerprint = "sha256:" + hashlib.sha256(robot_source.read_bytes()).hexdigest()
    metadata = replace(
        adapter.metadata,
        dataset=replace(
            adapter.metadata.dataset,
            gmr_revision=revision,
            robot_fingerprint=fingerprint,
        ),
        robot_fingerprint=fingerprint,
    )
    retargeter = GmrBelloRetargeter(
        adapter.env,
        metadata,
        GmrLiveRetargetConfig(passes_per_frame=1, settling_rounds=1),
    )

    standing = retargeter.standing(neutral_smplx_frame())

    assert standing.qpos.shape == (adapter.env.model.mj_model.nq,)
    assert standing.qvel.shape == (adapter.env.model.mj_model.nv,)
    assert np.all(np.isfinite(standing.qpos))
    assert np.all(standing.qvel == 0.0)
    assert standing.source_age_ms is None
    validate_twist2_reference(standing, adapter.env, HardwareTwist2Config())


def test_live_gmr_source_runs_retargeting_outside_policy_thread() -> None:
    source_frame = neutral_smplx_frame(sequence=1)
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    calls: list[str] = []

    class FakeStream:
        delivered = False

        def neutral(self):
            return neutral_smplx_frame()

        def wait_next(self, *, after_sequence, timeout_seconds):
            del after_sequence
            if not self.delivered:
                self.delivered = True
                return source_frame
            time.sleep(min(timeout_seconds, 0.001))
            return None

    class FakeRetargeter:
        def standing(self, _frame):
            return standing

        def retarget(self, frame):
            import threading

            calls.append(threading.current_thread().name)
            return replace(
                standing,
                sequence=frame.sequence,
                source_age_ms=0,
                received_at=time.monotonic(),
            )

        def reset_live(self):
            return None

    with LiveGmrReferenceSource(
        FakeStream(), FakeRetargeter(), poll_timeout_seconds=0.01
    ) as source:
        result = source.wait_until_ready(1.0)

        assert source.standing().sequence == 0
        assert source.latest().sequence == result.sequence == 1

    assert calls == ["bello-live-gmr"]


def test_twist2_control_uses_neutral_standing_and_returns_on_source_loss(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from bello_mujoco.hardware import twist2 as twist2_hardware

    coordinate_names = ("left", "right")
    frame = HardwareFrame(
        positions=np.zeros(2),
        velocities=np.zeros(2),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0)),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.zeros(3),
        max_motor_age_ms=0,
        imu_age_ms=0,
        max_group_age_ms=0,
        sample_span_ms=0.0,
        state_frame_round_trip_ms=0.0,
        group_states=("RUNNING", "RUNNING"),
        group_causes=("UNSPECIFIED", "UNSPECIFIED"),
        group_details=("", ""),
    )
    standing = Twist2ReferenceFrame(
        qpos=np.zeros(1),
        qvel=np.zeros(1),
        sequence=0,
        source_age_ms=None,
        received_at=0.0,
    )
    live = replace(standing, sequence=1)
    seen_sequences: list[int] = []
    resets: list[None] = []

    class FakeAdapter:
        env = SimpleNamespace()
        physical_coordinate_names = coordinate_names
        coordinate_index = {"left": 0, "right": 1}
        nominal_physical_target = np.zeros(2)
        kp = np.ones(2)
        kd = np.ones(2)
        control_timestep = 0.02

        def reset(self):
            resets.append(None)

        def step(self, _frame, reference):
            seen_sequences.append(reference.sequence)
            values = np.zeros(2)
            return SimpleNamespace(
                physical_target=values,
                observation=values,
                policy_observation=values,
                raw_action=values,
                action=values,
                command=values,
                action_clipped=False,
                reference_sequence=reference.sequence,
                reference_age_ms=None,
            )

    class FakeReferences:
        calls = 0

        def latest(self):
            self.calls += 1
            if self.calls == 1:
                return live
            raise ReferenceUnavailable("tracking lost")

    class FakeControl:
        previous_target = np.zeros(2)

        def ramp_to_nominal(self, _positions):
            return frame

        def wait_for_run(self, _prompt, *, operator_input, validate_hold):
            del operator_input
            validate_hold(frame)
            return True

        def read_frame(self, *, max_tilt_radians=None):
            del max_tilt_radians
            return frame

        def send_target(self, target):
            self.previous_target = np.asarray(target)

        def pace(self):
            return None

        def reset_deadline(self):
            return None

        def drain(self):
            return None

    control = FakeControl()
    monkeypatch.setattr(
        twist2_hardware,
        "armed_control_session",
        lambda *_args, **_kwargs: nullcontext(control),
    )

    def confirm_motion(expected, _prompt, hold_step, _operator_input):
        assert expected == "MOTION"
        assert hold_step()
        return True

    monkeypatch.setattr(
        twist2_hardware,
        "wait_for_operator_confirmation_while_holding",
        confirm_motion,
    )
    monkeypatch.setattr(
        twist2_hardware,
        "validate_twist2_reference",
        lambda *_args, **_kwargs: None,
    )

    control_physical_twist2(
        SimpleNamespace(),
        {},
        SimpleNamespace(),
        FakeReferences(),
        standing,
        frame,
        FakeAdapter(),
        HardwareTwist2Config(
            max_duration_seconds=0.05,
            policy_transition_seconds=0.001,
        ),
    )

    assert seen_sequences[:2] == [0, 1]
    assert seen_sequences[-1] == 0
    assert len(resets) >= 3
    output = capsys.readouterr().out
    assert "reference became unavailable" in output
    assert "neutral standing resumed" in output
